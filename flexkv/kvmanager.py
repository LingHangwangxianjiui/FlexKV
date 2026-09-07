# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
from typing import Optional, Tuple, List, Dict, Union, Iterable
import time

import numpy as np
import torch

from flexkv import c_ext
from flexkv.server.client import KVDPClient
from flexkv.server.server import KVServer, DPClient
from flexkv.kvtask import KVTaskEngine, KVResponse
from flexkv.common.config import ModelConfig, CacheConfig, GLOBAL_CONFIG_FROM_ENV, MooncakeTransferEngineConfig
from flexkv.integration.dynamo.collector import KVEventCollector
from flexkv.common.debug import eviction_log_aggregator, flexkv_logger
from flexkv.cache.redis_meta import RedisMeta


# ==============================================================================
# 本文件职责：FlexKV 面向用户的顶层 API 门面（Facade）
#
# 在系统链路中的位置：
#   用户/推理框架 -> 【KVManager 本文件】-> KVTaskEngine(kvtask.py) 或 KVDPClient(server/client.py)
#                 -> GlobalCacheEngine(控制面) -> TransferEngine(数据面) -> Worker -> c_ext
#
# 核心设计：双模式分叉
#   KVManager 本身不含业务逻辑，它只做两件事：
#     1. 参数归一化（torch.Tensor -> numpy）
#     2. 按 server_client_mode 把调用转发给两个后端之一：
#        - 直连模式：直接调用本进程的 KVTaskEngine（无 IPC 开销，v1.0 起的默认形态）
#        - 客户端-服务端模式：通过 ZMQ 转发给独立进程的 KVServer（多 DP rank / 多实例共享缓存时用）
#   因此下面几乎每个方法都是这个 if/else 结构，读的时候只要看懂一个，其余同理。
#
# 核心内容速查：
#   - KVManager.__init__ : 决定走哪种模式、起不起服务进程
#   - get_async / put_async : 一步到位的异步搬移（匹配 + 传输一起提交）
#   - get_match / put_match : 只做匹配、返回 mask，传输留待 launch 时再提交（两阶段，便于与调度重叠）
#   - prefetch_async      : 纯 CPU 侧预取，不占 GPU
#   - launch / wait / try_wait : 任务提交与完成等待
#   - reset               : 权重更新后失效全部分层缓存
#
# 阅读提示：
#   get_async 与 get_match 的区别是新手第一个坎。match 阶段只查 radix tree 判断"命中了哪些
#   token"，返回 mask；launch 阶段才真正提交传输。拆成两步是为了让推理框架的调度器在拿到
#   mask 后，有机会据此调整调度决策（比如只计算未命中的部分），再触发传输。
# ==============================================================================


class KVManager:
    """FlexKV 的用户入口，封装 KV Cache 的读取( get )、写入( put )与预取( prefetch )。

    两种运行形态（由 server_client_mode 决定，见 __init__）：
      - 直连模式：任务引擎 KVTaskEngine 住在当前进程，调用开销最小。
      - 客户端-服务端模式：任务引擎住在独立服务进程，本进程只持有一个 ZMQ 客户端，
        适用于 dp_size > 1 或多个实例共享同一份 CPU/SSD 缓存的场景。

    生命周期：构造 -> start() -> (get/put/launch/wait ...) -> shutdown()
    """

    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 dp_client_id: int = 0,
                 server_recv_port: str = "",
                 gpu_register_port: str = "",
                 event_collector: Optional[KVEventCollector] = None):
        """构造 KVManager 并完成后端选址。

        这里最关键的动作是决定 server_client_mode：只要 dp_size>1、instance_num>1，
        或环境变量显式要求，就走客户端-服务端模式。选完之后构造对应的后端对象。

        Args:
            model_config: 模型结构配置（层数、TP/DP 规模等），决定传输拓扑
            cache_config: 各级缓存容量与开关（CPU/SSD/远端）
            dp_client_id: 当前 DP rank 编号；embedded 模式下只有 rank 0 负责拉起服务进程
            server_recv_port: ZMQ 接收端口；留空则取 GLOBAL_CONFIG_FROM_ENV 的配置
            gpu_register_port: GPU 内存注册用的 IPC 端口；留空则由 server_recv_port 派生
            event_collector: KV 事件上报器，用于对接 Dynamo 的 KV 感知路由
        """
        # Use the curated ``__str__`` summaries. Dataclass repr includes
        # credential-bearing fields such as ``redis_password``.
        flexkv_logger.info(
            "[FlexKV-CONFIG] operation=config act=load status=success "
            "component=kv_manager commit=%s model_config=%s cache_config=%s",
            getattr(c_ext, "__git_commit__", "unknown"),
            model_config,
            cache_config,
        )
        self.model_config = model_config
        self.cache_config = cache_config

        if server_recv_port != "":
            self.server_recv_port = server_recv_port
        else:
            self.server_recv_port = GLOBAL_CONFIG_FROM_ENV.server_recv_port
        if gpu_register_port != "":
            self.gpu_register_port = gpu_register_port
        else:
            self.gpu_register_port = self.server_recv_port + "_gpu_register"

        flexkv_logger.info(
            f"[KVManager] IPC ports: server_recv_port={self.server_recv_port}, "
            f"gpu_register_port={self.gpu_register_port}"
        )

        # Multi-instance mode also requires server_client_mode
        self.server_client_mode = (model_config.dp_size > 1 or
                                   model_config.instance_num > 1 or
                                   GLOBAL_CONFIG_FROM_ENV.server_client_mode)
        self.server_launch_mode = GLOBAL_CONFIG_FROM_ENV.server_launch_mode
        if self.server_launch_mode not in ("embedded", "external"):
            raise ValueError(
                "FLEXKV_SERVER_LAUNCH_MODE must be embedded or external, "
                f"got {self.server_launch_mode!r}"
            )
        if self.server_launch_mode == "external" and not self.server_client_mode:
            raise ValueError(
                "FLEXKV_SERVER_LAUNCH_MODE=external requires server-client mode"
            )

        flexkv_logger.info(
            f"[KVManager] instance_num={model_config.instance_num}, dp_size={model_config.dp_size}, "
            f"server_client_mode={self.server_client_mode}, "
            f"server_launch_mode={self.server_launch_mode}"
        )

        self.redis_meta_client = None
        self.enable_mps = GLOBAL_CONFIG_FROM_ENV.enable_mps
        self.owns_mps = self.enable_mps and self.server_launch_mode != "external"

        if self.server_client_mode:
            if self.server_launch_mode == "embedded" and dp_client_id == 0:
                self.server_handle = KVServer.create_server(model_config=model_config,
                                                            cache_config=cache_config,
                                                            gpu_register_port=self.gpu_register_port,
                                                            server_recv_port=self.server_recv_port,
                                                            inherit_env=False)

            else:
                self.server_handle = None
            self.dp_client = KVDPClient(
                self.server_recv_port,
                model_config=model_config,
                dp_client_id=dp_client_id,
            )
        else:
            # In non-server_client_mode, create RedisMeta here and pass to KVTaskEngine
            if self.cache_config.enable_kv_sharing:
                flexkv_logger.info(f"[kv manager] initializing RedisMeta and connection to "
                                   f"{self.cache_config.redis_host}:{self.cache_config.redis_port}")
                self.redis_meta_client = RedisMeta(
                    self.cache_config.redis_host,
                    self.cache_config.redis_port,
                    self.cache_config.redis_password,
                    self.cache_config.local_ip,
                    node_ttl_seconds=self.cache_config.node_ttl_seconds,
                )
                self.redis_meta_client.init_meta()
                # update distributed_node_id
                self.cache_config.distributed_node_id = self.redis_meta_client.get_node_id()

            self.server_handle = None
            self.kv_task_engine = KVTaskEngine(
                model_config,
                self.cache_config,
                self.gpu_register_port,
                redis_meta=self.redis_meta_client,
                event_collector=event_collector,
            )

    def start(self) -> None:
        """启动后端，使其进入可服务状态。

        直连模式：启动 KVTaskEngine（拉起 worker 进程、初始化存储）。
        服务端模式：向服务进程发注册请求。注意 MPS（多进程服务）若开启且由本进程
        持有，也会在此拉起，用于让多个 worker 进程共享 GPU context。
        """
        if self.owns_mps:
            # try to start MPS
            subprocess.run(['nvidia-cuda-mps-control', '-d'], check=False)
            flexkv_logger.debug("MPS started")

        if not self.server_client_mode:
            self.kv_task_engine.start()
        else:
            # send the start request to the server
            self.dp_client.start_server_and_register()

    def is_ready(self) -> bool:
        """后端是否已完成初始化、可以接受请求。轮询用。"""
        if self.server_client_mode:
            return self.dp_client.is_ready()
        else:
            return self.kv_task_engine.is_ready()

    def shutdown(self) -> None:
        """优雅退出：先 flush 淘汰日志聚合器，再关闭后端。

        两种模式的收尾不同：external 模式下服务进程不由本进程持有，只反注册自己；
        embedded 模式下本进程负责关掉自己拉起的服务进程。MPS 守护进程不会自动停，
        需要人工执行 `echo quit | nvidia-cuda-mps-control`。
        """
        flexkv_logger.info("[FLEXKV] KVManager.shutdown begin.")
        eviction_log_aggregator.flush()
        if self.server_client_mode:
            if self.server_launch_mode == "external":
                self.dp_client.unregister()
            else:
                self.dp_client.shutdown()
                # Wait for the server process to exit after sending shutdown request
                if self.server_handle is not None:
                    self.server_handle.shutdown()
                    self.server_handle = None
        else:
            self.kv_task_engine.shutdown()

        if self.owns_mps:
            flexkv_logger.info(
                "MPS is enabled. To stop MPS daemon manually, run: "
                "'echo quit | nvidia-cuda-mps-control'"
            )
        flexkv_logger.info("[FLEXKV] KVManager.shutdown done.")

    def get_async(self,
                  token_ids: Union[torch.Tensor, np.ndarray],
                  slot_mapping: Union[torch.Tensor, np.ndarray],
                  token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
                  namespace: Optional[List[str]] = None,
                  ) -> int:
        """异步 get：一步到位提交「前缀匹配 + 数据回迁」，立即返回 task_id。

        这是最常用的入口。内部先查 radix tree 算出命中了多少 token，再把对应的
        传输子图提交给数据面，全程不阻塞调用方；随后用 wait/try_wait 取结果。

        与 get_match + launch 的两阶段写法相比，本方法把匹配和传输绑在一起提交，
        省事但失去了「拿到 mask 后再决定如何调度」的机会。

        Args:
            token_ids: 本次请求的完整 token 序列，用于前缀匹配
            slot_mapping: GPU 侧 KV Cache 槽位映射，指明数据搬回显存的哪个位置
            token_mask: 可选掩码，标记哪些 token 参与匹配
            namespace: 命名空间，用于多租户/多模型隔离

        Returns:
            task_id，后续用 wait/try_wait 查询完成状态
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        if isinstance(slot_mapping, torch.Tensor):
            slot_mapping = slot_mapping.numpy()
        if isinstance(token_mask, torch.Tensor):
            token_mask = token_mask.numpy()
        if self.server_client_mode:
            task_id = self.dp_client.get_async(token_ids,
                                               slot_mapping,
                                               token_mask,
                                               namespace=namespace)
        else:
            task_id, _ = self.kv_task_engine.get_async(
                token_ids=token_ids,
                slot_mapping=slot_mapping,
                token_mask=token_mask,
                namespace=namespace,
            )
        return task_id

    def get_match(self,
                  token_ids: Union[torch.Tensor, np.ndarray],
                  token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
                  cpu_only: bool = False,
                  namespace: Optional[List[str]] = None,
                  swa_aware: bool = False,
                  ) -> Tuple[int, np.ndarray]:
        """Match a prefix and build the load graph; return (task_id, mask).

        ``swa_aware=True`` clamps the Full-KV transfer to the reusable SWA window
        (from the same single match); the SWA window is the trailing block of the
        returned mask, which the caller reads directly. ``swa_aware=False``
        (default) is the plain path.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        if isinstance(token_mask, torch.Tensor):
            token_mask = token_mask.numpy()
        if self.server_client_mode:
            task_id, mask = self.dp_client.get_match(token_ids,
                                                     token_mask,
                                                     cpu_only=cpu_only,
                                                     namespace=namespace,
                                                     swa_aware=swa_aware)
        else:
            task_id, mask = self.kv_task_engine.get_match(
                token_ids=token_ids,
                token_mask=token_mask,
                cpu_only=cpu_only,
                namespace=namespace,
                swa_aware=swa_aware,
            )
        return task_id, mask

    def put_async(self,
                  token_ids: Union[torch.Tensor, np.ndarray],
                  slot_mapping: Union[torch.Tensor, np.ndarray],
                  token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
                  namespace: Optional[List[str]] = None,
                  ) -> int:
        """异步 put：把 GPU 上算好的 KV Cache 卸载到各级外部存储。

        与 get_async 同构、方向相反：匹配之后提交的是 D2H（显存->内存）以及后续的
        内存->SSD/远端写穿。主进程只负责发起，落盘等慢操作由数据面在后台完成，
        因此 put 的延迟可以与后续计算重叠。

        Returns:
            task_id
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        if isinstance(slot_mapping, torch.Tensor):
            slot_mapping = slot_mapping.numpy()
        if isinstance(token_mask, torch.Tensor):
            token_mask = token_mask.numpy()
        if self.server_client_mode:
            task_id = self.dp_client.put_async(token_ids, slot_mapping, token_mask,
                                               namespace=namespace)
        else:
            task_id, _ = self.kv_task_engine.put_async(
                token_ids=token_ids,
                slot_mapping=slot_mapping,
                token_mask=token_mask,
                namespace=namespace,
            )
        return task_id

    def put_match(self,
                  token_ids: Union[torch.Tensor, np.ndarray],
                  token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
                  namespace: Optional[List[str]] = None,
                  ) -> Tuple[int, np.ndarray]:
        """put 的两阶段写法之第一阶段：只做匹配，返回 mask，不提交传输。

        返回的 mask 标记了哪些 token 尚不在外部存储中（即真正需要写回的部分）。
        调用方据此决定调度，再用 launch 提交实际传输。

        Returns:
            (task_id, mask)
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        if isinstance(token_mask, torch.Tensor):
            token_mask = token_mask.numpy()
        if self.server_client_mode:
            task_id, mask = self.dp_client.put_match(token_ids, token_mask,
                                                     namespace=namespace)
        else:
            task_id, mask = self.kv_task_engine.put_match(
                token_ids=token_ids,
                token_mask=token_mask,
                namespace=namespace,
            )
        return task_id, mask

    def prefetch_async(self,
                       token_ids: np.ndarray,
                       namespace: Optional[List[str]] = None,
                       swa_aware: bool = False) -> int:
        """Launch prefetch; return the task_id.

        The prefetch is fire-and-forget at launch time. Callers poll progress
        via ``try_wait``/``wait`` — the returned ``KVResponse.return_mask`` is
        rewritten to the CPU-tree state at graph completion (post-commit), so
        ``sum(return_mask)`` is the authoritative usable-token count.

        ``swa_aware=True`` plans a joint Full+SWA REMOTE2H so the SWA snapshot
        lands on the local CPU SWA pool alongside the Full-KV prefix. The tree
        keeps the invariant "SWA present ⇒ Full ready up to this node" —
        partial Full or SWA failure frees the SWA slot; only the Full prefix
        (if any) stays on the tree.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        if self.server_client_mode:
            task_id = self.dp_client.prefetch_async(
                token_ids, namespace=namespace, swa_aware=swa_aware)
        else:
            task_id = self.kv_task_engine.prefetch_async(
                token_ids,
                namespace=namespace,
                swa_aware=swa_aware,
            )
        return task_id

    def launch(self,
               task_ids: Union[int, List[int]],
               slot_mappings: Union[np.ndarray, List[np.ndarray], torch.Tensor, List[torch.Tensor]],
               swa_slot_mappings: Optional[Union[np.ndarray, List[Optional[np.ndarray]], torch.Tensor, List[Optional[torch.Tensor]]]] = None,
               as_batch: bool = False,
               layerwise_transfer: bool = False,
               counter_id: int = 0) -> List[int]:
        """提交任务：把之前 match 阶段建好的传输图真正交给数据面执行。

        只有两阶段写法（get_match/put_match）才需要显式调用本方法；get_async/put_async
        内部已经包含了提交动作。

        Args:
            task_ids: 一个或多个任务 ID
            slot_mappings: GPU 侧槽位映射，与 task_ids 一一对应
            swa_slot_mappings: SWA（滑动窗口注意力）专用槽位映射，仅当注册了 SWA
                GPU 池且请求存在 SWA 复用窗口时才提供
            as_batch: 是否把多个任务融合成一个批次图提交，减少调度开销
            layerwise_transfer: 是否启用逐层传输（让 H2D 与 prefill 计算重叠）
            counter_id: 用于逐层传输的计数器分组

        Returns:
            被提交的任务 ID 列表
        """
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        if not isinstance(slot_mappings, List):
            slot_mappings = [slot_mappings]
        if isinstance(slot_mappings[0], torch.Tensor):
            slot_mappings = [slot_mapping.numpy() for slot_mapping in slot_mappings]
        # SWA GPU slot_mappings (optional): the connector supplies these only when
        # it registered an SWA GPU pool and the request has an SWA reuse window.
        if swa_slot_mappings is not None and not isinstance(swa_slot_mappings, List):
            swa_slot_mappings = [swa_slot_mappings]
        if isinstance(swa_slot_mappings, List):
            swa_slot_mappings = [
                sm.numpy() if isinstance(sm, torch.Tensor) else sm
                for sm in swa_slot_mappings
            ]
        if self.server_client_mode:
            return self.dp_client.launch_tasks(
                task_ids=task_ids,
                slot_mappings=slot_mappings,
                swa_slot_mappings=swa_slot_mappings,
                as_batch=as_batch,
                layerwise_transfer=layerwise_transfer,
                counter_id=counter_id,
            )
        else:
            return self.kv_task_engine.launch_tasks(
                task_ids,
                slot_mappings,
                swa_slot_mappings=swa_slot_mappings,
                as_batch=as_batch,
                layerwise_transfer=layerwise_transfer,
                counter_id=counter_id,
            )

    def cancel(self, task_ids: Union[int, List[int]]) -> None:
        """取消尚未完成的任务（例如推理请求被 abort 时）。

        取消后已分配的资源会回滚，相关 block 不会上树，避免脏数据被后续请求命中。
        """
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        if self.server_client_mode:
            self.dp_client.cancel_tasks(task_ids)
        else:
            self.kv_task_engine.cancel_tasks(task_ids)

    def wait(self,
             task_ids: Union[int, List[int]],
             timeout: float = 20.0,
             completely: bool = False) -> Dict[int, KVResponse]:
        """阻塞等待任务完成，直到全部结束或超时。

        Args:
            task_ids: 待等待的任务 ID
            timeout: 超时秒数
            completely: True 表示连「内存->SSD/远端」这类后台写穿也要等完成；
                False 则只要数据已到达 GPU（get）/ 已离开 GPU（put）就算完成

        Returns:
            {task_id: KVResponse}，KVResponse 里带 return_mask 描述实际完成的情况
        """
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        if self.server_client_mode:
            return self.dp_client.wait(task_ids, timeout, completely)
        else:
            return self.kv_task_engine.wait(task_ids, timeout, completely)

    def try_wait(self, task_ids: Union[int, List[int]]) -> Dict[int, KVResponse]:
        """非阻塞版本：查一下有哪些任务已完成，立即返回，不等。

        适合在推理主循环里每轮顺带轮询，避免阻塞调度。
        """
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        if self.server_client_mode:
            return self.dp_client.try_wait(task_ids)
        else:
            return self.kv_task_engine.try_wait(task_ids)

    # Only for testing
    def _clear_cpu_cache(self) -> None:
        """仅供测试：清空 CPU 级缓存。客户端-服务端模式下不支持。"""
        if self.server_client_mode:
            flexkv_logger.error("clear_cache is not supported in server client mode")
            return
        else:
            self.kv_task_engine._clear_cpu_cache()

    def reset(self) -> None:
        """Invalidate the cache across all tiers (CPU + SSD + remote): drop the
        radix tree and free the mempool.

        Call after a weight update so KV computed against stale weights is not
        reused. Works in both in-process and server-client mode. Cheap and
        idempotent (resetting an already-empty tree/mempool is a no-op).
        """
        if self.server_client_mode:
            self.dp_client.reset()
        else:
            self.kv_task_engine.reset_cache()
