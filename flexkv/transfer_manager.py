# ==============================================================================
# 本文件职责：传输管理器（GPU 注册服务端 + 传输引擎的引导装配 + 三种部署拓扑的句柄封装）。
#
# 它解决的核心问题：
#   推理框架（vLLM / SGLang / TRT-LLM）的 KV Cache 显存是**框架 worker 进程**分配的
#   torch.Tensor，而真正搬数据的 FlexKV worker 是另一批进程。两边不共享 CUDA 上下文，
#   普通 pickle 无法跨进程传递显存指针。因此必须有一条带外通道把 GPU 显存"登记"给
#   FlexKV：框架侧把每个 block 导出成 TensorSharedHandle（CUDA IPC / VMM 句柄），
#   经 ZMQ 推到本文件的注册端口，本文件再调用 StorageEngine.register_gpu_blocks，
#   由 StorageEngine 在 worker 侧 CUDA IPC import 成可访问的虚地址。整个过程称为
#   "GPU 内存注册"，它是 FlexKV 能对 GPU 做 DMA 的前提。
#
#   装配完成后，本文件建出 StorageEngine 与 TransferEngine，并把两者绑定成一个
#   TransferManager 实例。之后它退化成一个很薄的转发层：submit 下发给 TransferEngine，
#   wait 取回 CompletedOp。真正理解 DAG 语义的是上层的 GlobalCacheEngine。
#
# 在系统链路中的位置：
#   KVManager(kvmanager.py)
#     -> KVTaskEngine(kvtask.py)            任务层：命中判定、任务状态机
#       -> GlobalCacheEngine(cache/cache_engine.py)  控制面：产出 TransferOpGraph
#         -> 【本文件】TransferManagerHandle          跨进程/跨机拓扑 + GPU 注册服务端
#           -> TransferManager                       持有 StorageEngine + TransferEngine
#             -> TransferEngine(transfer/transfer_engine.py)  数据面调度
#               -> Worker(transfer/worker.py) -> c_ext        真正搬运
#
#   注意"回传"不是向上调用而是向上轮询：TransferEngine 把 CompletedOp 放进
#   completed_queue，本层的 wait() 抽出来交给 KVTaskEngine._get_completed_ops 聚合。
#   KVTaskEngine 持有 N 个 handle（本机 1 个 + 多机 1 个），同一个图要投 N 份、
#   完成通知要计够 N 份才算完成（见 kvtask.py 的 required_completed_count）。
#
# 核心内容速查：
#   * TransferManager                      GPU 注册服务端 + 引擎装配本体
#       - _register_gpu_blocks_via_socket  收齐 expected_gpus 份注册请求为止
#       - initialize_transfer_engine       注册 -> 建 StorageEngine -> 建 TransferEngine
#       - submit / submit_batch / wait     转发到 TransferEngine
#       - handle_gpu_control               sleep/wake 场景释放/重建 GPU 映射
#   * TransferManagerOnRemote              远端服务端本体（跑在独立进程里）
#   * TransferManagerHandleBase            抽象接口：start / is_ready / submit / wait
#   * TransferManagerIntraProcessHandle    thread 模式：同进程，无 IPC
#   * TransferManagerInterProcessHandle    process 模式（默认）：子进程 + Pipe
#   * TransferManagerMultiNodeHandle       remote 模式客户端：ZMQ 连远端
#   * TransferManagerHandle                门面：按 mode 选一个上面的 *_Handle
#
# 三种模式的取舍：
#   thread  ：最简单，但 TransferManager 与框架共享进程，worker 的 CUDA 上下文、
#             GIL、以及框架自己的显存操作会互相干扰；主要用来调试。
#   process ：默认。mp spawn 出独立进程隔离上述风险；代价是每次 submit 要走一次
#             Pipe pickle，所以用 submit_batch 摊薄，且 completed_queue 通过
#             selectors 监听其 _reader fd，做到事件驱动而非轮询。
#   remote  ：nnodes>1 时必选——Pipe 这类 IPC 跨不了机器，必须换成 TCP 打到对端节点
#             的 TransferManagerOnRemote 进程。TRT-LLM 场景也必选：它要先初始化 MPI，
#             MPI 不允许之后再 fork 子进程，于是本机也改成「Popen 起一个干净的远端
#             服务进程，再用 ZMQ 连过去」（见 create_process）。
#
# 阅读提示：
#   1. 看到 tuple key RegistrationKey = (dp_client_id, intra_client_id) 时记住：它是
#      **逻辑**身份，不等于 CUDA device_id；device_id 单独存在 gpu_device_id_mapping。
#      多个 registration 可能映射到同一个 WorkerKey（同一个 TP 组）。
#   2. gpu_control_socket 在 TransferManager.__init__ 里无条件 bind，所以**每种**模式
#      都必须有人去服务它，各自的服务者还不一样：
#        process -> 子进程 select/selector 循环里的 gpu_control 分支
#        thread  -> start_gpu_control_listener() 起的守护线程
#        remote  -> TransferManagerOnRemote._polling_worker 的 zmq.Poller
#      漏掉任何一处，vLLM sleep/wake 就会静默卡满客户端 120s 的 RCVTIMEO。
#   3. task_end_op_id 只对 remote 通路有意义：TransferEngine 不知道任务何时"够用了"，
#      只有远端副本能按 task_end_op 判断可否提前回报（见 MultiNodeHandle.submit）。
# ==============================================================================
import os
import multiprocessing as mp
import signal
import time
import queue
import selectors
from queue import Queue
from typing import Dict, Optional, List, Tuple, Any
from abc import ABC, abstractmethod
from multiprocessing import Process, Pipe, Event
from sympy.assumptions.assume import true
import torch
import zmq
import nvtx
import tempfile
import threading
import numpy as np
import textwrap
import subprocess
import pickle
import sys

from flexkv.common.transfer import TransferOpGraph, CompletedOp, WorkerKey
from flexkv.common.config import (
    CacheConfig, LayerGroupSpec, ModelConfig,
    recompute_cache_block_counts, GLOBAL_CONFIG_FROM_ENV,
)
from flexkv.common.debug import flexkv_logger
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.common.transfer import DeviceType
from flexkv.common.storage import KVCacheLayout
from flexkv.storage.storage_engine import StorageEngine
from flexkv.transfer.transfer_engine import TransferEngine
from flexkv.server.utils import get_zmq_socket
from flexkv.server.request import RegistrationKey, RegisterTPClientRequest, Response


class TransferManager:
    """传输服务本体：GPU 注册服务端 + StorageEngine/TransferEngine 的装配者。

    在整个链路中的职责：**把「框架 worker 进程里的显存」变成「数据面可以 DMA 的资源」。
    这是 FlexKV 唯一一处必须和推理框架握手的地方**——
      上游（框架 worker）：通过 flexkv.server.client.KVTPClient.register_to_server
                          把自己的 KV Cache blocks 导出成 TensorSharedHandle 推过来。
      本类：收到 N 份注册后建 StorageEngine（登记 GPU/CPU/SSD/REMOTE 各层 StorageHandle），
            再把它交给 TransferEngine。
      下游（TransferEngine）：只认 StorageHandle，不认 tensor / 不认 ZMQ。

    为什么它必须单独存在（回答"KVTaskEngine 和 TransferEngine 之间为什么还要一层"）：
      1. **显存注册天然是带外的、一次性的、收敛等待式的**。它不是"提交一个任务"，
         而是"等本节点 expected_gpus 个 worker 都来报到才能开工"。这个收齐语义
         既不适合塞进 TransferEngine（它只处理 DAG），也不适合塞进 KVTaskEngine
         （它可能跑在别的进程/别的节点里）。
      2. **多 DP rank / 多 TP rank 的 handle 归拢**。注册时按 RegistrationKey
         (dp_client_id, intra_client_id) 收集，装配时按键聚合成
         Dict[WorkerKey, List[StorageHandle]]——同一个 TP 组的 handles 进同一个
         WorkerKey 桶，TransferEngine 才能按 TP 组给每个 worker 各一份显存视图。
      3. **进程拓扑隔离**。本类可能被放到独立进程（默认）或远端进程，这样干活的
         CUDA 上下文不至于污染框架进程。选择逻辑不在本类，而在下面的 *_Handle。

    生命周期：
      __init__（只 bind  socket，不阻塞）
        -> initialize_transfer_engine()：阻塞等待 GPU 注册 -> 建 StorageEngine
           -> register_gpu_blocks -> 建 TransferEngine
        -> start()：拉起 TransferEngine 的 worker 子进程与调度线程
        -> submit / wait 业务期
        -> shutdown()
    """

    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 gpu_register_port: str):
        self.model_config = model_config
        self.cache_config = cache_config
        self.gpu_register_port = gpu_register_port
        self.instance_num = self.model_config.instance_num
        # Calculate total expected GPUs on this node across all instances
        # 中文：本节点预计来报到的 GPU 数 = 实例数 × 每实例每节点 GPU 数。
        # 它是 _register_gpu_blocks_via_socket 的收敛条件：注册数不到这里就不往下走。
        # 少卡/多卡/某个 worker 用了重复的 registration_key 都会卡在这一步。
        self.expected_gpus = self.instance_num * self.model_config.gpus_per_node

        # GPU 注册台账。四个 dict 同键：RegistrationKey = (dp_client_id, intra_client_id)
        self.all_gpu_layouts: Dict[RegistrationKey, KVCacheLayout] = {}
        # 值是从框架进程导出的 TensorSharedHandle 列表（每个元素 ≈ 一层 KV 的一整块
        # 显存的 IPC 句柄，本身不含数据，import 后才是可访问指针）
        self.all_gpu_blocks: Dict[RegistrationKey, List[TensorSharedHandle]] = {}
        # RegistrationKey -> WorkerKey(dp_client_id, pp_rank)：同 TP 组的多个 rank
        # 会折叠到同一个 WorkerKey，这是交给 TransferEngine 的分桶依据
        self.gpu_worker_key_mapping: Dict[RegistrationKey, WorkerKey] = {}
        # RegistrationKey -> 物理 CUDA device_id。逻辑身份 ≠ 设备号，两者分开存
        self.gpu_device_id_mapping: Dict[RegistrationKey, int] = {}

        # Multi-group storage for heterogeneous KV shapes, including DSA/NSA
        # indexer-as-group. None for uniform single-shape registrations.
        self.all_gpu_layouts_per_group: Dict[
            RegistrationKey, Optional[List[KVCacheLayout]]
        ] = {}
        self.all_gpu_blocks_per_group: Dict[
            RegistrationKey, Optional[List[List[TensorSharedHandle]]]
        ] = {}

        # SWA dedicated GPU pool (channel B): independent of the main-KV pool.
        # Logical registration key -> SWA handles / layout. Populated only when
        # the client provides swa_handles (DSv4 sliding-window-attention pool).
        self.all_swa_gpu_blocks: Dict[RegistrationKey, List[TensorSharedHandle]] = {}
        self.all_swa_gpu_layouts: Dict[RegistrationKey, KVCacheLayout] = {}
        self.all_swa_gpu_layouts_per_group: Dict[
            RegistrationKey, Optional[List[KVCacheLayout]]
        ] = {}
        self.all_swa_gpu_blocks_per_group: Dict[
            RegistrationKey, Optional[List[List[TensorSharedHandle]]]
        ] = {}
        self.swa_layer_groups: Optional[List[LayerGroupSpec]] = None

        # 注意 ctx 的 IO 线程数是 2 而不是默认 1：注册 PULL 与 gpu_control REP 是两个
        # 独立端点，单 IO 线程在两者同时排队时会串行化，让 poll 看起来像随机延迟。
        self.context = zmq.Context(2)
        # PULL + bind：框架侧是 PUSH，多 worker 可以把注册请求并发推到同一个端口
        self.recv_from_client = get_zmq_socket(
            self.context, zmq.SocketType.PULL, gpu_register_port, True)
        # GPU 控制的 REP 端点，服务于框架的 sleep/wake（如 vLLM sleep mode）：
        # 按约定派生端口，客户端用同一个 gpu_register_port 拼出来即可
        self.gpu_control_port = f"{gpu_register_port}_control"
        self.gpu_control_socket = get_zmq_socket(
            self.context, zmq.SocketType.REP, self.gpu_control_port, True
        )
        self._gpu_suspended = False
        self._pending_resume_registrations: Dict[
            RegistrationKey, RegisterTPClientRequest
        ] = {}
        # The REP socket above is bound unconditionally, so *every* deployment
        # mode has to service it -- an unserved REP endpoint turns a
        # suspend/resume call into a silent 120s RCVTIMEO stall on the client.
        # The subprocess mode drives it from its selector loop; the other two
        # modes use the listener thread below.
        self._gpu_control_shutdown = threading.Event()
        self._gpu_control_thread: Optional[threading.Thread] = None

        # 两个引擎都是延迟创建：initialize_transfer_engine() 收齐 GPU 注册后才建，
        # 在此之前它们为 None（GPU 控制请求会因此直接报 RuntimeError）。
        self.transfer_engine: Optional[TransferEngine] = None
        self.storage_engine: Optional[StorageEngine] = None
        flexkv_logger.info(f"Initialized TransferManager with config successfully, "
                           f"instance_num={self.instance_num}, expected_gpus={self.expected_gpus}")

    def _handle_gpu_blocks_registration(self, req: RegisterTPClientRequest) -> None:
        """登记一份 GPU 注册请求到本类的四份台账里。

        这是 GPU 内存注册的实际落库点：把 RegisterTPClientRequest 里的 handles /
        layout / device_id / WorkerKey 拆开存进对应的 dict，供 initialize_transfer_engine
        在收齐之后统一装配。

        注意本方法**不接触 CUDA**：它只保存 IPC 句柄对象。真正把显存映射进本进程
        地址空间是 StorageEngine.register_gpu_blocks 在 worker 侧完成的。

        Args:
            req: 框架 worker 经 ZMQ PUSH 过来的注册请求
        """
        registration_key = req.registration_key

        if registration_key in self.all_gpu_blocks:
            # A duplicate (dp_client_id, intra_client_id) means the framework
            # adapter handed us the same logical identity for two different
            # workers -- typically because it never passes intra_client_id and
            # every TP rank collapses onto ...0. Registration can then never
            # reach expected_gpus, so _register_gpu_blocks_via_socket spins
            # forever printing "Still waiting for GPU registrations: k/N".
            # Say so explicitly instead of leaving an unexplained hang.
            flexkv_logger.error(
                f"GPU worker {registration_key} has already registered. "
                f"A duplicate registration key from a different worker means "
                f"registration can never complete and init will hang. Check "
                f"that the framework adapter passes a per-worker-unique "
                f"intra_client_id (registered so far: "
                f"{sorted(self.all_gpu_blocks)}).")
        else:
            try:
                self.all_gpu_blocks[registration_key] = req.handles
                self.all_gpu_layouts[registration_key] = req.gpu_layout
                self.gpu_device_id_mapping[registration_key] = req.device_id
                self.gpu_worker_key_mapping[registration_key] = WorkerKey(
                    dp_client_id=req.dp_client_id,
                    pp_rank=req.pp_rank,
                )
                # Store multi-group info (None when uniform single-shape registration).
                # This covers heterogeneous shapes and DSA/NSA indexer-as-group.
                self.all_gpu_layouts_per_group[registration_key] = req.gpu_layouts
                self.all_gpu_blocks_per_group[registration_key] = req.handles_per_group
                # Store SWA GPU data if present.
                if getattr(req, "swa_handles", None) is not None and req.swa_layout is not None:
                    self.all_swa_gpu_blocks[registration_key] = req.swa_handles
                    self.all_swa_gpu_layouts[registration_key] = req.swa_layout
                    self.all_swa_gpu_layouts_per_group[registration_key] = (
                        req.swa_gpu_layouts
                    )
                    self.all_swa_gpu_blocks_per_group[registration_key] = (
                        req.swa_handles_per_group
                    )
                    if req.swa_layer_groups is not None:
                        if self.swa_layer_groups is None:
                            self.swa_layer_groups = req.swa_layer_groups
                        elif self.swa_layer_groups != req.swa_layer_groups:
                            raise ValueError(
                                "SWA layer groups differ across GPU registrations"
                            )
                    flexkv_logger.info(
                        f"GPU worker {registration_key}: registered SWA handles "
                        f"({len(req.swa_handles)} tensors, "
                        f"groups={len(req.swa_layer_groups or [])})"
                    )
                # Propagate layer_groups to model_config (first registration wins).
                # token_size_in_bytes / num_cpu_blocks recompute downstream depends on this.
                # 中文补充：layer_groups 描述异构层组（各组的层数/KV 头数/head_size 可能不同，
                # 也包含把 DSA/NSA indexer 当成一个组的情况）。它只有在第一个 worker 注册
                # 时才拿得到，而 KVTaskEngine 侧 CacheEngine 的 mempool 大小早就用它算过了，
                # 所以下游 recompute_cache_block_counts 必须与之严格一致，否则 CPU/SSD 层
                # 容量与 GPU 层对不上。"first registration wins" 约定了多 worker 不一致时取谁。
                if req.layer_groups is not None and self.model_config.layer_groups is None:
                    self.model_config.layer_groups = req.layer_groups
                    flexkv_logger.info(
                        f"Set model_config.layer_groups from GPU worker "
                        f"{registration_key}: "
                        f"{[(g.num_layers, g.num_kv_heads, g.head_size) for g in req.layer_groups]}"
                    )
            except Exception as e:
                flexkv_logger.error(
                    f"Failed to register GPU worker {registration_key}: {e}")

    # 中文补充：本方法服务框架的 sleep/wake 生命周期（典型场景是 vLLM sleep mode 释放显存）。
    #   suspend_gpu：让 TransferEngine 解掉所有 GPU 映射，否则那些显存页还被 FlexKV 持有
    #                引用，框架释放显存后会拿到非法地址。幂等——重复 suspend 返回 released=0。
    #   resume_gpu ：重新 import 显存。这里有个关键的**凑齐约束**：必须等 expected_gpus 个
    #                worker 都提交了新 registration，才一次性重映射（见
    #                _pending_resume_registrations 的计数判断）；没到齐时返回 ready=False，
    #                让调用方继续投下一个 rank。
    #   整个过程运行在 TransferManager 所在进程的地址空间，只动 TransferEngine 的内部映射，
    #   不搬运任何 KV 数据，因此不需要通知上层，也不会丢失已缓存数据。
    #
    #   Args:  request - dict，至少含 "type"（"suspend_gpu" | "resume_gpu"）；
    #                    resume_gpu 还需 "registration"（RegisterTPClientRequest）
    #   Returns: suspend 返回 {"ok", "released_mappings"}；
    #            resume  返回 {"ok", "ready", "registered", "imported_mappings"}
    #   Raises:  RuntimeError（引擎未初始化 / 未在 suspended 状态下 resume）、
    #            ValueError / TypeError / KeyError / NotImplementedError（请求不合法）
    def handle_gpu_control(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle synchronous sleep/wake mapping lifecycle requests."""
        if self.transfer_engine is None or self.storage_engine is None:
            raise RuntimeError("Transfer engine is not initialized")

        request_type = request.get("type")
        if request_type == "suspend_gpu":
            if not self._gpu_suspended:
                released = self.transfer_engine.suspend_gpu_mappings()
                self._gpu_suspended = True
                self._pending_resume_registrations.clear()
                flexkv_logger.info(
                    f"Suspended FlexKV GPU mappings: released={released}"
                )
            else:
                released = 0
            return {"ok": True, "released_mappings": released}

        if request_type != "resume_gpu":
            raise ValueError(f"Unknown GPU control request: {request_type}")
        if not self._gpu_suspended:
            raise RuntimeError("GPU mappings are not suspended")

        registration = request.get("registration")
        if not isinstance(registration, RegisterTPClientRequest):
            raise TypeError("resume_gpu requires RegisterTPClientRequest")
        if registration.registration_key not in self.all_gpu_blocks:
            raise KeyError(
                f"Unknown registration key {registration.registration_key}"
            )
        if (
            registration.layer_groups is not None
            or registration.handles_per_group is not None
            or registration.swa_handles is not None
        ):
            raise NotImplementedError(
                "GPU hot remap currently supports uniform main KV only"
            )
        self._pending_resume_registrations[
            registration.registration_key
        ] = registration

        ready = (
            len(self._pending_resume_registrations) == self.expected_gpus
        )
        imported = 0
        if ready:
            grouped_gpu_handles = {}
            for registration_key in sorted(
                self._pending_resume_registrations
            ):
                fresh = self._pending_resume_registrations[registration_key]
                self.all_gpu_blocks[registration_key] = fresh.handles
                self.all_gpu_layouts[registration_key] = fresh.gpu_layout
                handle = self.storage_engine.get_storage_handle(
                    DeviceType.GPU, fresh.device_id
                )
                handle.data = fresh.handles
                handle.kv_layout = fresh.gpu_layout
                worker_key = self.gpu_worker_key_mapping[registration_key]
                grouped_gpu_handles.setdefault(worker_key, []).append(handle)
            imported = self.transfer_engine.resume_gpu_mappings(
                grouped_gpu_handles
            )
            self._pending_resume_registrations.clear()
            self._gpu_suspended = False
            flexkv_logger.info(
                f"Resumed FlexKV GPU mappings: imported={imported}"
            )

        return {
            "ok": True,
            "ready": ready,
            "registered": (
                self.expected_gpus if ready
                else len(self._pending_resume_registrations)
            ),
            "imported_mappings": imported,
        }

    # 中文补充：ZMQ 暴露给 selectors / poller 的 fd 是**边沿触发**的——一次可读事件只保证
    #   "此刻队列非空"，并不保证一事件对应一消息。所以收到事件后必须在这里循环 recv 直到
    #   EAGAIN 为止，否则漏下的消息要等下一个请求进来才被处理，表现为 suspend/resume 随
    #   机卡住。本文件里所有监听 gpu_control_socket 的地方（三种模式各一处）都必须调它。
    #   Returns: 本轮实际处理的请求条数
    def drain_gpu_control_requests(self) -> int:
        """Drain all requests after a ZeroMQ FD edge notification."""
        processed = 0
        while True:
            try:
                request = self.gpu_control_socket.recv_pyobj(zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                response = self.handle_gpu_control(request)
            except Exception as e:
                flexkv_logger.exception("GPU mapping lifecycle request failed")
                response = {"ok": False, "error": str(e)}
            self.gpu_control_socket.send_pyobj(response)
            processed += 1
        return processed

    # 中文补充：只有 thread 模式会调用它。为什么另外两种不能调——
    #   process 模式：子进程的 selector 循环里已经注册了 gpu_control_socket
    #                 （见 _process_worker），同一 socket 被两处同时 poll 会让消息被随机
    #                 抢走、REP 状态机错乱，所以那里注释明确写了 "must not call this"。
    #   remote  模式：由 TransferManagerOnRemote._polling_worker 的 zmq.Poller 顺带服务。
    def start_gpu_control_listener(self) -> None:
        """Serve the GPU control REP socket from a dedicated thread.

        For the subprocess mode the selector loop in
        ``TransferManagerInterProcessHandle._process_worker`` already drains
        this socket, so it must not call this.  Thread mode and remote mode
        have no such loop: without this thread the bound REP endpoint accepts
        connections and then never answers, so a vLLM sleep/wake call blocks
        for the client's full 120s RCVTIMEO and then fails.
        """
        if self._gpu_control_thread is not None:
            return
        self._gpu_control_shutdown.clear()
        self._gpu_control_thread = threading.Thread(
            target=self._gpu_control_listener,
            name="flexkv-gpu-control",
            daemon=True,
        )
        self._gpu_control_thread.start()

    def _gpu_control_listener(self) -> None:
        poller = zmq.Poller()
        poller.register(self.gpu_control_socket, zmq.POLLIN)
        try:
            while not self._gpu_control_shutdown.is_set():
                try:
                    # Milliseconds -- zmq.Poller, unlike socket RCVTIMEO, does
                    # not take seconds.
                    # 中文补充：单位是毫秒。这里用 100ms 的超时而不是无限阻塞，是为了让
                    # daemon 线程能在 shutdown_flag 置位后最多 100ms 内退出循环。
                    if not poller.poll(timeout=100):
                        continue
                    self.drain_gpu_control_requests()
                except zmq.ZMQError:
                    # Context terminated during shutdown.
                    break
                except Exception:
                    if not self._gpu_control_shutdown.is_set():
                        flexkv_logger.exception(
                            "GPU control listener failed; retrying"
                        )
                        time.sleep(0.01)
        finally:
            try:
                poller.unregister(self.gpu_control_socket)
            except Exception:
                pass

    def stop_gpu_control_listener(self) -> None:
        self._gpu_control_shutdown.set()
        thread = self._gpu_control_thread
        self._gpu_control_thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def _register_gpu_blocks_via_socket(self) -> None:
        """收齐本节点所有 GPU 的注册请求才返回（GPU 内存注册的服务端循环）。

        为什么必须在这里等：StorageEngine 要为每个物理 GPU 建一个 GPU StorageHandle，
        TransferEngine 也要按 WorkerKey 给每个 TP worker 一份显存视图。少登记一张卡，就会
        在运行时访问到不存在的显存视图（而不是拿到一个干净的报错），所以宁可在初始化
        阶段阻塞在这里。

        隐含约定：各个 worker 的 registration_key 必须互不相同，否则登记数永远到不了
        expected_gpus（重复键会覆盖，不会计数+1），本函数会一直转。

        实现细节：recv 用 NOBLOCK 而不是阻塞 recv，是为了能周期性打日志——这是运维上最
        常看的初始化卡点；忙等间隔 1ms，是吞吐与 CPU 占用的折中（只在启动阶段用一次）。
        """
        try:
            flexkv_logger.info(f"GPU tensor registration server started on port {self.gpu_register_port}, "
                               f"expected {self.expected_gpus} GPUs to register "
                               f"(instance_num={self.instance_num}, gpus_per_node={self.model_config.gpus_per_node}, "
                               f"total_gpus={self.model_config.total_gpus}, nnodes={self.model_config.nnodes})")
            last_log_time = time.time()
            while len(self.all_gpu_blocks) < self.expected_gpus:
                try:
                    # Recv from: flexkv.server.client.KVTPClient.register_to_server
                    req = self.recv_from_client.recv_pyobj(zmq.NOBLOCK)
                except zmq.Again:
                    # Periodically log waiting status for debugging
                    now = time.time()
                    if now - last_log_time >= 5.0:
                        registered_keys = sorted(self.all_gpu_blocks.keys())
                        flexkv_logger.info(
                            f"Still waiting for GPU registrations: "
                            f"{len(self.all_gpu_blocks)}/{self.expected_gpus} registered "
                            f"(registered_keys={registered_keys}, "
                            f"port={self.gpu_register_port})")
                        last_log_time = now
                    time.sleep(0.001)
                    continue

                if isinstance(req, RegisterTPClientRequest):
                    flexkv_logger.info(f"Received GPU blocks registration request: {type(req)}, "
                                       f"registration_key={req.registration_key}, "
                                       f"device_id={req.device_id}, "
                                       f"dp_client_id={req.dp_client_id}, pp_rank={req.pp_rank}")
                    self._handle_gpu_blocks_registration(req)
                    flexkv_logger.info(f"GPU worker {req.registration_key} registered successfully, "
                                       f"waiting for {self.expected_gpus - len(self.all_gpu_blocks)} GPUs to register")
                else:
                    flexkv_logger.error(f"Unrecognized RequestType in SchedulerServer: {type(req)}")

            flexkv_logger.info(f"All {self.expected_gpus} GPUs registered successfully")

        except Exception as e:
            flexkv_logger.error(f"Error in GPU registration server: {e}")
            raise
        finally:
            pass
            # TODO: fix the socket close issue
            # self.recv_from_client.close()
            # self.context.term()

    def initialize_transfer_engine(self) -> None:
        """装配数据面：GPU 注册 -> StorageEngine -> register_gpu_blocks -> TransferEngine。

        这是本文件最重要的方法，五个阶段按顺序**不可换序**：
          1. _register_gpu_blocks_via_socket()  阻塞收齐 expected_gpus 份注册
          2. 从任一 GPU layout 取 num_layer 作为每个 PP stage 的层数，据此 new StorageEngine
          3. register_gpu_blocks()              把每张卡的 IPC 句柄交给 StorageEngine
          4. 按 WorkerKey 把 GPU StorageHandle 分组（TP 组 -> 一组 handle）
          5. new TransferEngine(...)            分组结果 + CPU/SSD/REMOTE handle 一起交付

        为什么 GPU 显存不归 cache_engine 管：GlobalCacheEngine 只负责 CPU 及以下的层，
        GPU 层的"存储"在概念上是框架已经分配好的那块 KV Cache，FlexKV 不能也不该重新
        分配它，只能"登记"进来。所以 register_gpu_blocks 走的是 allocate(raw_data=...) 的
        形式——GPUAllocator 不申请新显存，只把 TensorSharedHandle 映射成本进程可访问的
        虚地址，并记住它的 KVCacheLayout（形状/步长），从而具备按 block 做 DMA 的能力。

        隐含约定：所有 device 的 layer 数一致，所以 num_layer 取任意一个 layout 的值即可。
        """
        flexkv_logger.info("Initializing TransferEngine...")
        self._register_gpu_blocks_via_socket()

        assert len(self.all_gpu_layouts) == self.expected_gpus, \
            f"Expected {self.expected_gpus} GPU layouts, got {len(self.all_gpu_layouts)}"
        assert len(self.all_gpu_blocks) == self.expected_gpus, \
            f"Expected {self.expected_gpus} GPU blocks, got {len(self.all_gpu_blocks)}"
        num_layers_per_pp_stage = next(iter(self.all_gpu_layouts.values())).num_layer

        # Recompute block counts once layer_groups are known (heterogeneous /
        # multi-pool models).  Must match CacheEngine mempool sizing in the
        # main process — sglang DSv4 applies the same recompute before
        # KVManager; this path covers late discovery at GPU registration.
        recompute_cache_block_counts(self.model_config, self.cache_config)

        self.storage_engine = StorageEngine(
            self.model_config,
            self.cache_config,
            num_layers_per_pp_stage,
            swa_layer_groups=self.swa_layer_groups,
        )

        # 把每张卡的 GPU block 列表交给 StorageEngine 登记。
        # register_gpu_blocks 内部是 allocate(device_type=GPU, raw_data=gpu_blocks)，
        # GPUAllocator 不会新申请显存，而是把 TensorSharedHandle 通过 CUDA IPC 映射进
        # 本 worker 地址空间，并按 gpu_layout 记录 shape/stride，从而支持按 block 做 DMA。
        # 这是"注册"一词的实际含义：不是分配，而是导入 + 记录几何信息。
        #
        # 隐含约定：这里传入的 device_id 必须是**物理** device_id（存在
        # gpu_device_id_mapping 里），不能用 registration_key。多个逻辑 registration 可能
        # 映射到同一张卡（也可以不同），二者是独立的两个维度。
        # Logical registration identity is separate from the CUDA device ID.
        for registration_key, gpu_blocks_wrapper in self.all_gpu_blocks.items():
            device_id = self.gpu_device_id_mapping[registration_key]
            self.storage_engine.register_gpu_blocks(
                gpu_blocks_wrapper,
                self.all_gpu_layouts[registration_key],
                device_id,
                dtype=self.model_config.dtype,
            )

        # Register SWA dedicated GPU pool.
        for registration_key, swa_blocks in self.all_swa_gpu_blocks.items():
            device_id = self.gpu_device_id_mapping[registration_key]
            self.storage_engine.register_swa_gpu_blocks(
                swa_blocks,
                self.all_swa_gpu_layouts[registration_key],
                device_id,
                dtype=torch.uint8,
            )
            flexkv_logger.info(
                f"StorageEngine registered SWA GPU pool for device {device_id}"
            )

        # Group GPU handles by WorkerKey
        grouped_gpu_handles: Dict[WorkerKey, List] = {}
        # Per-group data, also keyed by WorkerKey, for multi-group support
        # (heterogeneous KV / indexer-as-group)
        grouped_gpu_blocks_per_group: Optional[Dict[WorkerKey, List]] = None
        grouped_gpu_layouts_per_group: Optional[Dict[WorkerKey, List]] = None
        has_multi_group = self.model_config.layer_groups is not None
        if has_multi_group:
            grouped_gpu_blocks_per_group = {}
            grouped_gpu_layouts_per_group = {}

        # 把一个 TP 组内的 GPU StorageHandle 收集成一个 list。TransferEngine 的 GPU worker
        # 是按 WorkerKey 分桶的（见 transfer_engine 的 _worker_map），所以这里是分桶的最后
        # 一道工序：RegistrationKey（每张卡一个）-> WorkerKey（每个 TP 组一个）。
        # sorted() 不是可有可无：list 内的顺序对应 TP rank，各卡必须按同一顺序排列，否则
        # 每个 worker 拿到的 tensor 会被错位解读。
        for registration_key in sorted(self.all_gpu_blocks.keys()):
            worker_key = self.gpu_worker_key_mapping[registration_key]
            device_id = self.gpu_device_id_mapping[registration_key]
            if worker_key not in grouped_gpu_handles:
                grouped_gpu_handles[worker_key] = []
            grouped_gpu_handles[worker_key].append(
                self.storage_engine.get_storage_handle(DeviceType.GPU, device_id))

            if has_multi_group:
                if worker_key not in grouped_gpu_blocks_per_group:
                    grouped_gpu_blocks_per_group[worker_key] = []
                    grouped_gpu_layouts_per_group[worker_key] = []
                grouped_gpu_blocks_per_group[worker_key].append(
                    self.all_gpu_blocks_per_group[registration_key])
                grouped_gpu_layouts_per_group[worker_key].append(
                    self.all_gpu_layouts_per_group[registration_key])

        # 三级/远端层的 handle：全部由 StorageEngine 自己按 cache_config 分配，与 GPU 注册
        # 无关。enable_* 为假时传 None，TransferEngine 据此跳过对应通路。
        # remote 额外排除 mooncake store：那种后端下远端读写由 mooncake 自己管，不走
        # StorageEngine 的 REMOTE handle。
        cpu_handle = self.storage_engine.get_storage_handle(DeviceType.CPU) \
            if self.cache_config.enable_cpu else None
        ssd_handle = self.storage_engine.get_storage_handle(DeviceType.SSD) \
            if self.cache_config.enable_ssd else None
        use_mooncake_store = self.cache_config.use_mooncake_store_backend
        remote_handle = (
            self.storage_engine.get_storage_handle(DeviceType.REMOTE) \
            if self.cache_config.enable_remote and not use_mooncake_store \
            else None
        )
        # Group SWA GPU handles by WorkerKey, mirroring the main-KV grouping,
        # so the dedicated SWA worker map can be built per TP group.
        swa_gpu_handles: Optional[Dict[WorkerKey, List]] = None
        swa_grouped_gpu_blocks_per_group: Optional[Dict[WorkerKey, List]] = None
        swa_grouped_gpu_layouts_per_group: Optional[Dict[WorkerKey, List]] = None
        if self.swa_layer_groups is not None:
            swa_grouped_gpu_blocks_per_group = {}
            swa_grouped_gpu_layouts_per_group = {}
        if self.storage_engine.has_storage_handle(DeviceType.CPU, is_swa=True):
            swa_gpu_handles = {}
            for registration_key in sorted(self.all_swa_gpu_blocks.keys()):
                device_id = self.gpu_device_id_mapping[registration_key]
                if self.storage_engine.get_storage_handle(DeviceType.GPU, device_id, is_swa=True):
                    worker_key = self.gpu_worker_key_mapping[registration_key]
                    if worker_key not in swa_gpu_handles:
                        swa_gpu_handles[worker_key] = []
                    swa_gpu_handles[worker_key].append(
                        self.storage_engine.get_storage_handle(DeviceType.GPU, device_id, is_swa=True))
                    if self.swa_layer_groups is not None:
                        swa_grouped_gpu_blocks_per_group.setdefault(
                            worker_key, []
                        ).append(self.all_swa_gpu_blocks_per_group[registration_key])
                        swa_grouped_gpu_layouts_per_group.setdefault(
                            worker_key, []
                        ).append(self.all_swa_gpu_layouts_per_group[registration_key])

        swa_cpu_handle =(
         self.storage_engine.get_storage_handle(DeviceType.CPU, is_swa=True)
         if self.storage_engine.has_storage_handle(DeviceType.CPU, is_swa=True)
         else None
         )
        swa_ssd_handle = (
         self.storage_engine.get_storage_handle(DeviceType.SSD, is_swa=True)
         if self.storage_engine.has_storage_handle(DeviceType.SSD, is_swa=True)
         else None
         )
        swa_remote_handle = (
         self.storage_engine.get_storage_handle(DeviceType.REMOTE, is_swa=True)
         if self.storage_engine.has_storage_handle(DeviceType.REMOTE, is_swa=True)
         else None
         )

        self.transfer_engine = TransferEngine(
            gpu_handles=grouped_gpu_handles,
            model_config=self.model_config,
            cache_config=self.cache_config,
            cpu_handle=cpu_handle,
            ssd_handle=ssd_handle,
            remote_handle=remote_handle,
            gpu_blocks_per_group=grouped_gpu_blocks_per_group,
            gpu_layouts_per_group=grouped_gpu_layouts_per_group,
            swa_gpu_handles=swa_gpu_handles,
            swa_cpu_handle=swa_cpu_handle,
            swa_ssd_handle=swa_ssd_handle,
            swa_remote_handle=swa_remote_handle,
            swa_layer_groups=self.swa_layer_groups,
            swa_gpu_blocks_per_group=swa_grouped_gpu_blocks_per_group,
            swa_gpu_layouts_per_group=swa_grouped_gpu_layouts_per_group,
        )
        flexkv_logger.info(
            f"Initialized TransferEngine successfully, "
            f"grouped_gpu_handles keys={list(grouped_gpu_handles.keys())}, "
            f"num_gpu_groups={len(grouped_gpu_handles)}"
        )

    def submit(self, transfer_graph: TransferOpGraph) -> None:
        """把一张 TransferOpGraph 交给 TransferEngine 执行（非阻塞，纯转发）。

        这是控制面到数据面的唯一入口。本层**不解析**图内容——TransferOpGraph 里的 op
        语义（哪个 tier 到哪个 tier、哪些 block）由 GlobalCacheEngine 填好，这里只是透传，
        所以 GPU block id / slot_mapping 必须在此之前已经通过 graph.set_gpu_blocks 就位。

        Args:
            transfer_graph: 由 GlobalCacheEngine 产出的 DAG
        """
        self.transfer_engine.submit_transfer_graph(transfer_graph)

    def submit_batch(self, transfer_graphs: List[TransferOpGraph]) -> None:
        """批量提交多张图，与 submit 等价但只走一次 IPC/调用边界。

        存在的意义是**摊薄跨进程开销**：process 模式下每次 submit 是一次 Pipe send + pickle，
        batch 化之后 N 张图只付一次。所以上层在能攒批的时候应当优先用它。

        Args:
            transfer_graphs: 多张 DAG
        """
        self.transfer_engine.submit_transfer_graph(transfer_graphs)

    def wait(self, timeout: Optional[float] = None) -> List[CompletedOp]:
        """取出已经完成/失败的图与 op 通知（完成回传的出口）。

        回传方向是本层 -> KVTaskEngine 的"拉"模型：TransferEngine 把 CompletedOp 推入
        completed_queue，本函数一次性抽干并返回；上层 _get_completed_ops 再做 N 路聚合。
        没有任何 callback / 反向调用，这样跨进程、跨节点时就不需要反向通道。

        Args:
            timeout: 队列为空时最多阻塞的秒数；None 表示无限等待，0 表示非阻塞捞一次
        Returns:
            本次取到的 CompletedOp 列表，可能为空
        """
        return self.transfer_engine.get_completed_graphs_and_ops(timeout)

    def start(self) -> None:
        """启动数据面：TransferEngine 的 worker 子进程 + 调度线程。

        注意 start 之前必须先 initialize_transfer_engine()，否则 self.transfer_engine
        还是 None —— 两个方法的顺序由调用方（各 *_Handle 的 start）保证。
        """
        self.transfer_engine.start()

    def shutdown(self) -> None:
        """停掉数据面。用 hasattr 兜底是为了允许"初始化一半失败"时也能安全调用。"""
        if hasattr(self, 'transfer_engine'):
            self.transfer_engine.shutdown()

class TransferManagerOnRemote(TransferManager):
    """
    TransferManager for remote mode, used for multi-node tensor parallelism.

    中文补充：它是 TransferManagerMultiNodeHandle 在**服务端**的伙伴，两者组成一对 C/S：
        MultiNodeHandle（本机 KVTaskEngine 进程，客户端）
            command_socket PUSH bind   ->  submit / submit_batch / set_slot_mapping / config
            result_socket  PULL bind   <-  CompletedOp
            query_socket   REQ  bind   <-> query_ready 探活
        TransferManagerOnRemote（远端进程，本类，服务端）
            同名三个 socket 全部 connect 到上面那些端口（方向相反）

    名字里的 "OnRemote" 指"跑在远端那个进程里的 TransferManager"，而不是"远端的句柄"。
    它继承 TransferManager 因此自动获得 GPU 注册服务端 + StorageEngine/TransferEngine 装配
    的全部能力，本类只额外加了：ZMQ 服务端三件套、把 config 同步给它的握手流程、
    以及 _polling_worker 这个单线程事件泵。

    为什么它要额外维护两对 pending 字典（_pending_graphs / _pending_slot_mappings）：
    跨 PP 场景下图到达时 GPU block id 是被清空的（不同 PP stage 的显存块号不一样），真
    正的 slot_mapping 由框架稍后单独送来。两个消息谁先到不确定，所以做了双向兜底匹配。
    """

    def __init__(
        self,
        master_host: str,
        master_ports: Tuple[str, str, str],
    ):
        # 注意这里故意**不**调用 TransferManager.__init__：本类先从 ZMQ 收 config，拿到
        # model_config / cache_config / gpu_register_port 之后，才在 _initialize_with_config
        # 的末尾 super().__init__(...)。构造是"两段式"的。
        self.master_host = master_host
        self.master_ports = master_ports
        flexkv_logger.info(
            f"[TransferManagerOnRemote] master endpoint: "
            f"host={master_host!r}, ports={master_ports}"
        )

        # LINGER=0：进程要能被随时杀掉。默认 LINGER=-1 会让未发出去的消息阻塞 socket
        # 关闭，在远端进程这种生命周期短、可能被 SIGKILL 的场景下会挂住退出流程。
        self.context = zmq.Context()
        self.command_socket = self.context.socket(zmq.PULL)
        self.command_socket.setsockopt(zmq.LINGER, 0)
        self.result_socket = self.context.socket(zmq.PUSH)
        self.result_socket.setsockopt(zmq.LINGER, 0)
        self.query_socket = self.context.socket(zmq.REP)
        self.query_socket.setsockopt(zmq.LINGER, 0)

        self._shutdown_flag = False
        self._is_ready = False

        # graph_id -> task_end_op_id。远端进程靠它判断"这条 op 完成是否足以提前回报"，
        # 因为 task_end_op 之后的 op 失败与否对上层已经不重要（详见 _polling_worker）。
        # key: graph_id, value: task_end_op_id
        self._active_graphs: Dict[int, int] = {}
        self._active_graphs_lock = threading.Lock()

        # 双向兜底配对表：图先到 -> 存图等 slot_mapping；slot_mapping 先到 -> 存它等图。
        # 两者都按 task_id（= graph_id）索引，凑齐后才真正 submit 给 TransferEngine。
        self._pending_graphs: Dict[int, Tuple[TransferOpGraph, int]] = {}
        self._pending_slot_mappings: Dict[int, np.ndarray] = {}
        self._pending_lock = threading.Lock()

        self._worker_thread: threading.Thread | None = None

        self._connect_to_master_transfer_manager()

        self._initialize_with_config()
        flexkv_logger.info("Initialized TransferManagerOnRemote with config successfully")

    def _connect_to_master_transfer_manager(self) -> None:
        """连到本机 Master 端 MultiNodeHandle bound 的三个端口。

        command/result/query 三端口的顺序必须与 master_ports 元组一致，也就是
        (command, result, query)。
        """
        try:
            command_addr = f"tcp://{self.master_host}:{self.master_ports[0]}"
            self.command_socket.connect(command_addr)
            flexkv_logger.debug(f"Connected to master command port at {command_addr}")

            result_addr = f"tcp://{self.master_host}:{self.master_ports[1]}"
            self.result_socket.connect(result_addr)
            flexkv_logger.debug(f"Connected to master result port at {result_addr}")

            query_addr = f"tcp://{self.master_host}:{self.master_ports[2]}"
            self.query_socket.connect(query_addr)
            flexkv_logger.debug(f"Connected to master query port at {query_addr}")

            flexkv_logger.debug("Successfully connected to master transfer manager")

        except Exception as e:
            flexkv_logger.error(f"Failed to connect to master transfer manager: {e}")
            raise

    def _initialize_with_config(self) -> None:
        """两段式构造的第二段：从 master 收一份 config，然后才初始化父类。

        为什么要走 ZMQ 而不是 pickle 命令行参数：本进程由 subprocess.Popen 起（见
        create_process），只有 stdout/stderr 与环境变量；把 ModelConfig/CacheConfig
        序列化进命令行既难维护又有长度限制，所以统一走 ZMQ 握手。

        这里是**阻塞 recv**：本进程启动时 master 一定已经 send_config_to_remotes 了，
        等不到就直接失败比带着半成品配置继续跑更安全。
        """
        flexkv_logger.info(f"Waiting for config from master at {self.master_host}:{self.master_ports[0]}")
        config_msg = self.command_socket.recv_pyobj()
        if isinstance(config_msg, dict) and config_msg.get('type') == 'config':
            self.model_config = config_msg.get('model_config')
            self.cache_config = config_msg.get('cache_config')
            self.gpu_register_port = config_msg.get('gpu_register_port')
            flexkv_logger.info(f"Received config from master, {self.model_config = }, \
                {self.cache_config = }, {self.gpu_register_port = }.")
        else:
            raise RuntimeError(f"Expected config message, got: {config_msg}")
        flexkv_logger.info("Received config from master successfully")
        super().__init__(self.model_config, self.cache_config, self.gpu_register_port)

    def _polling_worker(self) -> None:
        """远端进程的主事件泵：收命令 -> 收 GPU 控制 -> 探活 -> 捞完成，循环。

        一个线程串起四件事，靠 zmq.Poller 统一监听，避免为每个端点各起一个线程。
        本类是 TransferManager 的子类，其 __init__ 无条件 bind 了 gpu_control_socket，
        而远端进程里没有 selector 循环去服务它，所以必须在这里一并 poll —— 漏掉这一句
        sleep/wake 就会静默卡满客户端 120s 的 RCVTIMEO。

        完成回传的两条 return 分支是本类的语义核心：
          1. 图级完成/失败 -> 回报一次并注销 active_graphs（终态，必须上报）
          2. task_end_op 完成 或 带 block_results -> 回报一次但**保留**在图里
             （首个分支：让上层在整图跑完前就能返回成功）
             （第二个分支：Mooncake 的 per-block 结果必须在那一刻就回传，不能等到图终态）
        两个 if 是独立的（不是 elif），所以一条同时满足两者的 CompletedOp 会被发送两次；
        上层按图 id 做幂等聚合（见 KVTaskEngine._get_completed_ops 的计数逻辑）。
        """
        flexkv_logger.info("Polling worker thread started")

        poller = zmq.Poller()
        poller.register(self.command_socket, zmq.POLLIN)
        poller.register(self.query_socket, zmq.POLLIN)
        # Inherited from TransferManager.__init__, which binds it
        # unconditionally. Nothing else in this process serves it.
        poller.register(self.gpu_control_socket, zmq.POLLIN)

        while not self._shutdown_flag:
            try:
                socks = dict(poller.poll(timeout=0.001))

                if self.command_socket in socks:
                    try:
                        message = self.command_socket.recv_pyobj(zmq.NOBLOCK)

                        if isinstance(message, dict):
                            msg_type = message.get('type')
                            if msg_type == 'submit':
                                graph = message.get('graph')
                                # -1 表示"没有 task_end_op"，即整图跑完才算完成
                                task_end_op_id = message.get('task_end_op_id', -1)

                                if graph is not None:
                                    self._handle_submit(graph, task_end_op_id)
                                else:
                                    flexkv_logger.warning("Received submit message without graph")
                            elif msg_type == 'submit_batch':
                                # batch 通路拿不到 per-graph 的 task_end_op_id（见
                                # TransferManagerMultiNodeHandle.submit_batch 发的消息体），
                                # 统一记 -1，因此批量提交的图只能等整图完成才回报。
                                graphs = message.get('graphs', [])
                                for graph in graphs:
                                    graph_id = graph.graph_id
                                    with self._active_graphs_lock:
                                        self._active_graphs[graph_id] = -1
                                    self.submit(graph)
                            elif msg_type == 'set_slot_mapping':
                                task_id = message.get('task_id')
                                slot_mapping = message.get('slot_mapping')
                                self._handle_set_slot_mapping(task_id, slot_mapping)
                            else:
                                flexkv_logger.warning(f"Unexpected command message: {message}")
                        else:
                            flexkv_logger.warning(f"Unexpected command message type: {type(message)}")
                    except zmq.Again:
                        pass

                if self.gpu_control_socket in socks:
                    self.drain_gpu_control_requests()

                if self.query_socket in socks:
                    try:
                        query_msg = self.query_socket.recv_pyobj(zmq.NOBLOCK)

                        if isinstance(query_msg, dict) and query_msg.get('type') == 'query_ready':
                            response = {'ready': self._is_ready}
                            self.query_socket.send_pyobj(response)
                        else:
                            response = {'error': 'unknown query type'}
                            self.query_socket.send_pyobj(response)
                            flexkv_logger.warning(f"Unknown query message: {query_msg}")
                    except zmq.Again:
                        pass

                try:
                    completed = self.wait(timeout=0.001)

                    if completed:
                        with self._active_graphs_lock:
                            for completed_op in completed:
                                if completed_op.graph_id in self._active_graphs:
                                    task_end_op_id = self._active_graphs[completed_op.graph_id]

                                    if (completed_op.block_results is not None
                                            or (task_end_op_id != -1
                                                and completed_op.op_id == task_end_op_id)):
                                        # Preserve backend-specific completion details
                                        # (notably Mooncake per-block results).
                                        self.result_socket.send_pyobj(completed_op)
                                    if (completed_op.is_graph_completed()
                                            or completed_op.is_graph_failed()):
                                        self.result_socket.send_pyobj(completed_op)
                                        del self._active_graphs[completed_op.graph_id]

                except queue.Empty:
                    pass

            except Exception as e:
                if not self._shutdown_flag:
                    flexkv_logger.error(f"Error in polling worker: {e}")
                    time.sleep(0.01)

        poller.unregister(self.command_socket)
        poller.unregister(self.query_socket)
        poller.unregister(self.gpu_control_socket)

    def _handle_set_slot_mapping(self, task_id: int, slot_mapping: np.ndarray) -> None:
        """Handle set_slot_mapping message from FlexKVConnector.

        When the graph (with cleared GPU blocks) arrived earlier, we can immediately
        set_gpu_blocks and submit.  Otherwise, store the slot_mapping and wait
        for the graph to arrive later.
        """
        graph = None
        task_end_op_id = -1
        with self._pending_lock:
            if task_id in self._pending_graphs:
                # Graph already arrived, set GPU blocks and prepare for submit
                graph, task_end_op_id = self._pending_graphs.pop(task_id)
                graph.set_gpu_blocks(slot_mapping)
                flexkv_logger.debug(
                    f"[TransferManagerOnRemote] set_slot_mapping: "
                    f"graph for task_id={task_id} submitted (graph arrived first)"
                )
            else:
                # Graph not yet arrived, store slot_mapping for later matching
                self._pending_slot_mappings[task_id] = slot_mapping
                flexkv_logger.debug(
                    f"[TransferManagerOnRemote] set_slot_mapping: "
                    f"slot_mapping stored for task_id={task_id}, waiting for graph"
                )
                return

        # Submit graph to transfer engine
        with self._active_graphs_lock:
            self._active_graphs[graph.graph_id] = task_end_op_id
        self.submit(graph)

    def _handle_submit(self, graph: TransferOpGraph, task_end_op_id: int = -1) -> None:
        """Handle submit message with pending matching support.

        If slot_mapping already arrived, set_gpu_blocks and submit immediately.
        Otherwise, store graph in pending_graphs for later matching.
        """
        task_id = graph.graph_id  # Use graph_id as task_id for matching
        with self._pending_lock:
            if task_id in self._pending_slot_mappings:
                # slot_mapping already arrived, set GPU blocks and submit
                slot_mapping = self._pending_slot_mappings.pop(task_id)
                graph.set_gpu_blocks(slot_mapping)
                flexkv_logger.debug(
                    f"[TransferManagerOnRemote] submit: "
                    f"graph for task_id={task_id} submitted (slot_mapping arrived first)"
                )
            else:
                # slot_mapping not yet arrived, store graph and task_end_op_id for later matching
                self._pending_graphs[task_id] = (graph, task_end_op_id)
                flexkv_logger.debug(
                    f"[TransferManagerOnRemote] submit: "
                    f"graph stored for task_id={task_id}, waiting for slot_mapping"
                )
                return  # Don't submit yet, wait for slot_mapping

        # Submit graph to transfer engine
        with self._active_graphs_lock:
            self._active_graphs[graph.graph_id] = task_end_op_id
        self.submit(graph)

    def start(self) -> None:
        """装配 + 启动本远端传输服务，并把 _is_ready 置真对外可被 query_ready 探到。

        顺序要点：必须先 initialize_transfer_engine()（阻塞收 GPU 注册）再 super().start()，
        最后才置 _is_ready —— 顺序反过来会让 master 误判远端已就绪并开始 submit。
        """
        self.initialize_transfer_engine()
        super().start()

        self._is_ready = True

        self._worker_thread = threading.Thread(
            target=self._polling_worker, daemon=True
        )
        self._worker_thread.start()

        flexkv_logger.info("TransferManagerOnRemote started successfully")

    def shutdown(self) -> None:
        flexkv_logger.info("Shutting down TransferManagerOnRemote")

        self._shutdown_flag = True
        self._is_ready = False

        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

        super().shutdown()

        try:
            self.command_socket.close()
            self.result_socket.close()
            self.query_socket.close()
            self.context.term()
        except Exception as e:
            flexkv_logger.error(f"Error closing sockets: {e}")

        flexkv_logger.info("TransferManagerOnRemote shutdown complete")

    def __del__(self) -> None:
        if not self._shutdown_flag:
            self.shutdown()

    @classmethod
    def create_process(cls, **kwargs: Any) -> Process:
        """用 subprocess.Popen 起一个干净的独立进程跑本类，返回类 Process 的包装对象。

        为什么不用 multiprocessing：**避免与推理框架自己的 MPI 初始化冲突**。
        fork/spawn 出来的子进程会继承父进程部分状态，而 TensorRT-LLM 这类框架要先 init MPI，
        MPI 明确不允许 fork 之后继续使用。subprocess.Popen 起的是全新解释器 + 全新进程，
        不继承这些状态，两边互不干扰。这也正是 KVTaskEngine 在
        use_trtllm_subprocess 下把本机 handle 也切成 remote 模式的原因。

        两个细节：
          1. 类对象与 kwargs 通过 pickle 落到临时文件传递，避免把大对象塞进命令行。
             临时文件由子进程自己 unlink，父进程再起一个 daemon 线程
             wait() 后二次 unlink（幂等，两层保险）。
          2. 必须清掉 CUDA_VISIBLE_DEVICES：TransferManager 要对本节点**所有**物理 GPU
             做 CUDA IPC import，而框架进程往往只给每卡可见的单卡环境，留着会看不到
             其它卡，注册必然卡在 expected_gpus。

        Args:
            **kwargs: 传给本类 __init__ 的参数，通常是 master_host / master_ports
        Returns:
            SubprocessWrapper，实现了 pid / is_alive / terminate / join / close
        """
        # Serialize the class and kwargs
        cls_data = pickle.dumps(cls)
        kwargs_data = pickle.dumps(kwargs)

        # Create temporary files for serialized data
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.cls') as f:
            f.write(cls_data)
            cls_file = f.name

        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.kwargs') as f:
            f.write(kwargs_data)
            kwargs_file = f.name

        # Prepare environment - remove MPI-related variables to avoid conflicts
        env = os.environ.copy()
        # CRITICAL: Remove CUDA_VISIBLE_DEVICES to allow access to all GPUs
        # TransferManager needs to access all physical GPUs for IPC
        if 'CUDA_VISIBLE_DEVICES' in env:
            flexkv_logger.info(f"Removing CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} "
                               "for TransferManager subprocess")
            env.pop('CUDA_VISIBLE_DEVICES', None)

        # Create the subprocess script
        transfer_manager_script = textwrap.dedent(f'''
            import os
            import sys
            import pickle
            import tempfile
            from flexkv.common.debug import flexkv_logger

            # Immediately disable MPI to avoid conflicts
            os.environ['MPI4PY_RC_INITIALIZE'] = 'false'

            try:
                # Load the class and kwargs
                with open("{cls_file}", "rb") as f:
                    cls = pickle.load(f)

                with open("{kwargs_file}", "rb") as f:
                    kwargs = pickle.load(f)

                # Create and start TransferManagerOnRemote instance
                flexkv_logger.info(f"Creating TransferManagerOnRemote instance...")
                instance = cls(**kwargs)
                flexkv_logger.info(f"Starting TransferManagerOnRemote instance...")
                instance.start()
                flexkv_logger.info(f"TransferManager instance started successfully")

                # Keep running until worker thread exits
                if hasattr(instance, '_worker_thread') and instance._worker_thread is not None:
                    instance._worker_thread.join()

            except Exception as e:
                print(f"Error in TransferManager subprocess: {{e}}", file=sys.stderr)
                sys.exit(1)
            finally:
                # Clean up temporary files
                try:
                    os.unlink("{cls_file}")
                    os.unlink("{kwargs_file}")
                except Exception:
                    pass
        ''').strip()

        # Start the subprocess
        process = subprocess.Popen([
            sys.executable, '-c', transfer_manager_script
        ], env=env, stdout=None, stderr=None, text=True)  # None = inherit parent's stdout/stderr
        flexkv_logger.info(f"TransferManager subprocess started, PID: {process.pid}")

        # Clean up temporary files after subprocess completes
        def cleanup_files():
            # Wait for subprocess to complete before cleaning up files
            process.wait()
            try:
                os.unlink(cls_file)
                os.unlink(kwargs_file)
            except Exception:
                pass

        cleanup_thread = threading.Thread(target=cleanup_files, daemon=True)
        cleanup_thread.start()

        # Return a wrapper that mimics multiprocessing.Process interface
        class SubprocessWrapper:
            def __init__(self, popen_process):
                self._popen = popen_process
                self.pid = popen_process.pid

            def is_alive(self):
                return self._popen.poll() is None

            def terminate(self):
                self._popen.terminate()

            def join(self, timeout=None):
                return self._popen.wait(timeout)

            def close(self):
                # Close the subprocess pipes
                if self._popen.stdout:
                    self._popen.stdout.close()
                if self._popen.stderr:
                    self._popen.stderr.close()
                if self._popen.stdin:
                    self._popen.stdin.close()

        return SubprocessWrapper(process)

class TransferManagerHandleBase(ABC):
    """所有传输句柄的统一接口：上层只认它，不认具体部署方式。

    六个方法构成完整生命周期：
        start()    -> 装配/拉起（各实现具体语义不同：建对象 / 起子进程 / 起线程）
        is_ready() -> 探活，KVTaskEngine 用它决定能否开始 submit
        submit()   -> 投一张图
        submit_batch() -> 投一批图
        wait()     -> 拉回 CompletedOp
        shutdown() -> 释放资源

    上层（KVTaskEngine）持有的是 List[TransferManagerHandle]，同一张图要投给列表里每一个
    handle，同一个完成通知要计数到 len(list) 才算真的完成。因此**每个实现的 submit 与该实
    现的 wait 必须严格一一对应**——漏一份就会出现"任务永远不完"。
    """
    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        pass

    @abstractmethod
    def submit(self, transfer_graph: TransferOpGraph, task_end_op_id: int = -1) -> None:
        pass

    @abstractmethod
    def submit_batch(self, transfer_graphs: List[TransferOpGraph]) -> None:
        pass

    @abstractmethod
    def wait(self, timeout: Optional[float] = None) -> List[CompletedOp]:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass


class TransferManagerIntraProcessHandle(TransferManagerHandleBase):
    """thread 模式句柄：TransferManager 就活在调用方进程里，无 IPC。

    适用场景：调试、单测、以及不方便起子进程的集成环境。
    代价是 TransferManager（连同它 spawn 的 transfer worker）与推理框架挤在同一进程：
    GIL 竞争、以及框架自身的 CUDA 操作会互相干扰，所以生产默认是 process 模式。

    它的 GPU 控制端点没法靠 selector 循环服务（那是 process 模式专属），所以 start() 里
    额外起一个守护线程 start_gpu_control_listener()，shutdown() 里必须配对着停掉。
    """

    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 gpu_register_port: str):
        self.transfer_manager = TransferManager(model_config, cache_config, gpu_register_port)
        self._is_ready = False

    def start(self) -> None:
        self.transfer_manager.initialize_transfer_engine()
        self.transfer_manager.start()
        # No selector loop here (that is the subprocess mode), so the bound GPU
        # control REP socket needs its own listener.
        self.transfer_manager.start_gpu_control_listener()
        self._is_ready = True

    def is_ready(self) -> bool:
        return self._is_ready

    def submit(self, transfer_graph: TransferOpGraph, task_end_op_id: int = -1) -> None:
        """同进程投递：直接把图对象交给 TransferManager，零序列化开销。

        task_end_op_id 在本实现里被忽略——"提前返回"是远端通路才需要的优化
        （本机 submit/wait 之间没有跨进程延迟，没必要提前告知上层）。
        """
        self.transfer_manager.submit(transfer_graph)

    def submit_batch(self, transfer_graphs: List[TransferOpGraph]) -> None:
        self.transfer_manager.submit_batch(transfer_graphs)

    def wait(self, timeout: Optional[float] = None) -> List[CompletedOp]:
        return self.transfer_manager.wait(timeout)

    def shutdown(self) -> None:
        # 顺序：先停 GPU 控制监听线程，再停数据面。反过来的话，在监听器 join 的那 1 秒内
        # 仍可能进来 suspend/resume 请求，打到已经关掉的 TransferEngine 上。
        self.transfer_manager.stop_gpu_control_listener()
        self.transfer_manager.shutdown()


class TransferManagerInterProcessHandle(TransferManagerHandleBase):
    """process 模式句柄（默认）：TransferManager 跑在一个 spawn 出来的子进程里。

    解决什么问题：把 FlexKV 的 CUDA 上下文、worker 子进程、内存映射全部关进一个隔离进程，
    不与推理框架互相干扰；同时这个进程可以继续独占 CPU 做传输调度，不受框架主进程影响。

    进程间只有两条 Pipe：
        command_parent_conn -> command_child_conn : 上层下发 submit / submit_batch / shutdown
        result_child_conn   -> result_parent_conn : 子进程上报 CompletedOp 列表
    Pipe 自带 pickle，所以 TransferOpGraph 必须是可 pickle 的（它的 GPU block id 是
    numpy/int 数组， TensorSharedHandle 之类的显存句柄不在这条链路上——那是 ZMQ 注册通道的事）。

    为什么用 spawn 而不是 fork：子进程要自己建 CUDA 上下文给它的子 worker 用，而 fork 会把
    父进程（推理框架）的 CUDA 状态一起复制过来，是死锁与随机崩溃的经典来源。

    完成回传是**事件驱动**的：子进程用 selectors 把 command_conn 和
    transfer_engine.completed_queue 的 _reader fd 一起监听，不做忙轮询。详见 _process_worker。
    """

    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 gpu_register_port: str):
        # spawn 而不是 fork：见类注释。这也是 model_config/cache_config 必须可 pickle
        # 地作为 Process args 传进去的原因（spawn 不继承内存）。
        self.mp_ctx = mp.get_context('spawn')

        self.model_config = model_config
        self.cache_config = cache_config
        self.gpu_register_port = gpu_register_port

        self.command_parent_conn, self.command_child_conn = self.mp_ctx.Pipe()
        self.result_parent_conn, self.result_child_conn = self.mp_ctx.Pipe()

        # start_event：子进程已跑到程序入口（early）——表示"进程起来了"
        # ready_event：子进程已完成 GPU 注册 + 引擎装配（late）——表示"可以 submit 了"
        # 两者分开，是为了让上层区分"进程卡住"与"注册卡住"这两种都能导致永不 ready 的故障。
        self.process: Optional[Process] = None
        self.start_event = self.mp_ctx.Event()
        self.ready_event = self.mp_ctx.Event()

        self._completed_results: List[CompletedOp] = []

    def _start_process(self) -> None:
        if self.process is not None and self.process.is_alive():
            return

        flexkv_logger.debug(
            f"Spawning TransferManager subprocess: "
            f"tp_size={self.model_config.tp_size}, dp_size={self.model_config.dp_size}, "
            f"gpu_register_port={self.gpu_register_port}")
        self.process = self.mp_ctx.Process(
            target=self._process_worker,
            args=(self.model_config,
                  self.cache_config,
                  self.command_child_conn,
                  self.result_child_conn,
                  self.gpu_register_port,
                  self.ready_event,
                  self.start_event),
            daemon=False
        )
        self.process.start()
        flexkv_logger.debug(f"TransferManager subprocess spawned, pid={self.process.pid}")

    def _process_worker(self,
                        model_config: ModelConfig,
                        cache_config: CacheConfig,
                        command_conn,
                        result_conn,
                        gpu_register_port: str,
                        ready_event,
                        start_event) -> None:
        """子进程的入口：装配 TransferManager 后进入事件循环，退出前保证清理。

        三条语义要点：
          1. 本进程自带一套信号处理 —— 忽略 SIGINT 是为了让父进程的
             kill_process_tree 掌控生死，避免 Ctrl+C 与 SIGTERM 清理路径撕裂；
             收到 SIGTERM 直接抛 SystemExit，让下面那个 finally 有机会跑到
             transfer_manager.shutdown()，给 worker 做配对的 cudaHostUnregister。
          2. 事件循环靠 selectors 监听三个 fd：命令 Pipe、完成队列的 _reader、
             GPU 控制 socket。不做忙轮询。
          3. 只要父进程消失（Pipe 报 EOF/Broken），就必须**走完整 shutdown**而不是
             os._exit：漏掉 unregister 会在 GPU 上留下悬垂的页锁定映射，下次启动就报
             地址冲突。这是 finally 里那段注释反复强调的原因。
        """
        # Automatically reap child processes (daemon transfer workers) to
        # prevent zombie accumulation.  Use a handler that calls waitpid()
        # with WNOHANG so that multiprocessing.Process.join() still works
        # correctly (SIG_IGN would cause join() to raise ChildProcessError).
        def _reap_children(signum, frame):
            while True:
                try:
                    pid, _ = os.waitpid(-1, os.WNOHANG)
                    if pid == 0:
                        break
                except ChildProcessError:
                    break
        signal.signal(signal.SIGCHLD, _reap_children)

        # Ignore Ctrl+C (SIGINT): process-group SIGINT would race with parent
        # kill_process_tree(SIGKILL). Only SIGTERM / {'type':'shutdown'} should
        # trigger paired worker unregister. SIGKILL cannot be handled.
        def _on_sigterm(signum, frame):
            flexkv_logger.warning(
                f"TransferManager process received signal {signum}; exiting for graceful cleanup"
            )
            raise SystemExit(0)

        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, _on_sigterm)
        except Exception as e:
            flexkv_logger.warning(
                f"Failed to install TransferManager shutdown signal handlers: {e}"
            )

        try:
            flexkv_logger.debug(f"_process_worker started, pid={os.getpid()}, "
                               f"gpu_register_port={gpu_register_port}")
            start_event.set()
            os.environ['MPI4PY_RC_INITIALIZE'] = 'false'
            transfer_manager = TransferManager(model_config, cache_config, gpu_register_port)
            transfer_manager.initialize_transfer_engine()
            transfer_manager.start()
            flexkv_logger.debug("TransferEngine started successfully, setting ready_event")
            ready_event.set()

            # Setup selector for event-driven processing (complete zero polling!)
            sel = selectors.DefaultSelector()
            sel.register(command_conn.fileno(), selectors.EVENT_READ, data="command")
            # Also monitor completed_queue for finished ops (now it's mp.Queue with _reader)
            sel.register(transfer_manager.transfer_engine.completed_queue._reader,
                        selectors.EVENT_READ, data="finished_ops")
            sel.register(transfer_manager.gpu_control_socket,
                         selectors.EVENT_READ, data="gpu_control")

            flexkv_logger.info(
                "TransferManager daemon process started with selector-based "
                "event monitoring (command + finished_ops)"
            )

            should_exit = False
            while not should_exit:
                try:
                    # Event-driven: wait for command OR finished_ops.
                    # Graceful exit via {'type': 'shutdown'} command (or SIGTERM→SystemExit).
                    events = sel.select(timeout=None)

                    # Process all events
                    has_finished_ops = False

                    for key, mask in events:
                        if key.data == "command":
                            # New command available
                            inner_range = nvtx.start_range(message="TransferManagerInter.process_worker.req", color="red")
                            try:
                                request = command_conn.recv()
                            except (EOFError, BrokenPipeError, ConnectionResetError) as e:
                                # Parent (scheduler) died without sending shutdown.
                                # Break out and let finally run transfer_manager.shutdown()
                                # so workers get paired cudaHostUnregister.
                                flexkv_logger.warning(
                                    f"TransferManager command pipe closed ({e!r}); "
                                    "parent likely crashed. Exiting for graceful cleanup."
                                )
                                should_exit = True
                                nvtx.end_range(inner_range)
                                break
                            request_type = request.get('type')
                            if request_type == 'submit':
                                transfer_manager.submit(request['transfer_graph'])
                            elif request_type == 'submit_batch':
                                transfer_manager.submit_batch(request['transfer_graphs'])
                            elif request_type == 'shutdown':
                                flexkv_logger.info(
                                    "TransferManager received shutdown command; "
                                    "leaving event loop for graceful cleanup"
                                )
                                should_exit = True
                            else:
                                flexkv_logger.error(f"Unrecognized request type: {request_type}")
                            nvtx.end_range(inner_range)

                        elif key.data == "finished_ops":
                            # Selector reports finished_ops queue has data
                            has_finished_ops = True

                        elif key.data == "gpu_control":
                            # ZeroMQ exposes an edge-triggered FD, so one event
                            # must consume every request that is already queued.
                            transfer_manager.drain_gpu_control_requests()

                    # Only collect finished_ops if selector reported data available
                    if has_finished_ops and not should_exit:
                        inner_range = nvtx.start_range(message="TransferManagerInter.process_worker.results", color="red")
                        try:
                            # Directly get from completed_queue without timeout to avoid poll
                            finished_ops = []
                            completed_queue = transfer_manager.transfer_engine.completed_queue
                            while not completed_queue.empty():
                                try:
                                    finished_ops.append(completed_queue.get_nowait())
                                except queue.Empty:
                                    break

                            if finished_ops:
                                result_conn.send(finished_ops)
                        except Exception as e:
                            flexkv_logger.error(f"Error collecting finished ops: {e}")
                        nvtx.end_range(inner_range)

                except (EOFError, BrokenPipeError, ConnectionResetError) as e:
                    # Fallback: any IPC-broken exception bubbling up here also
                    # means the parent is gone — exit for graceful cleanup.
                    flexkv_logger.warning(
                        f"TransferManager IPC error ({e!r}); "
                        "parent likely crashed. Exiting for graceful cleanup."
                    )
                    should_exit = True
                except Exception as e:
                    flexkv_logger.error(f"Error in transfer manager process: {e}")

        except Exception as e:
            flexkv_logger.error(f"Failed to initialize transfer manager process: {e}")
        finally:
            # Cleanup selector (only if it was created)
            if 'sel' in locals():
                try:
                    sel.close()
                except Exception as e:
                    flexkv_logger.error(f"Error closing selector: {e}")

            # Gracefully shut down transfer engine and its worker subprocesses
            if 'transfer_manager' in locals():
                try:
                    flexkv_logger.info("TransferManager process: shutting down transfer engine")
                    transfer_manager.shutdown()
                except Exception as e:
                    flexkv_logger.error(f"Error shutting down transfer manager: {e}")

            try:
                command_conn.close()
            except Exception:
                pass
            try:
                result_conn.close()
            except Exception:
                pass
            flexkv_logger.info("TransferManager process cleanup complete")

    def start(self) -> None:
        """拉起子进程并等它跑到入口点。

        注意这里等的是 start_event（进程已启动）而**不是** ready_event：GPU 注册可能要等
        几十秒（要卡满意 expected_gpus），start 不应被它拖住；能不能 submit 由上层一次次
        轮询 is_ready() 判断。

        MPI4PY_RC_INITIALIZE 在 Process 创建前后被短暂关掉再恢复：spawn 时刻若让 mpi4py
        自动初始化，会在不需要 MPI 的子进程里白白拉起一套 MPI，且与框架的 MPI 冲突。
        """
        os.environ['MPI4PY_RC_INITIALIZE'] = 'false'
        self._start_process()
        self.start_event.wait()
        os.environ['MPI4PY_RC_INITIALIZE'] = 'true'

    def is_ready(self) -> bool:
        """子进程是否已完成 GPU 注册与引擎装配（对应 _process_worker 里的 ready_event）。"""
        return self.ready_event.is_set()

    def submit(self, transfer_graph: TransferOpGraph, task_end_op_id: int = -1) -> None:
        """把图 pickle 过 Pipe 发给子进程。非阻塞：Pipe send 只是入缓冲，返回不代表已执行。

        注意 task_end_op_id 在本实现里不透传（消息体里没有它）："提前返回"是远端通路才
        需要的优化，本机 Pipe 延迟极小，不必为此多一层状态。
        """
        nvtx_range = nvtx.start_range(message="TransferManagerInterProcessHandle.submit", color="green")
        self.command_parent_conn.send({
            'type': 'submit',
            'transfer_graph': transfer_graph
        })
        nvtx.end_range(nvtx_range)

    def submit_batch(self, transfer_graphs: List[TransferOpGraph]) -> None:
        # Batch submit to reduce IPC overhead
        nvtx_range = nvtx.start_range(
            message=f"TransferManagerInterProcessHandle.submit_batch count={len(transfer_graphs)}",
            color="green"
        )
        self.command_parent_conn.send({
            'type': 'submit_batch',
            'transfer_graphs': transfer_graphs
        })
        nvtx.end_range(nvtx_range)

    def wait(self, timeout: Optional[float] = None) -> List[CompletedOp]:
        """从结果 Pipe 拉回子进程攒好的 CompletedOp 批。

        子进程是"攒一批发一次"，所以这里收到的是一个 list；poll 拿到第一批后继续 poll
        把已经排队的后续批次一次性取干净，避免积压到下一次调用。
        EOFError 被吞掉不抛：父进程可能在 shutdown 竞态下先关掉了 Pipe 的那一端。
        """
        finished_ops: List[CompletedOp] = []
        try:
            if self.result_parent_conn.poll(timeout=timeout):
                received_ops = self.result_parent_conn.recv()
                finished_ops += received_ops
                while self.result_parent_conn.poll():
                    received_ops = self.result_parent_conn.recv()
                    finished_ops += received_ops
        except EOFError:
            pass

        return finished_ops

    def shutdown(self) -> None:
        """优雅关闭：先发 shutdown 命令给子进程（让它自己走清理），超时才升级为信号。

        三级降级是有意为之，不是冗余：
          send({'type':'shutdown'}) -> join(timeout) -> terminate() -> join(30) -> kill()
        因为 transfer worker 手上可能持有 GPU 页锁定映射和未写完的 SSD 数据，直接 SIGKILL
        会留下资源泄漏；而此刻又绝不能无限等待（父进程要退出）。超时常量来自
        GLOBAL_CONFIG_FROM_ENV，便于在不同部署里调。
        """
        if self.process is None:
            return

        if self.process.is_alive():
            try:
                flexkv_logger.info(
                    "Sending graceful shutdown command to TransferManager subprocess "
                    f"(pid={self.process.pid})"
                )
                self.command_parent_conn.send({'type': 'shutdown'})
            except (BrokenPipeError, OSError, EOFError) as e:
                flexkv_logger.warning(
                    f"Failed to send TransferManager shutdown command: {e}; "
                    "falling back to terminate"
                )
                self.process.terminate()

            timeout = float(GLOBAL_CONFIG_FROM_ENV.transfer_manager_shutdown_timeout_s)
            self.process.join(timeout=timeout)
            if self.process.is_alive():
                flexkv_logger.warning(
                    f"TransferManager still alive after {timeout:.0f}s graceful wait; "
                    "terminating"
                )
                self.process.terminate()
                self.process.join(timeout=30)
                if self.process.is_alive():
                    flexkv_logger.warning(
                        "TransferManager still alive after terminate; killing"
                    )
                    self.process.kill()
                    self.process.join()

        try:
            self.command_parent_conn.close()
        except Exception:
            pass
        try:
            self.result_parent_conn.close()
        except Exception:
            pass
        self.process = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass


class TransferManagerMultiNodeHandle(TransferManagerHandleBase):
    """remote 模式句柄（客户端）：本机 KVTaskEngine 与远端 TransferManagerOnRemote 之间的 ZMQ 桥梁。

    它**不持有**任何 StorageEngine / TransferEngine，GPU 注册也不在这里发生——所有重活都在
    远端那个进程里做。本类只负责：bind 三个端口、发 config、转 submit、收 CompletedOp、探活。

    角色对照（本类是左列）：
        本类（Master 侧，bind）              TransferManagerOnRemote（Remote 侧，connect）
        command_socket PUSH  ->              -> PULL   下发 config / submit / set_slot_mapping
        result_socket  PULL  <-              <- PUSH   上收 CompletedOp
        query_socket   REQ   <->             <-> REP   探活 query_ready

    注意 socket 方向与名字相反：命名是从"本句柄视角"出发的，command_socket 指"我用来发
    命令的 socket"，在 ZMQ 层却是 PUSH+bind。

    三个 query_socket 的 options 都有讲究：
        REQ_RELAXED=1 允许跳过严格的一问一答（上次超时后可以重发而不崩状态机）
        REQ_CORRELATE=1 给请求编号，避免丢包后把上次的答复当成这次的
        RCVTIMEO=1000   探活不能拖住主流程，1 秒内没收就当远端没就绪
    """

    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 gpu_register_port: str,
                 master_host: str,
                 master_ports: Tuple[str, str, str]):  # command, result, query
        self.model_config = model_config
        self.cache_config = cache_config
        self.gpu_register_port = gpu_register_port

        self.master_host = master_host
        self.master_ports = master_ports

        self.context = zmq.Context()
        self.command_socket = self.context.socket(zmq.PUSH)
        self.command_socket.setsockopt(zmq.LINGER, 0)
        self.result_socket = self.context.socket(zmq.PULL)
        self.result_socket.setsockopt(zmq.LINGER, 0)
        self.query_socket = self.context.socket(zmq.REQ)
        self.query_socket.setsockopt(zmq.LINGER, 0)
        self.query_socket.setsockopt(zmq.REQ_RELAXED, 1)
        self.query_socket.setsockopt(zmq.REQ_CORRELATE, 1)
        self.query_socket.setsockopt(zmq.RCVTIMEO, 1000)

        self._shutdown_flag = False
        self._connected = False

        # 结果缓冲区：ZMQ PULL 有一条后台 polling 线程收，wait() 从缓冲区取。
        # 为什么中间加一道：wait 是上层按自己的节奏轮询调用的，到达时刻与 ZMQ 无关；没有缓冲区就得让 ZMQ socket 自己堵着，导致 shutdown 时无法
        # 及时退出。改成"线程收 + Wait 取"后，wait 可以完全超时可控。
        self._result_buffer: List[CompletedOp] = []
        self._result_buffer_lock = threading.Lock()

        self._bind_master_ports()

        self._polling_thread: threading.Thread | None = None

    def _bind_master_ports(self) -> None:
        """bind 三个 TCP 端口并把自己标为 connected；失败要彻底关掉 context 再抛。

        bind（而非 connect）表明本句柄是拓扑里的 Master：远端进程会主动连过来，因此可以
        在远端进程还没起来时就先把 config 排到队列里（PUSH 在无人 connect 时也会缓存）。
        """
        try:
            command_addr = f"tcp://{self.master_host}:{self.master_ports[0]}"
            self.command_socket.bind(command_addr)
            flexkv_logger.info(f"Master bound command port at {command_addr}")

            result_addr = f"tcp://{self.master_host}:{self.master_ports[1]}"
            self.result_socket.bind(result_addr)
            flexkv_logger.info(f"Master bound result port at {result_addr}")

            query_addr = f"tcp://{self.master_host}:{self.master_ports[2]}"
            self.query_socket.bind(query_addr)
            flexkv_logger.info(f"Master bound query port at {query_addr}")

            self.result_socket.setsockopt(zmq.RCVTIMEO, 0)

            self._connected = True
            flexkv_logger.info("Master transfer manager ready for remote connections")

        except Exception as e:
            flexkv_logger.error(f"Master failed to bind ports: {e}")
            try:
                self.command_socket.close()
                self.result_socket.close()
                self.query_socket.close()
                self.context.term()
            except Exception:
                pass
            raise

    def send_config_to_remotes(self) -> None:
        """把 ModelConfig / CacheConfig / gpu_register_port 打包推给远端。

        这是**在本 handle 构造之后由上层显式调用**的（见 KVTaskEngine 的
        transfer_handles[-1]._handle.send_config_to_remotes()），因为远端进程必须先拿到
        config 才能构造 TransferManager（两段式构造的第一段）。放在 start() 里也行，但那样
        会要求远端进程在 start 之前就已经起来。

        gpu_register_port 被透传过去意味着：远端进程在自己机器上 bind 同一个 IPC 端口收
        GPU 注册——即每个节点各收齐自己那批 GPU，而不是跨节点汇总。
        """
        flexkv_logger.info(f"Sending config to remote at {self.master_host}:{self.master_ports[0]}")
        try:
            config_msg = {
                'type': 'config',
                'model_config': self.model_config,
                'cache_config': self.cache_config,
                'gpu_register_port': self.gpu_register_port
            }
            self.command_socket.send_pyobj(config_msg)
            flexkv_logger.info(f"Config sent to remote at {self.master_host}:{self.master_ports[0]}")
        except Exception as e:
            flexkv_logger.error(f"Failed to send config to remote: {e}")

    def _polling_worker(self) -> None:
        """后台收结果线程：把远端推来的 CompletedOp 攒进 _result_buffer 等 wait() 取走。

        这里用 NOBLOCK recv + 1ms 睡觉的轮询而不是 ZMQ Poller：线程需要在 _shutdown_flag
        变化后尽快退出，带 1ms 超时的轮询已经足够轻（这条通路的上报频率不高）。
        """
        while not self._shutdown_flag:
            try:
                result = self.result_socket.recv_pyobj(zmq.NOBLOCK)
                if isinstance(result, CompletedOp):
                    with self._result_buffer_lock:
                        self._result_buffer.append(result)
                else:
                    flexkv_logger.warning(f"Unexpected result format from remote: {result}")

            except zmq.Again:
                time.sleep(0.001)
            except Exception as e:
                if not self._shutdown_flag:
                    flexkv_logger.error(f"Error in polling thread: {e}")
                    time.sleep(0.01)

    def start(self) -> None:
        """只起后台收结果线程。远端可能还在拉起/收 GPU 注册，所以这里不做任何等待。"""
        self._polling_thread = threading.Thread(target=self._polling_worker, daemon=True)
        self._polling_thread.start()

    def is_ready(self) -> bool:
        """向远端发一次 query_ready 探活（1s 超时），返回远端是否已完成装配。

        上层 KVTaskEngine 会反复调它直到所有 handle 都就绪。返回 False 只代表"这一刻还没好
        "，不代表出错——远端可能还在等自己那批 GPU 注册。
        """
        if not self._connected:
            flexkv_logger.warning("Master not ready: ports not bound yet")
            return False

        try:
            query_msg = {'type': 'query_ready'}
            self.query_socket.send_pyobj(query_msg)

            response = self.query_socket.recv_pyobj()
            if response.get('ready'):
                return True
            else:
                flexkv_logger.warning(f"Remote not ready, response: {response}")
                return False

        except zmq.Again:
            flexkv_logger.warning("Timeout waiting for ready response from remote")
            return False
        except Exception as e:
            flexkv_logger.error(f"Error checking remote ready status: {e}")

            return False

    def submit(self, transfer_graph: TransferOpGraph, task_end_op_id: int = -1) -> None:
        """把图 + task_end_op_id 打包成 dict 发给远端。

        task_end_op_id 是这条通路特有的优化：远端收到后记进 _active_graphs，等该 op 完成就
        立刻回报，让上层不必等整图跑完（典型是"task_end 之后的 mooncake 上传异步收尾"）。
        -1 表示没有提前返回点。

        顺带一个隐含约定：上层在跨机 PP 场景下会先 clear_gpu_blocks() 再提交（因为不同 PP
        stage 的显存块号不同），真 slot_mapping 由框架稍后单独经 set_slot_mapping 补上；
        远端侧的两个 pending 字典负责这两种顺序的兜底。

        Args:
            transfer_graph: 要执行的 DAG
            task_end_op_id: 可提前回报的界碑 op id，-1 表示无
        """
        if not self._connected:
            flexkv_logger.warning("Not connected to remote transfer manager")
            return

        try:
            message = {
                'type': 'submit',
                'graph': transfer_graph,
                'task_end_op_id': task_end_op_id
            }
            self.command_socket.send_pyobj(message)

        except Exception as e:
            flexkv_logger.error(f"Failed to submit graph to remote: {e}")

    def submit_batch(self, transfer_graphs: List[TransferOpGraph]) -> None:
        if not self._connected:
            flexkv_logger.warning("Not connected to remote transfer manager")
            return

        try:
            message = {
                'type': 'submit_batch',
                'graphs': transfer_graphs
            }
            self.command_socket.send_pyobj(message)

        except Exception as e:
            flexkv_logger.error(f"Failed to submit batch graphs to remote: {e}")

    def wait(self, timeout: float | None = None) -> List[CompletedOp]:
        """从 _result_buffer 取走已到的结果，最多等 timeout 秒。

        注意这是一个忙轮询循环（1ms）：timeout=0 时只看缓冲一眼就返回，非 0 时也能在
        shutdown 置位后的毫秒级退出，比阻塞在 ZMQ 上更好控制。
        """
        start_time = time.time()
        results = []

        while True:
            with self._result_buffer_lock:
                if self._result_buffer:
                    results.extend(self._result_buffer)
                    self._result_buffer.clear()
                    break
                elif timeout is not None and (time.time() - start_time) >= timeout:
                    break

            time.sleep(0.001)

        return results

    def shutdown(self) -> None:
        flexkv_logger.info("Shutting down TransferManagerMultiNodeHandle")

        self._shutdown_flag = True

        if self._polling_thread is not None and self._polling_thread.is_alive():
            self._polling_thread.join(timeout=5.0)

        try:
            self.command_socket.close()
            self.result_socket.close()
            self.query_socket.close()
            self.context.term()
        except Exception as e:
            flexkv_logger.error(f"Error closing sockets: {e}")

        flexkv_logger.info("TransferManagerMultiNodeHandle shutdown complete")


class TransferManagerHandle:
    """传输句柄门面：上层唯一接触的类型，按 mode 选一个具体 *_Handle 委托。

    存在的意义是把"FlexKV 的数据面到底跑在哪"这个问题关在这一个构造函数里。
    KVTaskEngine 只写 TransferManagerHandle(model_config, cache_config, mode=...)，
    之后的 start/submit/wait/shutdown 完全一致。

    三种 mode 的选择依据：
        process（默认）-> TransferManagerInterProcessHandle
            独立子进程，隔离最好。适合绝大多数部署，包括 vLLM / SGLang。
        thread          -> TransferManagerIntraProcessHandle
            同进程，无 IPC，便于断点调试；代价是与框架共享进程。
        remote          -> TransferManagerMultiNodeHandle
            跨节点（nnodes>1）必选；以及 TRT-LLM 这种要先 init MPI、不允许再 fork 的
            框架——那时连本机也要改用 subprocess 拉起的远端服务（见 TransferManagerOnRemote
            .create_process）。

    Args:
        model_config / cache_config: 传给真正干活的那层
        gpu_register_port: GPU 注册的 IPC 端点。留 None 时会临时造一个 ipc:// 临时文件路径，
                           让同一台机上的一次性部署不必显式配端口
        mode: "process" | "thread" | "remote"
        **kwargs: remote 模式额外需要 master_host / master_ports
    """

    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 gpu_register_port: Optional[str] = None,
                 mode: str = "process",
                 **kwargs): # process or thread or remote
        flexkv_logger.debug(
            f"Creating TransferManagerHandle: mode={mode}, "
            f"tp_size={model_config.tp_size}, dp_size={model_config.dp_size}, "
            f"pp_size={model_config.pp_size}, nnodes={model_config.nnodes}, "
            f"gpu_register_port={gpu_register_port}")
        if gpu_register_port is None:
            gpu_register_port = f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
        if mode == "process":
            self._handle: TransferManagerHandleBase = TransferManagerInterProcessHandle(
                model_config, cache_config, gpu_register_port
            )
        elif mode == "thread":
            self._handle: TransferManagerHandleBase = TransferManagerIntraProcessHandle(
                model_config, cache_config, gpu_register_port
            )
        elif mode == "remote":
            master_host = kwargs["master_host"]
            master_ports = kwargs["master_ports"]
            self._handle: TransferManagerHandleBase = TransferManagerMultiNodeHandle(
                model_config, cache_config, gpu_register_port, master_host, master_ports
            )
        else:
            raise ValueError(f"Invalid mode: {mode}, must be process, thread or remote")

    def start(self) -> None:
        """委托给具体 handle 的 start。process 模式下这只是"起进程"，并不等于就绪。"""
        self._handle.start()

    def is_ready(self) -> bool:
        """数据面是否可以接收 submit。上层应轮询到全部 handle 都返回 True 才开工。"""
        return self._handle.is_ready()

    def submit(self, transfer_graph: TransferOpGraph, task_end_op_id: int = -1) -> None:
        """把 DAG 投递到本 handle 代表的那份数据面。非阻塞。

        上层对 List[TransferManagerHandle] 里的每一个都调用一次，并期待 len(list) 份
        完成通知（KVTaskEngine.required_completed_count），两者必须严格相等。
        """
        self._handle.submit(transfer_graph, task_end_op_id)

    def submit_batch(self, transfer_graphs: List[TransferOpGraph]) -> None:
        self._handle.submit_batch(transfer_graphs)

    def wait(self, timeout: Optional[float] = None) -> List[CompletedOp]:
        """捞回本路已完成的 CompletedOp；上层负责与其余路的结果按 (graph_id, op_id) 聚合。"""
        return self._handle.wait(timeout)

    def shutdown(self) -> None:
        self._handle.shutdown()
