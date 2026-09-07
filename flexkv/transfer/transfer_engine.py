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
# ==============================================================================
# 本文件职责：数据面的调度中枢。把控制面产出的传输 DAG(TransferOpGraph) 拆成一个个
# op，派发给具体的 Worker 子进程执行，并把完成情况汇聚回上层。
#
# 在系统链路中的位置：
#   KVManager -> KVTaskEngine -> GlobalCacheEngine(控制面：查索引、判命中、决定
#     "搬哪些 block / 走哪条通路"，产出 TransferOpGraph)
#     -> 【本文件 TransferEngine】(数据面：决定"派给哪个 worker、何时算搬完")
#       -> Worker(真正搬字节的独立进程) -> c_ext(csrc/bindings.cpp)
#
# 与控制面 GlobalCacheEngine 的分工（务必分清）：
#   控制面：认识 KV block 的语义，负责"搬什么"，编完 DAG 就撒手。
#   本文件：不认识 KV block，只认识 op / graph / transfer_type，负责"谁来搬"。
#   一句话：控制面决策，本文件执行调度。
#
# 核心内容速查：
#   - TransferEngine.__init__           : 建三个队列、pin_buffer、shutdown pipe
#   - TransferEngine._init_workers      : 按 TP/GDS/SWA/Layerwise 拓扑创建 worker
#   - TransferEngine._scheduler_loop    : 单线程 selectors 事件循环，本文件的心脏
#   - TransferEngine._assign_op_to_worker      : op -> worker 的路由（含 PP fan-out）
#   - TransferEngine.submit_transfer_graph     : 上层提交 DAG 的入口（非阻塞）
#   - TransferEngine.get_completed_graphs_and_ops : 上层回收完成通知的出口
#
# 阅读提示：
#   * 先读 flexkv/transfer/scheduler.py（本文件调用它做 DAG 就绪判定），再读本文件。
#     二者是上下游而非替代关系：scheduler.py 管"DAG 里哪些 op 已就绪可调度"，
#     本文件管"就绪的 op 派给哪个 worker 进程、完成事件怎么回收"。
#   * 跨进程通信全靠三个 mp.Queue + 每个 worker 一条 Pipe：
#       task_queue         : 上层 -> 调度线程（提交 DAG，支持 list 批量）
#       finished_ops_queue : worker -> 调度线程（op 完成/失败回报，多 worker 共享）
#       completed_queue    : 调度线程 -> 上层（CompletedOp 完成通知）
#     用 mp.Queue 而非 queue.Queue，是因为要拿它的 _reader fd 注册进 selectors。
#   * 一个 op 往往会"扇出"成多个 replica（PP 流水并行的每个 stage 一份），
#     靠 op.pending_count 归零判定整个 op 完成 —— 这是理解本文件的关键。
# ==============================================================================
import queue
import threading
import time
import multiprocessing as mp
import selectors
import os
from typing import Dict, List, Optional, Set, Tuple, Union

import contextlib
import nvtx
import numpy as np
import torch

from flexkv.common.debug import flexkv_logger
from flexkv.common.storage import StorageHandle
from flexkv.common.transfer import TransferOp, TransferOpGraph, TransferType, CompletedOp, WorkerKey
from flexkv.common.transfer import get_nvtx_range_color
from flexkv.transfer.scheduler import TransferScheduler
from flexkv.transfer import trace
from flexkv.transfer.worker import (
    WorkerHandle,
    CPUSSDDiskTransferWorker,
    CPURemoteTransferWorker,
    GPUCPUTransferWorker,
    tpGPUCPUTransferWorker,
    GDSTransferWorker,
    tpGDSTransferWorker,
    NixlTransferWorker,
    PEER2CPUTransferWorker,
    MooncakeStoreTransferWorker,
)
from flexkv.external.mooncake_store_keys import PoolKind
from flexkv.transfer.compression import build_compressors
from flexkv.transfer.layerwise import (
    LayerwiseTransferWorker,
    build_layerwise_eventfd_socket_path,
)
from flexkv.transfer.worker_op import WorkerTransferResult
from flexkv.common.config import (
    CacheConfig, LayerGroupSpec, ModelConfig, GLOBAL_CONFIG_FROM_ENV,
)
from flexkv.common.ring_buffer import SharedOpPool


def register_op_to_buffer(op: TransferOp, pin_buffer: SharedOpPool) -> None:
    """
    Register transfer operation to buffer with device type prefixes.

    Device type prefixes prevent hash collisions when different device types
    use the same block ID values (e.g., CPU block 0 vs SSD block 0).

    中文补充：给 op 的 src/dst block id 在共享 op_buffer(SharedOpPool) 里申请槽位，
    槽位号写回 op.src_slot_id / op.dst_slot_id。op 跨进程只传 block id 数组，
    真正的传输元数据放在共享内存槽里由 worker 按 slot_id 自取，省掉每 op 一次
    大数组 pickle 拷贝。
    transfer_type_to_devices 里的魔法数字是设备类型编号（1=GPU 2=CPU 3=SSD
    4=REMOTE 5=PEER_CPU 6=PEER_SSD），作为前缀参与槽位哈希，防止"CPU block 0
    与 SSD block 0"这类同号不同设备的 block 撞进同一个槽 —— 撞槽的表现是
    数据被写到别的 op 的 buffer 里，极难排查。
    LAYERWISE op 直接返回不占槽：它由 LayerwiseTransferWorker 自己管理数据。
    """
    if op.transfer_type == TransferType.LAYERWISE:
        return
    # Map TransferType to (src_device_type, dst_device_type) for hash prefix
    # This prevents hash collisions when different devices use the same block IDs
    transfer_type_to_devices = {
        TransferType.D2H: (1, 2),      # GPU -> CPU
        TransferType.H2D: (2, 1),      # CPU -> GPU
        TransferType.H2DISK: (2, 3),   # CPU -> SSD
        TransferType.DISK2H: (3, 2),   # SSD -> CPU
        TransferType.DISK2D: (3, 1),   # SSD -> GPU
        TransferType.D2DISK: (1, 3),   # GPU -> SSD
        TransferType.H2REMOTE: (2, 4), # CPU -> REMOTE
        TransferType.REMOTE2H: (4, 2), # REMOTE -> CPU
        TransferType.PEERH2H: (5, 2),  # PEER_CPU -> CPU
        TransferType.H2PEERH: (2, 5),  # CPU -> PEER_CPU
        TransferType.PEERSSD2H: (6, 2),# PEER_SSD -> CPU
        TransferType.H2PEERSSD: (2, 6),# CPU -> PEER_SSD
    }

    src_device, dst_device = transfer_type_to_devices.get(op.transfer_type, (0, 0))

    op.src_slot_id = pin_buffer.allocate_slot(op.src_block_ids, device_type_prefix=src_device)
    op.dst_slot_id = pin_buffer.allocate_slot(op.dst_block_ids, device_type_prefix=dst_device)

def free_op_from_buffer(op: TransferOp, pin_buffer: SharedOpPool) -> None:
    """释放 register_op_to_buffer 申请的槽位（register 的逆操作）。

    槽位号 -1 表示"没申请过"（未注册或 LAYERWISE），跳过即可，所以本函数对
    重复释放是安全的。
    约定：每个 op 要么在调度线程注册并由它释放（单 worker 通路），要么由
    _assign_*_to_worker 给每个 replica 各注册一份、各释放一份（PP fan-out 通路），
    两条路不能混用，否则会双重释放或漏释放。见 _op_buffer_registered_here。
    """
    if op.src_slot_id != -1:
        pin_buffer.free_slot(op.src_slot_id)
    if op.dst_slot_id != -1:
        pin_buffer.free_slot(op.dst_slot_id)


def _te_bounded_cuda_sync(timeout_s: float) -> None:
    """torch.cuda.synchronize() with a wall-clock cap.

    Runs the sync in a daemon thread so a wedged GPU cannot prevent
    TransferEngine.shutdown from returning. Failure / timeout is logged
    but not raised — this is called from the shutdown finally.

    中文补充：为什么不直接 torch.cuda.synchronize()——若 GPU 已经"卡死"，
    同步会永久阻塞，导致 TM 进程退不掉，只能靠父进程 SIGKILL。这里放进
    daemon 线程 + wait(timeout) 设上限：超时就放弃等待继续往下走。
    线程是 daemon 的，所以即使它永远卡住也不会拖住解释器退出。
    失败/超时只记日志不抛异常，因为调用点在 shutdown 的 finally 里。
    """
    if not (torch.cuda.is_available() and torch.cuda.is_initialized()):
        return
    done = threading.Event()
    err: List[BaseException] = []

    def _run() -> None:
        try:
            torch.cuda.synchronize()
        except BaseException as e:  # noqa: BLE001
            err.append(e)
        finally:
            done.set()

    t = threading.Thread(
        target=_run, name="flexkv-te-cuda-drain", daemon=True,
    )
    t.start()
    if not done.wait(timeout=timeout_s):
        flexkv_logger.warning(
            f"TransferEngine.shutdown: cuda synchronize did not finish in "
            f"{timeout_s:.0f}s (GPU likely wedged); continuing"
        )
        return
    if err:
        flexkv_logger.warning(
            f"TransferEngine.shutdown: cuda synchronize failed: {err[0]!r}"
        )

class TransferEngine:
    """数据面的调度中枢（整体定位见本文件头部注释）。

    在链路中的职责：承上启下。
      向上：控制面（GlobalCacheEngine / KVTaskEngine）通过 submit_transfer_graph
            丢进一张 TransferOpGraph，通过 get_completed_graphs_and_ops 取走
            CompletedOp 完成通知。本文件不认识 KV block 语义，只认 op/graph。
      向下：持有一批 WorkerHandle（每个 = 一个 worker 子进程 + 一条 Pipe），
            按 op.transfer_type 与集群拓扑把它们分配给对应 worker。

    关键设计：
      1. 单线程事件循环。_scheduler_loop 是唯一修改调度状态的线程，所有共享
         状态（op_id_to_op、_child_to_parent_op_id、scheduler、pin_buffer）
         都由它独占读写，因此内部完全不加锁；对外只通过 mp.Queue 交互。
      2. 两级映射决定"派给谁"：
           _worker_map     : TransferType -> WorkerHandle
                             或 Dict[WorkerKey, WorkerHandle]
           _swa_worker_map : 同上，SWA（滑动窗口）独立缓存池专用
         值为 dict 表示"按 WorkerKey(dp_client_id, pp_rank) 分桶"，通常意味着
         这条通路要做 PP fan-out；值为单个 handle 表示全局单例 worker
         （CPU<->SSD / CPU<->Remote 这类与 GPU 拓扑无关的通路）。
      3. 完成判定用 op.pending_count 归零，而不是"收到一次回报"：一个 op 扇出
         N 个 replica 就要收 N 次回报，保证 PP 各 stage 的完成是原子的。

    生命周期：start()（建 worker + 拉起调度线程）-> 业务期（submit / get）
    -> shutdown()（停线程 + 并行停 worker）。
    """

    # 中文：只做"建状态"，不建 worker（worker 在 start()/_init_workers 里才 spawn）。
    # 这里建好的几样东西贯穿整个生命周期：
    #   * mp_ctx = spawn：必须用 spawn 而非 fork —— worker 进程要建自己的 CUDA
    #     上下文，fork 会把父进程的 CUDA 状态一起复制过去，极易死锁/崩溃。
    #   * 三个 mp.Queue：task_queue（上层->调度线程）、finished_ops_queue
    #     （worker->调度线程，所有 worker 共用）、completed_queue（调度线程->上层）。
    #     用 mp.Queue 而非 queue.Queue 是为了把 _reader fd 注册进 selectors。
    #   * shutdown_read_fd/write_fd：一对 os.pipe()，专门用来零延迟唤醒
    #     阻塞在 select 上的调度线程（详见 _scheduler_loop）。
    #   * pin_buffer：跨进程共享的 op 元数据槽池（详见 register_op_to_buffer）。
    #   * scheduler：DAG 依赖调度器（详见 flexkv/transfer/scheduler.py）。
    def __init__(self,
        gpu_handles: Dict[WorkerKey, List[StorageHandle]],
        model_config: ModelConfig,
        cache_config: CacheConfig,
        cpu_handle: Optional[StorageHandle] = None,
        ssd_handle: Optional[StorageHandle] = None,
        remote_handle: Optional[StorageHandle] = None,
        gpu_blocks_per_group: Optional[Dict[WorkerKey, List]] = None,
        gpu_layouts_per_group: Optional[Dict[WorkerKey, List]] = None,
        swa_gpu_handles: Optional[Dict[WorkerKey, List[StorageHandle]]] = None,
        swa_cpu_handle: Optional[StorageHandle] = None,
        swa_ssd_handle: Optional[StorageHandle] = None,
        swa_remote_handle: Optional[StorageHandle] = None,
        swa_layer_groups: Optional[List[LayerGroupSpec]] = None,
        swa_gpu_blocks_per_group: Optional[Dict[WorkerKey, List]] = None,
        swa_gpu_layouts_per_group: Optional[Dict[WorkerKey, List]] = None,
        ):
        """
        Initialize transfer engine

        Args:
            gpu_handles: Dict mapping WorkerKey(dp_rank, pp_rank) -> list of GPU handles for that TP group
            model_config: global ModelConfig (parallelism sizes; no per-rank index)
            cache_config: global CacheConfig
            cpu_handle: CPU handle
            ssd_handle: Optional SSD handle
            remote_handle: Optional remote handle
            gpu_blocks_per_group: Per-group GPU handles, keyed by WorkerKey
            gpu_layouts_per_group: Per-group GPU layouts, keyed by WorkerKey
        """
        self.model_config: ModelConfig = model_config
        self.cache_config: CacheConfig = cache_config

        # 本 PP stage（本进程负责的层段）的层数，后面算传输字节数时要用：
        # 一次 op 只搬本 stage 的层，按全模型层数算会高估。
        first_handles = next(iter(gpu_handles.values()))
        self._num_layers_for_local_pp_stage = first_handles[0].kv_layout.num_layer

        # Use spawn context for CUDA compatibility
        self.mp_ctx = mp.get_context('spawn')

        # Initialize scheduler
        self.scheduler = TransferScheduler()
        # Use mp.Queue instead of queue.Queue to enable selector monitoring
        self.task_queue = self.mp_ctx.Queue()
        # Use mp.Queue for completed_queue to enable daemon process to monitor it via selector
        self.completed_queue = self.mp_ctx.Queue()
        self.finished_ops_queue = self.mp_ctx.Queue()
        self.op_id_to_op: Dict[int, TransferOp] = {}

        # Create shutdown pipe for zero-latency selector
        self.shutdown_read_fd, self.shutdown_write_fd = os.pipe()
        self.gpu_handle_groups = gpu_handles  # WorkerKey -> list of GPU handles for that TP group
        self._cpu_handle = cpu_handle
        self._ssd_handle = ssd_handle
        self._remote_handle = remote_handle
        self._gpu_blocks_per_group = gpu_blocks_per_group
        self._gpu_layouts_per_group = gpu_layouts_per_group

        # SWA handles and workers
        self._swa_gpu_handles = swa_gpu_handles
        self._swa_cpu_handle = swa_cpu_handle
        self._swa_ssd_handle = swa_ssd_handle
        self._swa_remote_handle = swa_remote_handle
        self._swa_layer_groups = (
            swa_cpu_handle.kv_layout.layer_groups
            if swa_cpu_handle is not None
            and swa_cpu_handle.kv_layout.layer_groups is not None
            else swa_layer_groups
        )
        self._swa_gpu_blocks_per_group = swa_gpu_blocks_per_group
        self._swa_gpu_layouts_per_group = swa_gpu_layouts_per_group
        if self._swa_layer_groups is not None and (
            self._swa_gpu_blocks_per_group is None
            or self._swa_gpu_layouts_per_group is None
        ):
            raise ValueError(
                "SWA multi-group layout is missing per-group GPU handles/layouts"
            )
        self._has_swa = (swa_gpu_handles is not None and len(swa_gpu_handles) > 0
                         and swa_cpu_handle is not None)
        self._cache_config = cache_config
        # TODO: is this correct?
        self._enable_pcfs_sharing = (
            GLOBAL_CONFIG_FROM_ENV.index_accel and cache_config.enable_kv_sharing
        )

        # 槽池容量 2048 = 同时在飞的 op 数上限（每个 op 占 src/dst 两个槽，
        # 槽位按 block id 内容哈希复用，相同内容的 op 会命中同一个槽）。
        # 槽位耗尽时分配会阻塞/失败，表现为调度线程整体卡住。
        self.pin_buffer = SharedOpPool(2048, self.cache_config.num_cpu_blocks)

        self.op_id_to_nvtx_range: Dict[int, str] = {}

        self.num_gpu_groups = len(self.gpu_handle_groups)
        self._running = False
        self._gpu_mappings_suspended = False

        self._child_id_to_child: Dict[int, TransferOp] = {}
        self._child_to_parent_op_id: Dict[int, int] = {}

        self._compressors = build_compressors(
            cpu_handle=self._cpu_handle,
            ssd_handle=self._ssd_handle,
            cache_config=self.cache_config,
            model_config=self.model_config,
            gpu_handle_groups=self.gpu_handle_groups,
            layerwise_enabled=GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer,
        )

        # Used for LAYERWISE PP fan-out: a parent op spawns one replica per PP
        # sibling worker; each replica's completion decrements the parent's
        # pending_count and the parent finalizes when count hits 0.
        self._child_id_to_child: Dict[int, TransferOp] = {}
        self._child_to_parent_op_id: Dict[int, int] = {}

        # Failure propagation state; see _handle_failed_op for the flow.
        # Graphs with a failed op, awaiting drain of their in-flight ops
        # before the graph-level failure message is emitted.
        self._failed_graph_ids: Set[int] = set()
        # Ops with at least one failed replica: must be discarded, not
        # finalized, when their pending_count drains to zero.
        self._failed_parent_op_ids: Set[int] = set()

    # 中文：给单个 worker 装配"多层组布局"(layer_groups) 参数。
    # 多层组 = 一张 KV 被切成若干层组（如主 KV 组 + MLA indexer 组），
    # 同一个 block id 在各组里的 layout 不同，worker 必须按组分别取
    # handle/layout 才不会搬错字节。
    # TP=1 时一个 WorkerKey 只对应一张卡，所以取下标 [0]。
    # 没配 layer_groups（或该 key 没有分组数据）返回空 dict —— 是"可选增强"，
    # worker 侧收到空就退回单组布局，不会报错。
    def _get_multi_group_kwargs_tp1(self, worker_key: WorkerKey) -> dict:
        """Get multi-group kwargs for TP=1 workers (GPUCPU / GDS)."""
        if (self.model_config.layer_groups is None or
                self._gpu_blocks_per_group is None or
                worker_key not in self._gpu_blocks_per_group):
            return {}
        # For TP=1, there's one device per WorkerKey
        # _gpu_blocks_per_group[worker_key][0] = per-group handle lists for that device
        per_device_group_blocks = self._gpu_blocks_per_group[worker_key][0]
        per_device_group_layouts = self._gpu_layouts_per_group[worker_key][0]
        if per_device_group_blocks is None or per_device_group_layouts is None:
            return {}
        return dict(
            layer_groups=self.model_config.layer_groups,
            gpu_blocks_per_group=per_device_group_blocks,
            gpu_layouts_per_group=per_device_group_layouts,
        )

    # 中文：TP>1 版本的多层组参数装配。与 tp1 版唯一的区别是这里多做一次转置：
    # _gpu_blocks_per_group 的入参是 [device][group]（每个 TP 卡一份，
    # 每份再按层组列 handle），而 worker 侧的执行单元是"层组"——一个层组要
    # 横跨 TP 组的所有卡同时搬，所以必须转成 [group][device] 传下去。
    # 维度搞反的表现是：数据能搬完不报错，但每张卡拿到的是别人的分片。
    def _get_multi_group_kwargs_tp(self, worker_key: WorkerKey) -> dict:
        """Get multi-group kwargs for TP>1 workers (tpGPUCPU / tpGDS)."""
        if (self.model_config.layer_groups is None or
                self._gpu_blocks_per_group is None or
                worker_key not in self._gpu_blocks_per_group):
            return {}
        # For TP>1, _gpu_blocks_per_group[worker_key] has tp_size entries (one per device)
        # Each entry is List[List[TensorSharedHandle]] (per-group handle lists for that device)
        per_device_data = self._gpu_blocks_per_group[worker_key]
        per_device_layouts = self._gpu_layouts_per_group[worker_key]
        if per_device_data[0] is None or per_device_layouts[0] is None:
            return {}

        num_groups = len(self.model_config.layer_groups)
        num_devices = len(per_device_data)

        # Restructure: from [device][group] -> [group][device]
        # gpu_blocks_per_group[group_idx][device_idx] = handles for that group on that device
        blocks_by_group = []
        layouts_by_group = []
        for gi in range(num_groups):
            group_blocks_per_device = [per_device_data[di][gi] for di in range(num_devices)]
            group_layouts_per_device = [per_device_layouts[di][gi] for di in range(num_devices)]
            blocks_by_group.append(group_blocks_per_device)
            layouts_by_group.append(group_layouts_per_device)

        return dict(
            layer_groups=self.model_config.layer_groups,
            gpu_blocks_per_group=blocks_by_group,
            gpu_layouts_per_group=layouts_by_group,
        )

    # 中文：SWA 专用池版本的多层组参数。逻辑与主 KV 的 tp1 版完全一致，
    # 只是数据源换成 _swa_*（SWA 层 / state sidecar 有自己独立的显存池、
    # CPU 池和 layout，尺寸与主 KV 不同，不能共用 handle）。
    def _get_swa_multi_group_kwargs_tp1(self, worker_key: WorkerKey) -> dict:
        """Return DSv4 SWA/state sidecar groups for a one-device worker."""
        if (
            self._swa_layer_groups is None
            or self._swa_gpu_blocks_per_group is None
            or self._swa_gpu_layouts_per_group is None
            or worker_key not in self._swa_gpu_blocks_per_group
        ):
            return {}
        per_device_blocks = self._swa_gpu_blocks_per_group[worker_key][0]
        per_device_layouts = self._swa_gpu_layouts_per_group[worker_key][0]
        if per_device_blocks is None or per_device_layouts is None:
            return {}
        return dict(
            layer_groups=self._swa_layer_groups,
            gpu_blocks_per_group=per_device_blocks,
            gpu_layouts_per_group=per_device_layouts,
        )

    # 中文：SWA 专用池版本的多层组参数（TP>1）。同样做 [device][group]
    # -> [group][device] 转置，理由见 _get_multi_group_kwargs_tp。
    def _get_swa_multi_group_kwargs_tp(self, worker_key: WorkerKey) -> dict:
        """Return SWA/state sidecar groups reshaped as [group][device]."""
        if (
            self._swa_layer_groups is None
            or self._swa_gpu_blocks_per_group is None
            or self._swa_gpu_layouts_per_group is None
            or worker_key not in self._swa_gpu_blocks_per_group
        ):
            return {}
        per_device_blocks = self._swa_gpu_blocks_per_group[worker_key]
        per_device_layouts = self._swa_gpu_layouts_per_group[worker_key]
        if per_device_blocks[0] is None or per_device_layouts[0] is None:
            return {}
        num_groups = len(self._swa_layer_groups)
        num_devices = len(per_device_blocks)
        return dict(
            layer_groups=self._swa_layer_groups,
            gpu_blocks_per_group=[
                [per_device_blocks[di][gi] for di in range(num_devices)]
                for gi in range(num_groups)
            ],
            gpu_layouts_per_group=[
                [per_device_layouts[di][gi] for di in range(num_devices)]
                for gi in range(num_groups)
            ],
        )

    # 中文：SWA 专用池版本的多层组参数。逻辑与主 KV 的 tp1 版完全一致，
    # 只是数据源换成 _swa_*（SWA 层 / state sidecar 有自己独立的显存池、
    # CPU 池和 layout，尺寸与主 KV 不同，不能共用 handle）。
    def _get_swa_multi_group_kwargs_tp1(self, worker_key: WorkerKey) -> dict:
        """Return DSv4 SWA/state sidecar groups for a one-device worker."""
        if (
            self._swa_layer_groups is None
            or self._swa_gpu_blocks_per_group is None
            or self._swa_gpu_layouts_per_group is None
            or worker_key not in self._swa_gpu_blocks_per_group
        ):
            return {}
        per_device_blocks = self._swa_gpu_blocks_per_group[worker_key][0]
        per_device_layouts = self._swa_gpu_layouts_per_group[worker_key][0]
        if per_device_blocks is None or per_device_layouts is None:
            return {}
        return dict(
            layer_groups=self._swa_layer_groups,
            gpu_blocks_per_group=per_device_blocks,
            gpu_layouts_per_group=per_device_layouts,
        )

    # 中文：SWA 专用池版本的多层组参数（TP>1）。同样做 [device][group]
    # -> [group][device] 转置，理由见 _get_multi_group_kwargs_tp。
    def _get_swa_multi_group_kwargs_tp(self, worker_key: WorkerKey) -> dict:
        """Return SWA/state sidecar groups reshaped as [group][device]."""
        if (
            self._swa_layer_groups is None
            or self._swa_gpu_blocks_per_group is None
            or self._swa_gpu_layouts_per_group is None
            or worker_key not in self._swa_gpu_blocks_per_group
        ):
            return {}
        per_device_blocks = self._swa_gpu_blocks_per_group[worker_key]
        per_device_layouts = self._swa_gpu_layouts_per_group[worker_key]
        if per_device_blocks[0] is None or per_device_layouts[0] is None:
            return {}
        num_groups = len(self._swa_layer_groups)
        num_devices = len(per_device_blocks)
        return dict(
            layer_groups=self._swa_layer_groups,
            gpu_blocks_per_group=[
                [per_device_blocks[di][gi] for di in range(num_devices)]
                for gi in range(num_groups)
            ],
            gpu_layouts_per_group=[
                [per_device_layouts[di][gi] for di in range(num_devices)]
                for gi in range(num_groups)
            ],
        )

    # 中文：给 LayerwiseTransferWorker 装配 SWA 参数。分两种形态：
    #   * 配了 _swa_layer_groups（DSv4 state sidecar 等多组形态）-> 走多组分支，
    #     传 swa_layer_groups + 按组/按卡的 handle；
    #   * 否则是"均匀 SWA"（所有 SWA 层同一 layout）-> 传 swa_gpu_blocks /
    #     swa_gpu_kv_layouts 列表（每卡一份）+ 单一 swa_dtype。
    # 关键点：layerwise 模式下 SWA 不会单独建 H2D worker，而是作为附加参数
    # 塞进 LAYERWISE worker，由它在逐层流水里顺带把 SWA 层一起搬上 GPU。
    def _get_layerwise_swa_kwargs(self, worker_key: WorkerKey) -> dict:
        """SWA args for LayerwiseTransferWorker (uniform or multi-group).

        When SWA is enabled, layerwise GET always binds SWA (and any C4 state
        sidecars) into the LAYERWISE worker rather than a standalone H2D worker.
        """
        if not self._has_swa:
            return {}
        swa_ssd_files = (
            self._swa_ssd_handle.get_file_list()
            if self._swa_ssd_handle is not None else None)
        swa_ssd_kv_layout = (
            self._swa_ssd_handle.kv_layout
            if self._swa_ssd_handle is not None else None)
        swa_num_blocks_per_file = (
            self._swa_ssd_handle.num_blocks_per_file
            if self._swa_ssd_handle is not None else 0)

        if self._swa_layer_groups is not None:
            mg = self._get_swa_multi_group_kwargs_tp(worker_key)
            if not mg:
                return {}
            return dict(
                swa_cpu_blocks=self._swa_cpu_handle.get_worker_tensor(),
                swa_cpu_kv_layout=self._swa_cpu_handle.kv_layout,
                swa_ssd_files=swa_ssd_files,
                swa_ssd_kv_layout=swa_ssd_kv_layout,
                swa_num_blocks_per_file=swa_num_blocks_per_file,
                swa_layer_groups=mg["layer_groups"],
                swa_gpu_blocks_per_group=mg["gpu_blocks_per_group"],
                swa_gpu_layouts_per_group=mg["gpu_layouts_per_group"],
            )

        return dict(
            swa_gpu_blocks=[
                h.get_tensor_handle_list()
                for h in self._swa_gpu_handles[worker_key]
            ],
            swa_cpu_blocks=self._swa_cpu_handle.get_worker_tensor(),
            swa_gpu_kv_layouts=[
                h.kv_layout for h in self._swa_gpu_handles[worker_key]
            ],
            swa_cpu_kv_layout=self._swa_cpu_handle.kv_layout,
            swa_dtype=self._swa_gpu_handles[worker_key][0].dtype,
            swa_ssd_files=swa_ssd_files,
            swa_ssd_kv_layout=swa_ssd_kv_layout,
            swa_num_blocks_per_file=swa_num_blocks_per_file,
        )

    def _init_workers(self) -> None:
        """按集群拓扑创建全部 Worker 子进程，建好路由表，最后拉起调度线程。

        本文件唯一的"建拓扑"入口，由 start() 调用；_running 为真时直接返回（只建一次）。

        产物（后续调度线程完全依赖这两张表派发 op）：
          self._worker_map     : TransferType -> WorkerHandle 或 Dict[WorkerKey, WorkerHandle]
          self._swa_worker_map : 同上，SWA（滑动窗口）专用池的那一份
        值为 WorkerHandle 表示"单实例"（一个进程服务全机，多见于与 GPU 拓扑
        无关的通路）；值为 dict 表示"按 WorkerKey(dp_client_id, pp_rank) 分桶"。

        Worker 种类 / 创建条件 / 粒度一览：

          | 通路(TransferType)   | Worker 类                        | 创建条件                                       | 粒度   |
          |----------------------|----------------------------------|------------------------------------------------|--------|
          | H2D   (CPU->GPU)     | GPUCPUTransferWorker             | 非 layerwise 且 TP==1                          | 按 key |
          | H2D                  | tpGPUCPUTransferWorker           | 非 layerwise 且 TP>1                           | 按 key |
          | D2H   (GPU->CPU)     | 同上两个类                        | 总是创建（layerwise 模式也建）                  | 按 key |
          | DISK2H(SSD->CPU)     | CPUSSDDiskTransferWorker         | 有 ssd_handle 且非 layerwise                   | 单实例 |
          | H2DISK(CPU->SSD)     | CPUSSDDiskTransferWorker         | 有 ssd_handle                                  | 单实例 |
          | REMOTE2H / H2REMOTE  | CPURemoteTransferWorker(读写各一) | 有 remote_handle                               | 单实例 |
          | REMOTE2H / H2REMOTE  | MooncakeStoreTransferWorker      | 无 remote_handle 但 use_mooncake_store_backend | 单实例 |
          | DISK2D / D2DISK      | NixlTransferWorker               | enable_gds 且 enable_nixl（要求 TP==1）        | 按 key |
          | DISK2D / D2DISK      | GDSTransferWorker                | enable_gds 且 TP==1                            | 按 key |
          | DISK2D / D2DISK      | tpGDSTransferWorker              | enable_gds 且 TP>1                             | 按 key |
          | LAYERWISE            | LayerwiseTransferWorker          | enable_layerwise_transfer                      | 按 key |
          | PEERH2H / PEERSSD2H  | PEER2CPUTransferWorker           | enable_kv_sharing 且 (enable_p2p_cpu 或 p2p_ssd)| 单实例 |
          | SWA 各通路            | 与主 KV 同构的类，绑 SWA 专用池    | _has_swa（有 swa_gpu_handles 且 swa_cpu_handle）| 同左   |

        三个拓扑开关如何影响创建决策：
          * TP（effective_tp_size_per_node）：决定用单卡 worker 还是 tp* worker。
            tp* worker 内部按 TP 组切分 block，一次 submit 覆盖整组所有卡，
            所以"按 key"的粒度是 TP 组而非单卡 —— 一个 op 只需发给一个 worker，
            不需要调用方自己按卡展开。
          * GDS：开启后 SSD<->GPU 走 GPUDirect Storage 直通（DISK2D/D2DISK），
            绕开 CPU 中转的 DISK2H+H2D 两步。GDS worker 与 CPU-SSD worker
            可以同时存在，走哪条路由控制面在 DAG 里决定（本文件只管派发）。
          * Layerwise：开启后不再建 H2D 与 DISK2H 的独立 worker，二者被
            LAYERWISE worker 吸收（末尾三条 assert 校验）。原因是逐层传输要求
            "读盘与拷 GPU 在同一进程内按层流水"，拆成两个进程做不到。
          * SWA：SWA 层的 KV 尺寸/layout 与主 KV 不同，放在独立池里，
            所以整套通路复制一份到 _swa_worker_map，但共用同一个
            finished_ops_queue（回收路径不区分主 KV / SWA）。

        Note:
            会阻塞等待每个 worker 的 ready_event：子进程要各自初始化 CUDA 上下文，
            这一步可能很慢；若子进程 init 失败会抛 RuntimeError（由 start() 回滚）。
            一张 worker 都没建出来会抛 ValueError，属于配置错误。
        """
        if self._running:
            return
        self._worker_map: Dict[TransferType, Union[WorkerHandle, Dict[WorkerKey, WorkerHandle]]] = {}

        assert self._cpu_handle is not None
        # When layerwise is on, SWA/state H2D is always fused into the LAYERWISE
        # worker (no standalone swa_multi_layer switch).
        _enable_layerwise = GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer
        # Use num_gpu_groups to support multi-instance mode
        # Use gpu_device_id from StorageHandle for correct CUDA device selection
        
        # H2D worker
        if not _enable_layerwise:
            if self.model_config.effective_tp_size_per_node == 1:
                self.h2d_workers: Dict[WorkerKey, WorkerHandle] = {
                    worker_key: GPUCPUTransferWorker.create_worker(
                        mp_ctx=self.mp_ctx,
                        finished_ops_queue=self.finished_ops_queue,
                        op_buffer_tensor=self.pin_buffer.get_buffer(),
                        gpu_blocks=gpu_handles[0].get_tensor_handle_list(),
                        cpu_blocks=self._cpu_handle.get_worker_tensor(),
                        gpu_kv_layout=gpu_handles[0].kv_layout,
                        cpu_kv_layout=self._cpu_handle.kv_layout,
                        dtype=gpu_handles[0].dtype,
                        gpu_device_id=gpu_handles[0].gpu_device_id,
                        use_ce_transfer_h2d=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d,
                        use_ce_transfer_d2h=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h,
                        transfer_num_cta_h2d=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_h2d,
                        transfer_num_cta_d2h=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_d2h,
                        compressor=self._compressors["gpu_cpu"],
                        **self._get_multi_group_kwargs_tp1(worker_key),
                    )
                    for worker_key, gpu_handles in self.gpu_handle_groups.items()
                }
            else:
                self.h2d_workers = {
                    worker_key: tpGPUCPUTransferWorker.create_worker(
                        mp_ctx=self.mp_ctx,
                        finished_ops_queue=self.finished_ops_queue,
                        op_buffer_tensor=self.pin_buffer.get_buffer(),
                        gpu_blocks=[gpu_handle.get_tensor_handle_list() for gpu_handle in gpu_handles],
                        cpu_blocks=self._cpu_handle.get_worker_tensor(),
                        gpu_kv_layouts=[gpu_handle.kv_layout for gpu_handle in gpu_handles],
                        cpu_kv_layout=self._cpu_handle.kv_layout,
                        dtype=gpu_handles[0].dtype,
                        tp_group_size=self.model_config.effective_tp_size_per_node,
                        use_ce_transfer_h2d=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d,
                        use_ce_transfer_d2h=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h,
                        transfer_num_cta_h2d=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_h2d,
                        transfer_num_cta_d2h=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_d2h,
                        compressor=self._compressors["gpu_cpu_tp"],
                        **self._get_multi_group_kwargs_tp(worker_key),
                    )
                    for worker_key, gpu_handles in self.gpu_handle_groups.items()
                }
            self._worker_map[TransferType.H2D] = self.h2d_workers

        # D2H worker
        if self.model_config.effective_tp_size_per_node == 1:
            self.d2h_workers: Dict[WorkerKey, WorkerHandle] = {
                worker_key: GPUCPUTransferWorker.create_worker(
                    mp_ctx=self.mp_ctx,
                    finished_ops_queue=self.finished_ops_queue,
                    op_buffer_tensor=self.pin_buffer.get_buffer(),
                    gpu_blocks=gpu_handles[0].get_tensor_handle_list(),
                    cpu_blocks=self._cpu_handle.get_worker_tensor(),
                    gpu_kv_layout=gpu_handles[0].kv_layout,
                    cpu_kv_layout=self._cpu_handle.kv_layout,
                    dtype=gpu_handles[0].dtype,
                    gpu_device_id=gpu_handles[0].gpu_device_id,
                    use_ce_transfer_h2d=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d,
                    use_ce_transfer_d2h=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h,
                    transfer_num_cta_h2d=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_h2d,
                    transfer_num_cta_d2h=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_d2h,
                    compressor=self._compressors["gpu_cpu"],
                    **self._get_multi_group_kwargs_tp1(worker_key),
                )
                for worker_key, gpu_handles in self.gpu_handle_groups.items()
            }
        else:
            self.d2h_workers = {
                worker_key: tpGPUCPUTransferWorker.create_worker(
                    mp_ctx=self.mp_ctx,
                    finished_ops_queue=self.finished_ops_queue,
                    op_buffer_tensor=self.pin_buffer.get_buffer(),
                    gpu_blocks=[gpu_handle.get_tensor_handle_list() for gpu_handle in gpu_handles],
                    cpu_blocks=self._cpu_handle.get_worker_tensor(),
                    gpu_kv_layouts=[gpu_handle.kv_layout for gpu_handle in gpu_handles],
                    cpu_kv_layout=self._cpu_handle.kv_layout,
                    dtype=gpu_handles[0].dtype,
                    tp_group_size=self.model_config.effective_tp_size_per_node,
                    use_ce_transfer_h2d=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d,
                    use_ce_transfer_d2h=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h,
                    transfer_num_cta_h2d=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_h2d,
                    transfer_num_cta_d2h=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_d2h,
                    compressor=self._compressors["gpu_cpu_tp"],
                    **self._get_multi_group_kwargs_tp(worker_key),
                )
                for worker_key, gpu_handles in self.gpu_handle_groups.items()
            }
        self._worker_map[TransferType.D2H] = self.d2h_workers

        if self._ssd_handle is not None and self._cpu_handle is not None:
            ssd_layer_groups = self.model_config.layer_groups
            # DISK2H worker
            if not _enable_layerwise:
                self.cpussd_read_worker: WorkerHandle = CPUSSDDiskTransferWorker.create_worker(
                    mp_ctx=self.mp_ctx,
                    finished_ops_queue=self.finished_ops_queue,
                    op_buffer_tensor = self.pin_buffer.get_buffer(),
                    cpu_blocks=self._cpu_handle.get_worker_tensor(),
                    ssd_files=self._ssd_handle.get_file_list(),
                    cpu_kv_layout=self._cpu_handle.kv_layout,
                    ssd_kv_layout=self._ssd_handle.kv_layout,
                    dtype=self._cpu_handle.dtype,
                    num_blocks_per_file=self._ssd_handle.num_blocks_per_file,
                    cache_config=self._cache_config,
                    compressor=self._compressors["cpu_ssd"],
                    layer_groups=ssd_layer_groups,
                )
                self._worker_map[TransferType.DISK2H] = self.cpussd_read_worker

            # H2DISK worker
            self.cpussd_write_worker: WorkerHandle = CPUSSDDiskTransferWorker.create_worker(
                mp_ctx=self.mp_ctx,
                finished_ops_queue=self.finished_ops_queue,
                op_buffer_tensor = self.pin_buffer.get_buffer(),
                cpu_blocks=self._cpu_handle.get_worker_tensor(),
                ssd_files=self._ssd_handle.get_file_list(),
                cpu_kv_layout=self._cpu_handle.kv_layout,
                ssd_kv_layout=self._ssd_handle.kv_layout,
                dtype=self._cpu_handle.dtype,
                num_blocks_per_file=self._ssd_handle.num_blocks_per_file,
                cache_config=self._cache_config,
                compressor=self._compressors["cpu_ssd"],
                layer_groups=ssd_layer_groups,
            )
            self._worker_map[TransferType.H2DISK] = self.cpussd_write_worker
        if self._remote_handle is not None and self._cpu_handle is not None:
            self.remotecpu_read_worker: WorkerHandle = CPURemoteTransferWorker.create_worker(
                mp_ctx=self.mp_ctx,
                finished_ops_queue=self.finished_ops_queue,
                op_buffer_tensor = self.pin_buffer.get_buffer(),
                cpu_blocks=self._cpu_handle.get_worker_tensor(),
                remote_file=self._remote_handle.get_file_list(),
                cpu_kv_layout=self._cpu_handle.kv_layout,
                remote_kv_layout=self._remote_handle.kv_layout,
                dtype=self._cpu_handle.dtype,
                remote_config_custom=self._remote_handle.remote_config_custom,
                enable_pcfs_sharing=self._enable_pcfs_sharing,
            )
            self.remotecpu_write_worker: WorkerHandle = CPURemoteTransferWorker.create_worker(
                mp_ctx=self.mp_ctx,
                finished_ops_queue=self.finished_ops_queue,
                op_buffer_tensor = self.pin_buffer.get_buffer(),
                cpu_blocks=self._cpu_handle.get_worker_tensor(),
                remote_file=self._remote_handle.get_file_list(),
                cpu_kv_layout=self._cpu_handle.kv_layout,
                remote_kv_layout=self._remote_handle.kv_layout,
                dtype=self._cpu_handle.dtype,
                remote_config_custom=self._remote_handle.remote_config_custom,
            )
            self._worker_map[TransferType.H2REMOTE] = self.remotecpu_write_worker
            self._worker_map[TransferType.REMOTE2H] = self.remotecpu_read_worker
        elif (getattr(self.cache_config, 'use_mooncake_store_backend', False)
              and self._cpu_handle is not None):
            self.mooncake_store_worker: WorkerHandle = MooncakeStoreTransferWorker.create_worker(
                mp_ctx=self.mp_ctx,
                finished_ops_queue=self.finished_ops_queue,
                op_buffer_tensor=self.pin_buffer.get_buffer(),
                cpu_blocks=self._cpu_handle.get_worker_tensor(),
                cpu_kv_layout=self._cpu_handle.kv_layout,
                dtype=self._cpu_handle.dtype,
                cache_config=self.cache_config,
                pool_kind=PoolKind.KV,
            )
            self._worker_map[TransferType.H2REMOTE] = self.mooncake_store_worker
            self._worker_map[TransferType.REMOTE2H] = self.mooncake_store_worker
            flexkv_logger.info(
                "[TransferEngine] mooncake-store workers created for H2REMOTE/REMOTE2H")
        if self.cache_config.enable_gds:
            assert self._ssd_handle is not None
            if self.cache_config.enable_nixl:
                flexkv_logger.info(
                    "[transfer_engine] GDS path using NixlTransferWorker (NIXL GDS_MT)"
                )
                if self.model_config.effective_tp_size_per_node != 1:
                    raise RuntimeError(
                        "enable_nixl requires effective_tp_size_per_node==1 (validated in KVTaskManager)"
                    )
                self.gds_workers: Dict[WorkerKey, WorkerHandle] = {
                    worker_key: NixlTransferWorker.create_worker(
                        mp_ctx=self.mp_ctx,
                        finished_ops_queue=self.finished_ops_queue,
                        op_buffer_tensor=self.pin_buffer.get_buffer(),
                        nixl_backend="GDS_MT",
                        ssd_files=self._ssd_handle.get_file_list(),
                        num_blocks_per_file=self._ssd_handle.num_blocks_per_file,
                        dtype=self._ssd_handle.dtype,
                        ssd_kv_layout=self._ssd_handle.kv_layout,
                        gpu_kv_layout=gpu_handles[0].kv_layout,
                        cpu_kv_layout=self._cpu_handle.kv_layout,
                        nixl_extra_config=self.cache_config.nixl_extra_config,
                        gpu_blocks=gpu_handles[0].get_tensor_handle_list(),
                        cpu_blocks=None,
                        gpu_device_id=gpu_handles[0].gpu_device_id,
                    )
                    for worker_key, gpu_handles in self.gpu_handle_groups.items()
                }
            elif self.model_config.effective_tp_size_per_node == 1:
                self.gds_workers: Dict[WorkerKey, WorkerHandle] = {
                    worker_key: GDSTransferWorker.create_worker(
                        mp_ctx=self.mp_ctx,
                        finished_ops_queue=self.finished_ops_queue,
                        op_buffer_tensor=self.pin_buffer.get_buffer(),
                        gpu_blocks=gpu_handles[0].get_tensor_handle_list(),
                        ssd_files=self._ssd_handle.get_file_list(),
                        num_blocks_per_file=self._ssd_handle.num_blocks_per_file,
                        gpu_kv_layout=gpu_handles[0].kv_layout,
                        ssd_kv_layout=self._ssd_handle.kv_layout,
                        dtype=self._ssd_handle.dtype,
                        gpu_device_id=gpu_handles[0].gpu_device_id,
                        **self._get_multi_group_kwargs_tp1(worker_key),
                    )
                    for worker_key, gpu_handles in self.gpu_handle_groups.items()
                }
            else:
                self.gds_workers = {
                    worker_key: tpGDSTransferWorker.create_worker(
                        mp_ctx=self.mp_ctx,
                        finished_ops_queue=self.finished_ops_queue,
                        op_buffer_tensor=self.pin_buffer.get_buffer(),
                        gpu_blocks=[gpu_handle.get_tensor_handle_list() for gpu_handle in gpu_handles],
                        ssd_files=self._ssd_handle.get_file_list(),
                        num_blocks_per_file=self._ssd_handle.num_blocks_per_file,
                        gpu_kv_layouts=[gpu_handle.kv_layout for gpu_handle in gpu_handles],
                        ssd_kv_layout=self._ssd_handle.kv_layout,
                        dtype=self._ssd_handle.dtype,
                        tp_group_size=self.model_config.effective_tp_size_per_node,
                        **self._get_multi_group_kwargs_tp(worker_key),
                    )
                    for worker_key, gpu_handles in self.gpu_handle_groups.items()
                }
            self._worker_map[TransferType.DISK2D] = self.gds_workers
            self._worker_map[TransferType.D2DISK] = self.gds_workers
        if GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer:
            ssd_files = {} if self._ssd_handle is None else self._ssd_handle.get_file_list()
            ssd_kv_layout = None if self._ssd_handle is None else self._ssd_handle.kv_layout
            num_blocks_per_file = 0 if self._ssd_handle is None else self._ssd_handle.num_blocks_per_file

            self.layerwise_workers: Dict[WorkerKey, WorkerHandle] = {}
            for worker_key, gpu_handles in self.gpu_handle_groups.items():
                _layerwise_eventfd_socket = build_layerwise_eventfd_socket_path(
                    dp_client_id=worker_key.dp_client_id,
                    pp_rank=worker_key.pp_rank,
                    model_config=self.model_config,
                )

                worker = LayerwiseTransferWorker.create_worker(
                    mp_ctx=self.mp_ctx,
                    finished_ops_queue=self.finished_ops_queue,
                    op_buffer_tensor=self.pin_buffer.get_buffer(),
                    gpu_blocks=[handle.get_tensor_handle_list() for handle in gpu_handles],
                    cpu_blocks=self._cpu_handle.get_worker_tensor(),
                    ssd_files=ssd_files,
                    gpu_kv_layouts=[handle.kv_layout for handle in gpu_handles],
                    cpu_kv_layout=self._cpu_handle.kv_layout,
                    ssd_kv_layout=ssd_kv_layout,
                    dtype=gpu_handles[0].dtype,
                    tp_group_size=self.model_config.effective_tp_size_per_node,
                    layerwise_eventfd_socket=_layerwise_eventfd_socket,
                    num_blocks_per_file=num_blocks_per_file,
                    use_ce_transfer_h2d=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d,
                    use_ce_transfer_d2h=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h,
                    h2d_cta_num=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_h2d,
                    d2h_cta_num=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_d2h,
                    # Fuse main-KV + uniform/multi-group SWA into one LAYERWISE op.
                    **self._get_layerwise_swa_kwargs(worker_key),
                    **self._get_multi_group_kwargs_tp(worker_key),
                )
                self.layerwise_workers[worker_key] = worker

                flexkv_logger.debug(
                    f"[TransferEngine] Created layerwise worker for {worker_key}: "
                    f"effective_tp_size_per_node={self.model_config.effective_tp_size_per_node}, "
                    f"layer_groups={'yes' if self.model_config.layer_groups else 'no'}, "
                    f"has_ssd={len(ssd_files) > 0}")

            self._worker_map[TransferType.LAYERWISE] = self.layerwise_workers

        if self.cache_config.enable_kv_sharing and self._cpu_handle is not None and (self.cache_config.enable_p2p_cpu \
            or (self._ssd_handle and self.cache_config.enable_p2p_ssd)):
            ## NOTE:if we have the cpu handle and enable p2p cpu transfer we need this worker
            ## (currently we inplement cpu and ssd distributed transfer in one worker)

            flexkv_logger.info("[transfer_engine] initializing the PEER2CPUTransferWorker!")
            self.cpu_remote_cpu_worker: WorkerHandle = PEER2CPUTransferWorker.create_worker(
                mp_ctx=self.mp_ctx,
                finished_ops_queue=self.finished_ops_queue,
                op_buffer_tensor = self.pin_buffer.get_buffer(),
                cpu_blocks=self._cpu_handle.get_worker_tensor(),
                cpu_kv_layout=self._cpu_handle.kv_layout,
                # TODO: get remote kv_layout, now we can assume that remote kv layout is same as current node
                remote_kv_layout=self._cpu_handle.kv_layout,
                dtype=self._cpu_handle.dtype,
                cache_config = self.cache_config,
                ssd_kv_layout = self._ssd_handle.kv_layout if self._ssd_handle else None,
                ssd_files = self._ssd_handle.get_file_list() if self._ssd_handle else None,
                num_blocks_per_file = self._ssd_handle.num_blocks_per_file if self._ssd_handle else 0,
                mooncake_config_path = (getattr(self.cache_config, 'mooncake_config_path', None)
                                        or os.environ.get("MOONCAKE_CONFIG_PATH")),
            )
            # NOTE: now peerH2H and peerSSD2H op use the same worker
            if self.cache_config.enable_p2p_cpu:
                self._worker_map[TransferType.PEERH2H] = self.cpu_remote_cpu_worker
            if self.cache_config.enable_p2p_ssd:
                self._worker_map[TransferType.PEERSSD2H] = self.cpu_remote_cpu_worker

        # ---- SWA dedicated worker map ----
        # Reuses GPUCPUTransferWorker / tpGPUCPUTransferWorker exactly like the
        # main-KV H2D/D2H workers, but bound to the dedicated SWA GPU/CPU pools
        # and submitting completion onto the shared finished_ops_queue.
        # Uniform SWA uses the legacy single-group worker. DSv4 state sidecars
        # reuse this channel with heterogeneous multi-group worker arguments.
        if self._has_swa:
            self._swa_worker_map: Dict[TransferType, Dict[WorkerKey, WorkerHandle]] = {}
            # When layerwise is on, SWA H2D always runs inside LAYERWISE
            # (uniform via launch_swa_h2d_layer_, multi-group via
            # launch_swa_mg_h2d_layer_). Standalone SWA H2D workers are only
            # created when layerwise transfer is disabled.
            if not _enable_layerwise:
                if self.model_config.effective_tp_size_per_node == 1:
                    self._swa_h2d_workers: Dict[WorkerKey, WorkerHandle] = {
                        worker_key: GPUCPUTransferWorker.create_worker(
                            mp_ctx=self.mp_ctx,
                            finished_ops_queue=self.finished_ops_queue,
                            op_buffer_tensor=self.pin_buffer.get_buffer(),
                            gpu_blocks=swa_handles[0].get_tensor_handle_list(),
                            cpu_blocks=self._swa_cpu_handle.get_worker_tensor(),
                            gpu_kv_layout=swa_handles[0].kv_layout,
                            cpu_kv_layout=self._swa_cpu_handle.kv_layout,
                            dtype=swa_handles[0].dtype,
                            gpu_device_id=swa_handles[0].gpu_device_id,
                            use_ce_transfer_h2d=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d,
                            use_ce_transfer_d2h=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h,
                            transfer_num_cta_h2d=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_h2d,
                            transfer_num_cta_d2h=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_d2h,
                            **self._get_swa_multi_group_kwargs_tp1(worker_key),
                        )
                        for worker_key, swa_handles in self._swa_gpu_handles.items()
                    }
                else:
                    self._swa_h2d_workers = {
                        worker_key: tpGPUCPUTransferWorker.create_worker(
                            mp_ctx=self.mp_ctx,
                            finished_ops_queue=self.finished_ops_queue,
                            op_buffer_tensor=self.pin_buffer.get_buffer(),
                            gpu_blocks=[h.get_tensor_handle_list() for h in swa_handles],
                            cpu_blocks=self._swa_cpu_handle.get_worker_tensor(),
                            gpu_kv_layouts=[h.kv_layout for h in swa_handles],
                            cpu_kv_layout=self._swa_cpu_handle.kv_layout,
                            dtype=swa_handles[0].dtype,
                            tp_group_size=self.model_config.effective_tp_size_per_node,
                            use_ce_transfer_h2d=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d,
                            use_ce_transfer_d2h=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h,
                            transfer_num_cta_h2d=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_h2d,
                            transfer_num_cta_d2h=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_d2h,
                            **self._get_swa_multi_group_kwargs_tp(worker_key),
                        )
                        for worker_key, swa_handles in self._swa_gpu_handles.items()
                    }
                self._swa_worker_map[TransferType.H2D] = self._swa_h2d_workers
                flexkv_logger.info("TransferEngine: swa H2D workers initialized")
            # D2H swa worker
            if self.model_config.effective_tp_size_per_node == 1:
                self._swa_d2h_workers: Dict[WorkerKey, WorkerHandle] = {
                    worker_key: GPUCPUTransferWorker.create_worker(
                        mp_ctx=self.mp_ctx,
                        finished_ops_queue=self.finished_ops_queue,
                        op_buffer_tensor=self.pin_buffer.get_buffer(),
                        gpu_blocks=swa_handles[0].get_tensor_handle_list(),
                        cpu_blocks=self._swa_cpu_handle.get_worker_tensor(),
                        gpu_kv_layout=swa_handles[0].kv_layout,
                        cpu_kv_layout=self._swa_cpu_handle.kv_layout,
                        dtype=swa_handles[0].dtype,
                        gpu_device_id=swa_handles[0].gpu_device_id,
                        use_ce_transfer_h2d=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d,
                        use_ce_transfer_d2h=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h,
                        transfer_num_cta_h2d=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_h2d,
                        transfer_num_cta_d2h=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_d2h,
                        **self._get_swa_multi_group_kwargs_tp1(worker_key),
                    )
                    for worker_key, swa_handles in self._swa_gpu_handles.items()
                }
            else:
                self._swa_d2h_workers = {
                    worker_key: tpGPUCPUTransferWorker.create_worker(
                        mp_ctx=self.mp_ctx,
                        finished_ops_queue=self.finished_ops_queue,
                        op_buffer_tensor=self.pin_buffer.get_buffer(),
                        gpu_blocks=[h.get_tensor_handle_list() for h in swa_handles],
                        cpu_blocks=self._swa_cpu_handle.get_worker_tensor(),
                        gpu_kv_layouts=[h.kv_layout for h in swa_handles],
                        cpu_kv_layout=self._swa_cpu_handle.kv_layout,
                        dtype=swa_handles[0].dtype,
                        tp_group_size=self.model_config.effective_tp_size_per_node,
                        use_ce_transfer_h2d=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d,
                        use_ce_transfer_d2h=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h,
                        transfer_num_cta_h2d=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_h2d,
                        transfer_num_cta_d2h=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_d2h,
                        **self._get_swa_multi_group_kwargs_tp(worker_key),
                        )
                    for worker_key, swa_handles in self._swa_gpu_handles.items()
                }

            self._swa_worker_map[TransferType.D2H] = self._swa_d2h_workers
            flexkv_logger.info("TransferEngine: swa D2H workers initialized")

            if self._swa_ssd_handle is not None and self._swa_cpu_handle is not None:
                self.swa_h2disk_worker: WorkerHandle = CPUSSDDiskTransferWorker.create_worker(
                    mp_ctx=self.mp_ctx,
                    finished_ops_queue=self.finished_ops_queue,
                    op_buffer_tensor=self.pin_buffer.get_buffer(),
                    cpu_blocks=self._swa_cpu_handle.get_worker_tensor(),
                    ssd_files=self._swa_ssd_handle.get_file_list(),
                    cpu_kv_layout=self._swa_cpu_handle.kv_layout,
                    ssd_kv_layout=self._swa_ssd_handle.kv_layout,
                    dtype=self._swa_cpu_handle.dtype,
                    num_blocks_per_file=self._swa_ssd_handle.num_blocks_per_file,
                    cache_config=self._cache_config,
                    layer_groups=self._swa_layer_groups,
                )
                self._swa_worker_map[TransferType.H2DISK] = self.swa_h2disk_worker

                self.swa_disk2h_worker: WorkerHandle = CPUSSDDiskTransferWorker.create_worker(
                    mp_ctx=self.mp_ctx,
                    finished_ops_queue=self.finished_ops_queue,
                    op_buffer_tensor=self.pin_buffer.get_buffer(),
                    cpu_blocks=self._swa_cpu_handle.get_worker_tensor(),
                    ssd_files=self._swa_ssd_handle.get_file_list(),
                    cpu_kv_layout=self._swa_cpu_handle.kv_layout,
                    ssd_kv_layout=self._swa_ssd_handle.kv_layout,
                    dtype=self._swa_cpu_handle.dtype,
                    num_blocks_per_file=self._swa_ssd_handle.num_blocks_per_file,
                    cache_config=self._cache_config,
                    layer_groups=self._swa_layer_groups,
                )
                self._swa_worker_map[TransferType.DISK2H] = self.swa_disk2h_worker
                flexkv_logger.info("TransferEngine: swa CPU<->SSD workers initialized")


            # ---- SWA CPU<->Remote workers -----------------------------------
            if (getattr(self.cache_config, 'use_mooncake_store_backend', False)
                    and self._swa_cpu_handle is not None):
                self.swa_mooncake_store_worker: WorkerHandle = (
                    MooncakeStoreTransferWorker.create_worker(
                        mp_ctx=self.mp_ctx,
                        finished_ops_queue=self.finished_ops_queue,
                        op_buffer_tensor=self.pin_buffer.get_buffer(),
                        cpu_blocks=self._swa_cpu_handle.get_worker_tensor(),
                        cpu_kv_layout=self._swa_cpu_handle.kv_layout,
                        dtype=self._swa_cpu_handle.dtype,
                        cache_config=self.cache_config,
                        pool_kind=PoolKind.SWA,
                        override_global_segment_size=0,
                    ))
                self._swa_worker_map[TransferType.REMOTE2H] = self.swa_mooncake_store_worker
                self._swa_worker_map[TransferType.H2REMOTE] = self.swa_mooncake_store_worker
                flexkv_logger.info(
                    "TransferEngine: swa mooncake-store workers initialized")
            elif self._swa_remote_handle is not None and self._swa_cpu_handle is not None:
                self.swa_remotecpu_read_worker: WorkerHandle = CPURemoteTransferWorker.create_worker(
                    mp_ctx=self.mp_ctx,
                    finished_ops_queue=self.finished_ops_queue,
                    op_buffer_tensor=self.pin_buffer.get_buffer(),
                    cpu_blocks=self._swa_cpu_handle.get_worker_tensor(),
                    remote_file=self._swa_remote_handle.get_file_list(),
                    cpu_kv_layout=self._swa_cpu_handle.kv_layout,
                    remote_kv_layout=self._swa_remote_handle.kv_layout,
                    dtype=self._swa_cpu_handle.dtype,
                    remote_config_custom=self._swa_remote_handle.remote_config_custom,
                    enable_pcfs_sharing=self._enable_pcfs_sharing,
                )
                self.swa_remotecpu_write_worker: WorkerHandle = CPURemoteTransferWorker.create_worker(
                    mp_ctx=self.mp_ctx,
                    finished_ops_queue=self.finished_ops_queue,
                    op_buffer_tensor=self.pin_buffer.get_buffer(),
                    cpu_blocks=self._swa_cpu_handle.get_worker_tensor(),
                    remote_file=self._swa_remote_handle.get_file_list(),
                    cpu_kv_layout=self._swa_cpu_handle.kv_layout,
                    remote_kv_layout=self._swa_remote_handle.kv_layout,
                    dtype=self._swa_cpu_handle.dtype,
                    remote_config_custom=self._swa_remote_handle.remote_config_custom,
                )
                self._swa_worker_map[TransferType.REMOTE2H] = self.swa_remotecpu_read_worker
                self._swa_worker_map[TransferType.H2REMOTE] = self.swa_remotecpu_write_worker
                flexkv_logger.info("TransferEngine: swa CPU<->Remote workers initialized")


            if self.cache_config.enable_gds and self._swa_ssd_handle is not None:
                if self.model_config.effective_tp_size_per_node == 1:
                    self._swa_gds_workers: Dict[WorkerKey, WorkerHandle] = {
                        worker_key: GDSTransferWorker.create_worker(
                            mp_ctx=self.mp_ctx,
                            finished_ops_queue=self.finished_ops_queue,
                            op_buffer_tensor=self.pin_buffer.get_buffer(),
                            gpu_blocks=swa_handles[0].get_tensor_handle_list(),
                            ssd_files=self._swa_ssd_handle.get_file_list(),
                            num_blocks_per_file=self._swa_ssd_handle.num_blocks_per_file,
                            gpu_kv_layout=swa_handles[0].kv_layout,
                            ssd_kv_layout=self._swa_ssd_handle.kv_layout,
                            dtype=swa_handles[0].dtype,
                            gpu_device_id=swa_handles[0].gpu_device_id,
                            **self._get_swa_multi_group_kwargs_tp1(worker_key),
                        )
                        for worker_key, swa_handles in self._swa_gpu_handles.items()
                    }
                else:
                    self._swa_gds_workers = {
                        worker_key: tpGDSTransferWorker.create_worker(
                            mp_ctx=self.mp_ctx,
                            finished_ops_queue=self.finished_ops_queue,
                            op_buffer_tensor=self.pin_buffer.get_buffer(),
                            gpu_blocks=[h.get_tensor_handle_list() for h in swa_handles],
                            ssd_files=self._swa_ssd_handle.get_file_list(),
                            num_blocks_per_file=self._swa_ssd_handle.num_blocks_per_file,
                            gpu_kv_layouts=[h.kv_layout for h in swa_handles],
                            ssd_kv_layout=self._swa_ssd_handle.kv_layout,
                            dtype=swa_handles[0].dtype,
                            tp_group_size=self.model_config.effective_tp_size_per_node,
                            **self._get_swa_multi_group_kwargs_tp(worker_key),
                        )
                        for worker_key, swa_handles in self._swa_gpu_handles.items()
                    }
                self._swa_worker_map[TransferType.DISK2D] = self._swa_gds_workers
                self._swa_worker_map[TransferType.D2DISK] = self._swa_gds_workers
                flexkv_logger.info("TransferEngine: swa GDS workers initialized")
            self._has_swa = True
            # Must mirror the create condition above.
            if not _enable_layerwise:
                flexkv_logger.info(
                    f"TransferEngine: swa workers initialized "
                    f"({len(self._swa_h2d_workers)} H2D + {len(self._swa_d2h_workers)} D2H)")
            else:
                flexkv_logger.info(
                    f"TransferEngine: swa inline workers initialized "
                    f"(H2D fused into layerwise, {len(self._swa_d2h_workers)} D2H)")

        if len(self._worker_map) == 0:
            raise ValueError("No workers initialized, please check the config")

        # 中文：spawn 出来的子进程要在自己进程里建 CUDA 上下文、import vLLM 的
        # VMM 映射，慢且可能失败（典型是 GPU0 上抢占显存导致 CUDA OOM）。
        # 所以不能无脑 wait()：每 5s 醒一次看进程是不是已经死了，死了立刻抛，
        # 否则会一直卡在启动阶段且看不到真正的报错。
        def _wait_worker_ready(
            worker: WorkerHandle,
            transfer_type: TransferType,
            worker_key: Optional[WorkerKey] = None,
        ) -> None:
            """Wait for ready_event, but fail fast if the process already died."""
            label = (
                f"{transfer_type.name} worker {worker.worker_id}"
                + (f" key={worker_key}" if worker_key is not None else "")
            )
            while not worker.ready_event.wait(timeout=5.0):
                if not worker.process.is_alive():
                    raise RuntimeError(
                        f"{label} died during init "
                        f"(exitcode={worker.process.exitcode}); "
                        f"see worker traceback above (often CUDA OOM from "
                        f"wrong-device context on GPU0)"
                    )
                flexkv_logger.debug(f"still waiting for {label} to ready")
            flexkv_logger.debug(f"{label} is ready")

        # Wait for all main KV workers to ready
        for transfer_type, worker in self._worker_map.items():
            if isinstance(worker, dict):
                for wk, w in worker.items():
                    _wait_worker_ready(w, transfer_type, wk)
            else:
                _wait_worker_ready(worker, transfer_type)

        # Wait for all SWA dedicated workers to be ready
        if self._has_swa:
            for transfer_type, worker in self._swa_worker_map.items():
                if isinstance(worker, dict):
                    for wk, w in worker.items():
                        _wait_worker_ready(w, transfer_type, wk)
                else:
                    _wait_worker_ready(worker, transfer_type)

        # Startup assertions: verify layerwise mode worker map consistency
        if _enable_layerwise:
            assert TransferType.H2D not in self._worker_map, \
                "H2D worker should not exist in layerwise mode (fused into layerwise worker)"
            assert TransferType.DISK2H not in self._worker_map, \
                "DISK2H worker should not exist in layerwise mode (fused into layerwise worker)"
            assert TransferType.LAYERWISE in self._worker_map, \
                "LAYERWISE worker must exist when layerwise transfer is enabled"

        # Start scheduler thread
        # 中文：_running 必须先置 True 再 start 线程，否则线程里的
        # while self._running 可能一进去就退出（竞态）。
        self._running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self._scheduler_thread.start()

    def _collect_worker_handles(self) -> List[WorkerHandle]:
        """把 _worker_map 与 _swa_worker_map 里所有 WorkerHandle 摊平成一个列表。

        两张 map 的值既可能是单个 WorkerHandle，也可能是 Dict[WorkerKey, ...]，
        这里统一摊平，供 shutdown / 回滚路径遍历。用 getattr 取 map 是因为
        _init_workers 可能中途失败（此时 _worker_map 还没建出来）。
        """
        handles: List[WorkerHandle] = []
        for worker in getattr(self, "_worker_map", {}).values():
            if isinstance(worker, dict):
                handles.extend(worker.values())
            else:
                handles.append(worker)
        for worker in getattr(self, "_swa_worker_map", {}).values():
            if isinstance(worker, dict):
                handles.extend(worker.values())
            else:
                handles.append(worker)
        return handles

    # 中文：并行停 worker。为什么必须并行——每个 worker 退出时要做
    # cudaHostUnregister 等清理，单进程耗时可能到秒级；串行关几十个 worker
    # 会成倍叠加，很容易超过上层的关停超时。并行后总耗时≈最慢的那个。
    def _shutdown_worker_handles(self, handles: List[WorkerHandle]) -> None:
        """Stop worker processes in parallel (send sentinel + join/unregister)."""
        if not handles:
            return
        flexkv_logger.info(
            f"TransferEngine: stopping {len(handles)} worker(s) in parallel"
        )

        def _shutdown_one(handle: WorkerHandle) -> None:
            try:
                handle.shutdown()
            except Exception as e:
                flexkv_logger.error(
                    f"Error shutting down worker {handle.worker_id}: {e}"
                )

        threads = [
            threading.Thread(
                target=_shutdown_one,
                args=(h,),
                name=f"flexkv-worker-shutdown-{h.worker_id}",
            )
            for h in handles
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # 中文：_init_workers 中途失败的补偿路径。spawn 是逐个建进程的，
    # 第 N 个失败时前 N-1 个已经起来并占着 GPU/共享内存，必须回收，
    # 否则失败重试会残留僵尸 worker 并泄漏显存。best-effort：任何异常都吞掉。
    def _rollback_init_workers(self, err: BaseException) -> None:
        """Best-effort cleanup when spawn/ready fails before engine is running."""
        handles = self._collect_worker_handles()
        if not handles:
            return
        flexkv_logger.error(
            f"TransferEngine init failed ({err}); "
            f"rolling back {len(handles)} already-created worker(s)"
        )
        self._shutdown_worker_handles(handles)
        self._worker_map = {}
        if hasattr(self, "_swa_worker_map"):
            self._swa_worker_map = {}

    def start(self) -> None:
        """启动引擎：建 worker 进程 + 拉起调度线程（阻塞，直到所有 worker ready）。

        调用者：KVTaskEngine / KVManager 初始化阶段。成功后本文件进入可服务状态，
        上层即可 submit_transfer_graph。失败会先回滚已建好的 worker 再抛出。
        """
        try:
            self._init_workers()
        except Exception as e:
            # Covers: mid-spawn exception, ready timeout/death, startup asserts.
            # Child-side __init__ failure also unpins inside the worker process;
            # this rolls back sibling workers that already became ready.
            if not self._running:
                self._rollback_init_workers(e)
            raise

    # ==========================================================================
    # 中文：本文件的心脏。单线程事件循环，唯一修改调度状态的线程。
    #
    # 监听什么（3 个 fd 全部注册进同一个 selectors，一次 select 同时等）：
    #   1. task_queue._reader         -> 上层提交了新 DAG（submit_transfer_graph）
    #   2. finished_ops_queue._reader -> 任意 worker 回报"某个 op 搬完了/失败了"
    #                                    （所有 worker 共用这一条队列，靠 payload 里的
    #                                     op_id 区分是谁、是父 op 还是 PP 副本）
    #   3. shutdown_read_fd           -> shutdown 唤醒管道（纯通知，无数据语义）
    # 用 mp.Queue 而不是 queue.Queue，就是为了拿它的 _reader fd 注册进 selectors；
    # select(timeout=None) 完全阻塞，所以没有轮询开销，也没有固定超时的延迟惩罚。
    #
    # 一轮循环做四件事（顺序有讲究）：
    #   ① 收新图    ：把 task_queue 里所有图一次性 get_nowait 抽干，交给
    #                  scheduler.add_transfer_graph（注意：这里只入 DAG，不派发）。
    #                  payload 可能是单个图也可能是 list（批量提交），两种都要支持。
    #   ② 收完成    ：把 finished_ops_queue 抽干。每个 payload 走两条分支之一：
    #                  - 是副本 op（在 _child_to_parent_op_id 里）：parent.pending_count--
    #                    ，归零才认为整个 op 完成（PP 各 stage 原子完成）。
    #                  - 是父 op：直接 pending_count--，归零即完成。
    #                  完成/失败都走 _finalize_or_discard / _handle_failed_op。
    #   ③ 推进 DAG  ：只有"有图完成或有新图"时才调 scheduler.schedule(finished_ops)，
    #                  拿回 (已完成的 graph_id 列表, 下一批就绪的 op)。
    #                  —— DAG 内部的依赖/就绪判定全在 scheduler.py，本文件只消费结果。
    #   ④ 派发 + 汇报：就绪 op 逐个 _assign_op_to_worker 发给具体 worker；
    #                  VIRTUAL op 不落盘不搬运，直接伪造一个完成通知塞回上层；
    #                  已完成的 graph_id 打包成 CompletedOp.completed_graph 推给上层。
    #
    # 为什么不加锁：op_id_to_op / _child_to_parent_op_id / scheduler / pin_buffer
    # 全部只在本线程读写，对外只通过 mp.Queue，因此无需任何锁。
    #
    # 出错不退出：整个循环体包在 try 里，异常只记日志 + sleep 1ms 继续，
    # 避免一次偶发错误让整个数据面停摆（日志里会打印各映射表前 16 个 key 便于定位）。
    # ==========================================================================
    def _scheduler_loop(self) -> None:
        """Event-driven scheduler loop using selectors (ZERO LATENCY with shutdown pipe)"""
        from flexkv.common.debug import flexkv_logger

        # Setup selector to monitor both queues simultaneously
        sel = selectors.DefaultSelector()

        # Register both queues for monitoring
        sel.register(self.task_queue._reader, selectors.EVENT_READ, data="new_graph")
        sel.register(self.finished_ops_queue._reader, selectors.EVENT_READ, data="finished_op")

        # Register shutdown pipe for zero-latency shutdown
        sel.register(self.shutdown_read_fd, selectors.EVENT_READ, data="shutdown")

        flexkv_logger.info("TransferEngine scheduler loop started with ZERO-LATENCY selector (timeout=None)")

        while self._running:
            try:
                # Complete blocking with NO TIMEOUT for zero latency!
                # Shutdown via pipe signal instead of timeout
                events = sel.select(timeout=None)

                new_graphs_num = 0
                finished_ops: List[TransferOp] = []
                should_shutdown = False

                # Process events from selector
                for key, mask in events:
                    if key.data == "shutdown":
                        # Shutdown signal received via pipe
                        flexkv_logger.info("Scheduler loop received shutdown signal via pipe")
                        should_shutdown = True
                        break

                    elif key.data == "new_graph":
                        # Process new transfer graphs (batch get all available)
                        nvtx_r1 = nvtx.start_range(message="transfer scheduler. get new graphs", color="orange")
                        # Get all available graphs in one go to reduce system calls
                        while True:
                            try:
                                transfer_graph = self.task_queue.get_nowait()
                                # Handle batch submission (list of graphs)
                                graphs = transfer_graph if isinstance(transfer_graph, list) else [transfer_graph]
                                for graph in graphs:
                                    self.scheduler.add_transfer_graph(graph)
                                new_graphs_num += len(graphs)
                            except queue.Empty:
                                break
                        nvtx.end_range(nvtx_r1)

                    elif key.data == "finished_op":
                        # Collect finished ops from main KV worker (batch get all available)
                        nvtx_r2 = nvtx.start_range(message="transfer scheduler. collect finished ops", color="orange")
                        # Get all available ops in one go to reduce system calls
                        while True:
                            try:
                                payload = self.finished_ops_queue.get_nowait()
                                # Payload forms:
                                #   WorkerTransferResult (partial block outcomes)
                                #   int (legacy success)
                                #   (op_id|WorkerTransferResult, ok)
                                #   (op_id|WorkerTransferResult, ok, metrics)
                                op_succeeded = True
                                metrics = None
                                block_results = None
                                if isinstance(payload, WorkerTransferResult):
                                    op_id = payload.transfer_op_id
                                    block_results = payload.block_results
                                elif isinstance(payload, tuple):
                                    if len(payload) >= 3:
                                        first, op_succeeded, metrics = (
                                            payload[0], payload[1], payload[2])
                                    else:
                                        first, op_succeeded = payload[0], payload[1]
                                    if isinstance(first, WorkerTransferResult):
                                        op_id = first.transfer_op_id
                                        block_results = first.block_results
                                    else:
                                        op_id = first
                                else:
                                    op_id = payload
                                if not op_succeeded:
                                    # Keep trace state consistent even on failure.
                                    trace.dec_inflight()
                                    trace.consume_submit_ns(op_id)
                                    self._handle_failed_op(op_id)
                                    continue
                                if op_id in self._child_to_parent_op_id:
                                    # Replica op (LAYERWISE PP fan-out): decrement parent's
                                    # pending_count and finalize parent when all replicas done.
                                    parent_op_id = self._child_to_parent_op_id.pop(op_id)
                                    child_op = self._child_id_to_child.pop(op_id)
                                    self._merge_block_results(child_op, block_results)
                                    free_op_from_buffer(child_op, self.pin_buffer)
                                    if op_id in self.op_id_to_nvtx_range:
                                        nvtx.end_range(self.op_id_to_nvtx_range.pop(op_id))
                                    self._emit_xfer_trace(op_id, metrics)
                                    parent_op = self.op_id_to_op[parent_op_id]
                                    self._merge_block_results(
                                        parent_op, child_op.block_results)
                                    parent_op.pending_count -= 1
                                    if parent_op.pending_count == 0:
                                        self._finalize_or_discard(parent_op, finished_ops)
                                    flexkv_logger.debug(
                                        f"[TransferEngine] child op {op_id} completed, "
                                        f"parent op {parent_op_id} pending_count={parent_op.pending_count}")
                                else:
                                    op = self.op_id_to_op[op_id]
                                    self._merge_block_results(op, block_results)
                                    op.pending_count -= 1
                                    self._emit_xfer_trace(op_id, metrics)
                                    if op.pending_count == 0:
                                        self._finalize_or_discard(op, finished_ops)
                            except queue.Empty:
                                break
                        nvtx.end_range(nvtx_r2)

                # Exit loop if shutdown requested
                if should_shutdown:
                    break

                # End NVTX ranges for finished ops
                for op in finished_ops:
                    nvtx_range = self.op_id_to_nvtx_range.pop(op.op_id, None)
                    if nvtx_range is not None:
                        nvtx.end_range(nvtx_range)

                # Schedule next operations
                nvtx_r3 = nvtx.start_range(message="transfer scheduler. schedule next ops", color="orange")
                if finished_ops or new_graphs_num > 0:
                    completed_graph_ids, next_ops = self.scheduler.schedule(finished_ops)
                    # Distribute new ops to workers
                    for op in next_ops:
                        # VIRTUAL op 是 DAG 里的"记账/汇合"节点（无实际数据搬运），
                        # 不发给任何 worker，直接伪造完成通知让上层推进状态机。
                        if op.transfer_type == TransferType.VIRTUAL:
                            self.completed_queue.put(CompletedOp(graph_id=op.graph_id, op_id=op.op_id))
                        else:
                            self.op_id_to_op[op.op_id] = op
                            # Unified rule for both main-KV and SWA paths:
                            # only register here when the resolved worker_map
                            # entry is a single worker (no PP fan-out). For
                            # dict-keyed entries (H2D/D2H), each replica is
                            # registered inside _assign_op_to_worker /
                            # _assign_swa_op_to_worker per PP sibling.
                            if self._op_buffer_registered_here(op):
                                register_op_to_buffer(op, self.pin_buffer)
                            self._assign_op_to_worker(op)
                    # Handle completed graphs
                    for graph_id in completed_graph_ids:
                        self.completed_queue.put(CompletedOp.completed_graph(graph_id))
                nvtx.end_range(nvtx_r3)

                # Outside the dispatch block: a tick may consist solely of a
                # failure report, with no finished op and no new graph.
                if self._failed_graph_ids:
                    self._emit_drained_graph_failures()

            except Exception as e:
                flexkv_logger.error(
                    f"Error in scheduler loop: {type(e).__name__}: {e!r} "
                    f"| op_id_to_op keys={list(self.op_id_to_op.keys())[:16]} "
                    f"(total={len(self.op_id_to_op)}) "
                    f"| child->parent keys={list(self._child_to_parent_op_id.keys())[:16]} "
                    f"(total={len(self._child_to_parent_op_id)}) "
                    f"| nvtx_range keys={list(self.op_id_to_nvtx_range.keys())[:16]} "
                    f"(total={len(self.op_id_to_nvtx_range)})",
                    exc_info=True,
                )
                time.sleep(0.001)  # Fallback on error

        # Cleanup
        sel.close()
        flexkv_logger.info("TransferEngine scheduler loop stopped")

    # 中文：pin_buffer 的"谁注册谁释放"判定——这是本文件最容易出 bug 的约定。
    #   单实例 worker 通路：父 op 自己在调度线程注册，也在这里释放（1 次）。
    #   dict(PP fan-out) 通路：父 op 根本不注册，每个副本各注册各释放（N 次）。
    # 两条路必须严格互斥，混用会导致槽位双重释放（别的 op 的数据被写花）
    # 或漏释放（槽位耗尽后 allocate 卡死）。派发、_finalize_op、_discard_failed_op
    # 三处共用这一个判定，保证口径一致。
    def _op_buffer_registered_here(self, op: TransferOp) -> bool:
        """The 'unified rule' shared by dispatch, _finalize_op and
        _discard_failed_op: a parent op's pin buffer is registered (and thus
        freed) at this level only when its worker_map entry resolves to a
        single worker. Dict-keyed entries (PP fan-out) register and free each
        replica individually."""
        if getattr(op, "is_swa", False):
            resolved_worker = self._swa_worker_map.get(op.transfer_type)
        else:
            resolved_worker = self._worker_map.get(op.transfer_type)
        return resolved_worker is not None and not isinstance(resolved_worker, dict)

    # 中文：op 的 pending_count 归零后的分流闸门。
    # 有副本失败过 -> _discard_failed_op（静默丢弃，绝不发完成通知，也绝不放行
    # 后继 op，否则上层会拿到"部分搬完"的 KV 并当成完整前缀用，直接算错结果）；
    # 全部成功 -> _finalize_op（发完成通知 + 让 scheduler 推进后继）。
    def _finalize_or_discard(self, op: TransferOp, finished_ops: List[TransferOp]) -> None:
        """Route a fully-drained op: discard if any replica of it failed,
        finalize (completion message + successor scheduling) otherwise."""
        if op.op_id in self._failed_parent_op_ids:
            self._discard_failed_op(op)
        else:
            self._finalize_op(op, finished_ops)

    def _emit_xfer_trace(self, op_id: int, metrics) -> None:
        """Print one ``[XFER]`` line for a completed op (transfer tracing).

        Combines the worker-computed timing metrics with the scheduler-side
        e2e (submit -> detect) and current backlog. All trace.* calls no-op
        when ``FLEXKV_TRANSFER_TRACE`` is unset, so this is safe to call
        unconditionally on every finished op.
        """
        e2e_ms = trace.consume_submit_ns(op_id)
        trace.dec_inflight()
        trace.record_xfer(op_id, metrics, e2e_ms)

    # 中文：失败传播链的起点。worker 报错 -> 这里把 op 标记失败、pending_count
    # 照样递减（否则永远收不齐）、graph 记入 _failed_graph_ids 并调
    # scheduler.fail_graph 让该图剩余 op 不再派发；等这张图在飞的 op 全部排干后，
    # 由 _emit_drained_graph_failures 统一发一次"整图失败"给上层。
    def _handle_failed_op(self, op_id: int) -> None:
        """A worker reported a failed transfer for ``op_id``.

        Bookkeeping mirrors the completion path (replica maps, pending
        counts, pin buffer, nvtx) so nothing leaks, but the op never reaches
        finished_ops (its successors must not run) and its graph is marked
        failed: the scheduler stops dispatching the graph's remaining ops,
        and once every already-dispatched op of the graph has drained the
        loop emits a graph-level failure to the task layer.
        """
        graph_id = None
        if op_id in self._child_to_parent_op_id:
            parent_op_id = self._child_to_parent_op_id.pop(op_id)
            child_op = self._child_id_to_child.pop(op_id)
            free_op_from_buffer(child_op, self.pin_buffer)
            if op_id in self.op_id_to_nvtx_range:
                nvtx.end_range(self.op_id_to_nvtx_range.pop(op_id))
            parent_op = self.op_id_to_op.get(parent_op_id)
            if parent_op is not None:
                graph_id = parent_op.graph_id
                parent_op.pending_count -= 1
                self._failed_parent_op_ids.add(parent_op_id)
                if parent_op.pending_count == 0:
                    self._discard_failed_op(parent_op)
        else:
            op = self.op_id_to_op.get(op_id)
            if op is not None:
                graph_id = op.graph_id
                op.pending_count -= 1
                self._failed_parent_op_ids.add(op_id)
                if op.pending_count == 0:
                    self._discard_failed_op(op)
        if graph_id is not None:
            flexkv_logger.error(
                f"[TransferEngine] transfer op {op_id} of graph {graph_id} "
                f"failed; failing the graph and draining its in-flight ops")
            self._failed_graph_ids.add(graph_id)
            self.scheduler.fail_graph(graph_id)

    def _discard_failed_op(self, op: TransferOp) -> None:
        """Release a fully-drained op that must not complete (it failed, or a
        replica of it did): same pin-buffer rule and cleanup as _finalize_op,
        minus the completion message and successor scheduling."""
        if self._op_buffer_registered_here(op):
            free_op_from_buffer(op, self.pin_buffer)
        if op.op_id in self.op_id_to_nvtx_range:
            nvtx.end_range(self.op_id_to_nvtx_range.pop(op.op_id))
        self.op_id_to_op.pop(op.op_id, None)
        self._failed_parent_op_ids.discard(op.op_id)

    # 中文：为什么要"等排干再报整图失败"——如果 op 一失败就立刻报图失败，
    # 上层的回滚（释放 block、回滚索引）会和还在飞的兄弟 op 的完成回调抢同一批
    # block，出现"回滚完了数据又被写回来"。所以这里等 op_id_to_op 里该图的
    # op 全部消失后才发通知。
    def _emit_drained_graph_failures(self) -> None:
        """Report each failed graph to the task layer once its dispatched ops
        have all drained, so the task's rollback never races an in-flight op's
        completion callback."""
        for graph_id in list(self._failed_graph_ids):
            if any(op.graph_id == graph_id for op in self.op_id_to_op.values()):
                continue
            self.completed_queue.put(CompletedOp.failed_graph(graph_id))
            self._failed_graph_ids.discard(graph_id)

    # 中文：op 真正完成时的收尾。三件事：
    #   1) 按统一规则释放 pin_buffer 槽位；
    #   2) 算这次传输的字节数（用于上层统计带宽/命中收益）并往 completed_queue
    #      推一个 CompletedOp —— 这是数据面回给上层的唯一出口；
    #   3) 把 op 放进 finished_ops，交给本轮末尾的 scheduler.schedule() 去解锁
    #      它的后继 op（注意：本文件不自己判断后继，那是 scheduler.py 的活）。
    # 字节数为什么要按"本 PP stage 的层数"折算：每张图的 block 只含本 stage
    # 负责的层，直接乘全模型层数会系统性高估。
    def _finalize_op(self, op: TransferOp, finished_ops: List[TransferOp]) -> None:
        """Finalize a completed op: release pin buffer, notify upper layer, and clean up.

        Called only when op.pending_count reaches 0, i.e., all PP-sibling replica
        workers have completed this op. This ensures atomic eviction semantics.
        """
        # Unified rule: free the parent op buffer here only if the parent itself
        # was registered upstream (single-worker path). For dict-keyed (PP fan-out)
        # entries the parent was never registered; each replica was registered and
        # freed individually in the scheduler's child completion path.
        if self._op_buffer_registered_here(op):
            free_op_from_buffer(op, self.pin_buffer)
        # Compute transfer metrics for this completed op.
        # Use layer_groups-aware token size so overlapping main/indexer groups
        # report their combined byte count.
        num_blocks = len(op.src_block_ids) if op.src_block_ids is not None else 0
        total_token_bytes = self.model_config.token_size_in_bytes
        total_layers = self.model_config.num_layers
        avg_bytes_per_layer = total_token_bytes // max(1, total_layers)
        token_size_in_bytes_per_pp_stage = self._num_layers_for_local_pp_stage * avg_bytes_per_layer
        num_bytes = num_blocks * self.cache_config.tokens_per_block * token_size_in_bytes_per_pp_stage
        transfer_type_str = op.transfer_type.value if op.transfer_type != TransferType.VIRTUAL else None
        self.completed_queue.put(CompletedOp(
            graph_id=op.graph_id,
            op_id=op.op_id,
            transfer_type=transfer_type_str,
            num_blocks=num_blocks,
            num_bytes=num_bytes,
            block_results=op.block_results,
        ))
        finished_ops.append(op)
        del self.op_id_to_op[op.op_id]

    # 中文：合并多个副本（PP 各段 / TP 各卡）回报的"每个 block 是否成功"。
    # 合并规则是逻辑与：任一个副本说某 block 失败，该 block 就整体算失败
    # （partial success 不能算命中，上层必须按未命中处理，否则会用残缺 KV 算错）。
    # 长度不一致视为全部失败并记 error —— 这是 worker 与本层对 block 数理解
    # 不一致的信号，不能静默放过。
    @staticmethod
    def _merge_block_results(
        op: TransferOp,
        block_results: Optional[Tuple[bool, ...]],
    ) -> None:
        """Accumulate per-worker outcomes; every participating worker must win."""
        if block_results is None:
            return
        normalized = tuple(bool(result) for result in block_results)
        if len(normalized) != len(op.src_block_ids):
            flexkv_logger.error(
                f"Completion result length mismatch for op {op.op_id}: "
                f"results={len(normalized)}, blocks={len(op.src_block_ids)}")
            normalized = (False,) * len(op.src_block_ids)
        if op.block_results is None:
            op.block_results = normalized
        else:
            op.block_results = tuple(
                old and new for old, new in zip(op.block_results, normalized))

    # 中文：PP fan-out 的"找兄弟"逻辑。WorkerKey = (dp_client_id, pp_rank)，
    # 同一个 dp_client_id 下的所有 pp_rank 就是同一请求的各流水段，它们必须
    # 收到同一份 op 副本（各段只搬自己那份层）。扁平化后 dp_client_id 已能
    # 唯一标识一个 DP 切片，所以这里只按它过滤。
    @staticmethod
    def _match_pp_siblings(
        worker_map: Dict[WorkerKey, WorkerHandle],
        dp_client_id: int,
    ) -> List[WorkerKey]:
        """Return every WorkerKey whose flat DP slice equals ``dp_client_id``.

        After flattening, a single int fully identifies the DP slice —
        PP siblings are the worker_keys that share it across pp_rank.
        """
        return [wk for wk in worker_map if wk.dp_client_id == dp_client_id]

    # 中文：逐层传输 op 的 PP fan-out。与普通 op 的差别在于副本类型是
    # LayerwiseTransferOp，一次副本里同时带 h2d 与 disk2h 两组 block id
    # （以及对应的 SWA id）—— 因为逐层流水要求"读盘"和"拷 GPU"由同一个
    # worker 进程内交错执行，不能拆成两个 op。
    def _assign_layerwise_op_to_workers(self, op: TransferOp) -> None:
        """Fan-out a LAYERWISE op symmetrically to every local PP-stage
        sibling worker matching ``op.dp_client_id``."""
        from flexkv.common.transfer import LayerwiseTransferOp
        assert isinstance(op, LayerwiseTransferOp)

        worker_map = self._worker_map[TransferType.LAYERWISE]
        assert isinstance(worker_map, dict), \
            "LAYERWISE worker map must be a Dict[WorkerKey, WorkerHandle]"

        sibling_keys = self._match_pp_siblings(worker_map, op.dp_client_id)
        if not sibling_keys:
            raise ValueError(
                f"No LAYERWISE worker found matching "
                f"dp_client_id={op.dp_client_id}; "
                f"available worker keys={list(worker_map.keys())}"
            )

        for wk in sibling_keys:
            replica = LayerwiseTransferOp(
                graph_id=op.graph_id,
                src_block_ids_h2d=op.src_block_ids_h2d.copy(),
                dst_block_ids_h2d=op.dst_block_ids_h2d.copy(),
                src_block_ids_disk2h=op.src_block_ids_disk2h.copy(),
                dst_block_ids_disk2h=op.dst_block_ids_disk2h.copy(),
                # SWA ids must be carried through PP fan-out replicas, otherwise
                # each PP sibling's worker would only see main-KV ids and the SWA
                # layer-fused branch in cpp would be silently skipped.
                swa_src_block_ids_h2d=op.swa_src_block_ids_h2d.copy(),
                swa_dst_block_ids_h2d=op.swa_dst_block_ids_h2d.copy(),
                swa_src_block_ids_disk2h=op.swa_src_block_ids_disk2h.copy(),
                swa_dst_block_ids_disk2h=op.swa_dst_block_ids_disk2h.copy(),
                dp_client_id=op.dp_client_id,
                counter_id=op.counter_id,
            )
            register_op_to_buffer(replica, self.pin_buffer)
            self._child_id_to_child[replica.op_id] = replica
            self._child_to_parent_op_id[replica.op_id] = op.op_id
            self.op_id_to_nvtx_range[replica.op_id] = nvtx.start_range(
                f"schedule {replica.transfer_type.name}_REPLICA op_id: {replica.op_id}, "
                f"graph_id: {replica.graph_id}, worker_key={wk}",
                color=get_nvtx_range_color(replica.graph_id))
            op.pending_count += 1
            worker_map[wk].submit_transfer(replica)
            flexkv_logger.debug(
                f"[TransferEngine] LAYERWISE fan-out: "
                f"parent_op_id={op.op_id}, replica_op_id={replica.op_id}, "
                f"worker_key={wk}, pending_count={op.pending_count}")

    # 中文：SWA op 的派发，与主 KV 的 _assign_op_to_worker 结构完全对称
    # （dict -> PP fan-out；单实例 -> 直接提交），唯一区别是查 _swa_worker_map。
    # SWA op 由控制面在构图时直接标 is_swa=True，本文件不做任何"从主 KV op
    # 派生 SWA op"的推断。
    def _assign_swa_op_to_worker(self, op: TransferOp) -> None:
        """Route a graph-built ``is_swa=True`` op to the SWA worker map.

        Structurally identical to the main-KV dispatch path:
          * dict worker_entry (H2D/D2H, keyed by WorkerKey for PP siblings)
            -> PP fan-out: derive one replica per sibling, register each,
               track in _child_to_parent_op_id, pending_count++ per replica.
            This is needed because each PP stage holds its own slice of SWA
            layers, exactly like the main-KV path: a single submit to one
            sibling would silently drop the other stages\' SWA data.
          * single-instance worker_entry (CPU<->SSD / CPU<->Remote)
            -> no fan-out; pending_count++ and submit op directly.
            register_op_to_buffer + op_id_to_op are done by the scheduler
            upstream for this branch, exactly like main-KV single-worker.
        """
        if op.transfer_type not in self._swa_worker_map:
            raise ValueError(f"Unsupported SWA transfer type: {op.transfer_type}")
        worker_entry = self._swa_worker_map[op.transfer_type]

        if isinstance(worker_entry, dict):
            sibling_keys = self._match_pp_siblings(worker_entry, op.dp_client_id)
            if not sibling_keys:
                raise ValueError(
                    f"No SWA_{op.transfer_type.name} worker found matching "
                    f"dp_client_id={op.dp_client_id}; "
                    f"available worker keys={list(worker_entry.keys())}"
                )
            for wk in sibling_keys:
                replica = TransferOp(
                    graph_id=op.graph_id,
                    transfer_type=op.transfer_type,
                    src_block_ids=op.src_block_ids.copy(),
                    dst_block_ids=op.dst_block_ids.copy(),
                    dp_client_id=op.dp_client_id,
                    is_swa=True,
                    mooncake_store_swa_block_hashes=(
                        list(op.mooncake_store_swa_block_hashes)
                        if op.mooncake_store_swa_block_hashes is not None else None),
                )
                register_op_to_buffer(replica, self.pin_buffer)
                self._child_id_to_child[replica.op_id] = replica
                self._child_to_parent_op_id[replica.op_id] = op.op_id
                self.op_id_to_nvtx_range[replica.op_id] = nvtx.start_range(
                    f"schedule SWA_{op.transfer_type.name}_REPLICA op_id: {replica.op_id}, "
                    f"graph_id: {replica.graph_id}, worker_key={wk}",
                    color=get_nvtx_range_color(replica.graph_id),
                )
                op.pending_count += 1
                worker_entry[wk].submit_transfer(replica)
                flexkv_logger.debug(
                    f"[TransferEngine] SWA_{op.transfer_type.name} fan-out: "
                    f"parent_op_id={op.op_id}, replica_op_id={replica.op_id}, "
                    f"worker_key={wk}, pending_count={op.pending_count}"
                )
        else:
            self.op_id_to_nvtx_range[op.op_id] = nvtx.start_range(
                f"schedule SWA_{op.transfer_type.name} op_id: {op.op_id}, "
                f"graph_id: {op.graph_id}, successors: {op.successors}",
                color=get_nvtx_range_color(op.graph_id),
            )
            op.pending_count += 1
            worker_entry.submit_transfer(op)
            flexkv_logger.debug(
                f"[TransferEngine] Submitted SWA op {op.op_id}: "
                f"type={op.transfer_type.name}, single-worker, "
                f"blocks={op.src_block_ids.size}, pending_count={op.pending_count}"
            )

    # ==========================================================================
    # 中文：op -> worker 的路由总入口（DAG 拆解的最后一跳）。
    #
    # "一张图怎么被拆成 op 发给 worker"的完整链路：
    #   控制面把一次 GET/PUT 编成 TransferOpGraph（节点=op，边=依赖）
    #   -> 上层 submit_transfer_graph 整图丢进 task_queue
    #   -> 调度线程塞进 scheduler（scheduler.py 负责算哪些 op 已就绪）
    #   -> 每轮 schedule() 返回"这一拍可以跑的 op 列表"
    #   -> 【本函数】逐个 op 决定"派给哪个 worker 进程"
    # 所以：拆 DAG 的是 scheduler.py，挑 worker 的是本函数，二者分工明确。
    #
    # 分配策略（两级，先按通路再按拓扑）：
    #   第一级：op.transfer_type 决定查哪张表的哪个键
    #           （H2D/D2H/DISK2H/H2DISK/REMOTE2H/... = 一条"设备对"通路）。
    #           换句话说：按"源设备->目的设备这条通路"选 worker 种类。
    #   第二级：取到的值决定发几份
    #           - 单个 WorkerHandle（CPU<->SSD、CPU<->Remote 等与 GPU 拓扑无关的
    #             通路）：pending_count=1，直接 submit。
    #           - Dict[WorkerKey, WorkerHandle]（H2D/D2H/GDS/LAYERWISE 等）：
    #             PP fan-out —— 按 op.dp_client_id 找出所有 pp_rank 兄弟，
    #             每个兄弟复制一份副本 op，各自 register 槽位、各自 submit，
    #             每份让父 op 的 pending_count +1。
    #           即"按 DP 切片选具体进程"，PP 各段各收一份，靠 pending_count 归零
    #           保证它们原子完成。
    #
    # 另外两条特殊路由：
    #   - op.is_swa 为真 -> 转 _assign_swa_op_to_worker，查 _swa_worker_map
    #     （SWA 池与主 KV 池是两套显存/内存，通路完全隔离）。
    #   - transfer_type == LAYERWISE -> 转 _assign_layerwise_op_to_workers。
    #   - transfer_type == VIRTUAL  -> 直接返回（DAG 记账节点，不落地搬运）。
    #
    # 隐含约定：pending_count 必须在本线程 submit 之前自增（本函数是唯一自增点），
    # 否则 worker 回报到达时计数还没设好，会提前归零误判完成。
    # ==========================================================================
    def _assign_op_to_worker(self, op: TransferOp) -> None:
        """Assign operation to appropriate worker."""
        if op.transfer_type == TransferType.VIRTUAL:
            return

        if op.is_swa:
            # SWA ops are built directly in the transfer graph (is_swa=True)
            # and routed to _swa_worker_map; they are NOT derived from main-KV
            # ops at dispatch time.
            self._assign_swa_op_to_worker(op)
            return

        if op.transfer_type not in self._worker_map:
            raise ValueError(f"Unsupported transfer type: {op.transfer_type}")

        if op.transfer_type == TransferType.LAYERWISE:
            self._assign_layerwise_op_to_workers(op)
            return

        worker = self._worker_map[op.transfer_type]
        if isinstance(worker, dict):
            sibling_keys = self._match_pp_siblings(worker, op.dp_client_id)
            if not sibling_keys:
                raise ValueError(
                    f"No MAIN_KV_{op.transfer_type.name} worker found matching "
                    f"dp_client_id={op.dp_client_id}; "
                    f"available worker keys={list(worker.keys())}"
                )
            for wk in sibling_keys:
                replica = TransferOp(
                    graph_id=op.graph_id,
                    transfer_type=op.transfer_type,
                    src_block_ids=op.src_block_ids.copy(),
                    dst_block_ids=op.dst_block_ids.copy(),
                    dp_client_id=op.dp_client_id,
                    mooncake_store_block_hashes=(
                        op.mooncake_store_block_hashes.copy()
                        if op.mooncake_store_block_hashes is not None else None),
                )
                register_op_to_buffer(replica, self.pin_buffer)
                self._child_id_to_child[replica.op_id] = replica
                self._child_to_parent_op_id[replica.op_id] = op.op_id
                self.op_id_to_nvtx_range[replica.op_id] = nvtx.start_range(
                    f"schedule {replica.transfer_type.name}_REPLICA op_id: {replica.op_id}, "
                    f"graph_id: {replica.graph_id}, worker_key={wk}",
                    color=get_nvtx_range_color(replica.graph_id))
                op.pending_count += 1
                worker[wk].submit_transfer(replica)
                flexkv_logger.debug(
                    f"[TransferEngine] MAIN_KV_{op.transfer_type.name} fan-out: "
                    f"parent_op_id={op.op_id}, replica_op_id={replica.op_id}, "
                    f"worker_key={wk}, pending_count={op.pending_count}")
        else:
            self.op_id_to_nvtx_range[op.op_id] = nvtx.start_range(
                f"schedule {op.transfer_type.name} "
                f"op_id: {op.op_id}, graph_id: {op.graph_id}, "
                f"successors: {op.successors}",
                color=get_nvtx_range_color(op.graph_id),
            )
            op.pending_count += 1
            worker.submit_transfer(op)

    # ==========================================================================
    # 中文：上层（KVTaskEngine）提交 DAG 的唯一入口 —— 数据面的"入口阀门"。
    # 非阻塞：只把图（或图的 list，批量提交）塞进 task_queue 就返回，
    # 真正的拆图、派发都在调度线程里做，所以提交方不会被 worker 的执行时间拖住。
    # 调用方需要记住 graph_id：后续靠 get_completed_graphs_and_ops 捞回来的
    # CompletedOp 里带 graph_id，据此判断哪个请求搬完了。
    # ==========================================================================
    def submit_transfer_graph(self, transfer_graph: Union[TransferOpGraph, List[TransferOpGraph]]) -> None:
        """Submit a transfer graph for execution"""
        nvtx_range = nvtx.start_range(message="TransferEngine.submit_transfer_graph", color="green")
        if not isinstance(transfer_graph, List):
            transfer_graph = [transfer_graph]
        self.task_queue.put(transfer_graph)
        nvtx.end_range(nvtx_range)

    # ==========================================================================
    # 中文：上层（KVTaskEngine 的结果线程）回收完成通知的唯一出口 —— "出口阀门"。
    #
    # 汇聚路径：worker 搬完 -> 写 finished_ops_queue -> 调度线程扣 pending_count
    #   -> 归零后 _finalize_op 往 completed_queue 推 CompletedOp
    #   -> 【本函数】一次性抽干 completed_queue 返回给上层。
    # 所以上层看到的 CompletedOp 是"已经把 PP 各段副本合并过"的完成事件，
    # 不需要自己关心 fan-out。
    #
    # 返回的 CompletedOp 有三种形态（见 common/transfer.py）：
    #   - 带 op_id + transfer_type/num_bytes：某个 op 搬完了，用于统计与精细回调
    #   - completed_graph(graph_id)：整张图跑完
    #   - failed_graph(graph_id)   ：整图失败（等该图在飞 op 排干后才发）
    #
    # 语义细节：timeout>0 时只"阻塞等第一个"、不等后续 —— 拿到第一个后就把队列里
    # 现成的全带走（批量摊薄系统调用）。这样既不会忙等烧 CPU，也不会为了凑批
    # 而人为增加延迟。
    # ==========================================================================
    def get_completed_graphs_and_ops(self, timeout: Optional[float] = None) -> List[CompletedOp]:
        """Drain completed ops, blocking up to ``timeout`` for the first one.

        The old early-return-on-empty ignored ``timeout`` and busy-spun the
        result thread at 100% CPU, starving the scheduler loop under high QPS.
        """
        completed_ops: List[CompletedOp] = []

        try:
            if timeout is None or timeout <= 0:
                # Non-blocking drain.
                if self.completed_queue.empty():
                    return completed_ops
                first_op = self.completed_queue.get_nowait()
            else:
                first_op = self.completed_queue.get(timeout=timeout)
            completed_ops.append(first_op)
        except queue.Empty:
            return completed_ops

        # Drain whatever else is immediately available.
        while not self.completed_queue.empty():
            try:
                completed_ops.append(self.completed_queue.get_nowait())
            except queue.Empty:
                break

        return completed_ops

    # 中文：GPU 热重映射（配合 vLLM sleep/wake 这类"引擎休眠-唤醒"场景）。
    # worker 启动时把 vLLM 的显存以 VMM 方式导入到自己进程；vLLM 休眠会作废
    # 这些映射，必须先让 worker 主动释放（suspend），唤醒时再导入新的（resume）。
    # 这样才能做到"不重建 worker 进程、不重新 spawn（省掉几十秒的 CUDA 初始化）"。
    # 前置校验很严格：有在飞传输、开了 layerwise / 多组布局 / SWA 都不支持。
    # 返回值是成功释放的映射数量，供上层核对。
    def suspend_gpu_mappings(self) -> int:
        """Drain worker pipes and release imported vLLM VMM mappings."""
        if self._gpu_mappings_suspended:
            return 0
        if GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer:
            raise NotImplementedError(
                "GPU hot remap does not support layerwise transfer"
            )
        if self.model_config.layer_groups is not None:
            raise NotImplementedError(
                "GPU hot remap does not support multi-group KV layouts"
            )
        if self._has_swa:
            raise NotImplementedError(
                "GPU hot remap does not support SWA KV pools"
            )
        if self.op_id_to_op or not self.task_queue.empty():
            raise RuntimeError(
                "Cannot suspend GPU mappings with transfers in flight"
            )
        released = 0
        for workers in (self.h2d_workers, self.d2h_workers):
            for worker in workers.values():
                released += int(worker.control("suspend_gpu"))
        self._gpu_mappings_suspended = True
        return released

    # 中文：suspend 的逆操作。要求传进来的 gpu_handle_groups 的 key 集合与启动时
    # 完全一致（拓扑变了必须重建引擎，不能热重映射），只是 handle 内容换成唤醒后
    # 的新 VMM handle。TP=1 传单卡 handle 列表，TP>1 传每卡一份的列表。
    def resume_gpu_mappings(
        self, gpu_handle_groups: Dict[WorkerKey, List[StorageHandle]]
    ) -> int:
        """Import fresh post-wake VMM handles into existing workers."""
        if not self._gpu_mappings_suspended:
            raise RuntimeError("GPU mappings are not suspended")
        if set(gpu_handle_groups) != set(self.gpu_handle_groups):
            raise ValueError(
                "GPU worker groups changed across sleep: "
                f"old={set(self.gpu_handle_groups)}, "
                f"new={set(gpu_handle_groups)}"
            )
        imported = 0
        for worker_key, handles in gpu_handle_groups.items():
            payload = (
                handles[0].get_tensor_handle_list()
                if self.model_config.effective_tp_size_per_node == 1
                else [handle.get_tensor_handle_list() for handle in handles]
            )
            imported += int(
                self.h2d_workers[worker_key].control("resume_gpu", payload)
            )
            imported += int(
                self.d2h_workers[worker_key].control("resume_gpu", payload)
            )
        self.gpu_handle_groups = gpu_handle_groups
        self._gpu_mappings_suspended = False
        return imported

    # 中文：关停顺序是有讲究的，改动前先想清楚每一步的目的：
    #   1) _running=False  + 写 shutdown 管道：把阻塞在 select(timeout=None) 的
    #      调度线程立刻唤醒（不写就只能等下一个事件，可能永远等不到）。
    #   2) join 调度线程：确保没有线程再往 worker 通道里塞新 op。
    #   3) 关管道 fd。
    #   4) 并行停 worker（见 _shutdown_worker_handles）。
    #   5) 抽干 finished_ops_queue：子进程写入的残留消息不清掉，mp.Queue 的
    #       feeder 线程可能让主进程 join 时卡住。
    #   6) empty_cache + 有界 cuda 同步（见 _te_bounded_cuda_sync）。
    def shutdown(self) -> None:
        """Shutdown the transfer engine"""
        try:
            if not self._running:
                return
            self._running = False

            # Send shutdown signal via pipe to wake up selector immediately
            try:
                os.write(self.shutdown_write_fd, b'1')
            except (OSError, BrokenPipeError) as e:
                # Pipe already closed, that's ok
                flexkv_logger.debug(f"Shutdown pipe already closed during write: {e}")

            self._scheduler_thread.join(timeout=5)

            # Close shutdown pipe
            try:
                os.close(self.shutdown_read_fd)
                os.close(self.shutdown_write_fd)
            except OSError as e:
                # Only ignore EBADF (bad file descriptor, already closed)
                if e.errno != 9:  # errno.EBADF = 9
                    flexkv_logger.warning(f"Unexpected error closing shutdown pipes: {e}")
                else:
                    flexkv_logger.debug(f"Shutdown pipes already closed: {e}")

            # Shutdown all workers in parallel so large cudaHostUnregister
            # work overlaps across processes instead of stacking timeouts.
            self._shutdown_worker_handles(self._collect_worker_handles())
        except Exception as e:
            flexkv_logger.error(f"Error during shutdown: {e}")
        finally:
            with contextlib.suppress(Exception):
                while not self.finished_ops_queue.empty():
                    self.finished_ops_queue.get_nowait()

            torch.cuda.empty_cache()
            # Bounded sync: a wedged GPU here would keep the TM process alive
            # past shutdown, forcing the parent-side SIGTERM/SIGKILL. Workers
            # have already unpinned; TM does not itself hold CPU pin refs.
            _te_bounded_cuda_sync(timeout_s=15.0) # hardcode for now
