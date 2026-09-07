# ==============================================================================
# 本文件职责：FlexKV 数据面的"搬运工"——真正把 KV Cache 字节从一处搬到另一处的执行体。
#
# 在系统链路中的位置（本文件属于【数据面 / 执行层】）：
#   KVManager(kvmanager.py)
#     -> KVTaskEngine(kvtask.py)                  任务编排 / 状态机
#       -> GlobalCacheEngine(cache/cache_engine.py) 控制面：产出 TransferOpGraph(DAG)
#         -> TransferEngine(transfer/transfer_engine.py) 数据面调度：DAG -> 单个 TransferOp
#           -> 【本文件 Worker】                    真正搬字节（本文件）
#             -> c_ext(csrc/bindings.cpp)          CUDA / io_uring / GDS / RDMA
#
# 组织方式（理解这点，这个 4000+ 行的文件就不难读）：
#   ** 一个类 = 一条物理通路 **。
#   每个 Worker 子类只负责"从 X 搬到 Y"这一件事，内部只做三件事：
#     1) __init__ 里解析出 X 侧和 Y 侧的内存布局（stride / chunk_size / 指针数组）；
#     2) _transfer_impl() 把 (src_block_ids, dst_block_ids) 翻译成一次 c_ext 调用；
#     3) launch_transfer() 做流绑定、计时、可选压缩，然后返回结果。
#   公共部分（进程生命周期、任务队列、完成回调、host memory pin/unpin）全部在
#   TransferWorkerBase 里用"模板方法模式"实现，子类只填钩子。
#
# 核心内容速查（按文件中的出现顺序）：
#   ┌── 基础 ─────────────────────────────────────────────────────────────────┐
#   │ TransferWorkerBase        抽象基类：进程模型 / 任务循环 / 完成回调 / pin │
#   │ WorkerHandle              父进程侧握着的句柄：submit / control / shutdown│
#   └────────────────────────────────────────────────────────────────────────┘
#   ┌── 主线（默认构建就生效，二次开发最先要读的四个）───────────────────────┐
#   │ GPUCPUTransferWorker      GPU <-> CPU    : H2D / D2H                    │
#   │ tpGPUCPUTransferWorker    GPU <-> CPU    : H2D / D2H（TP 多卡并行）     │
#   │ CPUSSDDiskTransferWorker  CPU <-> SSD    : H2DISK / DISK2H (io_uring)   │
#   │ (LayerwiseTransferWorker  GPU <-> CPU 分层流水，见同级 layerwise.py)    │
#   └────────────────────────────────────────────────────────────────────────┘
#   ┌── 旁支（需要编译宏 / 配置项才生效，默认不参与运行）────────────────────┐
#   │ CPURemoteTransferWorker   CPU <-> 远端存储 : H2REMOTE / REMOTE2H        │
#   │                           依赖 FLEXKV_ENABLE_CFS=1 编译（PCFS）         │
#   │ GDSTransferWorker         GPU <-> SSD    : D2DISK / DISK2D（GPUDirect） │
#   │ tpGDSTransferWorker       GPU <-> SSD    : D2DISK / DISK2D（TP 多卡）   │
#   │                           依赖 FLEXKV_ENABLE_GDS=1 编译                 │
#   │ NixlTransferWorker        GPU<->SSD(GDS_MT) 或 CPU<->SSD(POSIX/3FS)     │
#   │                           依赖 NIXL 后端可用 + nixl_backend 配置        │
#   │ PEER2CPUTransferWorker    跨机 对端CPU/SSD -> 本地CPU : PEERH2H/PEERSSD2H│
#   │                           依赖 enable_kv_sharing + Mooncake + Redis      │
#   │ MooncakeStoreTransferWorker CPU <-> Mooncake Store : H2REMOTE/REMOTE2H   │
#   │                           依赖 mooncake store 客户端                     │
#   └────────────────────────────────────────────────────────────────────────┘
#
# 进程模型（为什么是独立进程）：
#   Worker 跑在 **独立子进程** 里，原因是搬 KV 会长时间占住 CUDA 上下文 / io_uring /
#   RDMA 网卡，若与推理主进程同进程会阻塞 forward。父进程（TransferEngine）只握住
#   WorkerHandle：通过 Pipe 下发任务、通过共享 MPQueue 回收完成事件。
#   启动入口是类方法 TransferWorkerBase.create_worker -> _worker_process。
#
# 阅读提示 / 常见坑：
#   1. 任何 CUDA 调用之前必须先 ensure_cuda_device()/import_tensor_handles()，否则
#      每个 worker 都会在 GPU0 上建默认上下文，DP 场景下直接把 GPU0 撑爆。
#   2. 通过 CUDA IPC 导入的 tensor **必须** 用 keepalive 列表持有（见各
#      _multi_group_*_keepalive），C++ 侧只存了裸 data_ptr()，tensor 被 GC 后指针即悬空。
#   3. 所有 c_ext 调用都是"同步"语义（sync=True / 内部等待 io_uring 完成），
#      launch_transfer 返回即代表数据已落盘/落显存。
#   4. 完成回调不走 Pipe 原路返回，而是 put 到共享的 finished_ops_queue，由
#      TransferEngine._scheduler_loop 统一消费并回传 KVTaskEngine。
# ==============================================================================
import contextlib
import logging
import math
import os
import copy
import signal

import torch.multiprocessing as mp
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch.multiprocessing import Queue as MPQueue, Pipe as MPPipe
from multiprocessing.connection import Connection
from threading import Thread
from typing import List, Any, Dict, Union, Optional, Tuple

import numpy as np
import nvtx
import torch
import zmq
import json

from flexkv import c_ext

from flexkv.c_ext import transfer_kv_blocks, transfer_kv_blocks_ssd, TPTransferThreadGroup

# GDS imports are optional (only available when compiled with FLEXKV_ENABLE_GDS=1)
try:
    from flexkv.c_ext import transfer_kv_blocks_gds, TPGDSTransferThreadGroup
except ImportError:
    transfer_kv_blocks_gds = None
    TPGDSTransferThreadGroup = None

from flexkv.common.debug import flexkv_logger
from flexkv.common.memory_handle import TensorSharedHandle, release_vmm_tensor
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.transfer import TransferOp, TransferType, PartitionBlockType
from flexkv.common.transfer import get_nvtx_range_color, LayerwiseTransferOp
from flexkv.common.config import (
    CacheConfig, GLOBAL_CONFIG_FROM_ENV, MooncakeTransferEngineConfig, LayerGroupSpec,
)
from flexkv.storage.allocator import HugePageTensorHandle, materialize_worker_tensor
from flexkv.transfer.host_buffer import (
    allocate_host_buffer,
    cudaHostRegister,
    safe_cuda_host_unregister,
)


# 把当前进程绑到指定的 CUDA 设备。
# 必须早于一切 CUDA API（cudaHostRegister / CUDA IPC 导入 / Stream 创建）调用，
# 否则所有 worker 都会在 GPU0 上创建默认上下文。
def ensure_cuda_device(device: Union[int, torch.device, None]) -> None:
    """Bind this process's CUDA context before IPC import / host register / Stream.

    Workers must call this *before* any CUDA API. Otherwise the default device
    (usually GPU 0) gets a context from every worker, which under DP exhausts
    GPU0 and makes ``torch.cuda.Stream()`` OOM while ``ready_event.wait()`` hangs.
    """
    if device is None:
        return
    if isinstance(device, torch.device):
        if device.type != "cuda":
            return
        idx = 0 if device.index is None else int(device.index)
    else:
        idx = int(device)
        if idx < 0:
            return
    torch.cuda.set_device(idx)


# 通过 CUDA IPC 句柄把父进程的 GPU KV tensor 映射进本 worker 进程。
# 导入前先切到句柄所属设备，否则 IPC 映射会被建到 GPU0 上。
def import_tensor_handles(
    handles: List["TensorSharedHandle"],
) -> List[torch.Tensor]:
    """Import CUDA IPC tensors after switching to their owning device."""
    if handles:
        ensure_cuda_device(handles[0].device)
    return [h.get_tensor() for h in handles]


# 多 group（异构 KV，如主 KV + DSA indexer）场景下的一致性校验：
# 声明式 LayerGroupSpec 推导出的 chunk 字节数，必须与实际 GPU tensor 布局一致。
# 典型反例：page-packed 的 indexer（tpb=1，一行 8448B）被描述成 tpb=64，
# 若不拦截就会按错误 stride 提交给 c_ext，静默读错数据。
def _validate_multi_group_chunk_layout(
    group_chunk_size: int,
    layout_chunk_size: int,
    group_index: int,
    group_tpb: int,
    layout_tpb: int,
    head_size: int,
    compress_ratio: int,
) -> None:
    """Reject a transfer descriptor that disagrees with GPU storage."""
    if group_chunk_size != layout_chunk_size:
        raise ValueError(
            "Multi-group chunk/layout mismatch for group "
            f"{group_index}: group_chunk={group_chunk_size} B, "
            f"layout_chunk={layout_chunk_size} B, "
            f"group_tpb={group_tpb}, layout_tpb={layout_tpb}, "
            f"head_size={head_size}, compress_ratio={compress_ratio}"
        )



# 把一个已映射的 KV 大池切成多个 Mooncake 内存区（MR）。
# 背景：RDMA 传输对单块 MR 有大小上限（老版本 Mooncake 是 2 GiB），
# 且不允许一次传输跨两个 MR —— 所以必须在不切开 KV block 的前提下切分。
def _split_mooncake_registration_regions(
    base_ptr: int,
    logical_size: int,
    mapped_size: int,
    block_size: int,
    max_mr_size: int,
    size_alignment: int,
    pointer_alignment: int,
) -> List[Tuple[int, int]]:
    """Split a mapped KV pool without splitting KV blocks.

    Prefer regions aligned to the KV block, mapping alignment and HugePage
    size.  If that common alignment period is larger than the configured MR
    limit, keep block boundaries and relax only the derived sub-MR pointer and
    size alignment.  This is required by transports with a 2 GiB MR limit:
    older Mooncake releases cannot transfer one KV block across two MRs.
    """
    values = {
        "base_ptr": base_ptr,
        "logical_size": logical_size,
        "mapped_size": mapped_size,
        "block_size": block_size,
        "max_mr_size": max_mr_size,
        "size_alignment": size_alignment,
        "pointer_alignment": pointer_alignment,
    }
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if mapped_size < logical_size:
        raise ValueError(
            "HugePage mapped length is smaller than the logical CPU pool: "
            f"mapped={mapped_size}, logical={logical_size}"
        )
    if logical_size % block_size != 0:
        raise ValueError(
            "Logical CPU pool must contain whole KV blocks: "
            f"logical_size={logical_size}, block_size={block_size}"
        )
    if base_ptr % pointer_alignment != 0:
        raise ValueError(
            "Mooncake MR base pointer is not HugePage aligned: "
            f"ptr=0x{base_ptr:x}, alignment={pointer_alignment}"
        )
    if mapped_size % size_alignment != 0:
        raise ValueError(
            "Mooncake mapped size is not externally aligned: "
            f"mapped_size={mapped_size}, alignment={size_alignment}"
        )
    if mapped_size <= max_mr_size:
        return [(base_ptr, mapped_size)]

    regions: List[Tuple[int, int]] = []
    # 切分粒度取 block 大小、外部 size 对齐、hugepage 指针对齐三者的最小公倍数，
    # 这样每个子 MR 的起点仍然 hugepage 对齐、长度仍然是整数个 KV block。
    region_unit = math.lcm(block_size, size_alignment, pointer_alignment)
    aligned_region_size = (max_mr_size // region_unit) * region_unit
    # 若 MR 上限比一个对齐周期还小（极端配置），退化为"只保证 block 边界"的切法。
    use_block_boundary_fallback = aligned_region_size <= 0

    if use_block_boundary_fallback:
        mapped_padding = mapped_size - logical_size
        regular_region_size = (max_mr_size // block_size) * block_size
        final_logical_capacity = (
            (max_mr_size - mapped_padding) // block_size
        ) * block_size
        if regular_region_size <= 0 or final_logical_capacity <= 0:
            raise ValueError(
                "Mooncake max MR size cannot hold one KV block plus mapping tail: "
                f"max_mr_size={max_mr_size}, block_size={block_size}, "
                f"mapped_padding={mapped_padding}"
            )
        offset = 0
        while logical_size - offset > final_logical_capacity:
            required = logical_size - offset - final_logical_capacity
            size = min(
                regular_region_size,
                ((required + block_size - 1) // block_size) * block_size,
            )
            regions.append((base_ptr + offset, size))
            offset += size
        regions.append((base_ptr + offset, mapped_size - offset))
    else:
        offset = 0
        while offset < mapped_size:
            remaining = mapped_size - offset
            size = (
                remaining
                if remaining <= max_mr_size
                else aligned_region_size
            )
            regions.append((base_ptr + offset, size))
            offset += size

    for index, (ptr, size) in enumerate(regions):
        is_last = index == len(regions) - 1
        if not is_last and size % block_size != 0:
            raise ValueError(
                "Non-final Mooncake MR is not KV-block aligned: "
                f"index={index}, size={size}, block_size={block_size}"
            )
        if not use_block_boundary_fallback and ptr % pointer_alignment != 0:
            raise ValueError(
                "Mooncake MR pointer is not HugePage aligned: "
                f"ptr=0x{ptr:x}, alignment={pointer_alignment}"
            )
        if not use_block_boundary_fallback and size % size_alignment != 0:
            raise ValueError(
                "Mooncake MR size is not externally aligned: "
                f"size={size}, alignment={size_alignment}"
            )
        if size > max_mr_size:
            raise ValueError(
                "Mooncake MR exceeds configured maximum: "
                f"size={size}, max_mr_size={max_mr_size}"
            )

    if regions[-1][0] + regions[-1][1] != base_ptr + mapped_size:
        raise ValueError("Mooncake MR split does not cover mapped extent")
    if base_ptr + logical_size > regions[-1][0] + regions[-1][1]:
        raise ValueError("Mooncake MR split does not cover logical KV pool")
    return regions


def _register_mooncake_regions(
    client: Any, regions: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Register regions transactionally and roll back a partial failure."""
    registered: List[Tuple[int, int]] = []
    try:
        for ptr, size in regions:
            client.register_buffer(ptr, size)
            registered.append((ptr, size))
    except Exception:
        for ptr, _ in reversed(registered):
            try:
                client.unregister_buffer(ptr)
            except Exception as rollback_error:
                flexkv_logger.error(
                    "Mooncake MR rollback failed for "
                    f"ptr=0x{ptr:x}: {rollback_error}"
                )
        raise
    return registered


def _unregister_mooncake_regions(
    client: Any, regions: List[Tuple[int, int]]
) -> None:
    """Best-effort reverse-order cleanup for registered Mooncake MRs."""
    for ptr, size in reversed(regions):
        try:
            client.unregister_buffer(ptr)
        except Exception as error:
            flexkv_logger.error(
                "Mooncake MR unregister failed for "
                f"ptr=0x{ptr:x} size={size}: {error}"
            )

from flexkv.transfer.compression.common.strategy import (
    CompressionStrategy,
    NullCompressionStrategy,
)
from flexkv.transfer.worker_op import (
    WorkerLayerwiseTransferOp,
    WorkerTransferOp,
    WorkerTransferResult,
)
from flexkv.transfer import trace
trace.configure(GLOBAL_CONFIG_FROM_ENV.enable_transfer_trace)

from flexkv.mooncakeEngineWrapper import MoonCakeTransferEngineWrapper
from flexkv.external.mooncake_store_keys import PoolKind, build_key
from flexkv.external.mooncake_fault_inject import inject_mooncake_fault, is_mooncake_fault_inject_enabled
from flexkv.transfer.zmqHelper import NotifyMsg, NotifyStatus, SSDZMQServer, SSDZMQClient
from flexkv.cache.redis_meta import RedisMeta
from flexkv.transfer.utils import (
    group_blocks_by_node_and_segment,
    group_blocks_by_node,
    split_contiguous_blocks,
    RemoteSSD2HMetaInfo,
    NodeMetaInfo,
    RDMATaskInfo,
)
from flexkv.transfer.nixlutil import (
    NIXL_CPU_FILE_BACKENDS,
    NIXL_GPU_FILE_BACKENDS,
    NixlAgentSession,
    normalize_nixl_file_plugin_name,
    file_path_for_ssd_block,
    gpu_chunk_u8_view,
    kv_chunk_byte_offset_in_block,
    ssd_chunk_byte_offset_in_file,
)
try:
    from flexkv.c_ext import (
        transfer_kv_blocks_remote,
        shared_transfer_kv_blocks_remote_read,
    )
except ImportError:
    transfer_kv_blocks_remote = None
    shared_transfer_kv_blocks_remote_read = None


class TransferWorkerBase(ABC):
    """所有 Worker 的抽象基类：进程模型、任务队列、完成回调、host memory 生命周期。

    在链路中的职责：
        TransferEngine 把 TransferOpGraph 拆成单个 TransferOp 后，通过 Pipe 发给某个
        具体 Worker 子进程；本类提供"接收 -> 批量执行 -> 上报完成"这一公共骨架，
        字节搬运本身交给子类。

    关键设计 —— 模板方法模式：
        子类必须实现的钩子：
            - ``_transfer_impl()``  : 把 block id 列表翻译成一次底层（c_ext / RDMA）调用
            - ``launch_transfer()`` : 单个 op 的入口（绑流、计时、压缩、返回成功与否）
        子类可选复写的钩子：
            - ``__init__``          : 解析两侧布局、建 io_uring / GDS / RDMA 上下文
            - ``shutdown()``        : 释放外部资源（ZMQ / Mooncake / MR），必须回去调 super()
            - ``_control_xxx()``    : 扩展控制面指令（见 ``_handle_control``）
        基类已经固化、子类不要动的：
            ``create_worker`` / ``_worker_process`` / ``run`` / ``shutdown`` 的 pin-unpin 部分

    两条通信通道：
        - 下行：``transfer_conn``（Pipe 的收端），父进程 ``WorkerHandle`` 发 WorkerTransferOp；
                ``None`` 是优雅退出哨兵；dict 且 ``type=="control"`` 是控制面指令。
        - 上行：``finished_ops_queue``（多进程共享 Queue），完成/失败都往里 put。
    """
    _worker_id_counter = 0
    _worker_id_lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any):
        """在 __init__ 之前就把 shutdown() 需要的状态准备好（见下方注释）。"""
        # Allocate first so ``_worker_process`` can always hold a reference and
        # call shutdown() even when ``__init__`` fails mid-way after some pins.
        # 中文要点：__init__ 可能 pin 了一半就抛异常，此时 _worker_process 的 finally
        # 仍能拿到引用并调用 shutdown() 安全回滚，所以这些状态必须在 __new__ 里就绪。
        obj = super().__new__(cls)
        obj._host_registered = []
        obj._shutdown_done = False
        obj._op_buffer_pinned = False
        return obj

    def __init__(self,
                 worker_id: int,
                 transfer_conn: Connection,  # receive end of pipe
                 finished_ops_queue: MPQueue,
                 op_buffer_tensor: torch.Tensor):
        """子进程侧的初始化。子类 __init__ 必须先 super() 再碰任何 CUDA/IO 资源。

        Args:
            worker_id: 全局自增的 worker 编号（仅用于日志与 trace）
            transfer_conn: Pipe 收端，接收 WorkerTransferOp / 控制指令 / None 哨兵
            finished_ops_queue: 跨进程共享队列，用于把完成事件回传给 TransferEngine
            op_buffer_tensor: 共享的 block id 缓冲（pinned），避免每个 op 都拷一次 id 数组
        """
        self.worker_id = worker_id
        self.transfer_conn = transfer_conn  # receive end of pipe
        self.finished_ops_queue: MPQueue = finished_ops_queue

        self.op_buffer_tensor = op_buffer_tensor
        self._op_buffer_pinned = False
        # (tensor, label) pairs registered via _register_host_tensor / _pin_op_buffer.
        self._host_registered: List[Tuple[torch.Tensor, str]] = []
        self._shutdown_done = False

    def _register_host_tensor(self, tensor: torch.Tensor, label: str = "") -> None:
        """cudaHostRegister and track for paired unregister in shutdown().

        中文要点：把 tensor 锁页（pin）并登记到 _host_registered，
        shutdown() 时会逆序一一 unregister。所有需要被 CUDA DMA 直接访问的
        CPU 内存（KV 池、op_buffer、NIXL CPU 池）都必须走这里，不能裸调
        cudaHostRegister，否则进程退出时会漏解绑、把物理内存永久钉死。
        """
        size_gb = tensor.numel() * tensor.element_size() / (1024 ** 3)
        flexkv_logger.info(
            f"[worker {self.worker_id}] cudaHostRegister {label or 'host'}: "
            f"ptr=0x{tensor.data_ptr():x} size={size_gb:.3f} GiB"
        )
        cudaHostRegister(tensor)
        self._host_registered.append((tensor, label or "host"))

    def _pin_op_buffer(self) -> None:
        """Pin the shared op buffer after the worker has bound its CUDA device.

        Must not run before ``ensure_cuda_device`` / ``import_tensor_handles``,
        or every worker creates a default CUDA context on GPU0.

        中文要点：必须在 ensure_cuda_device() / import_tensor_handles() 之后调用，
        因为 cudaHostRegister 会隐式初始化 CUDA 上下文，提前调用会建在 GPU0 上。
        """
        if not self._op_buffer_pinned:
            self._register_host_tensor(self.op_buffer_tensor, "op_buffer")
            self._op_buffer_pinned = True

    def shutdown(self) -> None:
        """Unregister all host tensors pinned by this worker. Idempotent.

        Safe to call after a partially-failed ``__init__`` (only unregisters
        whatever was tracked in ``_host_registered``).

        中文要点：幂等，且对"__init__ 中途失败"也安全——只解绑已经登记的区域。
        子类的 shutdown() 应先释放自己的外部资源，再调用 super().shutdown()。
        """
        if getattr(self, "_shutdown_done", False):
            return
        self._shutdown_done = True
        registered = getattr(self, "_host_registered", None) or []
        worker_id = getattr(self, "worker_id", "-1")
        msg = (
            f"[worker {worker_id}] shutdown: unregistering "
            f"{len(registered)} host region(s)"
        )
        flexkv_logger.info(msg)
        # Drain in-flight CUDA work before unpinning host memory that
        # DMA / kernels may still be touching.
        #
        # 解绑前先排空在途 CUDA 工作（sync 会被包一个有超时上限的守护线程，见下）。
        # torch.cuda.synchronize() releases the GIL and blocks in the driver;
        # if the GPU is wedged (hung kernel, TDR, faulty NVLink) it can hang
        # forever. We run it in a daemon thread with a bounded join so a
        # wedged GPU cannot prevent cudaHostUnregister from firing — the
        # kernel behind the DMA is already dead, so proceeding with unpin
        # is the correct action; the sentinel thread dies with the process.
        self._drain_cuda_bounded(worker_id, timeout_s=30.0)
        # Unregister in reverse order of registration.
        while registered:
            tensor, label = registered.pop()
            safe_cuda_host_unregister(tensor, label=f"worker={worker_id} {label}")
        self._op_buffer_pinned = False
        self._host_registered = registered

    @staticmethod
    def _drain_cuda_bounded(worker_id: Any, timeout_s: float) -> None:
        """Best-effort torch.cuda.synchronize() with a wall-clock cap.

        Returns whether the sync actually completed. Failure / timeout is
        logged but not raised — unpin must proceed either way.

        中文要点：GPU 挂死（hung kernel / TDR）时 synchronize 会永久阻塞，
        所以放进守护线程并限时等待；超时也照常继续 unpin，因为 DMA 背后的
        kernel 进程本身已经要死了，线程随进程一起退出即可。
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
            target=_run,
            name=f"flexkv-worker-{worker_id}-cuda-drain",
            daemon=True,
        )
        t.start()
        if not done.wait(timeout=timeout_s):
            flexkv_logger.warning(
                f"[worker {worker_id}] cuda synchronize did not finish in "
                f"{timeout_s:.0f}s (GPU likely wedged); proceeding with unpin"
            )
            return
        if err:
            flexkv_logger.warning(
                f"[worker {worker_id}] cuda synchronize before unpin failed: "
                f"{err[0]!r}"
            )

    @classmethod
    def _get_worker_id(cls) -> int:
        """在父类进程侧自增地分配 worker 编号。需要加锁：create_worker 可能被多线程并发调用。"""
        with cls._worker_id_lock:
            worker_id = cls._worker_id_counter
            cls._worker_id_counter += 1
            return worker_id

    def _get_layer_ptrs(self, layer_blocks: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """把每层 KV tensor 的起始地址收集成一个 int64 的 pinned CPU 张量。

        c_ext 侧要的就是这个"指针数组"（见 bindings.cpp 里的 gpu_tensor_ptrs，
        要求 contiguous）。用 pinned memory 保证跨进程/跨设备拷贝时可直接 DMA。
        """
        if isinstance(layer_blocks, torch.Tensor):
            layer_blocks = [layer_blocks]
        layer_ptrs = torch.zeros(
            len(layer_blocks),
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        for lay_id in range(len(layer_blocks)):
            layer_ptrs[lay_id] = layer_blocks[lay_id][0].data_ptr()
        return layer_ptrs

    @staticmethod
    def _get_gpu_strides_from_tensor(
        tensor: torch.Tensor,
        tokens_per_block: int,
        dtype_size: int,
        kv_dim: int,
    ) -> tuple:
        """Compute (kv_stride, block_stride, layer_stride) in bytes from a GPU
        KV cache tensor's actual memory layout.

        Different attention backends use different dim orders for the 5D tensor:
          flash_attn:        [2, num_blocks, block_size, num_kv_heads, head_size]
          triton/flashinfer: [num_blocks, 2, block_size, num_kv_heads, head_size]

        Returns (gpu_kv_stride_bytes, gpu_block_stride_bytes, gpu_layer_stride_bytes).

        中文要点：不同 attention 后端 5D tensor 的维度顺序不同，靠"哪个维度 size==2
        （K/V）、哪个维度 size==tokens_per_block（block）"反推，比硬编码布局更稳。
        推断不出来时返回 None，调用方回退到 layout 声明的 stride。
        """
        if kv_dim == 1 or tensor.ndim != 5:
            return None  # caller should fall back to layout-based strides

        # Last 2 dims are always (num_kv_heads, head_size).
        # First 3 dims are a permutation of (num_blocks, kv_dim=2, block_size).
        dim_sizes = [tensor.shape[i] for i in range(3)]
        kv_dim_idx = None
        block_size_idx = None
        block_dim_idx = None

        # Identify kv_dim (size 2) and block_size (size tokens_per_block)
        for i in range(3):
            if dim_sizes[i] == 2 and kv_dim_idx is None:
                kv_dim_idx = i
        for i in range(3):
            if i != kv_dim_idx and dim_sizes[i] == tokens_per_block and block_size_idx is None:
                block_size_idx = i
        # Remaining dim is num_blocks
        for i in range(3):
            if i != kv_dim_idx and i != block_size_idx:
                block_dim_idx = i
                break

        if kv_dim_idx is None or block_dim_idx is None:
            return None  # ambiguous, fall back

        kv_stride = tensor.stride(kv_dim_idx) * dtype_size
        block_stride = tensor.stride(block_dim_idx) * dtype_size
        layer_stride = tensor.numel() * dtype_size
        return (kv_stride, block_stride, layer_stride)

    @classmethod
    def create_worker(cls,
                      mp_ctx: Any,
                      finished_ops_queue: MPQueue,
                      op_buffer_tensor: torch.Tensor,
                      *args: Any, **kwargs: Any) -> 'WorkerHandle':
        """Generic worker creation template method.

        在父进程侧执行：建 Pipe -> 起子进程 -> 返回 WorkerHandle。
        子进程 target 是类方法 ``cls._worker_process``（可 pickle），
        finished_ops_queue 与 op_buffer_tensor 都是可跨进程继承的共享对象。
        ready_event 用于等子进程 __init__ 完成后再继续，避免提任务时还没初始化完。
        """
        parent_conn, child_conn = mp_ctx.Pipe()  # create pipe
        ready_event = mp_ctx.Event()
        worker_id = cls._get_worker_id()

        process = mp_ctx.Process(
            target=cls._worker_process,
            args=(worker_id, child_conn, finished_ops_queue, op_buffer_tensor, ready_event, *args),
            kwargs=kwargs,
            daemon=True
        )
        process.start()

        return WorkerHandle(worker_id, parent_conn, process, ready_event)

    @classmethod
    def _worker_process(cls, worker_id: int, transfer_conn: Connection, finished_ops_queue: MPQueue,
                        op_buffer_tensor: torch.Tensor, ready_event: Any, *args: Any, **kwargs: Any) -> None:
        """Worker 子进程的入口（在子进程里执行）。

        启动流程：
            1. 装信号处理器：忽略 SIGINT（避免与推理主进程抢 Ctrl+C 导致 unpin 被打断、
               pinned 内存泄漏），SIGTERM 转成 SystemExit 走 finally 做优雅清理；
            2. 用 ``cls.__new__`` + ``__init__`` 而不是 ``cls(...)`` 构造：
               这样即使 __init__ 抛异常也能持有引用，finally 里照样能 shutdown()；
            3. ready_event.set() 通知父进程"已就绪"；
            4. worker.run() 进入任务循环；
            5. finally 里统一 shutdown() —— **唯一的** 一次清理，run() 内部不再自行清理。
        """
        # Note: MPI initialization prevention is handled by create_safe_process
        # Environment variables are set before this function is called.
        #
        # Use ``__new__`` + ``__init__`` (not ``cls(...)``) so we keep a live
        # reference if ``__init__`` raises after partial cudaHostRegister; the
        # ``finally`` block can still unpin. ``run()`` only exits the loop —
        # this finally owns shutdown().
        worker: Optional["TransferWorkerBase"] = None

        def _on_sigterm(signum: int, frame: Any) -> None:
            # Raise SystemExit so the ``finally`` below still runs shutdown().
            flexkv_logger.warning(
                f"[worker {worker_id}] received signal {signum}; exiting for graceful cleanup"
            )
            raise SystemExit(0)

        try:
            # Ignore Ctrl+C (SIGINT): the foreground process group receives it
            # together with sglang/tee. Workers must only unpin when the parent
            # sends a shutdown sentinel / SIGTERM, otherwise they race and get
            # SIGKILL mid-unregister, leaking pinned CPU buffers.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, _on_sigterm)
        except Exception as e:
            flexkv_logger.warning(
                f"[worker {worker_id}] failed to install shutdown signal handlers: {e}"
            )


        try:
            worker = cls.__new__(cls)
            worker.__init__(
                worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor, *args, **kwargs
            )
            ready_event.set()
            worker.run()
        except Exception as e:
            # Init / run failure: log then re-raise so process exitcode != 0.
            # SIGTERM → SystemExit is BaseException and bypasses this handler,
            # still hitting ``finally`` for unpin.
            flexkv_logger.error(
                f"[worker {worker_id}] exited with error during init/run: {e}"
            )
            raise
        finally:
            if worker is not None:
                try:
                    worker.shutdown()
                except Exception as e:
                    flexkv_logger.error(f"[worker {worker_id}] final shutdown error: {e}")

    @abstractmethod
    def _transfer_impl(
        self,
        src_block_ids: torch.Tensor,
        dst_block_ids: torch.Tensor,
        transfer_type: TransferType,
        **kwargs: Any
    ) -> None:
        """子类必须实现：把 (src_block_ids, dst_block_ids) 翻译成一次底层字节搬运。

        Args:
            src_block_ids / dst_block_ids: int64 的 block id 列表，一一对应，
                长度即本次要搬的 block 数（可能来自共享 op_buffer 的切片）
            transfer_type: 决定 src/dst 谁是 GPU、谁是 CPU/SSD/远端
            **kwargs: 个别通路有额外参数（如 CPURemote 的 src_block_node_ids、
                NIXL 的 layer_id / layer_granularity）
        Note:
            约定是"同步"语义：返回时数据必须已经落到位（c_ext 内部 sync=True）。
        """
        pass

    def get_transfer_block_ids(self,
                               transfer_op: WorkerTransferOp,
                               pinned: bool = True) ->tuple[torch.Tensor, torch.Tensor]:
        """
        Get transfer block ids from op buffer tensor or directly from op
        Args:
            transfer_op: WorkerTransferOp
            pinned: whether to pin the block ids tensor
        Returns:
            tuple[torch.Tensor, torch.Tensor]: src_block_ids and dst_block_ids

        中文要点：slot_id >= 0 表示 id 数组已经在共享 op_buffer 里（父进程写好的），
        这里只做零拷贝切片；slot_id == -1 表示 id 随 op 传过来（numpy），
        需要现场转 int64 并 pin——因为 c_ext 会拿它做 DMA/设备侧索引。
        """
        src_slot_id = transfer_op.src_slot_id
        dst_slot_id = transfer_op.dst_slot_id
        valid_block_num = transfer_op.valid_block_num

        if src_slot_id == -1:
            src_block_ids = torch.from_numpy(transfer_op.src_block_ids).to(dtype=torch.int64)
            if pinned:
                src_block_ids = src_block_ids.pin_memory()
        else:
            src_block_ids = self.op_buffer_tensor[src_slot_id, :valid_block_num]

        if dst_slot_id == -1:
            dst_block_ids = torch.from_numpy(transfer_op.dst_block_ids).to(dtype=torch.int64)
            if pinned:
                dst_block_ids = dst_block_ids.pin_memory()
        else:
            dst_block_ids = self.op_buffer_tensor[dst_slot_id, :valid_block_num]

        return src_block_ids, dst_block_ids

    def _log_transfer_performance(self,
                                  transfer_op: WorkerTransferOp,
                                  transfer_size: int,
                                  start_time: float,
                                  end_time: float,
                                  uncompressed_size: Optional[int] = None) -> None:
        """Emit one terminal record per transfer op."""
        # 这是一条结构化日志（[FlexKV-IO]），供外部采集 IO 带宽/压缩比；
        # 因此先判 is_enabled_for，避免关掉日志时还白算一遍。
        if not flexkv_logger.is_enabled_for(logging.INFO):
            return
        duration_s = max(end_time - start_time, 1e-9)
        is_layerwise = transfer_op.transfer_type == TransferType.LAYERWISE
        direction = "H2D" if is_layerwise else transfer_op.transfer_type.value
        blocks = (
            len(transfer_op.src_block_ids_h2d)
            if is_layerwise
            else transfer_op.valid_block_num
        )
        transfer_mode = "layerwise" if is_layerwise else "no-layerwise"
        bandwidth = transfer_size / duration_s / 1e9

        if (
            uncompressed_size is not None
            and transfer_size > 0
            and uncompressed_size != transfer_size
        ):
            flexkv_logger.info(
                "[FlexKV-IO] operation=transfer act=complete status=success "
                "direction=%s blocks=%d op_id=%d graph_id=%d mode=%s "
                "compressed_size=%.6gGB original_size=%.6gGB "
                "compression_ratio=%.2fx transfer_time=%.4fs "
                "bandwidth=%.2fGB/s",
                direction,
                blocks,
                transfer_op.transfer_op_id,
                transfer_op.transfer_graph_id,
                transfer_mode,
                transfer_size / (1024**3),
                uncompressed_size / (1024**3),
                uncompressed_size / transfer_size,
                duration_s,
                bandwidth,
            )
        else:
            flexkv_logger.info(
                "[FlexKV-IO] operation=transfer act=complete status=success "
                "direction=%s blocks=%d op_id=%d graph_id=%d mode=%s "
                "data_size=%.6gGB transfer_time=%.4fs bandwidth=%.2fGB/s",
                direction,
                blocks,
                transfer_op.transfer_op_id,
                transfer_op.transfer_graph_id,
                transfer_mode,
                transfer_size / (1024**3),
                duration_s,
                bandwidth,
            )

    @abstractmethod
    def launch_transfer(
        self, transfer_op: WorkerTransferOp
    ) -> Union[bool, WorkerTransferResult]:
        """子类必须实现：执行一个 op 并返回结果。

        典型实现：取 block ids -> 绑 CUDA stream -> 计时 -> 调 _transfer_impl()
        -> 打性能日志 -> 返回 True/False。

        Returns:
            bool: True 成功 / False 失败（大多数通路）
            WorkerTransferResult: 支持"部分成功"的后端（如 Mooncake）用逐 block 结果
        Note:
            由基类的 run() 调用；返回值决定往 finished_ops_queue 里 put 什么。
        """
        pass

    def _handle_control(self, command: str, payload: Any) -> Any:
        """控制面指令分发：把 command 映射到 ``_control_<command>`` 方法。

        例如 ``suspend_gpu`` -> ``_control_suspend_gpu``，用于 GPU 热重映射
        （把 VMM 映射释放掉，让别的进程/显存策略接管，再 resume 回来）。
        子类只要定义同名钩子即可扩展，不需要改这里。
        """
        handler = getattr(self, f"_control_{command}", None)
        if handler is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support control {command}"
            )
        return handler(payload)

    def _reply_control(self, op: Dict[str, Any]) -> None:
        """执行控制指令并把结果（或异常）沿 Pipe 回给 WorkerHandle.control()。

        与数据面不同：控制面是**同步请求-应答**，结果从 transfer_conn 原路发回，
        而不是走 finished_ops_queue。异常不能抛出（否则会打断 run 循环），
        只能塞进 reply["error"] 交给调用方决定。
        """
        request_id = op["request_id"]
        try:
            reply = {
                "type": "control_ack",
                "request_id": request_id,
                "result": self._handle_control(
                    op["command"], op.get("payload")
                ),
            }
        except Exception as exc:
            flexkv_logger.exception(
                f"Worker control {op.get('command')} failed"
            )
            reply = {
                "type": "control_ack",
                "request_id": request_id,
                "error": str(exc),
            }
        self.transfer_conn.send(reply)

    def run(self) -> None:
        """Main loop for the worker process.

        Exit paths (``None`` sentinel, pipe EOF, or return) do not unregister
        themselves — ``_worker_process`` owns a single ``shutdown()`` in its
        ``finally`` block so cleanup is not duplicated.
        """
        while True:
            try:
                if not self.transfer_conn.poll(timeout=0.0001):
                    continue

                op = self.transfer_conn.recv()
                if op is None:
                    return
                if not isinstance(op, dict):
                    op._received_ns = time.perf_counter_ns()

                # Drain any already-queued ops into one batch, then process.
                # The for-loop MUST sit outside the drain while: a single-op
                # submit leaves poll() False immediately, and a while-else
                # continue would otherwise drop the first op forever (which
                # stalls D2H → H2REMOTE and leaves mooncake PutStart=0).
                batch_ops = [op]
                stop_after_batch = False
                while self.transfer_conn.poll(timeout=0):
                    try:
                        op = self.transfer_conn.recv()
                    except EOFError:
                        # A closed pipe is readable. Preserve the batch already
                        # received, then exit after reporting its completions.
                        stop_after_batch = True
                        break
                    if op is None:
                        stop_after_batch = True
                        break
                    if not isinstance(op, dict):
                        op._received_ns = time.perf_counter_ns()
                    batch_ops.append(op)
                for op in batch_ops:
                    if isinstance(op, dict) and op.get("type") == "control":
                        self._reply_control(op)
                        continue
                    transfer_status = False
                    transfer_start_ns = time.perf_counter_ns()
                    nvtx_pushed = False
                    try:
                        nvtx.push_range(f"launch {op.transfer_type.name} op_id: {op.transfer_op_id}, "
                                            f"graph_id: {op.transfer_graph_id}",
                                            color=get_nvtx_range_color(op.transfer_graph_id))
                        nvtx_pushed = True
                        transfer_status = self.launch_transfer(op)
                    except Exception as e:
                        is_layerwise = op.transfer_type == TransferType.LAYERWISE
                        direction = "H2D" if is_layerwise else op.transfer_type.value
                        blocks = (
                            len(op.src_block_ids_h2d)
                            if is_layerwise
                            else op.valid_block_num
                        )
                        flexkv_logger.error(
                            "[FlexKV-IO] operation=transfer act=complete "
                            "status=failed direction=%s blocks=%d op_id=%d "
                            "graph_id=%d mode=%s transfer_time=%.4fs "
                            "error=%r",
                            direction,
                            blocks,
                            op.transfer_op_id,
                            op.transfer_graph_id,
                            "layerwise" if is_layerwise else "no-layerwise",
                            (time.perf_counter_ns() - transfer_start_ns) / 1e9,
                            str(e),
                            exc_info=True,
                        )
                    finally:
                        if nvtx_pushed:
                            nvtx.pop_range()
                    launched_ns = time.perf_counter_ns()
                    is_h2d = (op.transfer_type == TransferType.H2D
                              or op.transfer_type == TransferType.LAYERWISE)
                    metrics = trace.build_worker_metrics(
                        op,
                        getattr(op, "prof_submitted_ns", 0),
                        getattr(op, "_received_ns", launched_ns),
                        transfer_start_ns,
                        launched_ns,
                        self.worker_id,
                        getattr(self, "_bytes_per_block", 0),
                        getattr(self, "kv_dim", 2),
                        is_h2d,
                    )
                    if isinstance(transfer_status, WorkerTransferResult):
                        # Partial-capable backends report completion even when
                        # zero blocks succeeded, so the graph can clean up and
                        # the caller can fall back instead of hanging forever.
                        # Carry metrics so FLEXKV_TRANSFER_TRACE still works.
                        self.finished_ops_queue.put(
                            (transfer_status, True, metrics))
                    elif transfer_status:
                        self.finished_ops_queue.put(
                            (op.transfer_op_id, True, metrics))
                    else:
                        # Report the failure instead of dropping it: a
                        # dropped op leaves its graph incomplete forever
                        # and leaks every resource its plan holds. A bare
                        # int still means success, so the queue format
                        # stays compatible.
                        self.finished_ops_queue.put(
                            (op.transfer_op_id, False, None))
                if stop_after_batch:
                    # _worker_process owns the single shutdown() call in its
                    # finally block. Calling subclass shutdown here can repeat
                    # external unregister work before super()'s idempotence
                    # guard is reached.
                    return
            except EOFError:
                flexkv_logger.warning(
                    f"[worker {self.worker_id}] transfer pipe EOF; exiting run loop"
                )
                return
            except Exception as e:
                flexkv_logger.error(f"Error in worker run loop: {e}")

class WorkerHandle:
    """handle for worker process

    在链路中的职责（父进程侧的唯一入口）：
        TransferEngine 不直接碰子进程，只握住这个句柄。
        ┌── 下行（本句柄 -> 子进程）──────────────────────────────┐
        │ submit_transfer() : 通过 Pipe 发 WorkerTransferOp（异步，不等待）│
        │ control()         : 发 dict 控制指令并**阻塞等待**应答（同步）   │
        │ shutdown()        : 发 None 哨兵 -> join -> 超时则 terminate/kill│
        └────────────────────────────────────────────────────────┘
        上行（子进程 -> 引擎）不经过本句柄：worker 把完成事件直接 put 到
        共享的 finished_ops_queue，由 TransferEngine._scheduler_loop 消费。

    关键设计：
        - 一个 WorkerHandle 对应一个 Worker 子进程、一条 Pipe、一条物理通路。
        - ready_event 由子进程 __init__ 完成后 set，父进程可据此等待"就绪"。
    """
    def __init__(self, worker_id: int, transfer_conn: Connection, process: mp.Process, ready_event: Any):
        """仅由 TransferWorkerBase.create_worker 在父进程侧构造。

        Args:
            worker_id: 子进程编号
            transfer_conn: Pipe 的**发端**（子进程握收端）
            process: 已 start 的 daemon 子进程
            ready_event: 子进程初始化完成事件
        """
        self.worker_id = worker_id
        self.transfer_conn = transfer_conn
        self.process = process
        self.ready_event = ready_event

    def submit_transfer(self, op: Union[TransferOp, LayerwiseTransferOp]) -> None:
        """把一个 TransferOp 投递给 worker 子进程（非阻塞）。

        控制面产出的 TransferOp 会被包装成 WorkerTransferOp（可 pickle 的
        精简结构，block id 尽量走共享 op_buffer 而不是随消息拷贝）。
        若开启了 transfer trace，这里顺带记录 submit 时刻并增加 in-flight 计数。
        """
        if isinstance(op, LayerwiseTransferOp):
            worker_op = WorkerLayerwiseTransferOp(op)
        else:
            worker_op = WorkerTransferOp(op)
        if trace._TRACE_ON:
            submitted_ns = time.perf_counter_ns()
            worker_op.prof_submitted_ns = submitted_ns
            trace.set_submit_ns(op.op_id, submitted_ns)
            trace.inc_inflight()
        self.transfer_conn.send(worker_op)

    def control(
        self, command: str, payload: Any = None, timeout: float = 120.0
    ) -> Any:
        """同步下发一条控制指令并等待 worker 应答。

        与 submit_transfer 不同，这里**阻塞**：发完就 poll 等 reply，
        靠 request_id 对账防止串台。主要用于 GPU 热重映射
        （suspend_gpu / resume_gpu）。

        Raises:
            TimeoutError: worker 在 timeout 内没应答（默认 120s）
            RuntimeError: 回来的 request_id 对不上，或 worker 侧抛了异常
        """
        request_id = f"{self.worker_id}:{time.monotonic_ns()}"
        self.transfer_conn.send({
            "type": "control",
            "command": command,
            "payload": payload,
            "request_id": request_id,
        })
        if not self.transfer_conn.poll(timeout):
            raise TimeoutError(
                f"Worker {self.worker_id} timed out handling {command}"
            )
        reply = self.transfer_conn.recv()
        if reply.get("request_id") != request_id:
            raise RuntimeError(f"Unexpected worker control reply: {reply}")
        if "error" in reply:
            raise RuntimeError(
                f"Worker {self.worker_id} {command} failed: {reply['error']}"
            )
        return reply.get("result")

    def shutdown(self) -> None:
        """优雅停止 worker 子进程：None 哨兵 -> join -> terminate -> kill 三级降级。

        为什么必须先发 None 而不是直接 kill：worker 持有 pinned 内存和 VMM 映射，
        被 SIGKILL 掉的话 cudaHostUnregister / release_vmm 都来不及执行，
        物理页会被永久钉住（表现为宿主机内存缓慢泄漏）。所以先给
        worker_shutdown_timeout_s 秒让它自己走完 _worker_process 的 finally，
        超时才 terminate，再超时才 kill。
        """
        try:
            self.transfer_conn.send(None)
        except (BrokenPipeError, OSError, EOFError):
            pass  # Pipe already closed / peer gone

        timeout = float(GLOBAL_CONFIG_FROM_ENV.worker_shutdown_timeout_s)
        self.process.join(timeout=timeout)
        if self.process.is_alive():
            flexkv_logger.warning(
                f"[WorkerHandle] worker {self.worker_id} still alive after "
                f"{timeout:.0f}s graceful shutdown; force terminate"
            )
            self.process.terminate()
            self.process.join(timeout=30)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()

        try:
            self.transfer_conn.close()
        except Exception:
            pass

    def __del__(self) -> None:
        """兜底：句柄被 GC 时若子进程还活着，走一次优雅 shutdown。

        注意这里吞掉所有异常——解释器退出阶段调用 __del__ 时
        很多模块已经变成 None，抛异常只会打出无意义的噪音。
        """
        try:
            if getattr(self, "process", None) is not None and self.process.is_alive():
                self.shutdown()
        except Exception:
            pass

class GPUCPUTransferWorker(TransferWorkerBase):  # this worker only supports non-tp and non-dp case
    """GPU <-> CPU 通路（H2D / D2H），单卡、非 TP 场景的主线 worker。

    在链路中的职责：
        TransferEngine 里所有 GPU<->CPU 的 TransferOp 都落到这里。
        H2D = 命中前缀的 KV 从 CPU 内存拉回显存；D2H = 把新算出来的 KV 卸载到 CPU。

    传输机制（两条，由 use_ce_transfer_h2d / _d2h 决定）：
        1) Copy Engine（CE）路径：走 GPU 上独立的 DMA 拷贝引擎（cudaMemcpyAsync
           一族），不占用 SM，与正在跑的 forward kernel 真正并行。
           c_ext 侧还会按 ce_segment_threshold 把大块切段、按 ce_path_opt
           选路（一维 memcpy / 二维 memcpy2D / 分段 gather），
           ce_force_path=-1 表示交给 C++ 自动选。
        2) Kernel 路径：起一个自定义 kernel 逐 chunk 搬，transfer_num_cta
           控制起多少个 CTA。CE 不可用时（老驱动/老卡）退回这条。
        两条路径都通过同一个 c_ext.transfer_kv_blocks 入口，只是参数不同。

    线程模型：
        单进程单线程（本 worker 的子进程）。既没有线程池也没有后台 IO 线程；
        CUDA 工作全部投递到 self.transfer_stream 这条**非默认流**上，
        避免和推理主进程的默认流互相排队。c_ext 以 sync=True 调用，
        所以 launch_transfer 返回即代表数据已经在目标侧可见。

    调用 c_ext 的方式：
        _transfer_impl 把"两侧 stride + 指针数组 + block id 列表"一次性传给
        transfer_kv_blocks(...)，由 C++ 完成所有 block、所有 layer 的搬运；
        Python 侧不做逐 block 循环（那会是成千上万次 pybind 调用）。

    完成回调：
        launch_transfer 返回 True -> 基类 run() 把 (op_id, True, metrics)
        put 进 finished_ops_queue，TransferEngine._scheduler_loop 消费后
        推进 TransferOpGraph，最终通知 KVTaskEngine。

    注意：多卡 TP 场景请用 tpGPUCPUTransferWorker（本类只支持单卡，
    类名后面的注释已经写明了 non-tp and non-dp）。
    """
    def __init__(self,
                 worker_id: int,
                 transfer_conn: Connection,
                 finished_ops_queue: MPQueue,
                 op_buffer_tensor: torch.Tensor,
                 gpu_blocks: List[TensorSharedHandle],
                 cpu_blocks: Union[torch.Tensor, HugePageTensorHandle],
                 gpu_kv_layout: KVCacheLayout,
                 cpu_kv_layout: KVCacheLayout,
                 dtype: torch.dtype,
                 gpu_device_id: int,
                 use_ce_transfer_h2d: bool = False,
                 use_ce_transfer_d2h: bool = False,
                 transfer_num_cta_h2d: int = 4,
                 transfer_num_cta_d2h: int = 4,
                 compressor: Optional[CompressionStrategy] = None,
                 layer_groups: Optional[List[LayerGroupSpec]] = None,
                 gpu_blocks_per_group: Optional[List[List[TensorSharedHandle]]] = None,
                 gpu_layouts_per_group: Optional[List[KVCacheLayout]] = None) -> None:
        """解析 GPU/CPU 两侧的布局，并把两侧内存都"准备好"。

        顺序不能乱（每一步都依赖上一步）：
            1. super().__init__          建立 Pipe / 完成队列 / 清理登记
            2. ensure_cuda_device        绑定 GPU，**必须早于一切 CUDA 调用**
            3. _pin_op_buffer            pin 共享 block id 缓冲
            4. materialize + register    CPU KV 池锁页（hugepage 句柄 -> tensor）
            5. import_tensor_handles     通过 CUDA IPC 把父进程的 GPU KV 映射进来
            6. 算 stride                 GPU 侧优先从实际 tensor 反推（兼容不同后端布局）
            7. torch.cuda.Stream()       建专用传输流

        Args:
            gpu_blocks: 父进程的 GPU KV tensor 的 IPC 句柄列表（每层一个，或 K/V 各一个）
            cpu_blocks: CPU KV 池（可能是 HugePageTensorHandle，需 materialize）
            use_ce_transfer_h2d/_d2h: 分别控制两个方向是否走 Copy Engine
            transfer_num_cta_h2d/_d2h: kernel 路径下起多少个 CTA
            layer_groups / *_per_group: 异构 KV（如主 KV + DSA indexer）多分组布局
        """
        # initialize worker in a new process
        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)

        # Bind CUDA device BEFORE host-register / IPC import / Stream creation.
        ensure_cuda_device(gpu_device_id)

        self._pin_op_buffer()
        # Register CPU tensors with CUDA
        cpu_blocks = materialize_worker_tensor(cpu_blocks)
        flexkv_logger.info(f"Pinning CPU Memory: {cpu_blocks.numel() * cpu_blocks.element_size() / (1024 ** 3):.2f} GB")
        self._register_host_tensor(cpu_blocks, "cpu_kv_pool")

        self.gpu_device_id = gpu_device_id
        self._gpu_block_count = len(gpu_blocks)
        self.gpu_blocks = import_tensor_handles(gpu_blocks)
        # Get pointers first
        self.gpu_blocks_ptrs = self._get_layer_ptrs(self.gpu_blocks)
        self.gpu_tensor_ptrs = self.gpu_blocks_ptrs

        self.cpu_tensor = cpu_blocks

        self.dtype = dtype
        self.kv_dim = gpu_kv_layout.kv_dim
        self.num_kv_heads = gpu_kv_layout.num_kv_heads
        self.cpu_is_blockfirst = (
            cpu_kv_layout.type == KVCacheLayoutType.BLOCKFIRST
        )

        self.num_layers = gpu_kv_layout.num_layer
        self.layer_groups = layer_groups

        if layer_groups is not None and gpu_blocks_per_group is not None and gpu_layouts_per_group is not None:
            # Multi-group mode: compute per-group strides
            self._init_multi_group(
                gpu_blocks_per_group, gpu_layouts_per_group,
                cpu_kv_layout, layer_groups,
            )
        else:
            # Uniform mode: existing code path
            self.group_transfer_params = None

            # a chunk can be located by layer_id * layer_stride + kv_id * kv_stride + block_id * block_stride
            self.chunk_size_in_bytes = gpu_kv_layout.get_chunk_size() * self.dtype.itemsize
            # Bytes per KV block (all layers); used by transfer tracing for bw.
            self._bytes_per_block = self.chunk_size_in_bytes * self.num_layers * self.kv_dim

            # Compute GPU strides from actual tensor to handle different attention
            # backend layouts (flash_attn: [2,N,B,H,D], triton: [N,2,B,H,D]).
            gpu_strides = self._get_gpu_strides_from_tensor(
                self.gpu_blocks[0], gpu_kv_layout.tokens_per_block,
                self.dtype.itemsize, self.kv_dim,
            ) if len(self.gpu_blocks) > 1 else None
            if gpu_strides is not None:
                self.gpu_kv_stride_in_bytes = gpu_strides[0]
                self.gpu_block_stride_in_bytes = gpu_strides[1]
                self.gpu_layer_stride_in_bytes = gpu_strides[2]
            else:
                self.gpu_kv_stride_in_bytes = gpu_kv_layout.get_kv_stride() * self.dtype.itemsize
                self.gpu_block_stride_in_bytes = gpu_kv_layout.get_block_stride() * self.dtype.itemsize
                self.gpu_layer_stride_in_bytes = gpu_kv_layout.get_layer_stride() * self.dtype.itemsize

            self.cpu_layer_stride_in_bytes = cpu_kv_layout.get_layer_stride() * self.dtype.itemsize
            self.cpu_kv_stride_in_bytes = cpu_kv_layout.get_kv_stride() * self.dtype.itemsize
            self.cpu_block_stride_in_bytes = cpu_kv_layout.get_block_stride() * self.dtype.itemsize

        # gpu_block_type_ is a framework-level tag (0=VLLM, 1=TRTLLM, 2=SGLANG).
        # In multi-group mode all groups share the same per-layer GPU layout, so
        # judge from any one group; the flat self.gpu_blocks would over-count.
        if self.group_transfer_params is None:
            ref_blocks_len = len(self.gpu_blocks)
            ref_num_layers = self.num_layers
        else:
            ref_blocks_len = len(gpu_blocks_per_group[0])
            ref_num_layers = layer_groups[0].num_layers

        if ref_blocks_len == 1:
            self.gpu_block_type_ = 1
        elif ref_blocks_len == ref_num_layers:
            self.gpu_block_type_ = 0
        elif ref_blocks_len == ref_num_layers * 2:
            self.gpu_block_type_ = 2
        else:
            raise ValueError(
                f"Invalid GPU block type: ref_blocks_len={ref_blocks_len}, "
                f"ref_num_layers={ref_num_layers}"
            )

        self.transfer_stream = torch.cuda.Stream()
        self.transfer_num_cta_h2d = transfer_num_cta_h2d
        self.transfer_num_cta_d2h = transfer_num_cta_d2h
        self.use_ce_transfer_h2d = use_ce_transfer_h2d
        self.use_ce_transfer_d2h = use_ce_transfer_d2h

        self.ce_path_opt = GLOBAL_CONFIG_FROM_ENV.ce_path_opt
        self.ce_segment_threshold = GLOBAL_CONFIG_FROM_ENV.ce_segment_threshold
        self.ce_enable_memcpy2d = GLOBAL_CONFIG_FROM_ENV.enable_ce_memcpy2d

        self._compressor = compressor or NullCompressionStrategy()
        self._compressor.attach(self)

    def _init_multi_group(
        self,
        gpu_blocks_per_group: List[List[TensorSharedHandle]],
        gpu_layouts_per_group: List[KVCacheLayout],
        cpu_kv_layout: KVCacheLayout,
        layer_groups: List[LayerGroupSpec],
    ) -> None:
        """Initialize per-group transfer parameters for models with mixed KV shapes.

        中文要点：异构 KV（主 KV bf16 + DSA indexer uint8）在一个 CPU block 内
        按 group 依次排布。这里为每个 group 算出它自己的 GPU/CPU stride 和
        cpu_offset_bytes，之后 _transfer_impl 对每个 group 各发一次 c_ext 调用。
        """
        kv_dim = self.kv_dim
        tpb = cpu_kv_layout.tokens_per_block
        cpu_layout_type = cpu_kv_layout.type
        num_cpu_blocks = cpu_kv_layout.num_block

        # CPU buffer is sized in BYTES per block (see KVCacheLayout._compute_kv_shape).
        # cpu_kv_layout.get_block_stride() returns bytes_per_block directly for
        # multi-group BLOCKFIRST.  For LAYERFIRST, num_cpu_blocks is the contiguous
        # dim count and we keep elements-based per-group accumulation below.
        total_block_bytes = (
            cpu_kv_layout.get_block_stride()
            if cpu_layout_type == KVCacheLayoutType.BLOCKFIRST else None
        )

        self.group_transfer_params: list = []
        # Keep the imported CUDA-IPC tensors alive for the worker's lifetime.
        # _get_layer_ptrs() below records only their raw data_ptr()s; if the
        # tensors themselves were allowed to go out of scope, PyTorch would
        # release the underlying CUDA IPC mapping and the stored pointers would
        # dangle, so the per-group transfer would read/write freed device
        # memory (observed as the indexer group silently restoring zeros).
        self._multi_group_gpu_blocks_keepalive: list = []
        cpu_offset_bytes = 0  # byte offset of this group within a CPU block

        for gi, (g, gpu_layout) in enumerate(zip(layer_groups, gpu_layouts_per_group)):
            # Per-group dtype: indexer uses uint8 even when main KV is bf16/fp16.
            dtype_size_g = g.dtype.itemsize

            # Resolve GPU tensors for this group
            group_gpu_blocks = import_tensor_handles(gpu_blocks_per_group[gi])
            self._multi_group_gpu_blocks_keepalive.append(group_gpu_blocks)
            group_gpu_ptrs = self._get_layer_ptrs(group_gpu_blocks)

            # Compressed groups: GPU tensor's tokens dim equals tpb_g, not tpb.
            tpb_g = tpb // g.compress_ratio
            chunk_elements = tpb_g * g.num_kv_heads * g.head_size

            # GPU strides: compute from actual tensor to handle different
            # attention backend layouts (flash_attn vs triton/flashinfer).
            gpu_chunk_size = chunk_elements * dtype_size_g
            # Fail closed before submitting a native transfer if the
            # declarative LayerGroupSpec disagrees with the actual tensor
            # layout. This catches page-packed GLM DSA indexer buffers
            # (tpb=1, one 8448-byte row) being described as tpb=64.
            layout_chunk_size = gpu_layout.get_chunk_size() * dtype_size_g
            _validate_multi_group_chunk_layout(
                gpu_chunk_size,
                layout_chunk_size,
                gi,
                tpb_g,
                gpu_layout.tokens_per_block,
                g.head_size,
                g.compress_ratio,
            )
            t0 = group_gpu_blocks[0]
            gpu_strides = self._get_gpu_strides_from_tensor(t0, tpb_g, dtype_size_g, self.kv_dim)
            if gpu_strides is not None:
                gpu_kv_stride, gpu_block_stride, gpu_layer_stride = gpu_strides
            else:
                gpu_kv_stride = gpu_layout.get_kv_stride() * dtype_size_g
                gpu_block_stride = gpu_layout.get_block_stride() * dtype_size_g
                gpu_layer_stride = gpu_layout.get_layer_stride() * dtype_size_g

            # CPU strides: depend on layout type.  All values are in bytes; the
            # CPU buffer underlying self.cpu_tensor is uint8 for multi-group.
            if cpu_layout_type == KVCacheLayoutType.BLOCKFIRST:
                # BLOCKFIRST: [num_block, bytes_per_block]; within a block,
                # data is laid out group-by-group (each group's region holds
                # its own layer0_k, layer0_v, layer1_k, ... bytes).
                cpu_layer_stride = kv_dim * chunk_elements * dtype_size_g
                cpu_block_stride = total_block_bytes
                cpu_kv_stride = chunk_elements * dtype_size_g
            else:
                # LAYERFIRST: [all_layers, kv_dim, num_block, tpb, heads, head_dim]
                cpu_layer_stride = kv_dim * num_cpu_blocks * chunk_elements * dtype_size_g
                cpu_block_stride = chunk_elements * dtype_size_g
                cpu_kv_stride = num_cpu_blocks * chunk_elements * dtype_size_g

            self.group_transfer_params.append({
                'gpu_ptrs': group_gpu_ptrs,
                'chunk_size': gpu_chunk_size,
                'gpu_kv_stride': gpu_kv_stride,
                'gpu_block_stride': gpu_block_stride,
                'gpu_layer_stride': gpu_layer_stride,
                'cpu_layer_stride': cpu_layer_stride,
                'cpu_block_stride': cpu_block_stride,
                'cpu_kv_stride': cpu_kv_stride,
                'cpu_offset_bytes': cpu_offset_bytes,
                'num_layers': g.num_layers,
                'kv_dim': gpu_layout.kv_dim,
            })

            # Advance CPU byte offset for next group.
            if cpu_layout_type == KVCacheLayoutType.BLOCKFIRST:
                cpu_offset_bytes += g.num_layers * kv_dim * chunk_elements * dtype_size_g
            else:
                cpu_offset_bytes += (
                    g.num_layers * kv_dim * num_cpu_blocks * chunk_elements * dtype_size_g
                )

        flexkv_logger.info(
            f"Multi-group transfer initialized: {len(layer_groups)} groups, "
            f"total_block_bytes={total_block_bytes}"
        )

    def _control_suspend_gpu(self, payload: Any) -> int:
        """控制面：释放 GPU 侧的 VMM 映射（GPU 热重映射的第一步）。

        把指针数组清零 + 释放 CUDA VMM 映射，让别的使用方可以接管这块物理显存。
        之后必须由 _control_resume_gpu 按相同数量恢复，否则 worker 就废了。
        多分组布局不支持（每个 group 有自己的指针数组，无法一次性摘干净）。
        """
        if self.group_transfer_params is not None:
            raise NotImplementedError(
                "GPU hot remap does not support multi-group KV layouts"
            )
        if not self.gpu_blocks:
            return 0
        with torch.cuda.device(self.gpu_device_id):
            torch.cuda.synchronize()
        old_blocks = self.gpu_blocks
        self.gpu_blocks = []
        self.gpu_blocks_ptrs.zero_()
        self.gpu_tensor_ptrs = self.gpu_blocks_ptrs
        released = sum(release_vmm_tensor(tensor) for tensor in old_blocks)
        if released != len(old_blocks):
            raise RuntimeError(
                f"Expected {len(old_blocks)} VMM mappings, released {released}"
            )
        return released

    def _control_resume_gpu(
        self, gpu_blocks: List[TensorSharedHandle]
    ) -> int:
        """控制面：重新导入 GPU tensor 并恢复指针数组（suspend 的逆操作）。

        数量必须和 suspend 前一致（_gpu_block_count），否则 stride 与
        实际 tensor 数对不上，后续传输会静默错位。
        """
        if self.gpu_blocks:
            raise RuntimeError("GPU blocks are already registered")
        if len(gpu_blocks) != self._gpu_block_count:
            raise ValueError(
                f"Expected {self._gpu_block_count} GPU blocks, "
                f"got {len(gpu_blocks)}"
            )
        self.gpu_blocks = import_tensor_handles(gpu_blocks)
        self.gpu_blocks_ptrs = self._get_layer_ptrs(self.gpu_blocks)
        self.gpu_tensor_ptrs = self.gpu_blocks_ptrs
        return len(self.gpu_blocks)

    def _transfer_impl(
        self,
        src_block_ids: torch.Tensor,
        dst_block_ids: torch.Tensor,
        transfer_type: TransferType,
        **kwargs: Any,
    ) -> None:
        """把 block id 列表翻译成一次 transfer_kv_blocks 调用。

        方向映射：H2D 时 src=CPU/dst=GPU，D2H 时反过来；c_ext 只认
        "gpu_block_id_list / cpu_block_id_list"，不认方向枚举。

        两条分支：
            - 多分组：每个 group 一次调用，CPU 侧按 cpu_offset_bytes 切片
              （多分组下 cpu_tensor 是 uint8 字节池，切片即字节寻址）；
            - 统一布局：一次调用搬完整个模型的所有 layer。
        """
        assert src_block_ids.dtype == torch.int64
        assert dst_block_ids.dtype == torch.int64
        assert len(src_block_ids) == len(dst_block_ids)

        if transfer_type == TransferType.H2D:
            gpu_block_id_list = dst_block_ids
            cpu_block_id_list = src_block_ids
            use_ce_transfer = self.use_ce_transfer_h2d
            transfer_num_cta = self.transfer_num_cta_h2d
        elif transfer_type == TransferType.D2H:
            gpu_block_id_list = src_block_ids
            cpu_block_id_list = dst_block_ids
            use_ce_transfer = self.use_ce_transfer_d2h
            transfer_num_cta = self.transfer_num_cta_d2h
        else:
            raise ValueError(f"Invalid transfer type: {transfer_type} for GPUCPUTransferWorker")

        assert len(gpu_block_id_list) == len(cpu_block_id_list)

        if len(gpu_block_id_list) == 0:
            return

        if self.group_transfer_params is not None:
            # Multi-group transfer: one call per group
            for gp in self.group_transfer_params:
                gpu_ptrs = gp['gpu_ptrs'].contiguous().pin_memory()
                # Offset the CPU tensor to this group's region.  cpu_tensor is
                # uint8 in multi-group mode, so this slice is byte-addressed.
                cpu_tensor_for_group = self.cpu_tensor.view(-1)[
                    gp['cpu_offset_bytes']:
                ]

                transfer_kv_blocks(
                    gpu_block_id_list,
                    gpu_ptrs,
                    gp['gpu_kv_stride'],
                    gp['gpu_block_stride'],
                    gp['gpu_layer_stride'],
                    cpu_block_id_list,
                    cpu_tensor_for_group,
                    gp['cpu_kv_stride'],
                    gp['cpu_layer_stride'],
                    gp['cpu_block_stride'],
                    gp['chunk_size'],
                    0,                   # start_layer_id (always 0 within group)
                    gp['num_layers'],    # all layers in this group
                    transfer_num_cta,
                    transfer_type == TransferType.H2D,
                    use_ce_transfer,
                    self.kv_dim,
                    self.num_kv_heads,
                    self.gpu_block_type_,
                    True,  # sync
                    self.ce_path_opt,
                    self.ce_segment_threshold,
                    -1,  # ce_force_path
                    self.ce_enable_memcpy2d,
                    self.cpu_is_blockfirst,
                    enable_transfer_trace=GLOBAL_CONFIG_FROM_ENV.enable_transfer_trace,
                )
        else:
            # Uniform transfer: single call (whole-model)
            transfer_kv_blocks(
                gpu_block_id_list,
                self.gpu_blocks_ptrs,
                self.gpu_kv_stride_in_bytes,
                self.gpu_block_stride_in_bytes,
                self.gpu_layer_stride_in_bytes,
                cpu_block_id_list,
                self.cpu_tensor,
                self.cpu_kv_stride_in_bytes,
                self.cpu_layer_stride_in_bytes,
                self.cpu_block_stride_in_bytes,
                self.chunk_size_in_bytes,
                0,                  # start_layer_id (whole-model)
                self.num_layers,    # layer_granularity = all layers
                transfer_num_cta,
                transfer_type == TransferType.H2D,
                use_ce_transfer,
                self.kv_dim,
                self.num_kv_heads,
                self.gpu_block_type_,
                True,  # sync
                self.ce_path_opt,
                self.ce_segment_threshold,
                -1,  # ce_force_path
                self.ce_enable_memcpy2d,
                self.cpu_is_blockfirst,
                enable_transfer_trace=GLOBAL_CONFIG_FROM_ENV.enable_transfer_trace,
            )

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> bool:
        """执行一个 H2D/D2H op（由基类 run() 调用）。

        关键：整个传输跑在 self.transfer_stream 上而不是默认流，
        这样不会和推理主进程的 compute stream 互相排队。

        统一布局走 self._compressor.run(...)（压缩器内部会回调本对象的
        _transfer_impl，NullCompressionStrategy 则直接透传）；
        多分组不支持压缩，直接内联调用并自己算传输量。
        """
        nvtx_range = nvtx.start_range(
            message=f"GPUCPUWorker.launch_transfer[{transfer_op.transfer_op_id}]",
            color="purple")

        src_block_ids, dst_block_ids = self.get_transfer_block_ids(transfer_op)

        try:
            with torch.cuda.stream(self.transfer_stream):
                if self.group_transfer_params is not None:
                    # Multi-group (heterogeneous KV) path — compression is not
                    # supported here; issue the per-group transfers inline.
                    start_time = time.time()
                    self._transfer_impl(
                        src_block_ids,
                        dst_block_ids,
                        transfer_op.transfer_type,
                    )
                    end_time = time.time()
                    transfer_size = 0
                    for gp in self.group_transfer_params:
                        transfer_size += gp['chunk_size'] * gp['num_layers'] * transfer_op.valid_block_num * self.kv_dim
                    self._log_transfer_performance(
                        transfer_op,
                        transfer_size,
                        start_time,
                        end_time,
                    )
                else:
                    # Uniform path — supports (optional) nvcomp compression.
                    self._compressor.run(
                        self, src_block_ids=src_block_ids,
                        dst_block_ids=dst_block_ids, op=transfer_op)
        finally:
            nvtx.end_range(nvtx_range)

        return True

class tpGPUCPUTransferWorker(TransferWorkerBase):
    """GPU <-> CPU 通路（H2D / D2H）的 **TP 多卡** 版本。

    与非 TP 版本（GPUCPUTransferWorker）的差异 —— 这是理解本类的关键：
        1. 入参是"二维"的：gpu_blocks[card][layer]，每张卡一套 GPU KV tensor，
           gpu_kv_layouts 也是每张卡一份（各卡 stride 可能不同）。
        2. CPU 侧多了一个 **tp 维度**：TP 下每张卡只持有 1/tp_size 的 KV head，
           CPU 池里这些分片必须按 tp 维度拼接。约定是"tp 维永远紧跟 block 维"，
           因此 cpu_tp_stride = cpu_block_stride // tp_group_size；
           BLOCKFIRST 且 num_kv_heads > 1 时还要先 div_head(tp_group_size)。
        3. 不用 torch 的 stream，而是用 C++ 的 ``TPTransferThreadGroup``：
           它为**每张卡起一个专属线程**、各自 set_device 后并发下发拷贝。
           这是必需的——一次 cudaMemcpyAsync 只能作用于**当前设备**，
           想在单进程里同时驱动 8 张卡，就必须一卡一线程各自持上下文。
           因此本类没有 self.transfer_stream。
        4. 指针在 Python 侧解析好再传进 C++：spawn 出来的子进程里，
           pybind11 对跨进程 tensor 调 .data_ptr() 会报
           "Tensor that doesn't have storage"，所以在 Python 里取裸指针传进去。

    线程模型：
        本 worker 主线程（收任务、解析） + TPTransferThreadGroup 的 N 个
        per-GPU 传输线程（N = num_gpus）。调用 tp_group_transfer 时
        主线程下发、阻塞等待所有卡完成后返回（同步语义）。
    """
    def __init__(self,
                 worker_id: int,
                 transfer_conn: Connection,
                 finished_ops_queue: MPQueue,
                 op_buffer_tensor: torch.Tensor,
                 gpu_blocks: List[List[TensorSharedHandle]],
                 cpu_blocks: Union[torch.Tensor, HugePageTensorHandle],
                 gpu_kv_layouts: List[KVCacheLayout],
                 cpu_kv_layout: KVCacheLayout,
                 dtype: torch.dtype,
                 tp_group_size: int,
                 use_ce_transfer_h2d: bool = False,
                 use_ce_transfer_d2h: bool = False,
                 transfer_num_cta_h2d: int = 4,
                 transfer_num_cta_d2h: int = 4,
                 compressor: Optional[CompressionStrategy] = None,
                 layer_groups: Optional[List[LayerGroupSpec]] = None,
                 gpu_blocks_per_group: Optional[List[List[List[TensorSharedHandle]]]] = None,
                 gpu_layouts_per_group: Optional[List[List[KVCacheLayout]]] = None):
        """解析 N 张卡的布局，并建好 TPTransferThreadGroup。

        与 GPUCPUTransferWorker.__init__ 的三点差异：
            - ensure_cuda_device 用 gpu_blocks[0][0].device（主卡），
              后续每卡由 import_tensor_handles 各自 set_device；
            - stride 是"每卡一组"的列表而不是单个标量；
            - 最后构造 TPTransferThreadGroup（每卡一线程）而不是 torch Stream。

        Args:
            tp_group_size: 本节点上有效的 TP 规模
            gpu_kv_layouts: 与 gpu_blocks 一一对应，每卡一份 layout
            kv_shared_across_ranks_mode: 各 rank 的 KV 是否相同（相同则 D2H 只需搬一份）
        """

        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)
        assert len(gpu_blocks) == tp_group_size
        cpu_blocks = materialize_worker_tensor(cpu_blocks)
        # Bind primary GPU + pin op buffer before any CUDA IPC import.
        if gpu_blocks and gpu_blocks[0]:
            ensure_cuda_device(gpu_blocks[0][0].device)
        self._pin_op_buffer()
        # Handle tensor import for multi-process case — set_device per GPU first.
        imported_gpu_blocks = []
        for handles_in_one_gpu in gpu_blocks:
            imported_gpu_blocks.append(import_tensor_handles(handles_in_one_gpu))
        self._gpu_block_counts = [len(handles) for handles in gpu_blocks]
        self.gpu_blocks = imported_gpu_blocks
        self.dtype = dtype # note this should be quantized data type
        self.kv_dim = gpu_kv_layouts[0].kv_dim
        self.num_kv_heads = gpu_kv_layouts[0].num_kv_heads

        self.num_gpus = len(self.gpu_blocks)
        self.tp_group_size = tp_group_size
        self.layer_groups = layer_groups
        self.cpu_tensor = cpu_blocks

        flexkv_logger.info(f"Pinning CPU Memory: {cpu_blocks.numel() * cpu_blocks.element_size() / (1024 ** 3):.2f} GB")
        self._register_host_tensor(cpu_blocks, "tp_cpu_kv_pool")

        self.num_layers = gpu_kv_layouts[0].num_layer

        self.transfer_num_cta_h2d = transfer_num_cta_h2d
        self.transfer_num_cta_d2h = transfer_num_cta_d2h
        self.use_ce_transfer_h2d = use_ce_transfer_h2d
        self.use_ce_transfer_d2h = use_ce_transfer_d2h

        # Read KV shared across ranks D2H mode from global config
        self.kv_shared_across_ranks_mode = GLOBAL_CONFIG_FROM_ENV.kv_shared_across_ranks_mode
        flexkv_logger.debug(f"[tpGPUCPUTransferWorker] kv_shared_across_ranks_mode={self.kv_shared_across_ranks_mode}")

        if layer_groups is not None and gpu_blocks_per_group is not None and gpu_layouts_per_group is not None:
            self._init_tp_multi_group(
                gpu_blocks_per_group, gpu_layouts_per_group,
                cpu_kv_layout, layer_groups,
            )
        else:
            self.tp_group_transfer_groups = None

            # Compute GPU strides from actual tensor to handle different attention
            # backend layouts (flash_attn: [2,N,B,H,D], triton: [N,2,B,H,D]).
            # Each GPU may have different strides, so compute per-GPU.
            dtype_sz = self.dtype.itemsize
            tpb = gpu_kv_layouts[0].tokens_per_block
            self.gpu_chunk_sizes_in_bytes = []
            self.gpu_kv_strides_in_bytes = []
            self.gpu_block_strides_in_bytes = []
            self.gpu_layer_strides_in_bytes = []
            for i, gpu_kv_layout in enumerate(gpu_kv_layouts):
                gpu_strides = self._get_gpu_strides_from_tensor(
                    self.gpu_blocks[i][0], tpb, dtype_sz, self.kv_dim,
                ) if len(self.gpu_blocks[i]) > 1 else None
                if gpu_strides is not None:
                    kv_s, blk_s, layer_s = gpu_strides
                else:
                    kv_s = gpu_kv_layout.get_kv_stride() * dtype_sz
                    blk_s = gpu_kv_layout.get_block_stride() * dtype_sz
                    layer_s = gpu_kv_layout.get_layer_stride() * dtype_sz
                self.gpu_chunk_sizes_in_bytes.append(gpu_kv_layout.get_chunk_size() * dtype_sz)
                self.gpu_kv_strides_in_bytes.append(kv_s)
                self.gpu_block_strides_in_bytes.append(blk_s)
                self.gpu_layer_strides_in_bytes.append(layer_s)

            self.cpu_is_blockfirst = (
                cpu_kv_layout.type == KVCacheLayoutType.BLOCKFIRST
            )
            self.cpu_block_stride_in_bytes = cpu_kv_layout.get_block_stride() * self.dtype.itemsize
            self.cpu_chunk_size_in_bytes = cpu_kv_layout.get_chunk_size() * self.dtype.itemsize
            self.chunk_size_in_bytes = self.cpu_chunk_size_in_bytes
            # Bytes per KV block (all layers); used by transfer tracing for bw.
            self._bytes_per_block = self.chunk_size_in_bytes * self.num_layers * self.kv_dim
            # tp has effect on the layout of the cpu tensor
            # the tp dim should always be right after the block dim
            # on both blockfirst layout and layerfirst layout
            if cpu_kv_layout.type == KVCacheLayoutType.BLOCKFIRST and self.num_kv_heads > 1:
                cpu_kv_layout = cpu_kv_layout.div_head(self.tp_group_size)

            self.cpu_layer_stride_in_bytes = cpu_kv_layout.get_layer_stride() * self.dtype.itemsize
            self.cpu_kv_stride_in_bytes = cpu_kv_layout.get_kv_stride() * self.dtype.itemsize
            self.cpu_tp_stride_in_bytes = self.cpu_block_stride_in_bytes // self.tp_group_size

            # Resolve pointers in Python (where storage is valid); pass them to C++ so we avoid
            # "Tensor that doesn't have storage" when C++ calls .data_ptr() on tensors passed
            # across the pybind11 boundary from a spawn'd subprocess (shared memory / CUDA IPC).
            gpu_block_ptrs_flat = [
                self.gpu_blocks[i][j].data_ptr()
                for i in range(self.num_gpus)
                for j in range(len(self.gpu_blocks[i]))
            ]
            cpu_blocks_ptr = cpu_blocks.data_ptr()
            gpu_device_ids = [self.gpu_blocks[i][0].device.index for i in range(self.num_gpus)]
            num_tensors_per_gpu = len(self.gpu_blocks[0])

            self.tp_transfer_thread_group = TPTransferThreadGroup(
                self.num_gpus,
                gpu_block_ptrs_flat,
                num_tensors_per_gpu,
                cpu_blocks_ptr,
                self.num_layers,
                self.gpu_kv_strides_in_bytes,
                self.gpu_block_strides_in_bytes,
                self.gpu_layer_strides_in_bytes,
                self.gpu_chunk_sizes_in_bytes,
                gpu_device_ids,
                GLOBAL_CONFIG_FROM_ENV.ce_segment_threshold,
                GLOBAL_CONFIG_FROM_ENV.ce_path_opt,
                GLOBAL_CONFIG_FROM_ENV.enable_ce_memcpy2d,
                self.cpu_is_blockfirst,
                self.num_kv_heads,
                ce_gather_threads=GLOBAL_CONFIG_FROM_ENV.ce_gather_threads,
                ce_gather_nt=GLOBAL_CONFIG_FROM_ENV.ce_gather_nt,
            )

        self._compressor = compressor or NullCompressionStrategy()
        self._compressor.attach(self)

    def _init_tp_multi_group(
        self,
        gpu_blocks_per_group: List[List[List[TensorSharedHandle]]],
        gpu_layouts_per_group: List[List[KVCacheLayout]],
        cpu_kv_layout: KVCacheLayout,
        layer_groups: List[LayerGroupSpec],
    ) -> None:
        """Initialize per-group TPTransferThreadGroup instances.

        CPU buffer is byte-flat (uint8) in multi-group mode: each block has
        size kv_shape[1] = bytes_per_block (see KVCacheLayout._compute_kv_shape).
        Per-group strides use g.dtype.itemsize so groups with different element
        sizes (e.g. bf16 main + uint8 indexer) interleave correctly within a
        block.

        中文要点：与非 TP 的 _init_multi_group 思路一致，但每个 group 都要
        建一个**独立的** TPTransferThreadGroup（因为每组的 tensor 数/stride 不同），
        并把该组在 CPU block 内的字节偏移通过 cpu_blocks_ptr 传进去。
        """
        kv_dim = self.kv_dim
        tpb = cpu_kv_layout.tokens_per_block
        cpu_layout_type = cpu_kv_layout.type
        num_cpu_blocks = cpu_kv_layout.num_block

        # For BLOCKFIRST multi-group, get_block_stride() returns bytes_per_block
        # directly (already accounts for tp_size and per-group dtype sizes).
        total_block_bytes = (
            cpu_kv_layout.get_block_stride()
            if cpu_layout_type == KVCacheLayoutType.BLOCKFIRST else None
        )

        self.tp_group_transfer_groups: list = []
        # Keep imported CUDA-IPC tensors alive for the worker's lifetime:
        # TPTransferThreadGroup below stores only their raw data_ptr()s, so if
        # the tensors were dropped PyTorch would release the IPC mapping and the
        # pointers would dangle (mirrors GPUCPUTransferWorker._init_multi_group
        # and LayerwiseWorker._init_multi_group).
        self._multi_group_gpu_blocks_keepalive: list = []
        cpu_offset_bytes = 0

        for gi, g in enumerate(layer_groups):
            # Per-group dtype: indexer uses uint8 even when main KV is bf16/fp16.
            dtype_size_g = g.dtype.itemsize

            # gpu_blocks_per_group[gi] = list of per-GPU handle lists for this group
            # gpu_blocks_per_group[gi][gpu_idx] = handles for this group on GPU gpu_idx
            group_gpu_blocks_per_gpu = gpu_blocks_per_group[gi]

            # Import tensors from handles (bind CUDA device per GPU first)
            imported_group_blocks = []
            for handles_in_one_gpu in group_gpu_blocks_per_gpu:
                imported_group_blocks.append(import_tensor_handles(handles_in_one_gpu))
            self._multi_group_gpu_blocks_keepalive.append(imported_group_blocks)

            # Build flat pointer list for this group
            gpu_block_ptrs_flat = [
                imported_group_blocks[i][j].data_ptr()
                for i in range(self.num_gpus)
                for j in range(len(imported_group_blocks[i]))
            ]
            gpu_device_ids = [imported_group_blocks[i][0].device.index for i in range(self.num_gpus)]
            num_tensors_per_gpu = len(imported_group_blocks[0])

            # Compressed groups: GPU tensor's tokens dim equals tpb_g.
            tpb_g = tpb // g.compress_ratio

            # Per-group GPU strides: compute from actual tensor to handle different
            # attention backend layouts (flash_attn vs triton/flashinfer).
            group_gpu_layouts = gpu_layouts_per_group[gi]  # one layout per GPU
            gpu_kv_strides = []
            gpu_block_strides = []
            gpu_layer_strides = []
            gpu_chunk_sizes = []
            for i, layout in enumerate(group_gpu_layouts):
                gpu_strides = self._get_gpu_strides_from_tensor(
                    imported_group_blocks[i][0], tpb_g, dtype_size_g, self.kv_dim,
                ) if len(imported_group_blocks[i]) > 1 else None
                if gpu_strides is not None:
                    kv_s, blk_s, layer_s = gpu_strides
                else:
                    kv_s = layout.get_kv_stride() * dtype_size_g
                    blk_s = layout.get_block_stride() * dtype_size_g
                    layer_s = layout.get_layer_stride() * dtype_size_g
                gpu_kv_strides.append(kv_s)
                gpu_block_strides.append(blk_s)
                gpu_layer_strides.append(layer_s)
                gpu_chunk_sizes.append(layout.get_chunk_size() * dtype_size_g)

            chunk_elements = tpb_g * g.num_kv_heads * g.head_size

            # CPU strides for this group (all in bytes)
            if cpu_layout_type == KVCacheLayoutType.BLOCKFIRST:
                cpu_block_stride = total_block_bytes
                cpu_layer_stride = kv_dim * chunk_elements * dtype_size_g
                cpu_kv_stride = chunk_elements * dtype_size_g
                cpu_tp_stride = cpu_block_stride // self.tp_group_size
            else:
                cpu_block_stride = chunk_elements * dtype_size_g
                cpu_layer_stride = kv_dim * num_cpu_blocks * chunk_elements * dtype_size_g
                cpu_kv_stride = num_cpu_blocks * chunk_elements * dtype_size_g
                cpu_tp_stride = cpu_block_stride // self.tp_group_size

            # CPU tensor offset for this group (cpu_tensor is uint8 in multi-group)
            cpu_blocks_ptr = self.cpu_tensor.view(-1)[cpu_offset_bytes:].data_ptr()

            tp_thread_group = TPTransferThreadGroup(
                self.num_gpus,
                gpu_block_ptrs_flat,
                num_tensors_per_gpu,
                cpu_blocks_ptr,
                g.num_layers,
                gpu_kv_strides,
                gpu_block_strides,
                gpu_layer_strides,
                gpu_chunk_sizes,
                gpu_device_ids,
                GLOBAL_CONFIG_FROM_ENV.ce_segment_threshold,
                GLOBAL_CONFIG_FROM_ENV.ce_path_opt,
                GLOBAL_CONFIG_FROM_ENV.enable_ce_memcpy2d,
                (cpu_layout_type == KVCacheLayoutType.BLOCKFIRST),
                self.num_kv_heads,
            )

            self.tp_group_transfer_groups.append({
                'tp_thread_group': tp_thread_group,
                'cpu_kv_stride': cpu_kv_stride,
                'cpu_layer_stride': cpu_layer_stride,
                'cpu_block_stride': cpu_block_stride,
                'cpu_tp_stride': cpu_tp_stride,
                'cpu_offset_bytes': cpu_offset_bytes,
                'num_layers': g.num_layers,
                'chunk_size': chunk_elements * dtype_size_g,
            })

            # Advance CPU byte offset for next group
            if cpu_layout_type == KVCacheLayoutType.BLOCKFIRST:
                cpu_offset_bytes += g.num_layers * kv_dim * chunk_elements * dtype_size_g
            else:
                cpu_offset_bytes += (
                    g.num_layers * kv_dim * num_cpu_blocks * chunk_elements * dtype_size_g
                )

        flexkv_logger.info(
            f"TP multi-group transfer initialized: {len(layer_groups)} groups, "
            f"total_block_bytes={total_block_bytes}"
        )


    def _control_suspend_gpu(self, payload: Any) -> int:
        """控制面：释放所有卡的 VMM 映射。

        与非 TP 版的差异：指针不放在 Python 数组里，而是存在 C++ 的
        TPTransferThreadGroup 中，所以要先 update_gpu_block_ptrs(全 0)
        把 C++ 侧指针打空，再释放 Python 侧的 tensor。
        """
        if self.tp_group_transfer_groups is not None:
            raise NotImplementedError(
                "GPU hot remap does not support multi-group KV layouts"
            )
        if not self.gpu_blocks:
            return 0
        zero_ptrs = [0] * sum(self._gpu_block_counts)
        self.tp_transfer_thread_group.update_gpu_block_ptrs(zero_ptrs)
        old_blocks = self.gpu_blocks
        self.gpu_blocks = []
        released = sum(
            release_vmm_tensor(tensor)
            for blocks_in_one_gpu in old_blocks
            for tensor in blocks_in_one_gpu
        )
        expected = sum(self._gpu_block_counts)
        if released != expected:
            raise RuntimeError(
                f"Expected {expected} VMM mappings, released {released}"
            )
        return released

    def _control_resume_gpu(
        self, gpu_blocks: List[List[TensorSharedHandle]]
    ) -> int:
        """控制面：重新导入各卡 tensor 并把新的裸指针推回 C++ 线程组。"""
        if self.gpu_blocks:
            raise RuntimeError("GPU blocks are already registered")
        counts = [len(handles) for handles in gpu_blocks]
        if counts != self._gpu_block_counts:
            raise ValueError(
                f"Expected GPU block counts {self._gpu_block_counts}, got {counts}"
            )
        imported_gpu_blocks = [
            import_tensor_handles(handles) for handles in gpu_blocks
        ]
        gpu_block_ptrs_flat = [
            tensor.data_ptr()
            for blocks_in_one_gpu in imported_gpu_blocks
            for tensor in blocks_in_one_gpu
        ]
        self.tp_transfer_thread_group.update_gpu_block_ptrs(
            gpu_block_ptrs_flat
        )
        self.gpu_blocks = imported_gpu_blocks
        return len(gpu_block_ptrs_flat)

    def _transfer_impl(self,
                       src_block_ids: torch.Tensor,
                       dst_block_ids: torch.Tensor,
                       transfer_type: TransferType,
                       **kwargs: Any,
                       )->None:
        """把 block id 列表翻译成 TPTransferThreadGroup.tp_group_transfer 调用。

        CPU 侧四个 stride 中比非 TP 版多一个 cpu_tp_stride：
        C++ 靠它把第 r 张卡的 head 分片写到 CPU block 内正确的位置。
        GPU 侧的 stride/chunk_size 已经在 __init__ 里交给线程组了，这里不再传。
        """
        assert src_block_ids.dtype == torch.int64
        assert dst_block_ids.dtype == torch.int64
        assert len(src_block_ids) == len(dst_block_ids)

        if transfer_type == TransferType.H2D:
            gpu_block_id_list = dst_block_ids
            cpu_block_id_list = src_block_ids
            use_ce_transfer = self.use_ce_transfer_h2d
            transfer_num_cta = self.transfer_num_cta_h2d
        elif transfer_type == TransferType.D2H:
            gpu_block_id_list = src_block_ids
            cpu_block_id_list = dst_block_ids
            use_ce_transfer = self.use_ce_transfer_d2h
            transfer_num_cta = self.transfer_num_cta_d2h
        else:
            raise ValueError(f"Invalid transfer type: {transfer_type} for tpGPUCPUTransferWorker")


        assert len(gpu_block_id_list) == len(cpu_block_id_list)

        if len(gpu_block_id_list) == 0:
            return

        if self.tp_group_transfer_groups is not None:
            # Multi-group transfer: one call per group
            for gp in self.tp_group_transfer_groups:
                g_gpu = gpu_block_id_list
                g_cpu = cpu_block_id_list

                gp['tp_thread_group'].tp_group_transfer(
                    g_gpu,
                    g_cpu,
                    gp['cpu_kv_stride'],
                    gp['cpu_layer_stride'],
                    gp['cpu_block_stride'],
                    gp['cpu_tp_stride'],
                    transfer_num_cta,
                    transfer_type == TransferType.H2D,
                    use_ce_transfer,
                    0,                 # start_layer_id (always 0 within group)
                    gp['num_layers'],  # all layers in this group
                    self.kv_dim,
                    self.num_kv_heads,
                    self.kv_shared_across_ranks_mode,
                )
        else:
            self.tp_transfer_thread_group.tp_group_transfer(
                gpu_block_id_list,
                cpu_block_id_list,
                self.cpu_kv_stride_in_bytes,
                self.cpu_layer_stride_in_bytes,
                self.cpu_block_stride_in_bytes,
                self.cpu_tp_stride_in_bytes,
                transfer_num_cta,
                transfer_type == TransferType.H2D,
                use_ce_transfer,
                0,                  # start_layer_id (whole-model)
                self.num_layers,    # layer_granularity = all layers
                self.kv_dim,
                self.num_kv_heads,
                self.kv_shared_across_ranks_mode,
            )


    def launch_transfer(self, transfer_op: WorkerTransferOp) -> bool:
        """执行一个 TP 场景的 H2D/D2H op。

        注意这里**不**绑 torch stream：并发是由 C++ 侧 per-GPU 线程组完成的，
        Python 侧只负责解析和计时。
        """
        src_block_ids, dst_block_ids = self.get_transfer_block_ids(transfer_op)
        if self.tp_group_transfer_groups is not None:
            # Multi-group (heterogeneous KV) path — compression not supported here.
            start_time = time.time()
            self._transfer_impl(
                src_block_ids,
                dst_block_ids,
                transfer_op.transfer_type,
            )
            end_time = time.time()

            transfer_size = 0
            for gp in self.tp_group_transfer_groups:
                transfer_size += gp['chunk_size'] * gp['num_layers'] * transfer_op.valid_block_num * self.kv_dim

            self._log_transfer_performance(
                transfer_op,
                transfer_size,
                start_time,
                end_time,
            )
        else:
            # Uniform path — supports (optional) nvcomp compression.
            self._compressor.run(
                self, src_block_ids=src_block_ids,
                dst_block_ids=dst_block_ids, op=transfer_op)
        return True

class CPUSSDDiskTransferWorker(TransferWorkerBase):
    """CPU <-> 本地 SSD 通路（H2DISK / DISK2H），基于 **io_uring**。

    在链路中的职责：
        三级存储的第二级。D2H 把 KV 落到 CPU 后，由本 worker 再异步刷到
        本地 NVMe；命中时反过来 DISK2H 读回 CPU（之后再由 GPUCPU worker 拉上卡）。

    传输机制 —— io_uring：
        __init__ 里建 ``c_ext.SSDIOCTX(ssd_files, ..., iouring_entries,
        iouring_flags)``：这是 C++ 侧持有的一组 io_uring ring（每块盘一个），
        _entries 是 SQE 队列深度，_flags 透传给 io_uring_setup。
        每次 _transfer_impl 把"每个 block × 每个 layer × 每个 kv"的
        pread/pwrite 请求**批量**提交进 ring，然后内核侧由 C++ 等待 CQE
        全部完成才返回（所以 Python 侧依然是同步语义）。
        相比 libaio / 普通 pread：省掉每请求一次系统调用，且支持内核态轮询。

    向量化 I/O：
        transfer_kv_blocks_ssd 的倒数第三个参数 32 是 **并发线程数/队列深度**，
        C++ 会把 block 列表切片后并发提交；配合 ssd_io_opt（GLOBAL_CONFIG）
        可以启用更大粒度 / 合并相邻请求等优化。

    文件映射：
        多个 SSD 文件 round-robin 存放 block（round_robin=1 表示一个 block
        粒度轮转），num_blocks_per_file 决定 block_id -> (file, offset) 的换算。
        SSD 侧的 stride 用 ssd_kv_layout.div_block(num_files) 得到"每文件"布局。

    关于 hugepage：
        本通路**不**使用 hugepage 临时缓冲——CPU KV 池本身就是
        HugePageTensorHandle（见 materialize_worker_tensor），读写直接落在
        大页上，减少 TLB miss。真正用到 hugepage **临时**缓冲的是
        PEER2CPUTransferWorker（见 allocate_host_buffer 相关注释）。

    线程模型：单进程单线程，无 CUDA 参与（不需要绑 GPU、不建 stream）。
    """
    def __init__(self,
                 worker_id: int,
                 transfer_conn: Connection,
                 finished_ops_queue: MPQueue,
                 op_buffer_tensor: torch.Tensor,
                 cpu_blocks: Union[torch.Tensor, HugePageTensorHandle],
                 ssd_files: Dict[int, List[str]],  # ssd_device_id -> file_paths
                 cpu_kv_layout: KVCacheLayout,
                 ssd_kv_layout: KVCacheLayout,
                 dtype: torch.dtype,
                 num_blocks_per_file: int,
                 cache_config: CacheConfig,
                 compressor: Optional[CompressionStrategy] = None,
                 layer_groups: Optional[List[LayerGroupSpec]] = None):
        """解析 CPU/SSD 两侧布局并建好 io_uring 上下文。

        注意这里**不**调 ensure_cuda_device：本通路没有任何 CUDA 参与，
        但 op_buffer 仍然要 pin（c_ext 会 DMA 读 block id）。

        Args:
            ssd_files: {ssd_device_id: [文件路径...]}，多盘多文件轮转
            num_blocks_per_file: 每个文件放多少个 block（决定文件内偏移换算）
            cpu_kv_layout / ssd_kv_layout: 两侧必须是同一种 layout type
                （BLOCKFIRST / LAYERFIRST 不能混），否则直接 ValueError
        """
        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)
        self._pin_op_buffer()
        cpu_blocks = materialize_worker_tensor(cpu_blocks)
        self.ssd_files = ssd_files
        self.num_blocks_per_file = num_blocks_per_file
        self.num_files = sum(len(file_list) for file_list in ssd_files.values())

        self.num_layers = cpu_kv_layout.num_layer
        self.num_cpu_blocks = cpu_kv_layout.num_block
        self.round_robin = 1

        self.dtype = dtype

        self.cpu_blocks = cpu_blocks
        self.cpu_layer_ptrs = self._get_layer_ptrs(cpu_blocks)

        self.kv_dim = cpu_kv_layout.kv_dim
        self.num_kv_heads = cpu_kv_layout.num_kv_heads
        self.cpu_layout_type = cpu_kv_layout.type
        self.has_multi_group = layer_groups is not None

        if cpu_kv_layout.type != ssd_kv_layout.type:
            raise ValueError("no support for different CPU and SSD KV cache layout type")

        if self.has_multi_group:
            self._init_multi_group_ssd(cpu_kv_layout, ssd_kv_layout, layer_groups)
        else:
            ssd_kv_layout_per_file = ssd_kv_layout.div_block(self.num_files, padding=True)

            self.chunk_size_in_bytes = cpu_kv_layout.get_chunk_size() * self.dtype.itemsize
            self.block_stride_in_bytes = cpu_kv_layout.get_block_stride() * self.dtype.itemsize
            self.cpu_kv_stride_in_bytes = cpu_kv_layout.get_kv_stride() * self.dtype.itemsize
            self.cpu_layer_stride_in_bytes = cpu_kv_layout.get_layer_stride() * self.dtype.itemsize
            self.ssd_kv_stride_in_bytes = ssd_kv_layout_per_file.get_kv_stride() * self.dtype.itemsize
            self.ssd_layer_stride_in_bytes = ssd_kv_layout_per_file.get_layer_stride() * self.dtype.itemsize
            # Bytes per KV block (all layers); used by transfer tracing for bw.
            self._bytes_per_block = self.chunk_size_in_bytes * self.num_layers * self.kv_dim

        try:
            self.ioctx = c_ext.SSDIOCTX(ssd_files, len(ssd_files), GLOBAL_CONFIG_FROM_ENV.iouring_entries,
                GLOBAL_CONFIG_FROM_ENV.iouring_flags)
        except Exception as e:
            flexkv_logger.error(f"Error setting ssd ioctx: {e}\n")
            raise RuntimeError("SSD Worker init failed") from e

        self._compressor = compressor or NullCompressionStrategy()
        self._compressor.attach(self)

    def _init_multi_group_ssd(
        self,
        cpu_kv_layout: KVCacheLayout,
        ssd_kv_layout: KVCacheLayout,
        layer_groups: List[LayerGroupSpec],
    ) -> None:
        """Initialize CPU<->SSD multi-group parameters.

        CPU and SSD share an identical per-block byte layout (BLOCKFIRST),
        so multi-group SSD transfers move whole blocks as opaque blobs —
        no per-group / per-tp_rank slicing needed at the IO layer.

        中文要点：多分组下 CPU 与 SSD 的每 block 字节布局完全一致，
        所以整块可以当一个"不透明 blob"搬 —— 把 num_layers 说成 1、
        chunk_size = block_stride，C++ 就会对每个 block 只发一次
        block_stride 字节的 pread/pwrite。这样也顺带避开了高压缩比 group
        （如 DSv4 indexer，compress_ratio=128）产生 sub-4KiB 小 IO 的坑：
        NVMe 上小于 4K 的读写会触发读改写放大，吞吐掉一个数量级。
        """
        # Multi-group BLOCKFIRST: get_block_stride() returns bytes_per_block
        # directly (already accounts for tp_size and per-group dtype sizes).
        self.block_stride_in_bytes = cpu_kv_layout.get_block_stride()

        flexkv_logger.info(
            f"CPUSSDDiskTransferWorker multi-group initialized: {len(layer_groups)} groups, "
            f"block_stride={self.block_stride_in_bytes} bytes"
        )

    def _transfer_impl(
        self,
        src_block_ids: torch.Tensor,
        dst_block_ids: torch.Tensor,
        transfer_type: TransferType,
        **kwargs: Any,
    ) -> None:
        """把 block id 列表翻译成一次 transfer_kv_blocks_ssd 调用。

        参数里的 32 是并发度（C++ 侧把请求切片并发提交给 io_ring）；
        is_read 决定 pread(DISK2H) 还是 pwrite(H2DISK)。
        """
        assert src_block_ids.dtype == torch.int64
        assert dst_block_ids.dtype == torch.int64
        assert len(src_block_ids) == len(dst_block_ids)

        if transfer_type == TransferType.H2DISK:
            ssd_block_id_list = dst_block_ids
            cpu_block_id_list = src_block_ids
        elif transfer_type == TransferType.DISK2H:
            ssd_block_id_list = src_block_ids
            cpu_block_id_list = dst_block_ids
        else:
            raise ValueError(f"Invalid transfer type: {transfer_type} for CPUSSDDiskTransferWorker")

        is_read = (transfer_type == TransferType.DISK2H)
        cpu_base_ptr = self.cpu_layer_ptrs[0].item()

        if self.has_multi_group:
            # CPU and SSD share an identical per-block byte layout in multi-group
            # mode, so each block can be transferred as one opaque blob — no
            # per-group / per-tp_rank loop needed. num_layers=1,
            # layer_stride=chunk_size=block_stride, one KV region makes
            # the kernel issue exactly one pread/pwrite of block_stride bytes
            # per block, sidestepping the sub-4KiB chunk hazard for highly
            # compressed groups (e.g. DSv4 indexer at compress_ratio=128).
            one_layer_id = torch.tensor([0], dtype=torch.int32)
            transfer_kv_blocks_ssd(
                self.ioctx,
                one_layer_id,
                cpu_base_ptr,
                ssd_block_id_list,
                cpu_block_id_list,
                self.block_stride_in_bytes,
                0,
                self.block_stride_in_bytes,
                0,
                self.block_stride_in_bytes,
                self.block_stride_in_bytes,
                is_read,
                self.num_blocks_per_file,
                self.round_robin,
                32,
                True,
                ssd_io_opt=GLOBAL_CONFIG_FROM_ENV.ssd_io_opt,
            )
        else:
            layer_id_list = torch.arange(0, self.num_layers, dtype=torch.int32)

            transfer_kv_blocks_ssd(
                self.ioctx,
                layer_id_list,
                cpu_base_ptr,
                ssd_block_id_list,
                cpu_block_id_list,
                self.cpu_layer_stride_in_bytes,
                self.cpu_kv_stride_in_bytes,
                self.ssd_layer_stride_in_bytes,
                self.ssd_kv_stride_in_bytes,
                self.chunk_size_in_bytes,
                self.block_stride_in_bytes,
                is_read,
                self.num_blocks_per_file,
                self.round_robin,
                32,
                self.kv_dim,
                ssd_io_opt=GLOBAL_CONFIG_FROM_ENV.ssd_io_opt,
            )

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> bool:
        """执行一个 H2DISK / DISK2H op。

        统一布局走压缩器（SSD 上可以存压缩后的 KV，省带宽和容量）；
        多分组不支持压缩，按 block_stride × block 数直接算传输量。
        """
        src_block_ids, dst_block_ids = self.get_transfer_block_ids(transfer_op)
        if self.has_multi_group:
            # Multi-group (heterogeneous KV) path — compression not supported here.
            start_time = time.time()
            self._transfer_impl(
                src_block_ids,
                dst_block_ids,
                transfer_op.transfer_type,
            )
            end_time = time.time()
            # Total transfer size across all groups
            transfer_size = self.block_stride_in_bytes * transfer_op.valid_block_num
            self._log_transfer_performance(
                transfer_op,
                transfer_size,
                start_time,
                end_time,
            )
        else:
            # Uniform path — supports (optional) nvcomp compression.
            self._compressor.run(
                self, src_block_ids=src_block_ids,
                dst_block_ids=dst_block_ids, op=transfer_op)
        return True

class CPURemoteTransferWorker(TransferWorkerBase):
    """CPU <-> 远端共享存储（PCFS）通路：H2REMOTE / REMOTE2H【旁支，默认不生效】。

    启用条件：**需要以 FLEXKV_ENABLE_CFS=1 重新编译 c_ext**。
    未开启时 ``from flexkv.c_ext import transfer_kv_blocks_remote`` 会
    ImportError（本文件顶部已 try/except 兜成 None），
    __init__ 第一行就会抛 RuntimeError —— 默认构建下这条通路完全不参与运行。

    在链路中的职责：三级存储的第三级，跨节点共享的 KV 池，
    让别的机器算过的前缀不必重算。

    传输机制：不走 io_uring，而是走 PCFS（CFS 客户端）的用户态 SDK：
        - __init__ 里 c_ext.Pcfs(...) 建客户端，把每个远端文件
          lookup_or_create 成 nodeid，并 set 成全局实例；
        - _transfer_impl 调 transfer_kv_blocks_remote，由 C++ 侧
          做多线程远端读写（末尾的 32 是并发线程数）。
        - enable_pcfs_sharing 且是读时，改走 shared_transfer_kv_blocks_remote_read：
          按 src_block_node_ids 把 block 按"来自哪个远端文件"分组，
          一次批量读多个文件。

    线程模型：与 SSD worker 一样单进程单线程、无 CUDA。
    """
    def __init__(self,
                 worker_id: int,
                 transfer_conn: Connection,
                 finished_ops_queue: MPQueue,
                 op_buffer_tensor: torch.Tensor,
                 cpu_blocks: Union[List[torch.Tensor], torch.Tensor, HugePageTensorHandle],
                 remote_file: List[str],
                 cpu_kv_layout: KVCacheLayout,
                 remote_kv_layout: KVCacheLayout,
                 dtype: torch.dtype,
                 remote_config_custom: Dict[str, Any],
                 enable_pcfs_sharing: bool = False):
        """建 PCFS 客户端并完成远端文件的 lookup/create。

        Args:
            remote_file: 远端文件列表（block 在其间 round-robin 分布）
            remote_config_custom: 必须含 pcfs_fsid / pcfs_port / pcfs_ip /
                pcfs_parent_nodeid 四项，缺一即 RuntimeError
            enable_pcfs_sharing: 读时是否走"多文件共享批量读"路径
        """
        if transfer_kv_blocks_remote is None:
            raise RuntimeError("transfer_kv_blocks_remote not available, please build with FLEXKV_ENABLE_CFS=1")
        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)
        self._pin_op_buffer()

        cpu_blocks = materialize_worker_tensor(cpu_blocks)

        self.cpu_layer_ptrs = self._get_layer_ptrs(cpu_blocks)
        self.remote_files = remote_file
        self.num_remote_files = len(remote_file)

        self.num_layers = cpu_kv_layout.num_layer
        self.num_cpu_blocks = cpu_kv_layout.num_block
        self.num_remote_blocks = remote_kv_layout.num_block
        self.round_robin = 1
        self.enable_pcfs_sharing = enable_pcfs_sharing

        if self.num_remote_blocks % self.num_remote_files != 0:
            raise ValueError(f"num_remote_blocks {self.num_remote_blocks} "
                             f"is not divisible by num_remote_files {self.num_remote_blocks}")
        self.num_remote_blocks_per_file = self.num_remote_blocks // self.num_remote_files
        if self.num_remote_blocks_per_file % self.round_robin != 0:
            raise ValueError(f"num_remote_blocks_per_file {self.num_remote_blocks_per_file} "
                             f"is not divisible by round_robin {self.round_robin}")

        self.has_multi_group = (
            getattr(cpu_kv_layout, "layer_groups", None) is not None
        )
        if self.has_multi_group:
            if (
                cpu_kv_layout.type != KVCacheLayoutType.BLOCKFIRST
                or remote_kv_layout.type != KVCacheLayoutType.BLOCKFIRST
            ):
                raise ValueError(
                    "Multi-group CPU/remote transfer requires BLOCKFIRST layouts"
                )
            # A heterogeneous BLOCKFIRST block is already one byte-flat blob.
            # Present it to the existing remote kernel as one MLA layer.
            self.block_size = cpu_kv_layout.get_block_stride()
            self.num_layers = 1
        else:
            self.block_size = cpu_kv_layout.get_chunk_size()
        self.dtype = dtype

        self.kv_dim = 1 if self.has_multi_group else cpu_kv_layout.kv_dim
        self.num_kv_heads = cpu_kv_layout.num_kv_heads

        self.cpu_blocks = cpu_blocks

        self.cpu_layer_ptrs = self._get_layer_ptrs(cpu_blocks)

        self.cpu_layer_stride_in_bytes = (
            self.num_cpu_blocks * self.block_size * self.dtype.itemsize * self.kv_dim
        )
        self.remote_layer_stride_in_bytes = (
            self.num_remote_blocks * self.block_size * self.dtype.itemsize * self.kv_dim
        )
        self.remote_layer_stride_in_bytes_per_file = self.remote_layer_stride_in_bytes // self.num_remote_files
        self.cpu_kv_stride_in_bytes = (
            self.num_cpu_blocks * self.block_size * self.dtype.itemsize
        )
        self.remote_kv_stride_in_bytes = (
            self.num_remote_blocks * self.block_size * self.dtype.itemsize
        )
        self.remote_kv_stride_in_bytes_per_file = self.remote_kv_stride_in_bytes // self.num_remote_files
        self.remote_block_stride_in_bytes = self.block_size * self.dtype.itemsize
        self.cpu_block_stride_in_bytes = self.block_size * self.dtype.itemsize

        self.chunk_size_in_bytes = self.block_size * self.dtype.itemsize
        # Bytes per KV block (all layers); used by transfer tracing for bw.
        self._bytes_per_block = self.chunk_size_in_bytes * self.num_layers * self.kv_dim
        # 144115188075855883 only use int not c_types.u_int64
        if not remote_config_custom:
            raise RuntimeError("remote_config_custom is not provided")
        pcfs_fsid = remote_config_custom.get("pcfs_fsid")
        pcfs_port = remote_config_custom.get("pcfs_port")
        pcfs_ip = remote_config_custom.get("pcfs_ip")
        pcfs_parent_nodeid = remote_config_custom.get("pcfs_parent_nodeid")
        if None in (pcfs_fsid, pcfs_port, pcfs_ip, pcfs_parent_nodeid):
            raise RuntimeError("Some required PCFS config fields are missing")
        self.pcfs = c_ext.Pcfs(pcfs_fsid, pcfs_port, pcfs_ip, False, pcfs_parent_nodeid)
        if not self.pcfs.init():
            raise RuntimeError(f"PCFS init failed: fsid={pcfs_fsid}, ip={pcfs_ip}")
        self.file_nodeid_list = []
        need_create = False
        for remote_file_single in remote_file:
            nodeid = self.pcfs.lookup_or_create_file(
            remote_file_single,
            (self.remote_layer_stride_in_bytes_per_file * self.num_layers), need_create)
            if nodeid == 0:
                raise RuntimeError(f"lookup or create file failed for file: {remote_file_single}")
            self.file_nodeid_list.append(nodeid)

        c_ext.set_pcfs_instance(self.pcfs)

    def _transfer_impl(
        self,
        src_block_ids: torch.Tensor,
        dst_block_ids: torch.Tensor,
        transfer_type: TransferType,
        **kwargs: Any
    ) -> None:
        """把 block id 列表翻译成远端读写调用。

        注意：本方法没有返回值（CPURemoteTransferWorker.launch_transfer
        末尾也确实没有 return 语句），因此基类 run() 拿到的是 None，
        会按"失败"分支 put (op_id, False, None)。记录既有行为以便排查。
        """
        assert src_block_ids.dtype == torch.int64
        assert dst_block_ids.dtype == torch.int64
        assert len(src_block_ids) == len(dst_block_ids)

        # this means partial read hit cpu and other hit remote
        # or partial write hit remote and none hit cpu

        if transfer_type == TransferType.H2REMOTE:
            remote_block_id_list = dst_block_ids
            cpu_block_id_list = src_block_ids
        elif transfer_type == TransferType.REMOTE2H:
            remote_block_id_list = src_block_ids
            cpu_block_id_list = dst_block_ids
        else:
            raise ValueError(f"Invalid transfer type: {transfer_type} for CPUSSDDiskTransferWorker")

        layer_id_list = torch.arange(0, self.num_layers, dtype=torch.int32)
                # Use PCFS shared transfer for read operations when PCFS sharing is enabled
        if self.enable_pcfs_sharing and transfer_type == TransferType.REMOTE2H:
            # For PCFS sharing, we need to construct cfs_blocks_partition and cpu_blocks_partition
            # based on the file_nodeids from the transfer operation
            # Optional: per-source-block node ids for remote routing (numpy.ndarray)
            src_block_node_ids = kwargs.get("src_block_node_ids")
            if src_block_node_ids is not None and not isinstance(src_block_node_ids, np.ndarray):
                raise TypeError("src_block_node_ids must be a numpy.ndarray if provided")

            assert len(src_block_node_ids) == len(remote_block_id_list)

            # Construct cfs_blocks_partition and cpu_blocks_partition
            # This is a simplified implementation - in practice, you might need more sophisticated logic

            # Group blocks by file_nodeid (simplified grouping logic)
            files_set = set(src_block_node_ids)
            file_nodeids_list = list(files_set)

            # Initialize partitions with proper size
            cfs_blocks_partition = [[] for _ in range(len(file_nodeids_list))]
            cpu_blocks_partition = [[] for _ in range(len(file_nodeids_list))]

            # Create mapping from file_nodeid to partition index
            file2fid_dict = {file_nodeid: fid for fid, file_nodeid in enumerate(file_nodeids_list)}
            #因为每个flexkv的文件数量是相同的，所以total_file_num是相同的，后面用全局block_id计算block_id_in_file时，需要除以total_file_num
            total_file_num = len(self.file_nodeid_list)
            for i in range(len(remote_block_id_list)):
                file_nodeid = src_block_node_ids[i]
                fid = file2fid_dict[file_nodeid]

                # Calculate block_id_in_file using the same logic as C++
                # This should match the C++ implementation in pcfs.cpp
                block_id_in_file = int(
                    ((remote_block_id_list[i] / self.round_robin) / total_file_num)
                    * self.round_robin
                    + (remote_block_id_list[i] % self.round_robin)
                )

                cfs_blocks_partition[fid].append(block_id_in_file)
                cpu_blocks_partition[fid].append(cpu_block_id_list[i].item())

            # Use the new shared transfer function
            shared_transfer_kv_blocks_remote_read(
                file_nodeids_list,
                cfs_blocks_partition,
                cpu_blocks_partition,
                layer_id_list,
                self.cpu_layer_ptrs[0].item(),
                self.cpu_layer_stride_in_bytes,
                self.cpu_kv_stride_in_bytes,
                self.remote_layer_stride_in_bytes_per_file,
                self.remote_block_stride_in_bytes,
                self.remote_kv_stride_in_bytes_per_file,
                self.chunk_size_in_bytes,
                self.num_layers,
                self.kv_dim,
                num_threads_per_file=32,
            )
        else:
            transfer_kv_blocks_remote(
                self.file_nodeid_list,
                layer_id_list,
                self.cpu_layer_ptrs[0].item(),
                remote_block_id_list,
                cpu_block_id_list,
                self.cpu_layer_stride_in_bytes,
                self.cpu_kv_stride_in_bytes,
                self.remote_layer_stride_in_bytes_per_file,
                self.remote_block_stride_in_bytes,
                self.remote_kv_stride_in_bytes_per_file,
                self.chunk_size_in_bytes,
                self.num_layers,
                (transfer_type == TransferType.REMOTE2H),
                PartitionBlockType.SEQUENTIAL.value,
                self.round_robin,
                self.num_remote_blocks_per_file,
                False,
                32,
                self.kv_dim,
            )

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> bool:
        """执行一个 H2REMOTE / REMOTE2H op，并打性能日志。

        本通路不走压缩器（压缩由上层/远端侧负责），直接调 _transfer_impl。
        注意：见 _transfer_impl 的说明，本方法当前没有 return 值。
        """
        src_block_ids, dst_block_ids = self.get_transfer_block_ids(transfer_op)

        start_time = time.time()
        self._transfer_impl(
            src_block_ids,
            dst_block_ids,
            transfer_op.transfer_type,
            src_block_node_ids=transfer_op.src_block_node_ids,
        )
        end_time = time.time()
        transfer_size = self.chunk_size_in_bytes * self.num_layers * transfer_op.valid_block_num * self.kv_dim

        self._log_transfer_performance(
            transfer_op,
            transfer_size,
            start_time,
            end_time,
        )

class GDSTransferWorker(TransferWorkerBase):
    """GPU <-> 本地 SSD **直通**通路（D2DISK / DISK2D），基于 GPUDirect Storage【旁支，默认不生效】。

    启用条件：**需要以 FLEXKV_ENABLE_GDS=1 重新编译 c_ext**。
    未开启时文件顶部的 ``from flexkv.c_ext import transfer_kv_blocks_gds``
    会 ImportError，被 fallback 成 None（连同 TPGDSTransferThreadGroup），
    本类一旦被实例化就会在调用 transfer_kv_blocks_gds 时炸掉。
    默认构建不参与运行；CPU 侧仍走 CPUSSDDiskTransferWorker。

    与 CPUSSDDiskTransferWorker 的本质差异：
        GPUDirect Storage 让 NVMe 控制器通过 **DMA 直接读写显存**，
        数据不经过 CPU 内存、不做 bounce buffer，
        省掉 GPU->CPU->磁盘路径上的一次完整拷贝和一次 CPU 侧拷贝。
        代价是依赖 nvidia-fs 内核模块和兼容的 NVMe 驱动/文件系统。

    传输机制：
        __init__ 里建 c_ext.GDSManager(ssd_files, ...)，由它持有 GDS 的
        cuFile 句柄与注册过的显存缓冲区；is_ready() 为假即抛错。
        _transfer_impl 调 transfer_kv_blocks_gds，参数里除了两侧 stride，
        还有 ssd_copy_offset（多分组时该组在 SSD block 内的字节偏移）。

    线程模型：单进程单线程 + 一条专属 transfer_stream；
    GDS 的 DMA 由驱动异步完成，c_ext 返回即完成。
    """
    def __init__(
        self,
        worker_id: int,
        transfer_conn: Connection,
        finished_ops_queue: MPQueue,
        op_buffer_tensor: torch.Tensor,
        gpu_blocks: List[TensorSharedHandle],
        ssd_files: Dict[int, List[str]],
        num_blocks_per_file: int,
        gpu_kv_layout: KVCacheLayout,
        ssd_kv_layout: KVCacheLayout,
        dtype: torch.dtype,
        gpu_device_id: int = 0,
        layer_groups: Optional[List[LayerGroupSpec]] = None,
        gpu_blocks_per_group: Optional[List[List[TensorSharedHandle]]] = None,
        gpu_layouts_per_group: Optional[List[KVCacheLayout]] = None,
    ) -> None:
        """
        Initialize GDS Transfer Worker

        中文要点：先绑 GPU、再 pin op_buffer、再导入 GPU tensor（顺序同
        GPUCPUTransferWorker），然后建 GDSManager 并校验 is_ready()，
        最后建专用 stream。GPU 侧的 stride 同样优先从实际 tensor 反推。

        Args:
            ssd_files: {ssd_device_id: [路径...]}
            gpu_blocks: GPU KV tensor 的 IPC 句柄（GDS 需要注册这些显存）
        """
        # Initialize base class first
        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)

        ensure_cuda_device(gpu_device_id)
        self._pin_op_buffer()
        self.gpu_blocks = import_tensor_handles(gpu_blocks)
        self.gpu_blocks_ptrs = self._get_layer_ptrs(self.gpu_blocks)
        self.gpu_layer_ptrs = self.gpu_blocks_ptrs
        self.num_blocks_per_file = num_blocks_per_file
        self.num_files = sum(len(file_list) for file_list in ssd_files.values())

        # Use same round_robin as SSD transfer to ensure consistent block mapping
        self.round_robin = 1
        # Create GDSManager from file paths in this worker process
        self.gds_manager = c_ext.GDSManager(
            ssd_files,
            len(ssd_files),
            self.round_robin
        )

        if not self.gds_manager.is_ready():
            raise RuntimeError(f"Failed to initialize GDS Manager in worker {worker_id}: "
                               f"{self.gds_manager.get_last_error()}")

        self.dtype = dtype
        self.kv_dim = gpu_kv_layout.kv_dim
        self.num_kv_heads = gpu_kv_layout.num_kv_heads
        self.has_multi_group = layer_groups is not None

        # Layout information
        self.num_layers = gpu_kv_layout.num_layer

        if self.has_multi_group:
            self._init_multi_group_gds(
                gpu_kv_layout, ssd_kv_layout, layer_groups,
                gpu_blocks_per_group, gpu_layouts_per_group)
        else:
            gpu_kv_layout_per_layer = gpu_kv_layout.div_layer(self.num_layers)
            ssd_kv_layout_per_file = ssd_kv_layout.div_block(self.num_files, padding=True)

            # GPU layout calculations — compute strides from actual tensor to handle
            # different attention backend layouts (flash_attn vs triton/flashinfer).
            self.chunk_size_in_bytes = gpu_kv_layout_per_layer.get_chunk_size() * self.dtype.itemsize
            # Bytes per KV block (all layers); used by transfer tracing for bw.
            self._bytes_per_block = self.chunk_size_in_bytes * self.num_layers * self.kv_dim
            gpu_strides = self._get_gpu_strides_from_tensor(
                self.gpu_blocks[0], gpu_kv_layout.tokens_per_block,
                self.dtype.itemsize, self.kv_dim,
            ) if len(self.gpu_blocks) > 1 else None
            if gpu_strides is not None:
                self.gpu_kv_stride_in_bytes = gpu_strides[0]
                self.gpu_block_stride_in_bytes = gpu_strides[1]
                self.gpu_layer_stride_in_bytes = gpu_strides[2]
            else:
                self.gpu_kv_stride_in_bytes = gpu_kv_layout.get_kv_stride() * self.dtype.itemsize
                self.gpu_block_stride_in_bytes = gpu_kv_layout.get_block_stride() * self.dtype.itemsize
                self.gpu_layer_stride_in_bytes = gpu_kv_layout.get_layer_stride() * self.dtype.itemsize

            # SSD layout calculations
            self.ssd_layer_stride_in_bytes = ssd_kv_layout_per_file.get_layer_stride() * self.dtype.itemsize
            self.ssd_kv_stride_in_bytes = ssd_kv_layout_per_file.get_kv_stride() * self.dtype.itemsize
            self.ssd_block_stride_in_bytes = ssd_kv_layout_per_file.get_block_stride() * self.dtype.itemsize

        if len(self.gpu_blocks) == 1:
            self.gpu_block_type_ = 1  # TRTLLM
        elif len(self.gpu_blocks) == self.num_layers:
            self.gpu_block_type_ = 0  # VLLM
        elif len(self.gpu_blocks) == self.num_layers * 2:
            self.gpu_block_type_ = 2  # SGLANG
        else:
            raise ValueError(f"Invalid GPU block type: {len(self.gpu_blocks)}")

        # Set GPU device and create stream
        self.gpu_device_id = gpu_device_id
        self.transfer_stream = torch.cuda.Stream()

    def _init_multi_group_gds(
        self,
        gpu_kv_layout: KVCacheLayout,
        ssd_kv_layout: KVCacheLayout,
        layer_groups: List[LayerGroupSpec],
        gpu_blocks_per_group: Optional[List[List[TensorSharedHandle]]],
        gpu_layouts_per_group: Optional[List[KVCacheLayout]],
    ) -> None:
        """Initialize per-group GDS transfer parameters.

        SSD buffer is byte-flat (uint8) in multi-group mode; per-group strides
        use g.dtype.itemsize so groups with different element sizes (e.g.
        bf16 main + uint8 indexer) interleave correctly within a block.

        中文要点：每个 group 算出自己的 GPU/SSD stride 以及 ssd_copy_offset
        （该组在 SSD block 内的起始字节偏移），_transfer_impl 里对每个 group
        各发一次 transfer_kv_blocks_gds。
        """
        kv_dim = self.kv_dim
        tpb = ssd_kv_layout.tokens_per_block

        # Multi-group BLOCKFIRST: get_block_stride() returns bytes_per_block
        # directly (already accounts for tp_size and per-group dtype sizes).
        self.ssd_block_stride_in_bytes = ssd_kv_layout.get_block_stride()

        self.group_gds_params: list = []
        # Keep imported CUDA-IPC tensors alive: _get_layer_ptrs() records only
        # raw data_ptr()s below, so dropping the tensors would free the IPC
        # mapping and dangle the stored pointers.
        self._multi_group_gpu_blocks_keepalive: list = []
        ssd_offset_bytes = 0

        for gi, g in enumerate(layer_groups):
            # Per-group dtype: indexer uses uint8 even when main KV is bf16/fp16.
            dtype_size_g = g.dtype.itemsize
            # Compressed groups: tpb_g = tpb // compress_ratio.
            tpb_g = tpb // g.compress_ratio
            chunk_elements = tpb_g * g.num_kv_heads * g.head_size
            ssd_layer_stride = kv_dim * chunk_elements * dtype_size_g
            ssd_kv_stride = chunk_elements * dtype_size_g

            # GPU strides from per-group layout — compute from actual tensor to
            # handle different attention backend layouts (flash_attn vs triton).
            if gpu_layouts_per_group is not None:
                gpu_layout = gpu_layouts_per_group[gi]
                group_gpu_blocks = import_tensor_handles(gpu_blocks_per_group[gi])
                gpu_strides = self._get_gpu_strides_from_tensor(
                    group_gpu_blocks[0], tpb_g, dtype_size_g, self.kv_dim,
                ) if len(group_gpu_blocks) > 1 else None
                if gpu_strides is not None:
                    gpu_kv_stride, gpu_block_stride, gpu_layer_stride = gpu_strides
                else:
                    gpu_kv_stride = gpu_layout.get_kv_stride() * dtype_size_g
                    gpu_block_stride = gpu_layout.get_block_stride() * dtype_size_g
                    gpu_layer_stride = gpu_layout.get_layer_stride() * dtype_size_g
                gpu_chunk_size = chunk_elements * dtype_size_g
            else:
                gpu_kv_stride = self.gpu_kv_stride_in_bytes
                gpu_block_stride = self.gpu_block_stride_in_bytes
                gpu_layer_stride = self.gpu_layer_stride_in_bytes
                gpu_chunk_size = chunk_elements * dtype_size_g

            # GPU pointers for this group
            if gpu_blocks_per_group is not None:
                group_gpu_blocks = import_tensor_handles(gpu_blocks_per_group[gi])
                self._multi_group_gpu_blocks_keepalive.append(group_gpu_blocks)
                group_gpu_ptrs = self._get_layer_ptrs(group_gpu_blocks)
            else:
                group_gpu_ptrs = self.gpu_layer_ptrs

            self.group_gds_params.append({
                'num_layers': g.num_layers,
                'gpu_ptrs': group_gpu_ptrs,
                'gpu_kv_stride': gpu_kv_stride,
                'gpu_block_stride': gpu_block_stride,
                'gpu_layer_stride': gpu_layer_stride,
                'chunk_size': gpu_chunk_size,
                'ssd_layer_stride': ssd_layer_stride,
                'ssd_kv_stride': ssd_kv_stride,
                'ssd_copy_offset': ssd_offset_bytes,
            })

            ssd_offset_bytes += g.num_layers * kv_dim * chunk_elements * dtype_size_g

        flexkv_logger.info(
            f"GDSTransferWorker multi-group initialized: {len(layer_groups)} groups, "
            f"ssd_block_stride={self.ssd_block_stride_in_bytes} bytes"
        )

    def _transfer_impl(
        self,
        src_block_ids: torch.Tensor,
        dst_block_ids: torch.Tensor,
        transfer_type: TransferType,
        **kwargs: Any,
    ) -> None:
        """Implement actual transfer between GPU and SSD

        中文要点：GPU 与 SSD 的 block id 按方向互换后，交给
        transfer_kv_blocks_gds；失败时包一层 RuntimeError 抛出，
        由基类 run() 捕获并上报 failed（而不是让异常把 worker 循环带崩）。
        """
        assert src_block_ids.dtype == torch.int64
        assert dst_block_ids.dtype == torch.int64
        assert len(src_block_ids) == len(dst_block_ids)

        # SSD uses DISK2D/D2DISK transfer types (same as traditional SSD I/O)
        if transfer_type == TransferType.DISK2D:
            ssd_block_id_list = src_block_ids
            gpu_block_id_list = dst_block_ids
        elif transfer_type == TransferType.D2DISK:
            gpu_block_id_list = src_block_ids
            ssd_block_id_list = dst_block_ids
        else:
            raise ValueError(f"Invalid transfer type: {transfer_type} for GDSTransferWorker. "
                             f"Expected DISK2D or D2DISK.")

        if len(ssd_block_id_list) == 0:
            return

        is_read = (transfer_type == TransferType.DISK2D)

        try:
            if self.has_multi_group:
                for gp in self.group_gds_params:
                    g_gpu = gpu_block_id_list
                    g_ssd = ssd_block_id_list

                    layer_id_list = torch.arange(0, gp['num_layers'], dtype=torch.int32)
                    transfer_kv_blocks_gds(
                        self.gds_manager,
                        layer_id_list,
                        gp['gpu_ptrs'],
                        g_ssd,
                        g_gpu,
                        gp['gpu_kv_stride'],
                        gp['gpu_block_stride'],
                        gp['gpu_layer_stride'],
                        gp['ssd_layer_stride'],
                        self.ssd_block_stride_in_bytes,
                        gp['ssd_kv_stride'],
                        gp['chunk_size'],
                        gp['ssd_copy_offset'],
                        self.num_blocks_per_file,
                        gp['num_layers'],
                        is_read,
                        False,
                        self.kv_dim,
                        self.gpu_block_type_,
                        self.gpu_device_id,
                    )
            else:
                # Uniform: whole-model transfer
                layer_id_list = torch.arange(0, self.num_layers, dtype=torch.int32)
                transfer_kv_blocks_gds(
                    self.gds_manager,
                    layer_id_list,
                    self.gpu_layer_ptrs,
                    ssd_block_id_list,
                    gpu_block_id_list,
                    self.gpu_kv_stride_in_bytes,
                    self.gpu_block_stride_in_bytes,
                    self.gpu_layer_stride_in_bytes,
                    self.ssd_layer_stride_in_bytes,
                    self.ssd_block_stride_in_bytes,
                    self.ssd_kv_stride_in_bytes,
                    self.chunk_size_in_bytes,
                    0,
                    self.num_blocks_per_file,
                    self.num_layers,
                    is_read,
                    False,
                    self.kv_dim,
                    self.gpu_block_type_,
                    self.gpu_device_id,
                )

        except Exception as e:
            flexkv_logger.error(f"GDS transfer failed: {e}")
            raise RuntimeError(f"Failed to transfer KV blocks: {e}") from e

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> bool:
        """Launch a GDS transfer operation

        中文要点：整段跑在 transfer_stream 上；本通路不支持压缩，
        直接调 _transfer_impl 并按 group 累加传输量打日志。
        """
        src_block_ids, dst_block_ids = self.get_transfer_block_ids(transfer_op)

        with torch.cuda.stream(self.transfer_stream):
            start_time = time.time()
            self._transfer_impl(
                src_block_ids,
                dst_block_ids,
                transfer_op.transfer_type,
            )
            end_time = time.time()

            if self.has_multi_group:
                transfer_size = 0
                for gp in self.group_gds_params:
                    transfer_size += gp['chunk_size'] * gp['num_layers'] * transfer_op.valid_block_num * self.kv_dim
            else:
                transfer_size = self.chunk_size_in_bytes * self.num_layers * transfer_op.valid_block_num * self.kv_dim

            self._log_transfer_performance(
                transfer_op,
                transfer_size,
                start_time,
                end_time,
            )
        return True


class tpGDSTransferWorker(TransferWorkerBase):
    """GPU <-> 本地 SSD 直通通路的 **TP 多卡** 版本（D2DISK / DISK2D）【旁支，默认不生效】。

    启用条件：**需要以 FLEXKV_ENABLE_GDS=1 重新编译 c_ext**（同 GDSTransferWorker）。
    未开启时 TPGDSTransferThreadGroup 为 None，本类无法工作，默认构建不生效。

    与 GDSTransferWorker 的差异（同 TP 版 GPUCPU worker 的思路）：
        - 入参 gpu_blocks[card][layer]、gpu_kv_layouts[card] 都是"每卡一份"；
        - 用 C++ 的 TPGDSTransferThreadGroup：**每卡一个线程**各自持 CUDA
          上下文与 GDS 句柄，并发下发，因为一次 GDS 传输只能作用于当前设备；
        - SSD 侧多一个 ssd_tp_stride：TP 下每卡只存 1/tp_size 的 head，
          靠它在 block 内定位本卡分片；
        - 没有 transfer_stream，并发完全由 C++ 线程组负责。
    """
    def __init__(
        self,
        worker_id: int,
        transfer_conn: Connection,
        finished_ops_queue: MPQueue,
        op_buffer_tensor: torch.Tensor,
        gpu_blocks: List[List[TensorSharedHandle]],
        ssd_files: Dict[int, List[str]],
        num_blocks_per_file: int,
        gpu_kv_layouts: List[KVCacheLayout],
        ssd_kv_layout: KVCacheLayout,
        dtype: torch.dtype,
        tp_group_size: int,
        layer_groups: Optional[List[LayerGroupSpec]] = None,
        gpu_blocks_per_group: Optional[List[List[List[TensorSharedHandle]]]] = None,
        gpu_layouts_per_group: Optional[List[List[KVCacheLayout]]] = None,
    ) -> None:
        """
        Initialize TP GDS Transfer Worker

        Args:
            worker_id: Worker ID
            transfer_queue: Queue for incoming transfer operations
            finished_ops_queue: Queue for completed operations
            gpu_blocks: List of GPU memory block handles for each GPU in TP group
            ssd_files: Dict of SSD file paths
            num_blocks_per_file: Number of blocks per file
            gpu_kv_layouts: Layout of GPU KV cache
            ssd_kv_layout: Layout of SSD KV cache
            dtype: Data type
            tp_group_size: Effective tp-group size on this node
                (``effective_tp_size_per_node`` =
                ``tp_size_per_node × cp_size_per_node``).
            layer_groups: Optional per-group KV layouts for heterogeneous models
                (including DSA/NSA indexer-as-group).

        中文要点：与 tpGPUCPUTransferWorker.__init__ 同构 ——
        绑主卡 -> pin -> 逐卡 import -> 算每卡 stride -> 建
        TPGDSTransferThreadGroup（每卡一线程）。差别在最后不需要 torch Stream。
        """
        # Initialize base class first
        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)

        assert len(gpu_blocks) == tp_group_size
        if gpu_blocks and gpu_blocks[0]:
            ensure_cuda_device(gpu_blocks[0][0].device)
        self._pin_op_buffer()
        # Handle tensor import for multi-process case — set_device per GPU first.
        imported_gpu_blocks = []
        for handles_in_one_gpu in gpu_blocks:
            imported_gpu_blocks.append(import_tensor_handles(handles_in_one_gpu))
        self.gpu_blocks = imported_gpu_blocks
        self.num_blocks_per_file = num_blocks_per_file
        self.num_files = sum(len(file_list) for file_list in ssd_files.values())

        self.dtype = dtype
        self.kv_dim = gpu_kv_layouts[0].kv_dim
        self.num_kv_heads = gpu_kv_layouts[0].num_kv_heads
        self.num_gpus = len(self.gpu_blocks)
        self.tp_group_size = tp_group_size
        self.has_multi_group = layer_groups is not None

        # Layout information
        self.num_layers = gpu_kv_layouts[0].num_layer

        if self.has_multi_group:
            self._init_tp_multi_group_gds(
                gpu_kv_layouts, ssd_kv_layout, layer_groups,
                gpu_blocks_per_group, gpu_layouts_per_group,
                ssd_files)
        else:
            ssd_kv_layout_per_file = ssd_kv_layout.div_block(self.num_files, padding=True)
            self.ssd_chunk_size_in_bytes = ssd_kv_layout_per_file.get_chunk_size() * self.dtype.itemsize
            self.chunk_size_in_bytes = self.ssd_chunk_size_in_bytes
            self.ssd_block_stride_in_bytes = ssd_kv_layout_per_file.get_block_stride() * self.dtype.itemsize
            # Bytes per KV block (all layers); used by transfer tracing for bw.
            self._bytes_per_block = self.chunk_size_in_bytes * self.num_layers * self.kv_dim
            if self.num_kv_heads > 1:
                ssd_kv_layout_per_file = ssd_kv_layout_per_file.div_head(self.tp_group_size)

            # GPU layout calculations — compute strides from actual tensor to handle
            # different attention backend layouts (flash_attn vs triton/flashinfer).
            dtype_sz = self.dtype.itemsize
            tpb = gpu_kv_layouts[0].tokens_per_block
            self.gpu_chunk_sizes_in_bytes = []
            self.gpu_kv_strides_in_bytes = []
            self.gpu_block_strides_in_bytes = []
            self.gpu_layer_strides_in_bytes = []
            for i, gpu_kv_layout in enumerate(gpu_kv_layouts):
                gpu_strides = self._get_gpu_strides_from_tensor(
                    self.gpu_blocks[i][0], tpb, dtype_sz, self.kv_dim,
                ) if len(self.gpu_blocks[i]) > 1 else None
                if gpu_strides is not None:
                    kv_s, blk_s, layer_s = gpu_strides
                else:
                    kv_s = gpu_kv_layout.get_kv_stride() * dtype_sz
                    blk_s = gpu_kv_layout.get_block_stride() * dtype_sz
                    layer_s = gpu_kv_layout.get_layer_stride() * dtype_sz
                self.gpu_chunk_sizes_in_bytes.append(gpu_kv_layout.get_chunk_size() * dtype_sz)
                self.gpu_kv_strides_in_bytes.append(kv_s)
                self.gpu_block_strides_in_bytes.append(blk_s)
                self.gpu_layer_strides_in_bytes.append(layer_s)

            # SSD layout calculations
            self.ssd_layer_stride_in_bytes = ssd_kv_layout_per_file.get_layer_stride() * self.dtype.itemsize
            self.ssd_kv_stride_in_bytes = ssd_kv_layout_per_file.get_kv_stride() * self.dtype.itemsize
            self.ssd_tp_stride_in_bytes = (self.ssd_block_stride_in_bytes // self.tp_group_size
                                           if self.num_kv_heads > 1 else self.ssd_block_stride_in_bytes)

            # Resolve pointers in Python
            gpu_block_ptrs_flat = [
                self.gpu_blocks[i][j].data_ptr()
                for i in range(self.num_gpus)
                for j in range(len(self.gpu_blocks[i]))
            ]
            gpu_device_ids = [self.gpu_blocks[i][0].device.index for i in range(self.num_gpus)]
            num_tensors_per_gpu = len(self.gpu_blocks[0])

            # Create TP GDS Transfer Thread Group
            self.tp_gds_transfer_thread_group = TPGDSTransferThreadGroup(
                self.num_gpus,
                gpu_block_ptrs_flat,
                num_tensors_per_gpu,
                ssd_files,
                self.num_layers,
                self.gpu_kv_strides_in_bytes,
                self.gpu_block_strides_in_bytes,
                self.gpu_layer_strides_in_bytes,
                self.gpu_chunk_sizes_in_bytes,
                gpu_device_ids,
            )

    def _init_tp_multi_group_gds(
        self,
        gpu_kv_layouts: List[KVCacheLayout],
        ssd_kv_layout: KVCacheLayout,
        layer_groups: List[LayerGroupSpec],
        gpu_blocks_per_group: Optional[List[List[List[TensorSharedHandle]]]],
        gpu_layouts_per_group: Optional[List[List[KVCacheLayout]]],
        ssd_files: Dict[int, List[str]],
    ) -> None:
        """Initialize per-group TPGDSTransferThreadGroup instances.

        SSD buffer is byte-flat (uint8) in multi-group mode; per-group strides
        use g.dtype.itemsize so groups with different element sizes (e.g.
        bf16 main + uint8 indexer) interleave correctly within a block.

        中文要点：每个 group 建一个独立的 TPGDSTransferThreadGroup
        （因为各组的 tensor 数与 stride 不同），并记录该组在 SSD block 内的
        ssd_copy_offset；_transfer_impl 里对每个 group 各调一次 tp_group_transfer。
        """
        kv_dim = self.kv_dim
        tpb = ssd_kv_layout.tokens_per_block

        # Multi-group BLOCKFIRST: get_block_stride() returns bytes_per_block
        # directly (already accounts for tp_size and per-group dtype sizes).
        self.ssd_block_stride_in_bytes = ssd_kv_layout.get_block_stride()

        self.group_tp_gds_params: list = []
        # Keep imported CUDA-IPC tensors alive: only data_ptr()s are recorded
        # below, so dropping the tensors would free the IPC mapping and dangle
        # the stored pointers.
        self._multi_group_gpu_blocks_keepalive: list = []
        ssd_offset_bytes = 0

        gpu_device_ids = [self.gpu_blocks[i][0].device.index for i in range(self.num_gpus)]

        for gi, g in enumerate(layer_groups):
            # Per-group dtype: indexer uses uint8 even when main KV is bf16/fp16.
            dtype_size_g = g.dtype.itemsize
            # Compressed groups: tpb_g = tpb // compress_ratio.
            tpb_g = tpb // g.compress_ratio
            chunk_elements = tpb_g * g.num_kv_heads * g.head_size
            ssd_layer_stride = kv_dim * chunk_elements * dtype_size_g
            ssd_kv_stride = chunk_elements * dtype_size_g
            # TP stride for SSD: partition the block across TP ranks
            ssd_tp_stride = self.ssd_block_stride_in_bytes // self.tp_group_size if self.num_kv_heads > 1 \
                else self.ssd_block_stride_in_bytes

            # Per-group GPU strides and pointers
            if gpu_blocks_per_group is not None and gpu_layouts_per_group is not None:
                gpu_kv_strides = []
                gpu_block_strides = []
                gpu_layer_strides = []
                gpu_chunk_sizes = []
                gpu_ptrs_flat = []
                num_tensors = None

                for gpu_idx in range(self.num_gpus):
                    grp_layout = gpu_layouts_per_group[gi][gpu_idx]
                    grp_handles = gpu_blocks_per_group[gi][gpu_idx]
                    grp_tensors = [h.get_tensor() for h in grp_handles]
                    self._multi_group_gpu_blocks_keepalive.append(grp_tensors)

                    gpu_strides = self._get_gpu_strides_from_tensor(
                        grp_tensors[0], tpb_g, dtype_size_g, self.kv_dim,
                    ) if len(grp_tensors) > 1 else None
                    if gpu_strides is not None:
                        kv_s, blk_s, layer_s = gpu_strides
                    else:
                        kv_s = grp_layout.get_kv_stride() * dtype_size_g
                        blk_s = grp_layout.get_block_stride() * dtype_size_g
                        layer_s = grp_layout.get_layer_stride() * dtype_size_g
                    gpu_kv_strides.append(kv_s)
                    gpu_block_strides.append(blk_s)
                    gpu_layer_strides.append(layer_s)
                    gpu_chunk_sizes.append(chunk_elements * dtype_size_g)

                    for t in grp_tensors:
                        gpu_ptrs_flat.append(t.data_ptr())
                    if num_tensors is None:
                        num_tensors = len(grp_tensors)
            else:
                gpu_kv_strides = []
                gpu_block_strides = []
                gpu_layer_strides = []
                for i, layout in enumerate(gpu_kv_layouts):
                    gpu_strides = self._get_gpu_strides_from_tensor(
                        self.gpu_blocks[i][0], tpb_g, dtype_size_g, self.kv_dim,
                    ) if len(self.gpu_blocks[i]) > 1 else None
                    if gpu_strides is not None:
                        kv_s, blk_s, layer_s = gpu_strides
                    else:
                        kv_s = layout.get_kv_stride() * dtype_size_g
                        blk_s = layout.get_block_stride() * dtype_size_g
                        layer_s = layout.get_layer_stride() * dtype_size_g
                    gpu_kv_strides.append(kv_s)
                    gpu_block_strides.append(blk_s)
                    gpu_layer_strides.append(layer_s)
                gpu_chunk_sizes = [chunk_elements * dtype_size_g] * self.num_gpus
                gpu_ptrs_flat = [
                    self.gpu_blocks[i][j].data_ptr()
                    for i in range(self.num_gpus)
                    for j in range(len(self.gpu_blocks[i]))
                ]
                num_tensors = len(self.gpu_blocks[0])

            tp_gds_group = TPGDSTransferThreadGroup(
                self.num_gpus,
                gpu_ptrs_flat,
                num_tensors,
                ssd_files,
                g.num_layers,
                gpu_kv_strides,
                gpu_block_strides,
                gpu_layer_strides,
                gpu_chunk_sizes,
                gpu_device_ids,
            )

            self.group_tp_gds_params.append({
                'num_layers': g.num_layers,
                'tp_gds_group': tp_gds_group,
                'ssd_layer_stride': ssd_layer_stride,
                'ssd_kv_stride': ssd_kv_stride,
                'ssd_tp_stride': ssd_tp_stride,
                'ssd_copy_offset': ssd_offset_bytes,
            })

            ssd_offset_bytes += g.num_layers * kv_dim * chunk_elements * dtype_size_g

        flexkv_logger.info(
            f"tpGDSTransferWorker multi-group initialized: {len(layer_groups)} groups, "
            f"ssd_block_stride={self.ssd_block_stride_in_bytes} bytes"
        )

    def _transfer_impl(self,
                       src_block_ids: torch.Tensor,
                       dst_block_ids: torch.Tensor,
                       transfer_type: TransferType,
                       **kwargs: Any,
                       ) -> None:
        """把 block id 列表翻译成 TPGDSTransferThreadGroup.tp_group_transfer 调用。

        多分组时每组各调一次；统一布局时一次搬完所有 layer。
        SSD 侧除了 layer/kv/block stride 还多传一个 ssd_tp_stride。
        """
        assert src_block_ids.dtype == torch.int64
        assert dst_block_ids.dtype == torch.int64
        assert len(src_block_ids) == len(dst_block_ids)

        # GDS uses DISK2D/D2DISK transfer types
        if transfer_type == TransferType.D2DISK:
            gpu_block_ids = src_block_ids
            ssd_block_ids = dst_block_ids
            is_read = False
        elif transfer_type == TransferType.DISK2D:
            gpu_block_ids = dst_block_ids
            ssd_block_ids = src_block_ids
            is_read = True
        else:
            raise ValueError(f"Invalid transfer type: {transfer_type} for tpGDSTransferWorker. "
                             f"Expected DISK2D or D2DISK.")

        gpu_block_id_list = gpu_block_ids
        ssd_block_id_list = ssd_block_ids

        assert len(gpu_block_id_list) == len(ssd_block_id_list)

        if len(gpu_block_id_list) == 0:
            return

        if self.has_multi_group:
            for gp in self.group_tp_gds_params:
                gp['tp_gds_group'].tp_group_transfer(
                    gpu_block_id_list,
                    ssd_block_id_list,
                    gp['ssd_layer_stride'],
                    gp['ssd_kv_stride'],
                    self.ssd_block_stride_in_bytes,
                    gp['ssd_tp_stride'],
                    self.num_blocks_per_file,
                    is_read,
                    0,  # layer_id always 0 for per-group
                    gp['num_layers'],
                    self.kv_dim,
                    self.num_kv_heads,
                )
        else:
            self.tp_gds_transfer_thread_group.tp_group_transfer(
                gpu_block_id_list,
                ssd_block_id_list,
                self.ssd_layer_stride_in_bytes,
                self.ssd_kv_stride_in_bytes,
                self.ssd_block_stride_in_bytes,
                self.ssd_tp_stride_in_bytes,
                self.num_blocks_per_file,
                is_read,
                0,
                self.num_layers,
                self.kv_dim,
                self.num_kv_heads,
            )

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> bool:
        """Launch a TP GDS transfer operation

        中文要点：不绑 torch stream（并发由 C++ 每卡线程完成）；
        不支持压缩，直接调 _transfer_impl 后按 group 累加传输量打日志。
        """
        src_block_ids, dst_block_ids = self.get_transfer_block_ids(transfer_op)

        start_time = time.time()
        self._transfer_impl(
            src_block_ids,
            dst_block_ids,
            transfer_op.transfer_type,
        )
        end_time = time.time()

        if self.has_multi_group:
            transfer_size = 0
            for gp in self.group_tp_gds_params:
                transfer_size += gp['ssd_kv_stride'] * gp['num_layers'] * transfer_op.valid_block_num * self.kv_dim
        else:
            transfer_size = self.ssd_chunk_size_in_bytes * self.num_layers * transfer_op.valid_block_num * self.kv_dim

        self._log_transfer_performance(
            transfer_op,
            transfer_size,
            start_time,
            end_time,
        )

        return True


class NixlTransferWorker(TransferWorkerBase):
    """KV cache transfer via NIXL FILE backends: GDS_MT (GPU↔file) or POSIX / 3FS (CPU↔file).

    Both ``gpu_kv_layout`` and ``cpu_kv_layout`` are required so GPU, CPU, and SSD (per-file)
    byte strides are always defined; only the tensors needed for the chosen backend must be
    provided (``gpu_blocks`` for GDS_MT, ``cpu_blocks`` for POSIX/3FS).

    中文补充 —— 定位与启用条件【旁支，默认不生效】：
        本类把 SSD 这一级的 I/O 外包给 **NIXL**（NVIDIA 的跨存储/网络
        传输库），按 nixl_backend 分成两条完全不同的通路：
          - GDS_MT：GPU <-> 文件，transfer_type 用 DISK2D / D2DISK
          - POSIX / 3FS：CPU <-> 文件，transfer_type 用 DISK2H / H2DISK
        启用条件：需要 NIXL 后端可用 + 配置里指定 nixl_backend。
        默认构建（未启用 NIXL）这条通路不参与运行。

    与其它 worker 的最大差异：
        其它 worker 都是"一次 pybind 调用把所有 block 搬完"；
        本类在 Python 侧**逐 block × 逐 layer × 逐 kv** 展开成
        (地址, 长度, 文件路径, 文件内偏移) 的平铺列表，再一次性交给
        NixlAgentSession.xfer_vram_file / xfer_dram_file。
        即"Python 算描述、NIXL 批量执行"。
    """

    def __init__(
        self,
        worker_id: int,
        transfer_conn: Connection,
        finished_ops_queue: MPQueue,
        op_buffer_tensor: torch.Tensor,
        nixl_backend: str,
        ssd_files: Dict[int, List[str]],
        num_blocks_per_file: int,
        dtype: torch.dtype,
        ssd_kv_layout: KVCacheLayout,
        gpu_kv_layout: KVCacheLayout,
        cpu_kv_layout: KVCacheLayout,
        nixl_extra_config: Optional[Dict[str, Any]] = None,
        gpu_blocks: Optional[List[TensorSharedHandle]] = None,
        cpu_blocks: Optional[torch.Tensor] = None,
        gpu_device_id: int = 0,
    ) -> None:
        """校验后端合法性、算三侧 stride、建 NixlAgentSession 并注册内存/文件。

        GDS_MT 分支：绑 GPU -> import GPU tensor -> prepare_all_ssd_files
            -> prepare_vram_gpu（把显存注册给 NIXL）-> 建 transfer_stream。
        POSIX/3FS 分支：把 CPU 池 pin 住 -> prepare_dram_cpu -> 不需要 CUDA。

        Note:
            _pin_op_buffer 必须排在 ensure_cuda_device 之后（见基类说明），
            所以这里先判断后端再绑卡。
        """
        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)

        be = normalize_nixl_file_plugin_name(str(nixl_backend).upper())
        if be not in NIXL_GPU_FILE_BACKENDS and be not in NIXL_CPU_FILE_BACKENDS:
            raise ValueError(
                f"nixl_backend must be one of "
                f"{sorted(NIXL_GPU_FILE_BACKENDS | NIXL_CPU_FILE_BACKENDS)}, got {nixl_backend}"
            )
        if be in NIXL_GPU_FILE_BACKENDS and gpu_blocks is None:
            raise ValueError("GDS_MT requires gpu_blocks")
        if be in NIXL_CPU_FILE_BACKENDS and cpu_blocks is None:
            raise ValueError("POSIX/3FS require cpu_blocks")

        # Pin after optional GPU bind so GPU backends do not create a GPU0 context.
        if be in NIXL_GPU_FILE_BACKENDS:
            ensure_cuda_device(gpu_device_id)
        self._pin_op_buffer()
        if (
            gpu_kv_layout.num_layer != cpu_kv_layout.num_layer
            or gpu_kv_layout.kv_dim != cpu_kv_layout.kv_dim
            or gpu_kv_layout.num_kv_heads != cpu_kv_layout.num_kv_heads
        ):
            raise ValueError(
                "gpu_kv_layout and cpu_kv_layout must match on num_layer, "
                "kv_dim and num_kv_heads"
            )

        self.nixl_backend = be
        self.ssd_files = ssd_files
        self.num_blocks_per_file = num_blocks_per_file
        self.num_files = sum(len(fl) for fl in ssd_files.values())
        self.num_devices = len(ssd_files)
        self.num_files_per_device = len(ssd_files[0])
        self.round_robin = 1
        self.dtype = dtype

        self.num_layers = gpu_kv_layout.num_layer
        self.kv_dim = gpu_kv_layout.kv_dim
        self.num_kv_heads = gpu_kv_layout.num_kv_heads

        # SSD / file-side layout (same for every NIXL FILE backend).
        ssd_pf = ssd_kv_layout.div_block(self.num_files, padding=True)
        self.ssd_layer_stride_in_bytes = (
            ssd_pf.get_layer_stride() * self.dtype.itemsize
        )
        self.ssd_kv_stride_in_bytes = ssd_pf.get_kv_stride() * self.dtype.itemsize
        self.ssd_block_stride_in_bytes = (
            ssd_pf.get_block_stride() * self.dtype.itemsize
        )

        # GPU pool strides (per tensor layout).
        gpu_pl = gpu_kv_layout.div_layer(self.num_layers)
        self.gpu_chunk_size_in_bytes = gpu_pl.get_chunk_size() * self.dtype.itemsize
        self.gpu_kv_stride_in_bytes = (
            gpu_kv_layout.get_kv_stride() * self.dtype.itemsize
        )
        self.gpu_block_stride_in_bytes = (
            gpu_kv_layout.get_block_stride() * self.dtype.itemsize
        )
        self.gpu_layer_stride_in_bytes = (
            gpu_kv_layout.get_layer_stride() * self.dtype.itemsize
        )

        # CPU pool strides (DRAM side for POSIX / 3FS).
        self.cpu_chunk_size_in_bytes = (
            cpu_kv_layout.get_chunk_size() * self.dtype.itemsize
        )
        self.mem_block_stride_in_bytes = (
            cpu_kv_layout.get_block_stride() * self.dtype.itemsize
        )
        self.mem_kv_stride_in_bytes = (
            cpu_kv_layout.get_kv_stride() * self.dtype.itemsize
        )
        self.mem_layer_stride_in_bytes = (
            cpu_kv_layout.get_layer_stride() * self.dtype.itemsize
        )

        self._session = NixlAgentSession(be, nixl_extra_config or {})

        if be in NIXL_GPU_FILE_BACKENDS:
            self.gpu_blocks = import_tensor_handles(gpu_blocks)  # type: ignore[arg-type]
            if len(self.gpu_blocks) == 1:
                self.gpu_block_type_ = 1
            elif len(self.gpu_blocks) == self.num_layers:
                self.gpu_block_type_ = 0
            elif len(self.gpu_blocks) == self.num_layers * 2:
                self.gpu_block_type_ = 2
            else:
                raise ValueError(
                    f"Invalid GPU block count for NIXL: {len(self.gpu_blocks)}"
                )
            self.chunk_size_in_bytes = self.gpu_chunk_size_in_bytes
            self.gpu_device_id = gpu_device_id
            self.transfer_stream = torch.cuda.Stream()
            if not self._session.prepare_all_ssd_files(self.ssd_files):
                raise RuntimeError("NIXL: prepare_all_ssd_files failed")
            if not self._session.prepare_vram_gpu(self.gpu_blocks):
                raise RuntimeError("NIXL: prepare_vram_gpu failed")
        else:
            self.cpu_blocks = cpu_blocks  # type: ignore[assignment]
            flexkv_logger.info(
                f"NixlTransferWorker ({be}): pinning CPU pool "
                f"{cpu_blocks.numel() * cpu_blocks.element_size() / (1024 ** 3):.2f} GiB"
            )
            self._register_host_tensor(cpu_blocks, "nixl_cpu_pool")  # type: ignore[arg-type]
            if cpu_kv_layout.type != ssd_kv_layout.type:
                raise ValueError(
                    "CPU and SSD KV layout types must match for NIXL FILE transfer"
                )
            self.chunk_size_in_bytes = self.cpu_chunk_size_in_bytes
            if not self._session.prepare_all_ssd_files(self.ssd_files):
                raise RuntimeError("NIXL: prepare_all_ssd_files failed")
            if not self._session.prepare_dram_cpu(self.cpu_blocks):
                raise RuntimeError("NIXL: prepare_dram_cpu failed")

        # Bytes per KV block (all layers); used by transfer tracing for bw.
        self._bytes_per_block = self.chunk_size_in_bytes * self.num_layers * self.kv_dim

    def _transfer_impl(
        self,
        src_block_ids: torch.Tensor,
        dst_block_ids: torch.Tensor,
        transfer_type: TransferType,
        layer_id: int,
        layer_granularity: int,
        **kwargs: Any,
    ) -> None:
        """把 block id 列表展开成 NIXL 的批量传输描述并执行。

        支持按层切片（layer_id / layer_granularity），这是本类区别于其它
        worker 的地方：只搬某几层而不是整块，便于做分层流水。

        展开方式：对每个 block、每个 layer、每个 kv，分别算出
        - GPU 侧：gpu_chunk_u8_view 切出该 chunk 的 uint8 视图（GDS_MT）
          或 CPU 侧 kv_chunk_byte_offset_in_block 算出地址（POSIX/3FS）
        - 文件侧：ssd_chunk_byte_offset_in_file 算出文件内偏移
        最后一次性提交给 NIXL；失败抛出，由基类 run() 捕获上报。
        """
        assert src_block_ids.dtype == torch.int64
        assert dst_block_ids.dtype == torch.int64
        assert len(src_block_ids) == len(dst_block_ids)

        if layer_id == -1:
            layer_id = 0
        if layer_granularity == -1:
            layer_granularity = self.num_layers

        if self.nixl_backend in NIXL_GPU_FILE_BACKENDS:
            if transfer_type == TransferType.DISK2D:
                ssd_block_ids, mem_block_ids = src_block_ids, dst_block_ids
                direction = "READ"
            elif transfer_type == TransferType.D2DISK:
                mem_block_ids, ssd_block_ids = src_block_ids, dst_block_ids
                direction = "WRITE"
            else:
                raise ValueError(
                    f"GDS_MT NixlTransferWorker expects DISK2D or D2DISK, got {transfer_type}"
                )
        else:
            if transfer_type == TransferType.DISK2H:
                ssd_block_ids, mem_block_ids = src_block_ids, dst_block_ids
                direction = "READ"
            elif transfer_type == TransferType.H2DISK:
                mem_block_ids, ssd_block_ids = src_block_ids, dst_block_ids
                direction = "WRITE"
            else:
                raise ValueError(
                    f"POSIX/3FS NixlTransferWorker expects DISK2H or H2DISK, got {transfer_type}"
                )

        n = ssd_block_ids.numel()
        if n == 0:
            return

        kv_dim = self.kv_dim
        layer_end = layer_id + layer_granularity

        file_paths: List[str] = []
        region_offsets: List[int] = []
        region_lens: List[int] = []

        if self.nixl_backend in NIXL_GPU_FILE_BACKENDS:
            gpu_tensors: List[torch.Tensor] = []
            for i in range(n):
                ssd_b = int(ssd_block_ids[i].item())
                mem_b = int(mem_block_ids[i].item())
                path, block_in_file = file_path_for_ssd_block(
                    self.ssd_files,
                    ssd_b,
                    self.num_devices,
                    self.num_files_per_device,
                    self.round_robin,
                )
                for lid in range(layer_id, layer_end):
                    for kv in range(kv_dim):
                        sob = ssd_chunk_byte_offset_in_file(
                            lid,
                            kv,
                            block_in_file,
                            self.ssd_layer_stride_in_bytes,
                            self.ssd_kv_stride_in_bytes,
                            self.ssd_block_stride_in_bytes,
                            self.kv_dim,
                        )
                        gview = gpu_chunk_u8_view(
                            self.gpu_blocks,
                            self.gpu_block_type_,
                            self.num_layers,
                            mem_b,
                            lid,
                            kv,
                            self.gpu_kv_stride_in_bytes,
                            self.gpu_block_stride_in_bytes,
                            self.gpu_layer_stride_in_bytes,
                            self.chunk_size_in_bytes,
                            self.kv_dim,
                        )
                        gpu_tensors.append(gview)
                        file_paths.append(path)
                        region_offsets.append(sob)
                        region_lens.append(self.chunk_size_in_bytes)

            ok = self._session.xfer_vram_file(
                direction, gpu_tensors, file_paths, region_lens, region_offsets
            )
            if not ok:
                raise RuntimeError("NIXL GDS_MT transfer failed")
            torch.cuda.synchronize()
        else:
            base = self.cpu_blocks.data_ptr()
            dram_ptr_len: List[Tuple[int, int]] = []
            for i in range(n):
                ssd_b = int(ssd_block_ids[i].item())
                mem_b = int(mem_block_ids[i].item())
                path, block_in_file = file_path_for_ssd_block(
                    self.ssd_files,
                    ssd_b,
                    self.num_devices,
                    self.num_files_per_device,
                    self.round_robin,
                )
                for lid in range(layer_id, layer_end):
                    for kv in range(kv_dim):
                        cob = kv_chunk_byte_offset_in_block(
                            lid,
                            kv,
                            mem_b,
                            self.mem_layer_stride_in_bytes,
                            self.mem_kv_stride_in_bytes,
                            self.mem_block_stride_in_bytes,
                            self.kv_dim,
                        )
                        sob = ssd_chunk_byte_offset_in_file(
                            lid,
                            kv,
                            block_in_file,
                            self.ssd_layer_stride_in_bytes,
                            self.ssd_kv_stride_in_bytes,
                            self.ssd_block_stride_in_bytes,
                            self.kv_dim,
                        )
                        dram_ptr_len.append((base + cob, self.chunk_size_in_bytes))
                        file_paths.append(path)
                        region_offsets.append(sob)
                        region_lens.append(self.chunk_size_in_bytes)

            ok = self._session.xfer_dram_file(
                direction, dram_ptr_len, file_paths, region_lens, region_offsets
            )
            if not ok:
                raise RuntimeError(f"NIXL {self.nixl_backend} CPU↔file transfer failed")

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> bool:
        """执行一个 NIXL 文件传输 op，可只搬 [layer_id, +layer_granularity) 这几层。

        GDS_MT 需要在 transfer_stream 上跑（NIXL 的 VRAM 传输走 CUDA）；
        POSIX/3FS 是纯 CPU 路径，用 nullcontext 跳过流绑定。
        """
        lid = transfer_op.layer_id
        lg = transfer_op.layer_granularity
        if lid == -1:
            lid = 0
        if lg == -1:
            lg = self.num_layers

        src_block_ids, dst_block_ids = self.get_transfer_block_ids(transfer_op)

        # GDS_MT runs NIXL VRAM xfers on a dedicated stream; POSIX/3FS are CPU-only.
        stream_ctx = (
            torch.cuda.stream(self.transfer_stream)
            if self.nixl_backend in NIXL_GPU_FILE_BACKENDS
            else contextlib.nullcontext()
        )
        with stream_ctx:
            start_time = time.time()
            self._transfer_impl(
                src_block_ids,
                dst_block_ids,
                transfer_op.transfer_type,
                lid,
                lg,
            )
            end_time = time.time()
            kv_dim = self.kv_dim
            transfer_size = (
                self.chunk_size_in_bytes * lg * transfer_op.valid_block_num * kv_dim
            )
            self._log_transfer_performance(
                transfer_op, transfer_size, start_time, end_time
            )
        return True


class PEER2CPUTransferWorker(TransferWorkerBase):
    """跨机 P2P 通路：**对端的 CPU / SSD -> 本地 CPU**（PEERH2H / PEERSSD2H）【旁支，默认不生效】。

    启用条件：**需要 cache_config.enable_kv_sharing=1 + Mooncake + Redis**
    （enable_p2p_ssd 还要额外开 SSD 相关配置）。未启用时 __init__ 里
    Mooncake/Redis 那一段整体跳过，本 worker 退化成什么都不做。

    ┌── 一句话总览 ─────────────────────────────────────────────────────────┐
    │ 本类是全文件最复杂的 worker，因为它同时握着 **控制面** 和 **数据面**：│
    │   控制面 = ZMQ（交换元数据 / 通知），数据面 = RDMA（真正搬字节）。    │
    └──────────────────────────────────────────────────────────────────────┘

    控制面（ZMQ + Redis）：
        - Redis：节点注册表。每个节点把自己的 mooncake engine 地址、
          CPU/SSD 缓冲基址、ZMQ 监听地址写进 Redis（regist_node_meta），
          带 TTL 心跳；取值前先 is_node_active() 校验，防止往已挂节点 RDMA。
        - ZMQ：任务级信令。
            SSDZMQServer（本地监听 local_zmq_port，回调 ssd_handle_loop）
            SSDZMQClient（发往对端 local_zmq_port）
          用来传 RemoteSSD2HMetaInfo（要哪些 block、写到我哪、完成后通知谁）
          和 NotifyMsg（成功/失败回执）。

    数据面（RDMA / Mooncake）：
        - PEERH2H（对端 CPU -> 本地 CPU）：本端**主动**单向 RDMA read
          （transfer_sync_read / batch_transfer_sync_read）。控制面只在
          传输前用一次 Redis 取对端地址。
        - PEERSSD2H（对端 SSD -> 本地 CPU）：数据在对端磁盘上、本端够不着，
          所以改成"请对端代劳"——本端用 ZMQ 发一份 meta 给对端，
          对端的 ssd_handle_loop 收到后自己 io_uring 读盘到它的 hugepage
          临时缓冲，再**单向 RDMA write** 推到本端 CPU，最后 ZMQ 发回执。
          即：一次读 = 两次 ZMQ + 一次 RDMA write。

    为什么需要 hugepage 临时缓冲（tmp_cpu_buffer）：
        对端帮你读盘时，数据得先落在一块"本站可被 RDMA 直接读"的内存里。
        这块缓冲通过 allocate_host_buffer(use_hugepage=...) 分配：
        hugepage 能显著降低大块 RDMA 的 TLB miss；分配后同样要
        regist_buffer 注册给 Mooncake 才能被对端 RDMA 访问。
        它只有 num_tmp_cpu_blocks 个 block 大，所以超过这个数量的请求
        会被 ssd_handle_loop 直接拒绝（见其中的 TODO）。

    完成回调路径：
        launch_transfer -> op_parser 按"对端节点"切成多个 RDMATaskInfo
        -> 逐个 _batch_transfer_impl -> 汇总成 bool 返回
        -> 基类 run() put 进 finished_ops_queue。
        注意 PEERSSD2H 的完成与否取决于对端 ZMQ 回执（wait_transfer_notify），
        并且有 RDMA_TRANSFER_TIMEOUT_SECONDS 兜底，防止对端失联时永久阻塞。

    本地行为 vs 远端行为的分工（本类方法可按此分组）：
        本地（主动发起）：op_parser / _dist_cpu_op_parser / _dist_ssd_op_parser
                          / _batch_transfer_impl / launch_transfer
        远端（被动服务）：ssd_handle_loop / copy_ssd_data_to_dram
                          / write_data_back_to_peer / meta_info_parser
        公共：get_cpu_buffer_block_start_ptr / gen_task_id / Redis 三件套
    """
    def __init__(self,
        worker_id: int,
        transfer_conn: Connection,
        finished_ops_queue: MPQueue,
        op_buffer_tensor: torch.Tensor,
        cpu_blocks: Union[torch.Tensor, HugePageTensorHandle],
        cpu_kv_layout: KVCacheLayout,
        remote_kv_layout: KVCacheLayout,
        dtype: torch.dtype,
        cache_config: CacheConfig,
        ssd_kv_layout: KVCacheLayout = None,
        ssd_files: Dict[int, List[str]] = None,  # ssd_device_id -> file_paths
        num_blocks_per_file: int = 0,
        mooncake_config_path: str = None,
    ):
        """初始化顺序（step1~step4，注释里也是这么标号的）：

            step1：建 Redis 客户端，连上节点信息表并扫描活跃节点
                   （必须先 connect，否则 is_node_active 永远为假）
            step2：建 Mooncake 传输引擎（config 优先取参数 > cache_config >
                   环境变量，因为 spawn 的子进程可能丢环境变量）
            step3：把本地 CPU 池 regist_buffer 给 Mooncake
                   （不注册对端就 RDMA 不到这块内存）
            step3.5（enable_p2p_ssd 时）：分配 hugepage tmp 缓冲、
                   起 ZMQ server/client、建 io_uring ioctx
            step4：把本节点的元信息注册进 Redis
                   —— 必须在 step3.5 之后，这样注册的 ssd_buffer_base_ptr
                      才是 tmp_cpu_buffer 的真实地址

        注意：本类**不** pin CPU 池给 CUDA（见 MooncakeStoreTransferWorker 的
        说明），但 Mooncake 会自己做 RDMA 内存注册。
        """
        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)
        self._pin_op_buffer()
        cpu_blocks = materialize_worker_tensor(cpu_blocks)
        self.cpu_layer_ptrs = self._get_layer_ptrs(cpu_blocks)
        self.num_layers = cpu_kv_layout.num_layer
        self.num_cpu_blocks = cpu_kv_layout.num_block
        # For multi-group layouts, get_chunk_size() is invalid;
        # use get_block_stride() which works for both single and multi-group BLOCKFIRST.
        if getattr(cpu_kv_layout, "layer_groups", None) is not None:
            self.block_size = cpu_kv_layout.get_block_stride()
        else:
            self.block_size = cpu_kv_layout.get_chunk_size()
        self.dtype = dtype
        self.cpu_kv_layout = cpu_kv_layout
        self.remote_kv_layout = remote_kv_layout

        self.kv_dim = cpu_kv_layout.kv_dim
        self.num_kv_heads = cpu_kv_layout.num_kv_heads
        # Bytes per KV block (all layers); used by transfer tracing for bw.
        self._bytes_per_block = self.block_size * self.dtype.itemsize * self.num_layers * self.kv_dim

        self.cpu_blocks = cpu_blocks  ## shared memory
        self.cache_config = cache_config
        self.dst_buffer_ptr = self.cpu_blocks.data_ptr()

        self.mooncake_transfer_engine = None
        # self.zmq_listen_addr = ""

        self.zmq_listen_addr = (
            f"tcp://{cache_config.local_zmq_ip}:{cache_config.local_zmq_port}"
        )

        ## initialize distributed environment
        if self.cache_config.enable_kv_sharing:
            # step1: initialize the redis meta client for node info
            self.redis_meta_client = RedisMeta(
                self.cache_config.redis_host,
                self.cache_config.redis_port,
                self.cache_config.redis_password,
                self.cache_config.local_ip,
                node_ttl_seconds=getattr(self.cache_config, 'node_ttl_seconds', 0),
            )
            self.redis_meta_client.set_node_id(self.cache_config.distributed_node_id)

            # Connect nodeinfo so the listener/heartbeat threads start and
            # current_node_id_set is populated — required for is_node_active()
            # checks during P2P transfers.
            if not self.redis_meta_client.nodeinfo.connect():
                flexkv_logger.warning(
                    "PEER2CPUTransferWorker: failed to connect RedisNodeInfo listener"
                )
            else:
                self.redis_meta_client.nodeinfo.scan_active_nodes()

            # Persistent NodeMetaInfo Pool for skip redis operation when getting
            # NodeMetaInfo according to node_id
            # assuming that every flexkv progress has unique node id
            self.node_metas: Dict[int, NodeMetaInfo] = {}
            assert self.redis_meta_client is not None


            # step2: initialize mooncake transfer engine for the whole flexkv
            # NOTE: prefer explicit parameter > cache_config > env variable
            # (spawn subprocesses may lose env vars, but cache_config is pickle-serialized)
            if mooncake_config_path is None:
                mooncake_config_path = getattr(self.cache_config, 'mooncake_config_path', None)
            if mooncake_config_path is None:
                mooncake_config_path = os.environ.get("MOONCAKE_CONFIG_PATH")
            if mooncake_config_path is None:
                raise RuntimeError(
                    "MOONCAKE_CONFIG_PATH is not set. Please either pass mooncake_config_path "
                    "parameter, set cache_config.mooncake_config_path, or set the "
                    "MOONCAKE_CONFIG_PATH environment variable."
                )
            self.mooncake_config = MooncakeTransferEngineConfig.from_file(
                mooncake_config_path
            )
            self.mooncake_transfer_engine = MoonCakeTransferEngineWrapper(
                self.mooncake_config
            )
            assert (
                self.mooncake_transfer_engine is not None
            ), "PEER2CPUTransferWorker: initilaize mooncake transfer engine failed"

            # step3: register local cpu buffer to mooncake transfer engine
            total_cpu_blocks_size = (
                self.cpu_blocks.numel() * self.cpu_blocks.element_size()
            )
            regist_buffer_status = self.mooncake_transfer_engine.regist_buffer(
                self.cpu_blocks.data_ptr(), total_cpu_blocks_size
            )
            assert (
                regist_buffer_status == 0
            ), "PEER2CPUTransferWorker: regist cpu buffer to mooncake transfer engine"

        ## when enable p2p ssd, we need start a zmq server to recive the meta info from remote node,
        # and allocate a cpu buffer for ssd to cpu copy
        if self.cache_config.enable_p2p_ssd:
            assert ssd_kv_layout is not None, "Invalid ssd kv layout!"
            ## init the cpu buffer for ssd to cpu copy
            # NOTE: now we allocate 500 blocks for test
            self.tmp_cpu_buffer_layout = KVCacheLayout(
                type=self.cpu_kv_layout.type,
                num_layer=self.cpu_kv_layout.num_layer,
                num_block=self.cache_config.num_tmp_cpu_blocks,
                tokens_per_block=self.cpu_kv_layout.tokens_per_block,
                num_head=self.cpu_kv_layout.num_head,
                head_size=self.cpu_kv_layout.head_size,
                kv_dim=self.cpu_kv_layout.kv_dim,
                num_kv_heads=self.cpu_kv_layout.num_kv_heads,
                _kv_shape=self.cpu_kv_layout.kv_shape,
            )
            # Allocate the temporary SSD->CPU staging buffer.
            #
            # Two backends are supported:
            #  (a) HugePage-backed mmap (when ``cache_config.use_hugepage_tmp_buffer``
            #      is True and the kernel has huge pages reserved). We still need
            #      to pin it for CUDA via ``cudaHostRegister`` because the region
            #      is not allocated through PyTorch's pinned-memory allocator.
            #  (b) Pinned ``torch.empty`` (the original behavior, default).
            tmp_num_elements = self.tmp_cpu_buffer_layout.get_total_elements()
            self._tmp_cpu_buffer_handle = allocate_host_buffer(
                num_elements=tmp_num_elements,
                dtype=self.dtype,
                use_hugepage=self.cache_config.use_hugepage_tmp_buffer,
                hugepage_size_bytes=self.cache_config.hugepage_size_bytes,
            )
            self.tmp_cpu_buffer = self._tmp_cpu_buffer_handle.tensor

            self.mooncake_transfer_engine.regist_buffer(
                self.tmp_cpu_buffer.data_ptr(),
                self.tmp_cpu_buffer.numel() * self.tmp_cpu_buffer.element_size(),
            )

            ## start the zmq server and client
            self.zmq_server = SSDZMQServer(cache_config.local_zmq_ip, cache_config.local_zmq_port, self.ssd_handle_loop)
            self.zmq_client = SSDZMQClient(cache_config.local_zmq_ip, cache_config.local_zmq_port+1)

            ## ssd copy to temp cpu buffer related
            self.ssd_files = ssd_files
            self.num_blocks_per_file = num_blocks_per_file
            self.num_files = sum(len(file_list) for file_list in ssd_files.values())

            ssd_kv_layout_per_file = ssd_kv_layout.div_block(self.num_files, padding=True)

            self.chunk_size_in_bytes = (
                self.tmp_cpu_buffer_layout.get_chunk_size() * self.dtype.itemsize
            )
            self.block_stride_in_bytes = (
                self.tmp_cpu_buffer_layout.get_block_stride() * self.dtype.itemsize
            )
            self.cpu_kv_stride_in_bytes = (
                self.tmp_cpu_buffer_layout.get_kv_stride() * self.dtype.itemsize
            )
            self.cpu_layer_stride_in_bytes = (
                self.tmp_cpu_buffer_layout.get_layer_stride() * self.dtype.itemsize
            )
            self.ssd_kv_stride_in_bytes = (
                ssd_kv_layout_per_file.get_kv_stride() * self.dtype.itemsize
            )
            self.ssd_layer_stride_in_bytes = (
                ssd_kv_layout_per_file.get_layer_stride() * self.dtype.itemsize
            )


            self.round_robin = 1
            # initialize ssd ioctx
            try:
                self.ioctx = c_ext.SSDIOCTX(
                    ssd_files,
                    len(ssd_files),
                    GLOBAL_CONFIG_FROM_ENV.iouring_entries,
                    GLOBAL_CONFIG_FROM_ENV.iouring_flags,
                )
            except Exception as e:
                flexkv_logger.error(f"Error setting ssd ioctx: {e}\n")
                raise RuntimeError("SSD Worker init failed") from e

        ## step4: regist node info into redis server
        ## Must be done after P2P SSD init so we can register the correct
        ## ssd_buffer_base_ptr (tmp_cpu_buffer) when P2P SSD is enabled.
        if self.cache_config.enable_kv_sharing:
            ssd_buffer_ptr = (
                self.tmp_cpu_buffer.data_ptr()
                if self.cache_config.enable_p2p_ssd
                else 0
            )
            self.regist_node_meta(
                self.cpu_blocks.data_ptr(),
                ssd_buffer_ptr,
                self.zmq_listen_addr,
            )

        ## unique task id counter for remote ssd to cpu transfer task
        self.remote_ssd_task_id_counter = 0
        self.task_id_lock = threading.Lock()

    #============================ common part ========================
    def gen_task_id(self) -> int:
        """
        generate a unique task id for remote ssd to cpu transfer task
        Returns:
            int: task id

        中文要点：task_id 用于把"我发的请求"和"对端回的回执"对上号
        （wait_transfer_notify 按 (peer_engine_addr, task_id) 匹配），
        所以它必须在本进程内单调递增且加锁保护。
        """
        with self.task_id_lock:
            old_value = self.remote_ssd_task_id_counter
            self.remote_ssd_task_id_counter += 1
            return old_value

    def shutdown(self):
        """Best-effort cleanup; tolerant of partially-failed ``__init__``.

        中文要点：本类持有的外部资源比其它 worker 多得多，必须逐个拆掉，
        且每个都可能不存在（__init__ 中途失败时），所以全程 getattr + try：
            ZMQ server/client -> Mooncake 注销 CPU 池 -> 注销 tmp 缓冲
            -> 释放 hugepage handle -> Redis 摘掉本节点元信息
        最后**必须**调 super().shutdown() 解绑 op_buffer。
        """
        try:
            zmq_server = getattr(self, "zmq_server", None)
            if zmq_server is not None:
                zmq_server.shutdown()
            zmq_client = getattr(self, "zmq_client", None)
            if zmq_client is not None:
                zmq_client.shutdown()
            engine = getattr(self, "mooncake_transfer_engine", None)
            cpu_blocks = getattr(self, "cpu_blocks", None)
            if engine is not None and cpu_blocks is not None:
                try:
                    engine.unregist_buffer(cpu_blocks.data_ptr())
                except Exception as e:
                    flexkv_logger.warning(
                        f"PEER2CPUTransferWorker unregist cpu buffer failed: {e}"
                    )
                cache_config = getattr(self, "cache_config", None)
                if cache_config is not None and getattr(cache_config, "enable_p2p_ssd", False):
                    tmp_buf = getattr(self, "tmp_cpu_buffer", None)
                    if tmp_buf is not None:
                        try:
                            engine.unregist_buffer(tmp_buf.data_ptr())
                        except Exception as e:
                            flexkv_logger.warning(
                                f"PEER2CPUTransferWorker unregist tmp buffer failed: {e}"
                            )
                    tmp_handle = getattr(self, "_tmp_cpu_buffer_handle", None)
                    if tmp_handle is not None:
                        try:
                            tmp_handle.release()
                        except Exception as e:
                            flexkv_logger.warning(
                                f"PEER2CPUTransferWorker release tmp handle failed: {e}"
                            )
            if getattr(self, "redis_meta_client", None) is not None:
                try:
                    self.unregist_node_meta()
                except Exception as e:
                    flexkv_logger.warning(
                        f"PEER2CPUTransferWorker unregist_node_meta failed: {e}"
                    )
        except Exception as e:
            flexkv_logger.error(f"PEER2CPUTransferWorker shutdown error: {e}")
        finally:
            super().shutdown()

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> bool:
        """执行一个 PEERH2H / PEERSSD2H op（本地主动侧入口）。

        与其它 worker 不同：一次 op 可能横跨**多个对端节点**，
        所以 op_parser 先按节点拆成多个 RDMATaskInfo，这里串行逐个执行；
        任何一个节点失败就整体判定失败（但已传成功的部分不会回滚）。

        返回 False 时基类 run() 会 put (op_id, False)，上层据此回退或重算。
        """
        task_info_list = self.op_parser(transfer_op)

        start_time = time.time()
        transfered_size = 0
        transfer_finished = True

        for task_info in task_info_list:
            # NOTE: here one task_info represent data transfer from one node
            ret = self._batch_transfer_impl(
                task_info,
                transfer_op.transfer_type,
            )
            if not ret:
                transfer_finished = False
                break
            transfered_size += task_info.data_size

        end_time = time.time()

        self._log_transfer_performance(
            transfer_op,
            transfered_size,
            start_time,
            end_time,
        )
        return transfer_finished

    # Timeout for a single RDMA batch transfer (seconds).
    # Prevents indefinite blocking when a remote node becomes unreachable
    # but its node:<id> TTL hasn't expired yet.
    RDMA_TRANSFER_TIMEOUT_SECONDS = 30

    def _batch_transfer_impl(self,
        task_info: RDMATaskInfo,
        transfer_type: TransferType,
        **kwargs,):
        """执行一个"到某个对端节点"的批量传输（launch_transfer 实际调用的版本）。

        PEERH2H：调 Mooncake 的 batch_transfer_sync_read 一次性搬完该节点的
            所有段。这里特意包一层 ThreadPoolExecutor 是为了**能超时**：
            Mooncake 的同步读在对端失联但 Redis TTL 未过期时会永久阻塞，
            靠 future.result(timeout=RDMA_TRANSFER_TIMEOUT_SECONDS) 兜底。
        PEERSSD2H：数据在对端磁盘上，本端搬不动，改为
            step1 构造 RemoteSSD2HMetaInfo（含写到我哪个 CPU 地址、
            完成后通知我哪个 ZMQ 地址）-> step2 ZMQ 发给对端
            -> step3 阻塞等回执。（真正的搬运动作发生在对端的
            ssd_handle_loop 里：读盘 + RDMA write 推回来。）
        """
        if transfer_type == TransferType.PEERH2H:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self.mooncake_transfer_engine.batch_transfer_sync_read,
                    task_info.peer_engine_addr, task_info.src_ptrs, task_info.dst_ptrs, task_info.data_lens
                )
                try:
                    ret = future.result(timeout=self.RDMA_TRANSFER_TIMEOUT_SECONDS)
                except concurrent.futures.TimeoutError:
                    flexkv_logger.error(
                        f"RDMA batch transfer to {task_info.peer_engine_addr} timed out "
                        f"after {self.RDMA_TRANSFER_TIMEOUT_SECONDS}s"
                    )
                    return False
            if ret != 0:
                flexkv_logger.error(f"RDMA transfer failed with error code: {ret}")
                return False
        elif transfer_type == TransferType.PEERSSD2H:
          # remote ssd to local cpu transfer by two side zmq and one side rdma write
            # step1: construct the meta info
            remote_ssd_to_cpu_meta = RemoteSSD2HMetaInfo(
                task_id=task_info.task_id,
                cpu_block_ids=task_info.dst_block_ids,
                ssd_block_ids=task_info.src_block_ids,
                peer_engine_addr=task_info.local_engine_addr,
                peer_cpu_base_ptr=self.dst_buffer_ptr,
                peer_zmq_status_addr=self.zmq_client.get_addr(),
                data_size=task_info.data_size,
            )
            #flexkv_logger.info(
            #    f"[PEERSSD2H] Sending meta: task_id={task_info.task_id}, "
            #    f"ssd_block_ids={task_info.src_block_ids}, cpu_block_ids={task_info.dst_block_ids}, "
            #    f"peer_engine_addr={task_info.local_engine_addr}, peer_zmq_addr={task_info.peer_zmq_addr}"
            #)
            ## step2: send the meta info to remote node
            if not self.zmq_client.send_meta_info(remote_ssd_to_cpu_meta, task_info.peer_zmq_addr):
                flexkv_logger.error(
                    f"Send remote ssd to cpu meta info to {task_info.peer_zmq_addr} failed"
                )
                return False

            ## step3: wait for remote node to send data transfer complete notify
            ret = self.zmq_client.wait_transfer_notify(
                task_info.peer_engine_addr, task_info.task_id
            )
            if not ret:
                flexkv_logger.error(
                    f"Wait remote ssd to cpu transfer task {task_info.task_id} "
                    f"notify from {task_info.peer_engine_addr} failed with error code: {ret}"
                )
                return False
        else:
            raise ValueError(
                f"Invalid transfer type: {transfer_type} for PEER2CPUTransferWorker"
            )
        return True

    def _transfer_impl(
        self,
        task_info: RDMATaskInfo,
        transfer_type: TransferType,
        **kwargs,
    ):
        """单段（非批量）版本：与 _batch_transfer_impl 流程一致，
        但 PEERH2H 走的是逐指针的 transfer_sync_read（一个指针一次调用），
        而 _batch_transfer_impl 用 batch_transfer_sync_read 一次搞定。
        当前 launch_transfer 走的是 batch 版本。
        """
        if transfer_type == TransferType.PEERH2H:
            # remote cpu to local cpu transfer by one-side rdma read
            for i in range(len(task_info.src_ptrs)):
                ret = self.mooncake_transfer_engine.transfer_sync_read(
                    task_info.peer_engine_addr,
                    task_info.src_ptrs[i],
                    task_info.dst_ptrs[i],
                    task_info.data_lens[i],
                )
                if ret != 0:
                    flexkv_logger.error(f"transfer_sync_write failed with error code: {ret}")
                    return False
        elif transfer_type == TransferType.PEERSSD2H:
            # remote ssd to local cpu transfer by two side zmq and one side rdma write
            # step1: construct the meta info
            remote_ssd_to_cpu_meta = RemoteSSD2HMetaInfo(
                task_id=task_info.task_id,
                cpu_block_ids=task_info.dst_block_ids,
                ssd_block_ids=task_info.src_block_ids,
                peer_engine_addr=task_info.local_engine_addr,
                peer_cpu_base_ptr=self.dst_buffer_ptr,
                peer_zmq_status_addr=self.zmq_client.get_addr(),
                data_size=task_info.data_size,
            )
            flexkv_logger.info(
                f"[_transfer_impl] Sending task_id={task_info.task_id}, "
                f"ssd_block_ids={task_info.src_block_ids}, "
                f"cpu_block_ids={task_info.dst_block_ids} to {task_info.peer_zmq_addr}"
            )
            ## step2: send the meta info to remote node
            if not self.zmq_client.send_meta_info(remote_ssd_to_cpu_meta, task_info.peer_zmq_addr):
                flexkv_logger.error(
                    f"Send remote ssd to cpu meta info to {task_info.peer_zmq_addr} failed"
                )
                return False

            ## step3: wait for remote node to send data transfer complete notify
            ret = self.zmq_client.wait_transfer_notify(
                task_info.peer_engine_addr, task_info.task_id
            )
            if not ret:
                flexkv_logger.error(
                    f"Wait remote ssd to cpu transfer task {task_info.task_id} "
                    f"notify from {task_info.peer_engine_addr} failed with error code: {ret}"
                )
                return False

        else:
            raise ValueError(
                f"Invalid transfer type: {transfer_type} for PEER2CPUTransferWorker"
            )

        return True

    def op_parser(
        self, transfer_op: WorkerTransferOp
    ) -> List[RDMATaskInfo]:
        """
        parse the transfer op to a list of RDMATaskInfo
        1. group the blocks by remote node id, each segment is a list of
           continuous blocks (segment is the smallest transmission unit)
        2. using corresponding distributed op parser to parse the op and create RDMATaskInfo for each segment
        5. return the list of RDMATaskInfo
        Parameters:
            transfer_op (WorkerTransferOp): the transfer op to be parsed
        Returns:
            List[RDMATaskInfo]: the list of RDMATaskInfo

        中文要点：这是"逻辑地址 -> 物理 RDMA 描述"的翻译层。
        PEERH2H 用 group_blocks_by_node_and_segment（先按节点、再按**连续段**
        分组，一段一次 RDMA，减少请求数）；
        PEERSSD2H 只用 group_blocks_by_node（不分段，因为整包交给对端去读）。
        """
        assert (
            transfer_op.transfer_type == TransferType.PEERH2H
            or transfer_op.transfer_type == TransferType.PEERSSD2H
        ), f"PEER2CPUTransferWorker only support PEERH2H or PEERSSD2H, but get {transfer_op.transfer_type}"

        src_block_ids, dst_block_ids = self.get_transfer_block_ids(transfer_op, False)

        assert len(src_block_ids) == len(dst_block_ids)

        src_block_node_ids = transfer_op.src_block_node_ids
        # Convert to plain list — the compiled utils.so (pybind11) requires list,
        # not numpy.ndarray.
        if hasattr(src_block_node_ids, 'tolist'):
            src_block_node_ids = src_block_node_ids.tolist()

        # step1: group the blocks by remote node id and remote block source type,
        # each segment is a list of continuous blocks
        #flexkv_logger.info(
        #    f"[PEER2CPUTransferWorker] src_block_ids: {src_block_ids} \n \
        #                        dst_block_ids: {dst_block_ids} \n \
        #                        src_block_node_ids: {src_block_node_ids} \n"
        #)
        task_info_list = []

        if transfer_op.transfer_type == TransferType.PEERH2H:
            groups = group_blocks_by_node_and_segment(
                src_block_ids, dst_block_ids, src_block_node_ids
            )
            task_info_list = self._dist_cpu_op_parser(groups)
        elif transfer_op.transfer_type == TransferType.PEERSSD2H:
            groups = group_blocks_by_node(
                src_block_ids, dst_block_ids, src_block_node_ids
            )
            task_info_list = self._dist_ssd_op_parser(groups)
        else:
            raise RuntimeError(
                f"Unsurpported transfer_type {transfer_op.transfer_type} in PEER2CPUTransferWorker"
            )

        return task_info_list

    #========================== distrbuted ssd related ==========================
    #========================== local behaviors
    def _dist_ssd_op_parser(self, groups: Dict[int, Dict[str, List[int]]]):
        """
        Distributed ssd op parser
        1. for each segment, get the remote ssd blocks and local cpu blocks
        2. create RDMATaskInfo for each segment
        Args:
            groups (Dict[int, Dict[str, List[int]]]): the grouped blocks

        Returns:
            task_info_list: the list of RDMATaskInfo, each task refers to one data transfer operation

        中文要点：SSD 场景下 RDMATaskInfo 里只需带 block id（不填指针）,
        因为实际搬运在对端完成；peer_engine_addr 这里填的是**对端的**
        engine 地址（用于 ZMQ 寻址），而 task_id 用来等对端回执。
        """
        ## parse ssd
        # TODO: now we only support blockwise layout, need support layerwise layout

        task_info_list = []

        for node_id, segment in groups.items():
            ##NOTE: for ssd scenario, each node will only have one set of src and dst block ids
            peer_node_info = self.get_node_meta(node_id)
            if peer_node_info is None:
                return []
            peer_zmq_addr = peer_node_info.zmq_addr
            peer_engine_addr = peer_node_info.engine_addr
            assert (
                peer_zmq_addr != ""
            ), f"Node {node_id} zmq addr not found in redis server"

            src_blocks = segment["src"]
            dst_blocks = segment["dst"]
            assert len(src_blocks) == len(dst_blocks)

            data_size = self.cpu_kv_layout.get_block_stride() * self.dtype.itemsize * len(src_blocks)
            ssd_task_id = self.gen_task_id()
            task_info_list.append(
                RDMATaskInfo(
                    ssd_task_id,
                    self.mooncake_transfer_engine.get_engine_addr(),
                    # for ssd transfer, peer engine addr refers to local mooncake engine
                    peer_engine_addr,
                    peer_zmq_addr,
                    None,
                    None,
                    src_blocks,
                    dst_blocks,
                    [], # not used in ssd transfer
                    data_size=data_size
                )
            )
        return task_info_list


    #=============================remote behaviors

    def meta_info_parser(self, recv_msg: str):
        """控制面：把对端发来的 JSON 反序列化成 RemoteSSD2HMetaInfo。"""
        recv_dict = json.loads(recv_msg)
        return RemoteSSD2HMetaInfo.from_dict(recv_dict)

    def ssd_handle_loop(self):
        """**远端被动侧**的主循环：替对端把本地 SSD 数据读出来并 RDMA 推回去。

        由 SSDZMQServer 在独立线程里驱动（见 __init__ 里传入的回调）。

        处理一个请求的四个阶段：
            step1 收 ZMQ 消息 -> meta_info_parser 反序列化 -> 立刻回 "OK"
                  （先应答，避免对端死等）
            step2 校验：block 数合法、且不超过 num_tmp_cpu_blocks
                  （超限目前直接回失败，代码里有对应 TODO）
            step3 copy_ssd_data_to_dram：用 io_uring 把 SSD 数据读到本地
                  hugepage 临时缓冲（tmp_cpu_buffer）；同时按"最长连续段"
                  切分（split_contiguous_blocks），一段一次批量 RDMA
            step4 write_data_back_to_peer：batch_transfer_sync_write
                  **单向 RDMA 写**把数据推进对端 CPU 缓冲 -> ZMQ 发回执

        无论成功失败都必须发回执（包括异常分支），否则对端的
        wait_transfer_notify 会一直挂着，整个 graph 永远完不成。
        """
        flexkv_logger.info(
            f"Node {self.cache_config.distributed_node_id} Listening on {self.zmq_listen_addr}"
        )
        while not self.zmq_server.shutdown_event.is_set():
            recv_meta = None
            failure_msg = None
            try:
                ## step1: recv and parse the message into meta info
                try:
                    message = self.zmq_server.listen_socket.recv().decode("utf-8")
                except zmq.Again:
                    time.sleep(0.001)
                    continue
                if not message:
                    self.zmq_server.listen_socket.send(b"ERROR")
                    continue

                recv_meta = self.meta_info_parser(message)
                if not recv_meta:
                    self.zmq_server.listen_socket.send(b"ERROR")
                    flexkv_logger.warning("Can not parse RemoteSSD2HMetaInfo using recieved message")
                    continue

                flexkv_logger.info(
                    f"[ssd_handle_loop] Received task_id={recv_meta.task_id}, "
                    f"ssd_block_ids={recv_meta.ssd_block_ids}, "
                    f"cpu_block_ids={recv_meta.cpu_block_ids}"
                )

                self.zmq_server.listen_socket.send(b"OK")

                failure_msg = NotifyMsg(
                    mooncake_engine_addr=self.mooncake_transfer_engine.get_engine_addr(),
                    task_id=recv_meta.task_id,
                    status=NotifyStatus.FAIL,
                )
                success_msg = NotifyMsg(
                    mooncake_engine_addr=self.mooncake_transfer_engine.get_engine_addr(),
                    task_id=recv_meta.task_id,
                    status=NotifyStatus.SUCCESS,
                )

                # step2: ckeck the recieved info, early return if check error
                nvtx_range = nvtx.start_range(message="ssd_handle_loop. check and load_data", color="orange")
                if len(recv_meta.ssd_block_ids) == 0 or len(recv_meta.cpu_block_ids) == 0 \
                    or len(recv_meta.cpu_block_ids)!=len(recv_meta.ssd_block_ids):
                        flexkv_logger.warning(
                            "Invalid cpu_block_ids or ssd_block_ids, skipping this transfer..."
                        )
                        self.zmq_server.send_transfer_status(recv_meta.peer_zmq_status_addr, failure_msg)
                        continue

                # TODO: we need to support dynamic temp buffer or split the ssd
                # transfer request if number of ssd blocks is larger than
                # num_tmp_cpu_blocks. Now we just refuse this transfer by
                # returning a failure status.
                if len(recv_meta.ssd_block_ids)>self.cache_config.num_tmp_cpu_blocks:
                    flexkv_logger.warning(
                            f"The number of ssd_block_ids is larger than "
                            f"{self.cache_config.num_tmp_cpu_blocks}, can not do transfer now"
                        )
                    self.zmq_server.send_transfer_status(recv_meta.peer_zmq_status_addr, failure_msg)
                    continue

                ## step3: do copy data from ssd to cpu
                # NOTE: this block ids is a corresponding relationship with
                # self.tmp_cpu_buffer, for every transfer req we reuse the local cpu buffer
                local_cpu_buffer_block_ids = torch.arange(0, len(recv_meta.ssd_block_ids), dtype = torch.int64)
                local_cpu_start_idx = 0

                # seperate the blocks to get the longest continuous blocks
                groups = split_contiguous_blocks(recv_meta.ssd_block_ids, recv_meta.cpu_block_ids)

                all_copy_complete = True
                src_ptr_list = []
                dst_ptr_list = []
                data_size_list = []

                for item in groups:
                    # in this loop we do two things:
                    # 1. copy ssd data to cpu for each segment
                    # 2. calculate the start ptr of local cpu blocks and dst cpu blocks for each segment and record them
                    ssd_block_ids_per_seg = torch.tensor(item["src"], dtype=torch.int64)
                    dst_cpu_block_ids_per_seg = torch.tensor(item["dst"], dtype=torch.int64)

                    if len(ssd_block_ids_per_seg) == 0:
                        all_copy_complete = False
                        break
                    # get corresponding temp cpu block ids
                    local_cpu_buffer_block_ids_per_seg = local_cpu_buffer_block_ids[
                        local_cpu_start_idx: local_cpu_start_idx + len(ssd_block_ids_per_seg)
                    ]
                    local_cpu_start_idx += len(ssd_block_ids_per_seg)

                    layer_id_list = torch.arange(
                        0, self.num_layers, dtype=torch.int32
                    )
                    if not self.copy_ssd_data_to_dram(
                        layer_id_list, ssd_block_ids_per_seg, local_cpu_buffer_block_ids_per_seg
                    ):
                        flexkv_logger.error("Copy ssd data to dram failed!")
                        all_copy_complete = False
                        break

                    src_ptrs, src_block_size = self.get_cpu_buffer_block_start_ptr(
                        local_cpu_buffer_block_ids_per_seg,
                        self.tmp_cpu_buffer.data_ptr(),
                    )

                    dst_ptrs, dst_block_size = self.get_cpu_buffer_block_start_ptr(
                        dst_cpu_block_ids_per_seg,
                        recv_meta.peer_cpu_base_ptr,
                    )
                    assert src_block_size == dst_block_size, "Block size mismatch between src and dst"

                    for _ in range(len(src_ptrs)):
                        data_size_list.append(src_block_size * len(local_cpu_buffer_block_ids_per_seg))
                    src_ptr_list.extend(src_ptrs)
                    dst_ptr_list.extend(dst_ptrs)
                    assert len(src_ptr_list) == len(data_size_list) and len(dst_ptr_list) == len(data_size_list)

                nvtx.end_range(nvtx_range)
                nvtx_range = nvtx.start_range(message="ssd_handle_loop. write_data_back_to_peer", color="orange")
                ## step4: do rdma transfer and send notify
                if not all_copy_complete:
                    self.zmq_server.send_transfer_status(recv_meta.peer_zmq_status_addr, failure_msg)
                    continue

                if not self.write_data_back_to_peer(
                    recv_meta.peer_engine_addr, src_ptr_list, dst_ptr_list, data_size_list
                ):
                    self.zmq_server.send_transfer_status(recv_meta.peer_zmq_status_addr, failure_msg)
                    flexkv_logger.error("Failed to write data back to peer")
                    continue

                self.zmq_server.send_transfer_status(recv_meta.peer_zmq_status_addr, success_msg)
                nvtx.end_range(nvtx_range)
            except Exception as e:
                flexkv_logger.error(f"Unexpected error in ssd_handle_loop: {e}")
                # Send failure notify so the peer doesn't block waiting forever
                try:
                    if recv_meta is not None:
                        self.zmq_server.send_transfer_status(
                            recv_meta.peer_zmq_status_addr, failure_msg
                        )
                except Exception:
                    pass
                time.sleep(0.001)

    def copy_ssd_data_to_dram(
        self, layer_id_list: torch.Tensor, ssd_block_id_list: torch.Tensor, cpu_block_id_list: torch.Tensor
    ):
        """数据面（远端侧）：用 io_uring 把本地 SSD 的 block 读进 tmp_cpu_buffer。

        走的是和 CPUSSDDiskTransferWorker 同一个 transfer_kv_blocks_ssd，
        区别是目标缓冲是 hugepage 临时缓冲而不是 CPU KV 池——
        因为这些数据马上要被 RDMA 推走，不占 KV 池的 block。
        """
        assert len(ssd_block_id_list) == len(cpu_block_id_list)
        flexkv_logger.info(f"copy ssd blocks:{ssd_block_id_list} to cpu blocks: {cpu_block_id_list}" )
        try:
            transfer_kv_blocks_ssd(
                self.ioctx,
                layer_id_list,
                self.tmp_cpu_buffer.data_ptr(),  ## copy ssd data to tmp cpu buffer
                ssd_block_id_list,
                cpu_block_id_list,
                self.cpu_layer_stride_in_bytes,
                self.cpu_kv_stride_in_bytes,
                self.ssd_layer_stride_in_bytes,
                self.ssd_kv_stride_in_bytes,
                self.chunk_size_in_bytes,
                self.block_stride_in_bytes,
                True,
                self.num_blocks_per_file,
                self.round_robin,
                32,
                self.kv_dim,
                ssd_io_opt=GLOBAL_CONFIG_FROM_ENV.ssd_io_opt,
            )
        except Exception as e:
            flexkv_logger.error(f"Copy data from ssd to cpu failed: {e}")
            return False
        return True

    def write_data_back_to_peer(
        self,
        peer_address: str,
        src_ptr_list: List[int],
        dst_ptr_list: List[int],
        data_size_list: List[int]
    ):
        """数据面（远端侧）：**单向 RDMA write** 把临时缓冲里的数据推进对端 CPU。

        注意方向：PEERH2H 是本端 read 对端，PEERSSD2H 是本端（这里指服务方）
        write 对端。用 write 是因为数据在本端手上、对端的目标地址由
        meta 里的 peer_cpu_base_ptr 给出。
        """
        flexkv_logger.info(
            f"Write data back to peer from src: {src_ptr_list} to {dst_ptr_list}"
        )
        ret = self.mooncake_transfer_engine.batch_transfer_sync_write(
            peer_address, src_ptr_list, dst_ptr_list, data_size_list
        )
        return ret == 0


    #============================== distrbuted cpu related ==========================

    def _dist_cpu_op_parser(
        self,
        groups: Dict[int, List[Dict[str, List[int]]]],
    ):
        """
        Distributed cpu op parser
        1. for each segment, get the remote cpu ptrs and local cpu ptrs
        2. create RDMATaskInfo for each segment

        Inputs:
            groups (Dict[int, List[Dict[str, List[int]]]]): the grouped blocks

        Returns:
            task_info_list: the list of RDMATaskInfo, each task refers to the data transfer of one node
        """

        task_info_list = []

        for node_id, segments in groups.items():
            # step1: get the remote meta info
            src_meta = self.get_node_meta(node_id)
            if src_meta is None:
                # Skip this node's blocks instead of aborting all nodes.
                # In multi-node P2P, one dead node should not prevent fetching
                # blocks from other healthy nodes.
                flexkv_logger.warning(
                    f"[PEER2CPUTransferWorker] Skipping node {node_id}: "
                    f"meta unavailable, will skip {len(segments)} segment(s)"
                )
                continue
            peer_engine_addr = src_meta.engine_addr
            src_ptr_list = []
            dst_ptr_list = []
            data_size_list = []
            for seg in segments:
                src_blocks = seg["src"]
                dst_blocks = seg["dst"]

                # step2: calculate the src and dst block start ptrs
                src_block_start_ptrs, src_data_size_per_block = (
                    self.get_cpu_buffer_block_start_ptr(
                        src_blocks,
                        src_meta.cpu_bufer_base_ptr,  # the cpu buffer ptr on remote machine
                    )
                )


                dst_block_start_ptrs, dst_data_size_per_block = (
                    self.get_cpu_buffer_block_start_ptr(
                        dst_blocks,
                        self.dst_buffer_ptr,  # the cpu buffer ptr on local machine
                    )
                )

                assert (
                    src_data_size_per_block == dst_data_size_per_block
                ), "src and dst blocks have different layout"


                for _ in range(len(src_block_start_ptrs)):
                    data_size = src_data_size_per_block * len(src_blocks)
                    data_size_list.append(data_size)
                src_ptr_list.extend(src_block_start_ptrs)
                dst_ptr_list.extend(dst_block_start_ptrs)
                assert len(data_size_list) == len(src_ptr_list) and len(data_size_list) == len(dst_ptr_list)

            flexkv_logger.info(
                f"[PEER2CPUTransferWorker]: remote cpu op parser "
                f"src_ptr_list: {src_ptr_list}, dst_ptr_list: {dst_ptr_list} "
            )
            # step3: create RDMATaskInfo for each segment
            # NOTE: block wise layout: only one start ptr for each segment
            #       layer wise layout: multiple start ptrs for each segment,
            #       the number of start ptrs equals num_layers * kv_dim
            task_info_list.append(
                  RDMATaskInfo(
                    0,
                    "",
                    peer_engine_addr,
                    "",
                    src_ptr_list,
                    dst_ptr_list,
                    [],    # src_block_ids unused for PEERH2H (uses ptrs)
                    [],    # dst_block_ids unused for PEERH2H (uses ptrs)
                    data_size_list,
                    data_size = sum(data_size_list)
                )
            )

        return task_info_list

    #================================== utils =================================
    def get_cpu_buffer_block_start_ptr(
        self,
        cpu_blocks: List[int],
        cpu_base_ptr: int,
    ) -> Tuple[List[int], int]:
        """
        Get the cpu buffer block start ptrs for the given cpu blocks.
        We have two layout types in flexkv, layerwise and blockwise.
        1) For layerwise layout, although the cpu blocks are continuous, we need to
        calculate the start ptrs for each layer and each kv dim. So
        the number of start ptrs equals self.num_layers * kv_dim.
        2) For blockwise layout, the cpu blocks are continuous, so we only need to
        calculate the start ptr for the first block. So
        the number of start ptrs is 1.
        3) For other layout types, raise error.

        Parameters:
            cpu_blocks (List[int]): the list of cpu block ids, continuous
            cpu_base_ptr (int): the base ptr of the cpu buffer
        Returns:
            Tuple(List[int], int): the list of cpu buffer block start ptrs and
                data size per block (used for calculate total data size)
        """

        # assuming that remote cpu buffer layout is the same as local cpu buffer layout
        assert self.cpu_kv_layout.type == self.remote_kv_layout.type
        src_block_ptrs = []

        # Get the first block ID and handle different input types
        if isinstance(cpu_blocks, torch.Tensor):
            block_id_int = int(cpu_blocks[0].item())
        elif isinstance(cpu_blocks, list):
            first_elem = cpu_blocks[0]
            if isinstance(first_elem, torch.Tensor):
                block_id_int = int(first_elem.item())
            else:
                block_id_int = int(first_elem)
        else:
            raise ValueError(f"Invalid cpu_blocks type: {type(cpu_blocks)}")

        if self.cpu_kv_layout.type == KVCacheLayoutType.LAYERFIRST:
            for layer_id in range(0, self.num_layers):
                for kv_id in range(self.kv_dim):
                    element_offset = (
                        (
                            ((layer_id * self.kv_dim) + kv_id)
                            * self.cpu_kv_layout.num_block
                            + block_id_int
                        )
                        * self.cpu_kv_layout.get_block_stride()
                        * self.dtype.itemsize
                    )
                    src_block_ptrs.append(cpu_base_ptr + element_offset)

        elif self.cpu_kv_layout.type == KVCacheLayoutType.BLOCKFIRST:
            block_volume = self.cpu_kv_layout.get_block_stride()
            element_offset = block_id_int * block_volume * self.dtype.itemsize
            src_block_ptrs.append(cpu_base_ptr + element_offset)
        else:
            raise ValueError(f"Invalid KVCacheLayoutType: {self.cpu_kv_layout.type}")
        data_size_per_block = self.cpu_kv_layout.get_block_stride() * self.dtype.itemsize

        return  src_block_ptrs, data_size_per_block

    ### redis client helper functions
    def regist_node_meta(
        self, cpu_buffer_base_ptr: int, ssd_buffer_base_ptr: int, zmq_addr: str
    ):
        self.redis_meta_client.regist_node_meta(
            self.redis_meta_client.get_node_id(),
            self.mooncake_transfer_engine.get_engine_addr(),
                                                zmq_addr, cpu_buffer_base_ptr, ssd_buffer_base_ptr)
        #NOTE: maybe useless
        node_meta_info = NodeMetaInfo(
            self.redis_meta_client.get_node_id(),
            self.mooncake_transfer_engine.get_engine_addr(),
            zmq_addr,
            cpu_buffer_base_ptr,
            ssd_buffer_base_ptr
        )
        self.node_metas[self.redis_meta_client.get_node_id()] = node_meta_info
        flexkv_logger.info(f"Registered node {self.redis_meta_client.get_node_id()} to Redis.")

    def unregist_node_meta(self, node_id: int = None) -> None:
        self.redis_meta_client.unregist_node_meta(self.redis_meta_client.get_node_id())
        flexkv_logger.info(f"Unregistered node {self.redis_meta_client.get_node_id()} from Redis.")

    def get_node_meta(self, node_id: int) -> Optional[NodeMetaInfo]:
        """Get the node meta info by node id.

        Before returning cached or freshly-fetched meta, we verify that the
        node is still active (its node:<id> key exists in Redis and has not
        expired).  This prevents RDMA transfers to stale addresses after a
        remote node has crashed.
        """
        # ===== Active-node validation (Scheme 4) =====
        if not self.redis_meta_client.is_node_active(node_id):
            # Node is no longer active – purge cached meta if any
            if node_id in self.node_metas:
                del self.node_metas[node_id]
                flexkv_logger.warning(
                    f"Node {node_id} is no longer active, removed cached meta."
                )
            else:
                flexkv_logger.warning(
                    f"Node {node_id} is not active, skipping meta fetch."
                )
            return None

        if node_id not in self.node_metas:
            ## fetch from redis
            node_redis_data = self.redis_meta_client.get_node_meta(node_id)
            if not node_redis_data:
                flexkv_logger.error(f"Node {node_id} meta not found in Redis.")
                return None

            node_meta = NodeMetaInfo.from_dict(node_redis_data)

            self.node_metas[node_id] = node_meta
            flexkv_logger.info(f"Fetched node {node_id} meta from Redis.")

        return self.node_metas[node_id]

class MooncakeStoreTransferWorker(TransferWorkerBase):
    """Mooncake-store remote KV I/O worker (main KV and SWA pools)."""

    def __init__(
        self,
        worker_id: int,
        transfer_conn: Connection,
        finished_ops_queue: MPQueue,
        op_buffer_tensor: torch.Tensor,
        cpu_blocks: Union[List[torch.Tensor], torch.Tensor, HugePageTensorHandle],
        cpu_kv_layout: "KVCacheLayout",
        dtype: torch.dtype,
        cache_config: "CacheConfig",
        pool_kind: PoolKind = PoolKind.KV,
        override_global_segment_size: Optional[int] = None,
    ) -> None:
        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)
        self.pp_rank = int(getattr(cache_config, 'mooncake_store_pp_rank', 0) or 0)
        self.pp_size = int(getattr(cache_config, 'mooncake_store_pp_size', 1) or 1)
        self.node_layer_start = int(getattr(cache_config, 'mooncake_store_node_layer_start', 0) or 0)
        self.node_layer_end = int(getattr(cache_config, 'mooncake_store_node_layer_end', 0) or 0)
        self.total_layers = int(getattr(cache_config, 'mooncake_store_total_layers', 0) or 0)
        self.pool_kind = pool_kind

        mapped_size = (
            int(cpu_blocks.aligned)
            if isinstance(cpu_blocks, HugePageTensorHandle)
            else None
        )
        cpu_blocks = materialize_worker_tensor(cpu_blocks)
        # Mooncake owns the RDMA registration and this worker only uses host
        # pointers. A second CUDA/HIP registration of the same shared pool is
        # redundant and can exhaust the host mapping budget for large caches.
        flexkv_logger.info(
            "[MooncakeStoreTransferWorker] skip CUDA host registration for "
            "the CPU KV pool; Mooncake owns the external MR"
        )
        self.cpu_layer_ptrs = self._get_layer_ptrs(cpu_blocks)
        self.num_layers: int = cpu_kv_layout.num_layer
        self.num_cpu_blocks: int = cpu_kv_layout.num_block
        self.dtype = dtype
        self.cpu_kv_layout = cpu_kv_layout
        assert self.cpu_kv_layout.type == KVCacheLayoutType.BLOCKFIRST
        self.kv_dim = cpu_kv_layout.kv_dim
        self.num_kv_heads = cpu_kv_layout.num_kv_heads
        self.cpu_blocks = cpu_blocks
        self.cache_config = cache_config
        self._cpu_buffer = cpu_blocks[0] if isinstance(cpu_blocks, (list, tuple)) else cpu_blocks
        # Opaque whole-block I/O: multi-group CPU layout is byte-flat
        # ([num_block, bytes_per_block]); get_chunk_size() is invalid there.
        self.block_size_bytes = self._block_size_bytes(cpu_kv_layout, dtype)
        # Bytes per KV block (all layers); used by transfer tracing for bw.
        self._bytes_per_block = self.block_size_bytes

        from flexkv.external.mooncake_store_utils import MooncakeStoreClient, MooncakeStoreConfig
        store_config = MooncakeStoreConfig.from_file(
            self.cache_config,
            override_global_segment_size=override_global_segment_size,
        )
        self.mooncake_client = MooncakeStoreClient(store_config)
        self._mooncake_registered_regions: List[Tuple[int, int]] = []
        base_ptr = self._cpu_buffer.data_ptr()
        logical_size = self._cpu_buffer.numel() * self._cpu_buffer.element_size()
        if mapped_size is None:
            regions = [(base_ptr, logical_size)]
        else:
            hugepage_size = int(self.cache_config.hugepage_size_bytes)
            size_alignment = int(
                os.getenv(
                    "FLEXKV_HUGEPAGE_MAPPING_ALIGNMENT_BYTES",
                    str(hugepage_size),
                )
            )
            max_mr_size = int(self.cache_config.mooncake_max_mr_size_bytes)
            regions = _split_mooncake_registration_regions(
                base_ptr=base_ptr,
                logical_size=logical_size,
                mapped_size=mapped_size,
                block_size=self.block_size_bytes,
                max_mr_size=max_mr_size,
                size_alignment=size_alignment,
                pointer_alignment=hugepage_size,
            )
        flexkv_logger.info(
            "[MooncakeStoreTransferWorker] registering external MRs: "
            f"logical_size={logical_size} mapped_size={mapped_size or logical_size} "
            f"regions={regions}"
        )
        self._mooncake_registered_regions = _register_mooncake_regions(
            self.mooncake_client, regions
        )

    def shutdown(self) -> None:
        """Best-effort cleanup; tolerant of partially-failed ``__init__``."""
        try:
            client = getattr(self, "mooncake_client", None)
            regions = getattr(self, "_mooncake_registered_regions", [])
            if client is not None:
                _unregister_mooncake_regions(client, regions)
            self._mooncake_registered_regions = []
        finally:
            super().shutdown()

    @staticmethod
    def _block_size_bytes(cpu_kv_layout: "KVCacheLayout", dtype: torch.dtype) -> int:
        """Bytes per CPU block for mooncake put/get addressing.

        Multi-group BLOCKFIRST stores ``bytes_per_block`` directly in
        ``kv_shape[1]`` (via ``get_block_stride()``) — do not multiply by
        ``dtype.itemsize``. Single-group layouts still use element count ×
        itemsize.
        """
        if cpu_kv_layout.layer_groups is not None:
            return int(cpu_kv_layout.get_block_stride())
        return int(cpu_kv_layout.get_elements_per_block() * dtype.itemsize)

    def _transfer_impl(self, cpu_ptrs, block_sizes, keys,
                       transfer_type: TransferType) -> List[bool]:
        if transfer_type == TransferType.H2REMOTE:
            put_results = self.mooncake_client.batch_put(keys, cpu_ptrs, block_sizes)
            if not all(put_results):
                flexkv_logger.warning(f"Mooncake-store batch put partially failed: {put_results}")
            return put_results
        elif transfer_type == TransferType.REMOTE2H:
            get_results = self.mooncake_client.batch_get(keys, cpu_ptrs, block_sizes)

            if is_mooncake_fault_inject_enabled():
                get_results = inject_mooncake_fault(get_results, transfer_type)
                flexkv_logger.info(f"Mooncake-store batch get results after fault injection: {get_results}")

            if not all(get_results):
                flexkv_logger.warning(
                    f"Mooncake-store batch get partially failed: {get_results}")
            return get_results
        else:
            raise ValueError(
                f"MooncakeStoreTransferWorker only supports H2REMOTE/REMOTE2H, got {transfer_type}")

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> WorkerTransferResult:
        expected_blocks = len(transfer_op.src_block_ids)
        block_sizes: List[int] = []
        start_time = time.time()
        try:
            if self.pool_kind == PoolKind.SWA:
                cpu_ptrs, block_sizes, keys = self._preprocess_swa(transfer_op)
            else:
                cpu_ptrs, block_sizes, keys = self._preprocess_kv(transfer_op)
            if len(keys) != expected_blocks:
                raise ValueError(
                    "Mooncake key count does not match transfer block count: "
                    f"keys={len(keys)}, blocks={expected_blocks}")
            block_results = self._transfer_impl(
                cpu_ptrs, block_sizes, keys, transfer_op.transfer_type)
            if len(block_results) != expected_blocks:
                raise ValueError(
                    "Mooncake result count does not match transfer block count: "
                    f"results={len(block_results)}, blocks={expected_blocks}")
            normalized_results = tuple(bool(result) for result in block_results)
        except Exception:
            # A completed failure must still reach the scheduler. Otherwise the
            # graph never completes and its reserved cache blocks remain leaked.
            flexkv_logger.error(
                "Mooncake transfer failed; reporting all blocks unsuccessful "
                f"for op_id={transfer_op.transfer_op_id}",
                exc_info=True,
            )
            normalized_results = (False,) * expected_blocks
        end_time = time.time()
        try:
            transfer_size = sum(block_sizes)
            self._log_transfer_performance(
                transfer_op, transfer_size, start_time, end_time)
        except Exception:
            flexkv_logger.error(
                "Mooncake transfer performance logging failed; reporting the "
                f"operation unsuccessful for op_id={transfer_op.transfer_op_id}",
                exc_info=True,
            )
            normalized_results = (False,) * expected_blocks
        if not all(normalized_results):
            flexkv_logger.warning(
                "Mooncake transfer partially failed: "
                f"op_id={transfer_op.transfer_op_id}, "
                f"successful={sum(normalized_results)}/{len(normalized_results)}")
        return WorkerTransferResult(
            transfer_op_id=transfer_op.transfer_op_id,
            block_results=normalized_results,
        )

    def _preprocess_kv(self, transfer_op: WorkerTransferOp):
        cpu_block_ids = (
            transfer_op.dst_block_ids
            if transfer_op.transfer_type == TransferType.REMOTE2H
            else transfer_op.src_block_ids
        )
        assert transfer_op.mooncake_store_block_hashes is not None
        block_size_bytes = self.block_size_bytes
        base_ptr = self._cpu_buffer.data_ptr()
        cpu_ptrs, block_sizes, keys = [], [], []
        for i, blk_id in enumerate(cpu_block_ids):
            key = build_key(
                transfer_op.mooncake_store_block_hashes[i],
                PoolKind.KV,
                pp_rank=self.pp_rank,
                pp_size=self.pp_size,
                node_layer_start=self.node_layer_start,
                node_layer_end=self.node_layer_end,
                total_layers=self.total_layers,
            )
            cpu_ptrs.append(base_ptr + int(blk_id) * block_size_bytes)
            block_sizes.append(block_size_bytes)
            keys.append(key)
        return cpu_ptrs, block_sizes, keys

    def _preprocess_swa(self, transfer_op: WorkerTransferOp):
        """Build (cpu_ptrs, sizes, keys) for the SWA mooncake lane.

        Each request contributes one (CPU slot id, tail_hash) pair; after batch
        merging, ``cpu_block_ids`` and ``mooncake_store_swa_block_hashes`` both
        hold N entries in the same order (one key per slot).
        """
        tail_hashes = transfer_op.mooncake_store_swa_block_hashes
        if tail_hashes is None:
            raise ValueError(
                "SWA mooncake transfer requires mooncake_store_swa_block_hashes")
        cpu_block_ids = (
            transfer_op.dst_block_ids
            if transfer_op.transfer_type == TransferType.REMOTE2H
            else transfer_op.src_block_ids
        )
        if len(tail_hashes) != len(cpu_block_ids):
            raise ValueError(
                "SWA mooncake transfer requires len(swa_block_hashes) == "
                f"len(cpu_block_ids): got {len(tail_hashes)} vs {len(cpu_block_ids)}")
        block_size_bytes = self.block_size_bytes
        base_ptr = self._cpu_buffer.data_ptr()
        cpu_ptrs: List[int] = []
        block_sizes: List[int] = []
        keys: List[str] = []
        for i, blk_id in enumerate(cpu_block_ids):
            cpu_ptrs.append(base_ptr + int(blk_id) * block_size_bytes)
            block_sizes.append(block_size_bytes)
            keys.append(build_key(
                str(tail_hashes[i]),
                PoolKind.SWA,
                pp_rank=self.pp_rank,
                pp_size=self.pp_size,
                node_layer_start=self.node_layer_start,
                node_layer_end=self.node_layer_end,
                total_layers=self.total_layers,
            ))
        return cpu_ptrs, block_sizes, keys

    def _postprocess(self, transfer_op: WorkerTransferOp) -> None:
        pass
