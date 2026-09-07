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
# 本文件职责：FlexKV 的**控制面核心** —— 决定「数据该从哪搬到哪、搬哪些 block」。
#
# 【控制面 vs 数据面 —— 阅读本文件的第一要义】
#   本文件**不搬运任何一个字节**。它只做三件事：
#       1. 在各级缓存的 radix tree 上做前缀匹配（match），算出命中了多少 block；
#       2. 决定搬运路径（CPU / SSD / REMOTE -> CPU 内存 -> GPU）与所需中转 block；
#       3. 把决策编译成一张传输 DAG —— TransferOpGraph（节点 = TransferOp，
#          边 = 依赖），连同"传输完成后要执行的回调"一起交给调用方。
#   真正的搬运由数据面完成：TransferEngine -> Worker -> c_ext（DMA / GDS / RDMA）。
#   因此本文件里所有的 src_block_ids / dst_block_ids 都只是"物理 block 编号"，
#   不是 KV 数据本身。
#
# 在系统链路中的位置：
#   KVManager(kvmanager.py) -> KVTaskEngine(kvtask.py)
#     -> 【本文件 GlobalCacheEngine】 -> TransferOpGraph
#     -> TransferEngine(transfer/transfer_engine.py) -> Worker -> c_ext
#
# 三级缓存与索引：
#   CPU 内存 / 本地 SSD / 远端存储每个 tier 各持有一个 CacheEngine 实例，
#   GlobalCacheEngine 按 DeviceType 存在 cpu_cache_engine / ssd_cache_engine /
#   remote_cache_engine（见 __init__ 约 912 行起），并在 cache_engines 字典里维护
#   DeviceType -> engine 的映射。每个 engine = RadixTree（前缀匹配）
#   + Mempool（block 分配与淘汰触发）。
#
# 核心内容速查表：
#   - CacheEngineAccel       : 单 tier 引擎，radix 索引为 C++ 实现（CRadixTreeIndex）
#   - CacheEngine            : 单 tier 引擎，radix 索引为 Python 实现（RadixTreeIndex）
#   - HierarchyLRCacheEngine : 分布式版（P2P / Redis 元信息），见 cache/hie_cache_engine.py
#   - CacheStrategy          : 单次请求的开关（是否忽略 GPU / SSD / REMOTE / GDS）
#   - GlobalCacheEngine      : 顶层编排，本文件主角
#       .get()                  : 读路径入口，返回 (DAG, return_mask, callback,
#                                 op_callback_dict, task_end_op_id)
#       .put()                  : 写路径入口，把 GPU 上的新 KV 下沉到 CPU / SSD / REMOTE
#       ._get_impl_local()      : 只在 CPU / SSD（含 peer）里匹配，不查远端
#       ._get_impl_global()     : CPU / SSD / REMOTE 三级一起匹配（远端参与）
#       ._put_impl_local()      : 写入 CPU / SSD
#       ._put_impl_global()     : 写入 CPU / SSD / REMOTE
#       ._commit_deferred_insert(): 传输完成后才把 block 挂上 radix tree（延迟上树）
#       ._transfer_callback()   : 整图完成后的收尾（解锁 / 置 ready / 回收 buffer）
#       ._abort_transfer_plan() : 图未提交就被取消时的回滚
#       ._op_callback()         : 单个 op 完成后把对应节点置为 ready
#
# 五种核心语义（贯穿全文件，先弄懂再读代码）：
#   - match  : 前缀匹配。返回命中 block 数、命中节点上的 physical_blocks，
#              以及 last_node / last_ready_node 等插入锚点。
#   - insert : 把一段 (hash 序列 -> physical blocks) 挂到 radix tree 上。
#              is_ready=False 表示"节点已上树，但数据还没写进去"。
#   - evict  : 由 take() 在空间不足时触发，按 LRU / LFU / SLRU 等策略淘汰节点，
#              把 block 回收进 mempool。
#   - lock   : 对 radix 节点加引用计数锁，使它在本次请求完成前不可被淘汰。
#   - ready  : set_ready(node, True, len) 把节点标记为"数据已就绪、可被后续请求
#              命中"。未 ready 的块不计入 num_ready_matched_blocks。
#
# 阅读提示 / 常见坑：
#   1. fragment 命名法：fragment1 = 只有 CPU 命中的部分；fragment2 = CPU 未命中但
#      SSD 命中的部分；fragment3 = CPU / SSD 都未命中但远端命中的部分。
#      fragment12 = fragment1 + fragment2，fragment123 同理。
#   2. SSD / 远端的数据不会直接进 GPU，必须先落到 CPU 内存（DISK2H / REMOTE2H），
#      再统一 H2D 上 GPU —— CPU 内存是唯一的中转层（GDS 例外，可 DISK2D 直通）。
#   3. 上树时机：多数路径在**规划阶段**就 insert（is_ready=False），等 op 完成回调
#      再 set_ready；mooncake 远端路径例外，走"延迟上树"
#      （DeferredCacheInsert + _commit_deferred_insert），见相关注释。
#   4. 规划阶段与回调阶段操作同一棵 radix tree，而 pybind 会释放 GIL，
#      故相关方法都套了 @_synchronized_cache_tree 做串行化。
#   5. GPU 侧 block 由 slot_mapping 换算而来（slot_mapping_to_block_ids），
#      本文件不负责 GPU 显存的分配与淘汰。
# ==============================================================================

import logging
import threading
import time
from functools import partial, wraps
from queue import Queue
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass, field, replace

import numpy as np
import nvtx
import torch
from flexkv.c_ext import CRadixNode, CRadixTreeIndex, CMatchResult
from flexkv.cache.hie_cache_engine import HierarchyLRCacheEngine
from flexkv.cache.redis_meta import RedisMeta, dist_available

from flexkv.cache.mempool import Mempool
from flexkv.cache.radixtree import RadixTreeIndex, RadixNode, MatchResult
from flexkv.cache.swa_cache_engine import SWAOpConstructor
from flexkv.common.block import SequenceMeta, format_block_hash
from flexkv.common.config import CacheConfig, ModelConfig, GLOBAL_CONFIG_FROM_ENV, SWAPoolConfig
from flexkv.common.transfer import (
    CompletedOp,
    CompletionAwareCallback,
    DeviceType,
    TransferOpGraph,
    TransferOp,
    TransferType,
    add_virtual_op_for_multiple_finished_ops,
)
from flexkv.common.debug import (
    eviction_log_aggregator,
    flexkv_logger,
    summarize_id_tensor,
)
from flexkv.common.type import MatchResultAccel
from flexkv.integration.dynamo.collector import KVEventCollector
from flexkv.metrics import FlexKVMetricsCollector, init_global_collector, get_global_collector

DEVICE_TYPE: List[str] = ['CPU', 'GPU', 'SSD', 'REMOTE']
_VALID_EVICTION_POLICIES = {'lru', 'lfu', 'slru', 'fifo', 'mru', 'filo'}


# 为什么需要这把锁：radix tree 的 match / insert / evict 经 pybind 进入 C++ 时会
# 释放 GIL，规划阶段（get/put）与完成阶段（回调里的 rematch + insert）因此可能
# 并发踩同一棵树。这里用可重入锁把"规划"和"完成期变更"两段都串起来。
def _synchronized_cache_tree(method: Callable) -> Callable:
    """Serialize radix-tree planning and completion-time mutations."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._cache_tree_lock:
            return method(self, *args, **kwargs)
    return wrapped


# 记录一次 mooncake（远端 KV store）REMOTE2H 的逐 block 成败，由 CompletionAwareCallback
# 在 op 完成时回填。远端读可能"整体报成功但部分 block 失败"，所以下游只能按
# successful_prefix() 得到的最长**连续成功前缀**来用，不能按总数用。
@dataclass
class MooncakeLoadResult:
    """Per-request Mooncake outcome populated by the REMOTE2H callback."""

    block_results: Optional[Tuple[bool, ...]] = None

    def record(self, completed_op: Optional[CompletedOp]) -> None:
        self.block_results = (
            None if completed_op is None else completed_op.block_results)

    def successful_prefix(self, expected_blocks: int) -> int:
        if (self.block_results is None
                or len(self.block_results) != expected_blocks):
            return 0
        prefix = 0
        for succeeded in self.block_results:
            if not succeeded:
                break
            prefix += 1
        return prefix


# "延迟上树"这条路径上，radix tree 最终真正挂载了多少远端 block 的回执。
# 注意它和 MooncakeLoadResult 的区别：传输成功 != 树上可见（rematch/insert 可能
# 被并发写入挤掉），所以预取任务的 return_mask 要取两者较小值。
@dataclass
class DeferredPublishResult:
    """CPU radix publication outcome for a deferred Mooncake load.

    Transfer ``block_results`` alone are not enough for prefetch
    ``return_mask``: rematch/insert may discard staging even when REMOTE2H
    succeeded. Task finalize takes
    ``min(transfer_prefix, published_remote_blocks)``.
    """

    published_remote_blocks: Optional[int] = None
    failed: bool = False
    reason: str = ""

    def record(
            self,
            published_remote_blocks: int,
            reason: str = "ok",
            failed: bool = False) -> None:
        self.published_remote_blocks = int(published_remote_blocks)
        self.reason = reason
        self.failed = bool(failed)

    def record_failure(self, reason: str = "error") -> None:
        self.record(0, reason=reason, failed=True)


# 一条"待发布"的上树记录：规划阶段先把 block 分配出来但**不挂到 radix tree 上**，
# 等整图传输完成（_transfer_callback）后再由 _commit_deferred_insert 决定挂多少。
# 这样设计的原因见 _commit_deferred_insert 的注释。
@dataclass(frozen=True)
class DeferredCacheInsert:
    """Detached tier blocks published by the graph-completion callback.

    ``load_result`` is present for Mooncake reads, where only the longest
    successful remote prefix is valid.  PUT staging has already completed all
    of its graph consumers when this record is committed, so ``None`` means the
    complete staged range is valid.

    ``swa_anchor_block`` / ``swa_load_result`` support joint Full+SWA prefetch:
    the SWA snapshot is a single-block window keyed at position
    ``swa_anchor_block`` (i.e. J-1 for a joint match of length J). At commit
    time the joint guard mounts the SWA slot ONLY when the Full commit reaches
    ``swa_anchor_block+1`` AND the SWA REMOTE2H reported success; otherwise the
    slot is freed and only the Full prefix is published to the tree.

    ``publish_result`` (CPU Mooncake loads) is filled by
    ``_commit_deferred_insert`` / ``_transfer_callback`` so prefetch finalize
    can report the mounted prefix rather than transfer-only success.
    """

    device_type: DeviceType
    sequence_meta: SequenceMeta
    physical_blocks: np.ndarray
    staged_start_block: int
    remote_start_block: int
    requested_end_block: int
    load_result: Optional[MooncakeLoadResult] = None
    swa_slot: int = -1
    publish_to_peer: bool = False
    swa_anchor_block: int = -1
    swa_load_result: Optional[MooncakeLoadResult] = None
    publish_result: Optional[DeferredPublishResult] = None


@dataclass
class GetTransferPlan:
    """一次 GET 规划的完整产物 —— 控制面交给数据面的全部信息。

    字段含义：
        transfer_graph            : 要执行的传输 DAG
        finished_ops_ids          : 「跑完即代表本请求完成」的 op（会合成一个虚拟汇点）
        node_to_unlock            : 规划期锁住的 radix 节点，完成后解锁
        op_callback_dict          : op_id -> 回调（多为 set_ready / SWA 锁释放）
        buffer_to_free            : 未能上树、完成后要归还 mempool 的中转 block
        num_gpu_blocks_to_transfer: 本次真的往 GPU 搬了多少 block（决定 return_mask）
        deferred_inserts          : 需要延迟上树的记录
        swa_reservation           : SWA 读占用的源节点 pin / 暂存槽，完成后释放
    """
    transfer_graph: TransferOpGraph
    finished_ops_ids: List[int]
    node_to_unlock: Dict[DeviceType, Tuple[object, int]]
    op_callback_dict: Dict[int, Callable]
    buffer_to_free: Dict[DeviceType, np.ndarray]
    num_gpu_blocks_to_transfer: int
    deferred_inserts: List[DeferredCacheInsert] = field(default_factory=list)
    # SWA read reservation held by this plan; released by an op callback on the
    # normal path, or by the abort path when the plan is cancelled unlaunched.
    swa_reservation: Optional["SWAReadReservation"] = None

    @classmethod
    def empty(cls) -> "GetTransferPlan":
        return cls(
            transfer_graph=TransferOpGraph.create_empty_graph(),
            finished_ops_ids=[],
            node_to_unlock={},
            op_callback_dict={},
            buffer_to_free={},
            num_gpu_blocks_to_transfer=0,
        )


@dataclass
class PutTransferPlan:
    """一次 PUT 规划的完整产物，结构同 GetTransferPlan。

    差异字段：
        skipped_gpu_blocks: GPU 上已缓存（无需重复下沉）的头部 block 数，
                            用于把 return_mask 右移到真正搬运的区间
        swa_slots_to_free : 预留了但尚未挂载到节点上的 SWA 槽位，
                            回滚路径必须把它们还给 host pool
    """
    transfer_graph: TransferOpGraph
    finished_ops_ids: List[int]
    node_to_unlock: Dict[DeviceType, Tuple[object, int]]
    op_callback_dict: Dict[int, Callable]
    buffer_to_free: Dict[DeviceType, np.ndarray]
    num_gpu_blocks_to_transfer: int
    skipped_gpu_blocks: int
    deferred_inserts: List[DeferredCacheInsert] = field(default_factory=list)
    # SWA slots reserved for this put but not yet mounted (publication happens
    # in an op callback); the abort path must return them to the host pool.
    swa_slots_to_free: List[Tuple[DeviceType, int]] = field(default_factory=list)

    @classmethod
    def empty(cls) -> "PutTransferPlan":
        return cls(
            transfer_graph=TransferOpGraph.create_empty_graph(),
            finished_ops_ids=[],
            node_to_unlock={},
            op_callback_dict={},
            buffer_to_free={},
            num_gpu_blocks_to_transfer=0,
            skipped_gpu_blocks=0,
        )


@dataclass
class SWAReadSource:
    """SWA（Sliding Window Attention）读路径选中的快照来源。

    SWA 只需一个"窗口快照"而非完整前缀，所以按 tier 各自独立选源，
    与 Full-KV 的 fragment 划分解耦。
    """
    hit_blocks: int = 0
    host_slot: int = -1
    node: Optional[object] = None
    device_type: Optional[DeviceType] = None
    engine: Optional[object] = None
    # Key-addressed REMOTE tier (mooncake-store): the tail hash of the hit
    # block is the sole remote handle — no radix node / host slot exists on
    # that tier, so pin / unlock / evict do not apply to the source.
    mooncake_tail_hash: Optional[str] = None

    @property
    def is_mooncake(self) -> bool:
        return self.mooncake_tail_hash is not None

    @property
    def found(self) -> bool:
        if self.hit_blocks <= 0 or self.device_type is None:
            return False
        if self.is_mooncake:
            return self.device_type == DeviceType.REMOTE
        return self.host_slot >= 0 and self.node is not None


# SWA 读期间持有的资源集合：源节点 pin + 可能存在的 CPU 暂存槽 + 对应的 H2D op。
# 传输完成或计划取消时统一释放（_swa_release_load_lock）。
@dataclass(frozen=True)
class SWAReadReservation:
    """Pinned SWA source plus any transient CPU staging slot and graph op."""
    source: SWAReadSource
    staging_slot: int
    h2d_id: int


# 一个"二选一且只执行一次"的完成句柄：正常完成走 __call__()，计划被取消走 abort()。
# 二者互斥由 _consumed 保证 —— 防止 cancel 与 completion 并发导致重复解锁 / 重复回收。
class TransferPlanHandle:
    """Completion callback for a planned get/put, with an abort path.

    Calling the handle (the pre-existing contract for ``task.callback``)
    publishes the plan's results exactly as the plain completion partial did.
    ``abort()`` instead rolls back a plan whose graph was never launched:
    unlock, drop unready inserts, recycle staging, release SWA state. At most
    one of the two may run, and only once; the handle enforces that so a
    cancel racing a completion cannot double-unlock.
    """

    __slots__ = ("_complete", "_abort", "_consumed")

    def __init__(self, complete: Callable[[], None], abort: Callable[[], None]):
        self._complete = complete
        self._abort = abort
        self._consumed = False

    def __call__(self) -> None:
        if self._consumed:
            return
        self._consumed = True
        self._complete()

    def abort(self) -> None:
        if self._consumed:
            return
        self._consumed = True
        self._abort()

    @property
    def keywords(self) -> Dict:
        """Expose completion-partial metadata for existing callback users."""
        return getattr(self._complete, "keywords", {})


class CacheEngineAccel:
    """单个缓存层级（CPU / SSD / REMOTE 之一）的缓存引擎 —— C++ radix 索引版。

    职责（纯控制面）：维护"本 tier 上有哪些 KV block"，提供
    match / insert / take / recycle / lock / set_ready 这组语义。它只记录
    block 编号与树结构，不接触任何 KV 字节。

    与 CacheEngine（Python 索引版，见下）的差异 —— 二者接口完全一致、
    可互换，差异集中在索引实现与数据表示：
        1. 索引：CRadixTreeIndex（C++，csrc/radix_tree.h） vs RadixTreeIndex（Python）
        2. 进出索引的张量：torch.Tensor（int64） vs np.ndarray
        3. match 返回值：MatchResultAccel vs MatchResult
           （Accel 版额外带 block_node_ids、SWA 命中信息）
        4. take/evict 需要用预分配的 torch 缓冲区接收结果，且 evict 会
           **就地 resize** 缓冲区（见 take 内注释）；Python 版直接返回 numpy 数组
        5. 本版是维护中的主路径；CacheEngine 是 legacy Python mirror

    组成：
        index   : radix tree，做前缀匹配、承载节点锁与 ready 标记
        mempool : 本 tier 的物理 block 池，负责分配 / 回收 / 触发淘汰
        swa_pool: SWA 快照槽位池（可选），槽位挂在 radix 节点上（node-mounted），
                  与 Full-KV 共用同一棵树做淘汰，两个池不会漂移
    """

    def __init__(self,
                 device_type: DeviceType,
                 num_total_blocks: int,
                 tokens_per_block: int,
                 evict_ratio: float,
                 hit_reward_seconds: int = 0,
                 evict_start_threshold: float = 1.0,
                 eviction_policy: str = "lru",
                 event_collector: Optional[KVEventCollector] = None,
                 metrics_collector = None,
                 protected_threshold = 2,
                 swa_config: Optional["SWAPoolConfig"] = None):
        if not isinstance(device_type, DeviceType):
            raise ValueError(f"Unknown device type: {device_type}")
        if num_total_blocks <= 0:
            raise ValueError(f"Invalid num_total_blocks: {num_total_blocks}")
        if tokens_per_block <= 0 or (tokens_per_block & (tokens_per_block - 1)) != 0:
            raise ValueError(f"Invalid tokens_per_block: {tokens_per_block}, "
                              f"tokens_per_block must be a power of 2")
        if eviction_policy not in _VALID_EVICTION_POLICIES:
            raise ValueError(f"Invalid eviction_policy: '{eviction_policy}'. "
                              f"Supported policies: {sorted(_VALID_EVICTION_POLICIES)}")
        if not isinstance(protected_threshold, int) or protected_threshold < 1:
            raise ValueError(f"Invalid protected_threshold: {protected_threshold}. "
                              f"protected_threshold must be an integer >= 1")

        self.device_type = device_type

        # C++ radix 索引：前缀匹配 / 插入 / 淘汰 / 节点锁全在这里面完成
        self.index = CRadixTreeIndex(tokens_per_block, num_total_blocks, hit_reward_seconds, eviction_policy,
                                     protected_threshold)

        self.mempool = Mempool(num_total_blocks=num_total_blocks)

        self.tokens_per_block = tokens_per_block
        self.num_total_blocks = num_total_blocks
        self.evict_ratio = evict_ratio
        self.evict_start_threshold = evict_start_threshold

        self.event_collector = event_collector
        self._metrics_collector = metrics_collector

        # SWA (Sliding Window Attention) — NODE-MOUNTED on the Full-KV radix
        # tree (hicache / sglang style), NOT a standalone index. The radix nodes
        # carry the SWA slot / tombstone / lock (see csrc/radix_tree.h and
        # flexkv/cache/radixtree.py); this engine only owns the SWA host-pool
        # (slot bytes + free-list) and the slot alloc/free/drain plumbing. SWA
        # and Full eviction are UNIFIED through the one tree so the two pools
        # never drift. Thisengine owns SWA initialization for its tier; init_swa()
        #  remains public for tests and explicit embedding.
        self.swa_pool = None
        tier_swa_config = (swa_config.for_cache_tier(device_type)
                           if swa_config is not None else None)
        if tier_swa_config is not None:
            self.init_swa(tier_swa_config)

    def init_swa(self, swa_config: "SWAPoolConfig") -> None:
        """Initialize the SWA host pool for node-mounted SWA on this engine."""
        from flexkv.swa.swa_host_pool import SWAHostPool
        self.swa_pool = SWAHostPool(swa_config)

    @property
    def swa_enabled(self) -> bool:
        return self.swa_pool is not None

    def _alloc_swa_slot(self, protected_node=None) -> int:
        """Allocate one SWA slot; evict SWA-LRU once when the pool is full."""
        if self.swa_pool is None:
            return -1
        slot = self.swa_pool.allocate()
        if slot is not None:
            return slot
        # can not allocate SWA slot, evict SWA-LRU once
        if protected_node is not None:
            self.lock_node(protected_node)
        try:
            self._evict_swa_slots(1)
        finally:
            if protected_node is not None:
                self.unlock(protected_node)
        slot = self.swa_pool.allocate()
        return slot if slot is not None else -1

    def _free_swa_slot(self, slot: int) -> None:
        """Return one detached SWA slot to this tier's pool."""
        self.swa_pool.free(int(slot))

    def _drain_unmounted_swa_slots(self) -> None:
        """Return slots detached by radix-tree structural changes to the pool."""
        if self.swa_pool is None:
            return
        for slot in self.index.drain_freed_swa_slots():
            self._free_swa_slot(slot)

    def _pin_swa_node(self, node) -> None:
        self.index.lock(node)
        try:
            node.inc_swa_lock_ref()
        except Exception:
            self.index.unlock(node)
            raise

    def _evict_swa_slots(self, num_swa_evicted: int) -> int:
        """Evict node-mounted SWA slots through the C++ radix tree."""
        if self.swa_pool is None:
            return 0
        free_before = self.swa_pool.num_free
        start_ns = time.perf_counter_ns()
        evicted_full = torch.zeros(0, dtype=torch.int64)
        num_freed = self.index.evict_swa(evicted_full, num_swa_evicted)
        if evicted_full.numel() > 0:
            self.mempool.recycle_blocks(evicted_full.numpy())
        self._drain_unmounted_swa_slots()
        free_after = self.swa_pool.num_free
        eviction_log_aggregator.record(
            tier=DEVICE_TYPE[self.device_type].lower(),
            scope="swa",
            reason="capacity",
            requested_blocks=num_swa_evicted,
            required_blocks=num_swa_evicted,
            evicted_blocks=num_freed,
            free_blocks_before=free_before,
            free_blocks_after=free_after,
            total_blocks=self.swa_pool.num_slots,
            duration_ms=(time.perf_counter_ns() - start_ns) / 1e6,
            target_met=num_freed >= num_swa_evicted,
        )
        return num_freed

    def reset(self) -> None:
        self.index.reset()
        self.mempool.reset()
        # The tree reset bulk-deletes all nodes (their SWA slots are not
        # buffered), so re-arm the SWA pool as fully free to avoid a leak.
        if self.swa_pool is not None:
            self.swa_pool.reset()

    def match(self, sequence_meta: SequenceMeta) -> MatchResultAccel:
        """在本 tier 的 radix tree 上做最长前缀匹配（只命中"已 ready"的部分）。

        Returns:
            MatchResultAccel，含 num_matched_blocks / num_ready_matched_blocks、
            命中的 physical_blocks，以及 last_node、last_ready_node 等插入锚点，
            供后续 insert 复用以避免二次遍历。
        Note:
            本方法只读不写树结构（但会更新 LRU 命中信息），可并发安全调用的前提
            是调用方持有 _cache_tree_lock。
        """
        sequence_meta.gen_hashes()
        match_result = self.index.match_prefix(torch.from_numpy(sequence_meta.block_hashes).to(torch.int64),
                                              sequence_meta.num_blocks, True)
        # physical blocks (torch.Tensor -> numpy, zero-copy on CPU)
        phys = match_result.physical_blocks.cpu().numpy()
        # optional block_node_ids
        try:
            bnis = getattr(match_result, "block_node_ids", None)
            if isinstance(bnis, torch.Tensor) and bnis.numel() > 0:
                bnids_np = bnis.cpu().numpy()
            else:
                bnids_np = None
        except Exception:
            bnids_np = None
        return MatchResultAccel(
            num_ready_matched_blocks=match_result.num_ready_matched_blocks,
            num_matched_blocks=match_result.num_matched_blocks,
            last_ready_node=match_result.last_ready_node,
            last_node=match_result.last_node,
            last_node_matched_length=match_result.last_node_matched_length,
            physical_blocks=phys,
            block_node_ids=bnids_np,
            matched_pos="remote" if self.device_type == DeviceType.REMOTE else "local",
            # SWA node-mount: carry the SWA hit found on the SAME forward pass so
            # the SWA-aware get can reuse it (no second match_prefix walk).
            last_swa_node=getattr(match_result, "last_swa_node", None),
            swa_hit_blocks=int(getattr(match_result, "swa_hit_blocks", 0) or 0),
        )

    def insert(self,
               sequence_meta: SequenceMeta,
               physical_block_ids: torch.Tensor,
               num_insert_blocks: int = -1,
               is_ready: bool = True,
               match_result: Optional[MatchResultAccel] = None) -> Optional[CRadixNode]:
        """把 (block hash 序列 -> physical blocks) 挂上 radix tree。

        Args:
            num_insert_blocks  : 要插入的 block 数；-1 表示整个序列
            is_ready           : False = 节点先上树占位、数据尚未写入。
                                 未 ready 的块不会被后续 match 计入
                                 num_ready_matched_blocks，从而避免"命中脏数据"
            match_result       : 复用上一次 match 的锚点，省掉一次树的查找
        Returns:
            插入/分裂后的叶子节点；冲突或空插入时返回 None
        """
        sequence_meta.gen_hashes()
        if match_result is None:
            node = self.index.insert(torch.from_numpy(physical_block_ids).to(torch.int64),
                                     torch.from_numpy(sequence_meta.block_hashes).to(torch.int64),
                                     sequence_meta.num_blocks,
                                     num_insert_blocks,
                                     is_ready)
        else:
            node = self.index.insert(torch.from_numpy(physical_block_ids).to(torch.int64),
                                     torch.from_numpy(sequence_meta.block_hashes).to(torch.int64),
                                     sequence_meta.num_blocks,
                                     num_insert_blocks,
                                     is_ready,
                                     match_result.last_node,
                                     match_result.num_matched_blocks,
                                     match_result.last_node_matched_length)

        if self.event_collector is not None:
            self.event_collector.publish_stored(
                block_hashes=sequence_meta.block_hashes[:None if num_insert_blocks == -1 else num_insert_blocks],
                block_size=self.tokens_per_block,
                medium=DEVICE_TYPE[self.device_type]
            )

        return node

    def lock_node(self, node: CRadixNode) -> None:
        self.index.lock(node)

    def unlock(self, node: CRadixNode) -> None:
        self.index.unlock(node)

    def set_ready(self, node: CRadixNode, ready: bool, ready_length: int) -> None:
        self.index.set_ready(node, ready, ready_length)

    # 与 CacheEngine.take 逻辑完全一致（详见那里的注释），差异仅在索引实现：
    # 这里用 torch 缓冲区接收 evict 结果，且 evict 会就地 resize 缓冲区。
    def take(self,
             num_required_blocks: int,
             protected_node: Optional[CRadixNode] = None,
             strict: bool = True) -> np.ndarray:
        # Calculate current utilization
        utilization = ((self.mempool.num_total_blocks - self.mempool.num_free_blocks)
                       / self.mempool.num_total_blocks) if self.mempool.num_total_blocks > 0 else 0

        # Proactive eviction: trigger when utilization exceeds threshold OR when blocks are needed
        should_evict = (utilization >= self.evict_start_threshold) or \
            (num_required_blocks > self.mempool.num_free_blocks)

        if should_evict:
            if protected_node is not None:
                self.index.lock(protected_node)

            # Calculate how many blocks to evict
            # Goal: maintain free blocks above (1 - evict_start_threshold) ratio
            target_free_blocks = int(self.mempool.num_total_blocks * (1.0 - self.evict_start_threshold))
            evict_to_reach_target = max(0, target_free_blocks - self.mempool.num_free_blocks)

            evict_block_num = max(
                num_required_blocks - self.mempool.num_free_blocks,  # At least meet current demand
                evict_to_reach_target,                               # Or reach target free ratio
                # Or minimum evict_ratio
                int(self.mempool.num_total_blocks * self.evict_ratio) if self.evict_ratio > 0 else 0
            )

            if evict_block_num > 0:
                free_before = self.mempool.num_free_blocks
                start_ns = time.perf_counter_ns()
                target_blocks = torch.zeros(evict_block_num, dtype=torch.int64)
                evicted_block_hashes = torch.zeros(evict_block_num, dtype=torch.int64)
                # evict() resizes both tensors in-place to the actual freed count
                # (which may EXCEED evict_block_num when the I2 tombstone cascade
                # frees ancestors) and returns that count. Trust it, don't assume
                # evict_block_num.
                num_evicted = self.index.evict(target_blocks, evicted_block_hashes, evict_block_num)
                if target_blocks.numel() != num_evicted:
                    target_blocks.resize_(num_evicted)
                    evicted_block_hashes.resize_(num_evicted)
                target_blocks = target_blocks.numpy()
                self.mempool.recycle_blocks(target_blocks)

                # SWA node-mount: full eviction may have connected-freed SWA
                # slots (record_freed_swa_slot in split/evict). Return them to the
                # SWA host pool so the two pools stay in lock-step (I1). No-op when
                # SWA is disabled.
                self._drain_unmounted_swa_slots()

                # Record eviction metrics
                if self._metrics_collector is not None and num_evicted > 0:
                    self._metrics_collector.record_eviction(DEVICE_TYPE[self.device_type].lower(), num_evicted)

                if self.event_collector is not None:
                    self.event_collector.publish_removed(
                        block_hashes=evicted_block_hashes.numpy(),
                        medium=DEVICE_TYPE[self.device_type]
                    )
                free_after = self.mempool.num_free_blocks
                target_met = free_after >= max(
                    num_required_blocks, target_free_blocks
                )
                sample_block_hashes = None
                batch_level = logging.DEBUG if target_met else logging.WARNING
                if flexkv_logger.is_enabled_for(batch_level):
                    sample_block_hashes = [
                        format_block_hash(value)
                        for value in evicted_block_hashes[:3].tolist()
                    ]
                eviction_log_aggregator.record(
                    tier=DEVICE_TYPE[self.device_type].lower(),
                    scope="full",
                    reason=(
                        "capacity"
                        if num_required_blocks > free_before
                        else "threshold"
                    ),
                    requested_blocks=num_required_blocks,
                    required_blocks=evict_block_num,
                    evicted_blocks=num_evicted,
                    free_blocks_before=free_before,
                    free_blocks_after=free_after,
                    total_blocks=self.mempool.num_total_blocks,
                    duration_ms=(time.perf_counter_ns() - start_ns) / 1e6,
                    sample_block_hashes=sample_block_hashes,
                    target_met=target_met,
                )
            if protected_node is not None:
                self.index.unlock(protected_node)

        if strict and num_required_blocks > self.mempool.num_free_blocks:
            raise RuntimeError(f"Not enough free blocks to take, "
                               f"required: {num_required_blocks}, "
                               f"available: {self.mempool.num_free_blocks}")
        num_allocated_blocks = min(num_required_blocks, self.mempool.num_free_blocks)
        allocated_blocks = self.mempool.allocate_blocks(num_allocated_blocks)

        # Record allocation metrics
        if self._metrics_collector is not None and num_allocated_blocks > 0:
            self._metrics_collector.record_allocation(DEVICE_TYPE[self.device_type].lower(), num_allocated_blocks)

        return allocated_blocks

    def recycle(self, physical_blocks: np.ndarray) -> None:
        self.mempool.recycle_blocks(physical_blocks)
        self._drain_unmounted_swa_slots()

    def rollback_unready_insert(self, node: Optional[CRadixNode]) -> int:
        """Undo an ``is_ready=False`` insert whose completion callback will
        never run (plan aborted before launch). No-op for ready nodes, so it is
        safe to call on every entry of a plan's node_to_unlock: only nodes this
        plan inserted are unready. Recycles the freed blocks."""
        if node is None:
            return 0
        freed = torch.zeros(node.size(), dtype=torch.int64)
        num_freed = self.index.remove_unready_leaf(node, freed)
        if freed.numel() != num_freed:
            freed.resize_(num_freed)
        if num_freed > 0:
            self.recycle(freed.numpy())
        return num_freed

class CacheEngine:
    """单个缓存层级（CPU / SSD / REMOTE 之一）的缓存引擎 —— Python radix 索引版。

    与 CacheEngineAccel 接口完全一致、可互换，索引改为 Python 实现的
    RadixTreeIndex（flexkv/cache/radixtree.py），进出用 np.ndarray。
    两者的选择开关在 GlobalCacheEngine.__init__：
        enable_p2p_* -> HierarchyLRCacheEngine（分布式版）
        index_accel  -> CacheEngineAccel
        否则          -> 本类（legacy Python mirror）

    引擎维护三样东西：
        index    : RadixTreeIndex，前缀匹配 + 节点锁 + ready 标记
        mempool  : Mempool，本 tier 物理 block 的分配 / 回收 / 空闲量追踪
        swa_pool : SWA 快照槽位池（可选）

    对外语义速查（全局通用，不只是本类）：
        match(sequence_meta)                  -> MatchResult，前缀匹配
        insert(..., is_ready=False)           -> 上树占位（数据未就绪）
        take(n, protected_node)               -> 申请 n 个 block，不够就先淘汰
        recycle(blocks)                       -> 归还 block 给 mempool
        lock_node(node) / unlock(node)        -> 锁住节点，防止被淘汰
        set_ready(node, True, ready_length)   -> 标记数据就绪，此后才可被命中
    """

    def __init__(self,
                 device_type: DeviceType,
                 num_total_blocks: int,
                 tokens_per_block: int,
                 evict_ratio: float,
                 hit_reward_seconds: int = 0,
                 evict_start_threshold: float = 1.0,
                 eviction_policy: str = "lru",
                 event_collector: Optional[KVEventCollector] = None,
                 metrics_collector = None,
                 protected_threshold = 2,
                 swa_config: Optional["SWAPoolConfig"] = None):
        if not isinstance(device_type, DeviceType):
            raise ValueError(f"Unknown device type: {device_type}")
        if num_total_blocks <= 0:
            raise ValueError(f"Invalid num_total_blocks: {num_total_blocks}")
        if tokens_per_block <= 0 or (tokens_per_block & (tokens_per_block - 1)) != 0:
            raise ValueError(f"Invalid tokens_per_block: {tokens_per_block}, "
                              f"tokens_per_block must be a power of 2")
        if eviction_policy not in _VALID_EVICTION_POLICIES:
            raise ValueError(f"Invalid eviction_policy: '{eviction_policy}'. "
                              f"Supported policies: {sorted(_VALID_EVICTION_POLICIES)}")
        if not isinstance(protected_threshold, int) or protected_threshold < 1:
            raise ValueError(f"Invalid protected_threshold: {protected_threshold}. "
                              f"protected_threshold must be an integer >= 1")

        self.device_type = device_type

        self.index = RadixTreeIndex(tokens_per_block=tokens_per_block,
                                    hit_reward_seconds=hit_reward_seconds,
                                    eviction_policy=eviction_policy,
                                    protected_threshold=protected_threshold)

        self.mempool = Mempool(num_total_blocks=num_total_blocks)

        self.tokens_per_block = tokens_per_block
        self.num_total_blocks = num_total_blocks
        self.evict_ratio = evict_ratio
        self.evict_start_threshold = evict_start_threshold

        self.event_collector = event_collector
        self._metrics_collector = metrics_collector

        # Legacy Python mirror. Keep the SWA helpers local to this class; the
        # C++ CacheEngineAccel path is the maintained path.
        self.swa_pool = None
        self.tier_swa_config = (swa_config.for_cache_tier(device_type)
                           if swa_config is not None else None)
        if self.tier_swa_config is not None:
            self.init_swa(self.tier_swa_config)

    def init_swa(self, swa_config: "SWAPoolConfig") -> None:
        """Initialize the SWA host pool for node-mounted SWA on this engine."""
        from flexkv.swa.swa_host_pool import SWAHostPool
        self.swa_pool = SWAHostPool(swa_config)

    @property
    def swa_enabled(self) -> bool:
        return self.tier_swa_config is not None and self.tier_swa_config.enabled \
               and self.swa_pool is not None

    def _alloc_swa_slot(self, protected_node=None) -> int:
        """Allocate one SWA slot; evict SWA-LRU once when the pool is full."""
        if self.swa_pool is None:
            return -1
        slot = self.swa_pool.allocate()
        if slot is not None:
            return slot
        if protected_node is not None:
            self.lock_node(protected_node)
        try:
            self._evict_swa_slots(1)
        finally:
            if protected_node is not None:
                self.unlock(protected_node)
        slot = self.swa_pool.allocate()
        return slot if slot is not None else -1

    def _free_swa_slot(self, slot: int) -> None:
        """Return one detached SWA slot to this tier's pool."""
        self.swa_pool.free(int(slot))

    def _drain_unmounted_swa_slots(self) -> None:
        """Return slots detached by radix-tree structural changes to the pool."""
        if self.swa_pool is None:
            return
        for slot in self.index.drain_freed_swa_slots():
            self._free_swa_slot(slot)

    def _pin_swa_node(self, node) -> None:
        self.index.lock(node)
        try:
            node.inc_swa_lock_ref()
        except Exception:
            self.index.unlock(node)
            raise

    def _evict_swa_slots(self, num_swa_evicted: int) -> int:
        """Evict node-mounted SWA slots through the Python radix tree."""
        if self.swa_pool is None:
            return 0
        free_before = self.swa_pool.num_free
        start_ns = time.perf_counter_ns()
        evicted_full, num_freed = self.index.evict_swa(num_swa_evicted)
        if evicted_full.size > 0:
            self.mempool.recycle_blocks(evicted_full)
        self._drain_unmounted_swa_slots()
        free_after = self.swa_pool.num_free
        eviction_log_aggregator.record(
            tier=DEVICE_TYPE[self.device_type].lower(),
            scope="swa",
            reason="capacity",
            requested_blocks=num_swa_evicted,
            required_blocks=num_swa_evicted,
            evicted_blocks=num_freed,
            free_blocks_before=free_before,
            free_blocks_after=free_after,
            total_blocks=self.swa_pool.num_slots,
            duration_ms=(time.perf_counter_ns() - start_ns) / 1e6,
            target_met=num_freed >= num_swa_evicted,
        )
        return num_freed

    def reset(self) -> None:
        self.index.reset()
        self.mempool.reset()
        if self.swa_pool is not None:
            self.swa_pool.reset()

    def match(self, sequence_meta: SequenceMeta) -> MatchResult:
        """在本 tier 的 radix tree 上做最长前缀匹配。

        Args:
            sequence_meta: 本次请求的 token 序列元信息，内部会惰性生成 block_hashes
        Returns:
            MatchResult：num_matched_blocks（含未 ready）/ num_ready_matched_blocks
            （可立即使用）、physical_blocks（命中块编号）、last_node / last_ready_node
        Note:
            只有 num_ready_matched_blocks 个块是真能读的；未 ready 的块是别人正在
            写、还没写完的。调用方（GlobalCacheEngine）一律按 ready 数量切分。
        """
        match_result = self.index.match_prefix(sequence_meta,
                                              update_cache_info=True)
        return match_result

    def insert(self,
               sequence_meta: SequenceMeta,
               physical_block_ids: np.ndarray,
               num_insert_blocks: int = -1,
               is_ready: bool = True,
               match_result: Optional[MatchResult] = None) -> Optional[RadixNode]:
        """把 (block hash 序列 -> physical blocks) 挂上 radix tree。

        Args:
            physical_block_ids : 这些 hash 对应的物理 block 编号
            num_insert_blocks  : 插入多少个 block；-1 = 整条序列
            is_ready           : False 表示"节点先上树占位，数据还没写完"。
                                 此时节点可见但不可命中，写完由 set_ready 打开
            match_result       : 复用上次 match 的锚点，省一次树查找
        Returns:
            新插入（或分裂出的）叶子节点；插入失败 / 冲突时返回 None
        """
        node = self.index.insert(sequence_meta,
                                 physical_block_ids,
                                 num_insert_blocks=num_insert_blocks,
                                 is_ready=is_ready,
                                 match_result=match_result)
        if self.event_collector is not None:
            self.event_collector.publish_stored(
                block_hashes=sequence_meta.block_hashes[:None if num_insert_blocks == -1 else num_insert_blocks],
                                                block_size=self.tokens_per_block,
                                                medium=DEVICE_TYPE[self.device_type])
        return node

    def lock_node(self, node: RadixNode) -> None:
        self.index.lock(node)

    def unlock(self, node: RadixNode) -> None:
        self.index.unlock(node)

    def set_ready(self, node: RadixNode, ready: bool, ready_length: int) -> None:
        self.index.set_ready(node, ready, ready_length)

    def take(self,
             num_required_blocks: int,
             protected_node: Optional[RadixNode] = None,
             strict: bool = True) -> np.ndarray:
        """申请 num_required_blocks 个空闲 block，空间不足时先淘汰再分配。

        Args:
            num_required_blocks: 需要的 block 数
            protected_node     : 淘汰期间要保护的节点（通常是本次 match 命中的
                                 last_node）。淘汰可能误伤它，故先 lock 再 evict
            strict             : True 时若最终仍不够则 raise RuntimeError；
                                 False 则返回能拿到的数量（可能少于请求量）
        Returns:
            分配到的 block 编号数组（长度 <= num_required_blocks）
        Note:
            淘汰是"预防式"的：只要利用率超过 evict_start_threshold 或当前需求
            得不到满足就触发，一次多淘汰一些，避免每来一个请求就淘汰一次。
        """
        # Calculate current utilization
        utilization = ((self.mempool.num_total_blocks - self.mempool.num_free_blocks)
                       / self.mempool.num_total_blocks) if self.mempool.num_total_blocks > 0 else 0

        # Proactive eviction: trigger when utilization exceeds threshold OR when blocks are needed
        should_evict = (utilization >= self.evict_start_threshold) or \
            (num_required_blocks > self.mempool.num_free_blocks)

        if should_evict:
            if protected_node is not None:
                self.index.lock(protected_node)

            # Calculate how many blocks to evict
            # Goal: maintain free blocks above (1 - evict_start_threshold) ratio
            target_free_blocks = int(self.mempool.num_total_blocks * (1.0 - self.evict_start_threshold))
            evict_to_reach_target = max(0, target_free_blocks - self.mempool.num_free_blocks)

            evict_block_num = max(
                num_required_blocks - self.mempool.num_free_blocks,  # At least meet current demand
                evict_to_reach_target,                               # Or reach target free ratio
                # Or minimum evict_ratio
                int(self.mempool.num_total_blocks * self.evict_ratio) if self.evict_ratio > 0 else 0
            )
            if evict_block_num > 0:
                free_before = self.mempool.num_free_blocks
                start_ns = time.perf_counter_ns()
                # 淘汰由 radix tree 按 LRU/LFU/SLRU 策略挑节点，返回被踢掉的
                # block 编号与对应 hash（hash 用于向外部发 KV 事件）
                evicted_blocks, evicted_block_hashes = self.index.evict(evict_block_num)
                self.mempool.recycle_blocks(evicted_blocks)

                # SWA node-mount: return connected-freed SWA slots to the pool (I1).
                self._drain_unmounted_swa_slots()

                # Record eviction metrics
                if self._metrics_collector is not None and len(evicted_blocks) > 0:
                    self._metrics_collector.record_eviction(DEVICE_TYPE[self.device_type].lower(), len(evicted_blocks))

                if self.event_collector is not None:
                    self.event_collector.publish_removed(block_hashes=evicted_block_hashes,
                                                         medium=DEVICE_TYPE[self.device_type])
                free_after = self.mempool.num_free_blocks
                target_met = free_after >= max(
                    num_required_blocks, target_free_blocks
                )
                sample_block_hashes = None
                batch_level = logging.DEBUG if target_met else logging.WARNING
                if flexkv_logger.is_enabled_for(batch_level):
                    sample_block_hashes = [
                        format_block_hash(value)
                        for value in evicted_block_hashes[:3].tolist()
                    ]
                eviction_log_aggregator.record(
                    tier=DEVICE_TYPE[self.device_type].lower(),
                    scope="full",
                    reason=(
                        "capacity"
                        if num_required_blocks > free_before
                        else "threshold"
                    ),
                    requested_blocks=num_required_blocks,
                    required_blocks=evict_block_num,
                    evicted_blocks=len(evicted_blocks),
                    free_blocks_before=free_before,
                    free_blocks_after=free_after,
                    total_blocks=self.mempool.num_total_blocks,
                    duration_ms=(time.perf_counter_ns() - start_ns) / 1e6,
                    sample_block_hashes=sample_block_hashes,
                    target_met=target_met,
                )
            if protected_node is not None:
                self.index.unlock(protected_node)

        if strict and num_required_blocks > self.mempool.num_free_blocks:
            raise RuntimeError("Not enough free blocks to take, ",
                               f"required: {num_required_blocks}, "
                               f"available: {self.mempool.num_free_blocks}")
        num_allocated_blocks = min(num_required_blocks, self.mempool.num_free_blocks)
        allocated_blocks = self.mempool.allocate_blocks(num_allocated_blocks)

        # Record allocation metrics
        if self._metrics_collector is not None and num_allocated_blocks > 0:
            self._metrics_collector.record_allocation(DEVICE_TYPE[self.device_type].lower(), num_allocated_blocks)

        return allocated_blocks

    def recycle(self, physical_blocks: np.ndarray) -> None:
        self.mempool.recycle_blocks(physical_blocks)
        self._drain_unmounted_swa_slots()

    def rollback_unready_insert(self, node: Optional[RadixNode]) -> int:
        """Undo an ``is_ready=False`` insert whose completion callback will
        never run (plan aborted before launch). No-op for ready nodes, so it is
        safe to call on every entry of a plan's node_to_unlock: only nodes this
        plan inserted are unready. Recycles the freed blocks."""
        if node is None:
            return 0
        freed = self.index.remove_unready_leaf(node)
        if freed.size > 0:
            self.recycle(freed)
        return int(freed.size)

@dataclass
class CacheStrategy:
    """单次 get/put 的策略开关，用来临时屏蔽某些介质或通道。

    典型用法：预取任务设 ignore_gpu=True（只搬到 CPU，不上 GPU）；
    compute 侧的正常 GET 则全开。
    """
    # if True, will not put or get blocks from GPU
    ignore_gpu: bool = False
    # if True, will not put or get blocks from SSD
    ignore_ssd: bool = False
    # if True, will not get blocks from REMOTE
    ignore_remote: bool = False
    # if True, will not use GDS
    ignore_gds: bool = False

DEFAULT_CACHE_STRATEGY = CacheStrategy()

CPUONLY_CACHE_STRATEGY = CacheStrategy(ignore_gpu=False, ignore_ssd=True, ignore_remote=True, ignore_gds=True)


# mooncake（远端 KV store）启用时的 GET 策略修正：预取（ignore_gpu=True）才允许
# 走 REMOTE2H；compute 侧的 GPU 绑定 GET 强制忽略远端，宁可当作 CPU miss 去重算，
# 也不能在图里插一条阻塞的 REMOTE2H。
def resolve_get_cache_strategy(
        use_mooncake_store_backend: bool,
        temp_cache_strategy: CacheStrategy) -> CacheStrategy:
    """Apply mooncake prefetch/compute split to a GET CacheStrategy.

    Prefetch tasks set ``ignore_gpu=True`` and may pull REMOTE2H.
    Compute retrieve (GPU-bound GET) must stay on local CPU/SSD when mooncake
    is enabled — remote misses become CPU misses / recompute, never an
    in-graph REMOTE2H.
    """
    if (use_mooncake_store_backend
            and not temp_cache_strategy.ignore_gpu
            and not temp_cache_strategy.ignore_remote):
        return replace(temp_cache_strategy, ignore_remote=True)
    return temp_cache_strategy


class GlobalCacheEngine:
    """顶层缓存编排器 —— 整个 FlexKV 控制面的大脑。

    职责：
        把一次 get / put 请求翻译成一张传输 DAG（TransferOpGraph）+ 一组回调，
        自己**不搬运任何字节**。搬运由数据面 TransferEngine / Worker 执行。

    它手里有什么：
        cpu_cache_engine    : CPU 内存层（DeviceType.CPU）
        ssd_cache_engine    : 本地 SSD 层（DeviceType.SSD）
        remote_cache_engine : 远端存储层（DeviceType.REMOTE）
        cache_engines       : {DeviceType -> engine} 映射
        _cache_tree_lock    : 串行化"规划"与"完成期树变更"的可重入锁
        每个 engine 可以是 CacheEngineAccel / CacheEngine / HierarchyLRCacheEngine
        / MooncakeStoreCacheEngine 之一，按配置三选一（见 __init__）。

    两条主路径：
        get()  -> _get_impl_local()  （不查远端：CPU/SSD 及 peer 节点）
               -> _get_impl_global() （CPU/SSD/REMOTE 三级一起匹配）
        put()  -> _put_impl_local()  （下沉到 CPU/SSD）
               -> _put_impl_global() （再下沉到 REMOTE）

    三个关键设计（新手最容易困惑的地方）：
        1) 所有数据都要经 CPU 内存中转：SSD/REMOTE 的块先 DISK2H/REMOTE2H 落到
           CPU，再统一 H2D 上 GPU。只有 GDS 例外（DISK2D 直通 GPU）。
        2) 上树与搬理解耦：规划阶段就把目标节点 insert 到 radix 树（is_ready=
           False 占位），传输完成回调再 set_ready 打开可见性。
        3) mooncake 远端走"延迟上树"：规划阶段完全不上树，等图完成后
           _commit_deferred_insert 重新 match 再决定挂多少（见该方法注释）。
    """

    def __init__(self, cache_config: CacheConfig, model_config: ModelConfig, redis_meta: RedisMeta = None,
                 event_collector: Optional[KVEventCollector] = None):
        # pybind releases the GIL around radix match/insert/evict. Protect the
        # tree across planning and callback-time rematch+insert transactions.
        self._cache_tree_lock = threading.RLock()
        self.cache_config = cache_config
        self.model_config = model_config
        self.tokens_per_block = cache_config.tokens_per_block

        self.cpu_cache_engine = None
        self.ssd_cache_engine = None
        self.remote_cache_engine = None
        self.use_mooncake_store_backend = cache_config.use_mooncake_store_backend
        if self.use_mooncake_store_backend:
            # Product rule (M0): mooncake REMOTE2H runs in prefetch only;
            # compute retrieve matches local ready tiers and does H2D.
            flexkv_logger.info(
                "Mooncake store enabled: REMOTE2H is prefetch-only; "
                "compute GET forces ignore_remote=True"
            )

        self.index_accel = GLOBAL_CONFIG_FROM_ENV.index_accel
        if cache_config.enable_kv_sharing:
            assert redis_meta is not None
            self.redis_meta = redis_meta
            self.node_id = self.redis_meta.get_node_id()
            self.enable_kv_sharing = True
        else:
            self.enable_kv_sharing = False
        self.cache_engines = {}

        self.evict_ratio = GLOBAL_CONFIG_FROM_ENV.evict_ratio
        self.evict_start_threshold = GLOBAL_CONFIG_FROM_ENV.evict_start_threshold
        self.hit_reward_seconds = GLOBAL_CONFIG_FROM_ENV.hit_reward_seconds
        self.eviction_policy = GLOBAL_CONFIG_FROM_ENV.eviction_policy
        self.protected_threshold = GLOBAL_CONFIG_FROM_ENV.slru_protected_threshold

        # Initialize metrics collector for cache engine monitoring (before creating CacheEngines)
        self._metrics_collector = get_global_collector()
        if self._metrics_collector is None:
            self._metrics_collector = init_global_collector()

        need_dist = (
            (cache_config.enable_cpu and cache_config.enable_p2p_cpu)
            or (cache_config.enable_ssd and cache_config.enable_p2p_ssd)
            or (cache_config.enable_remote and cache_config.enable_kv_sharing)
        )
        if need_dist and not dist_available():
            raise RuntimeError(
                "Config enables distributed KV cache (P2P/Redis), but FlexKV was built without it. "
                "Rebuild with FLEXKV_ENABLE_P2P=1 and install Redis dependencies "
                "(e.g. libhiredis-dev, redis-tools). See README for full list."
            )

        # 三种引擎实现的选择开关（每个 tier 独立三选一）：
        #   enable_p2p_*  -> HierarchyLRCacheEngine（分布式，走 Redis 元信息 + P2P）
        #   index_accel   -> CacheEngineAccel      （C++ radix 索引，主路径）
        #   否则           -> CacheEngine           （Python radix 索引，legacy）
        # REMOTE tier 还多一种：mooncake store 后端（use_mooncake_store_backend）。
        if cache_config.enable_cpu:
            if cache_config.enable_p2p_cpu:
                self.cpu_cache_engine = HierarchyLRCacheEngine.from_cache_config(
                    cache_config, self.node_id, DeviceType.CPU, meta=self.redis_meta)
            elif self.index_accel:
                self.cpu_cache_engine = CacheEngineAccel(
                    device_type=DeviceType.CPU,
                    num_total_blocks=cache_config.num_cpu_blocks,
                    tokens_per_block=cache_config.tokens_per_block,
                    evict_ratio=self.evict_ratio,
                    hit_reward_seconds=self.hit_reward_seconds,
                    evict_start_threshold=self.evict_start_threshold,
                    eviction_policy=self.eviction_policy,
                    event_collector=event_collector,
                    metrics_collector=self._metrics_collector,
                    protected_threshold=self.protected_threshold,
                    swa_config=cache_config.swa,
                )
            else:
                self.cpu_cache_engine = CacheEngine(
                    device_type=DeviceType.CPU,
                    num_total_blocks=cache_config.num_cpu_blocks,
                    tokens_per_block=cache_config.tokens_per_block,
                    evict_ratio=self.evict_ratio,
                    hit_reward_seconds=self.hit_reward_seconds,
                    evict_start_threshold=self.evict_start_threshold,
                    eviction_policy=self.eviction_policy,
                    event_collector=event_collector,
                    metrics_collector=self._metrics_collector,
                    protected_threshold=self.protected_threshold,
                    swa_config=cache_config.swa,
                )
            self.cache_engines[DeviceType.CPU] = self.cpu_cache_engine
        if cache_config.enable_ssd:
            if cache_config.enable_p2p_ssd:
                self.ssd_cache_engine = HierarchyLRCacheEngine.from_cache_config(
                    cache_config, self.node_id, DeviceType.SSD, meta=self.redis_meta)
            elif self.index_accel:
                self.ssd_cache_engine = CacheEngineAccel(
                    device_type=DeviceType.SSD,
                    num_total_blocks=cache_config.num_ssd_blocks,
                    tokens_per_block=cache_config.tokens_per_block,
                    evict_ratio=self.evict_ratio,
                    hit_reward_seconds=self.hit_reward_seconds,
                    evict_start_threshold=self.evict_start_threshold,
                    eviction_policy=self.eviction_policy,
                    event_collector=event_collector,
                    metrics_collector=self._metrics_collector,
                    protected_threshold=self.protected_threshold,
                    swa_config=cache_config.swa,
                )
            else:
                self.ssd_cache_engine = CacheEngine(
                    device_type=DeviceType.SSD,
                    num_total_blocks=cache_config.num_ssd_blocks,
                    tokens_per_block=cache_config.tokens_per_block,
                    evict_ratio=self.evict_ratio,
                    hit_reward_seconds=self.hit_reward_seconds,
                    evict_start_threshold=self.evict_start_threshold,
                    eviction_policy=self.eviction_policy,
                    event_collector=event_collector,
                    metrics_collector=self._metrics_collector,
                    protected_threshold=self.protected_threshold,
                    swa_config=cache_config.swa,
                )
            self.cache_engines[DeviceType.SSD] = self.ssd_cache_engine
        if cache_config.enable_remote:
            if self.use_mooncake_store_backend:
                from flexkv.external.mooncake_store_utils import MooncakeStoreCacheEngine
                self.remote_cache_engine = MooncakeStoreCacheEngine(
                    cache_config=cache_config,
                )
            elif cache_config.enable_kv_sharing:
                # Build PCFSCacheEngine from CacheConfig directly (replacing RemotePCFSCacheEngine) TODO
                self.remote_cache_engine = HierarchyLRCacheEngine.from_cache_config(
                    cache_config, self.node_id, DeviceType.REMOTE, meta=self.redis_meta)
            elif self.index_accel:
                self.remote_cache_engine = CacheEngineAccel(
                    device_type=DeviceType.REMOTE,
                    num_total_blocks=cache_config.num_remote_blocks,
                    tokens_per_block=cache_config.tokens_per_block,
                    evict_ratio=self.evict_ratio,
                    hit_reward_seconds=self.hit_reward_seconds,
                    evict_start_threshold=self.evict_start_threshold,
                    eviction_policy=self.eviction_policy,
                    event_collector=None,
                    metrics_collector=self._metrics_collector,
                    protected_threshold=self.protected_threshold,
                    swa_config=cache_config.swa,
                )
            else:
                self.remote_cache_engine = CacheEngine(
                    device_type=DeviceType.REMOTE,
                    num_total_blocks=cache_config.num_remote_blocks,
                    tokens_per_block=cache_config.tokens_per_block,
                    evict_ratio=self.evict_ratio,
                    hit_reward_seconds=self.hit_reward_seconds,
                    evict_start_threshold=self.evict_start_threshold,
                    eviction_policy=self.eviction_policy,
                    event_collector=None,
                    metrics_collector=self._metrics_collector,
                    protected_threshold=self.protected_threshold,
                    swa_config=cache_config.swa,
                )
            self.cache_engines[DeviceType.REMOTE] = self.remote_cache_engine

        # SWA peer-op builder. Per-tier match/slot resolution is fused into the
        # Full-KV get/put implementations; this helper only appends SWA ops.
        self.swa_op_constructor = SWAOpConstructor(self)

        #TODO move this to kvmanager.start()
        self.start()

        # 空计划工厂：任何"规划失败 / 无需搬运"的早退路径都返回它 ——
        # 空图 + 空 mask + 空回调，让上层的任务收尾逻辑保持统一。
        self._empty_get_return: Callable[[int], GetTransferPlan] = \
            lambda request_id: GetTransferPlan.empty()
        self._empty_put_return: Callable[[int], PutTransferPlan] = \
            lambda request_id: PutTransferPlan.empty()

        # Update initial mempool stats
        self._update_mempool_metrics()

    def start(self) -> None:
        if self.cpu_cache_engine and self.cache_config.enable_p2p_cpu:
            self.cpu_cache_engine.start()
        if self.ssd_cache_engine and self.cache_config.enable_p2p_ssd:
            self.ssd_cache_engine.start()
        if self.remote_cache_engine and self.cache_config.enable_3rd_remote:
            self.remote_cache_engine.start()

    @_synchronized_cache_tree
    def reset(self) -> None:
        if self.cpu_cache_engine:
            self.cpu_cache_engine.reset()
        if self.ssd_cache_engine:
            self.ssd_cache_engine.reset()
        if self.remote_cache_engine:
            self.remote_cache_engine.reset()

    def _update_mempool_metrics(self) -> None:
        """Update memory pool metrics for all cache engines."""
        if self._metrics_collector is None:
            return
        for device_type, engine in self.cache_engines.items():
            if hasattr(engine, 'mempool'):
                device_label = DEVICE_TYPE[device_type].lower()
                self._metrics_collector.update_mempool_stats(
                    device_label,
                    engine.mempool.num_total_blocks,
                    engine.mempool.num_free_blocks
                )

    @_synchronized_cache_tree
    def get(self,
            request_id: int,
            token_ids: np.ndarray,
            token_mask: np.ndarray,
            slot_mapping: np.ndarray,
            dp_client_id: int,
            temp_cache_strategy: CacheStrategy = DEFAULT_CACHE_STRATEGY,
            namespace: Optional[List[str]] = None,
            swa_aware: bool = False) \
                 -> Tuple[TransferOpGraph, np.ndarray, Callable, Dict, int]:
        """GET 主入口：为一次前缀复用请求规划出传输 DAG。

        完整决策流程：
            1. 对齐到 block 粒度：不足一个 block 的尾部 token 直接放弃
               （KV Cache 以 block 为最小单位，半个块无法复用）。
            2. 由 token_mask 求出需要搬运的 block 区间
               [block_start_idx, block_end_idx)，并把 slot_mapping 换算成
               GPU 侧的目标 block 编号。
            3. 分派到 _get_impl_local / _get_impl_global 做真正的匹配与建图。
            4. 给所有 finished_ops 加一个虚拟汇点，得到 task_end_op_id
               （上层据此判断任务何时完成）。
            5. 计算 return_mask：告诉上层"哪些 token 不用重算了"。
            6. 锁住规划期用到的 radix 节点，并打包完成/回滚回调。

        Args:
            token_ids / token_mask : 本次请求的 token 与"需要 KV"的掩码。
                                     mask 为 True 表示该 token 需要从缓存拉
            slot_mapping           : GPU 侧 KV Cache 的槽位映射，用于换算
                                     数据要写到 GPU 的哪些 block
            dp_client_id           : 数据并行客户端 id，随 op 透传给数据面
            temp_cache_strategy    : 本次请求的临时策略（是否忽略 GPU/SSD/REMOTE）
            swa_aware              : 是否同时规划 SWA（滑动窗口注意力）快照

        Returns:
            (transfer_graph, return_mask, callback, op_callback_dict, task_end_op_id)
            - return_mask 为 True 的 token 已由本图搬到位，无需重算
            - callback 即 TransferPlanHandle：正常完成调它，取消则调 .abort()

        Side effects:
            可能已在各级 radix 树上 insert 了 is_ready=False 的占位节点、
            分配了 CPU/SSD 中转 block、并对匹配节点加了锁 —— 这些全部由
            callback / abort 负责收尾。
        """
        self._check_input(token_ids, token_mask, slot_mapping)

        # 只按整 block 处理：尾部不足一个 block 的 token 无法复用，直接掩掉
        aligned_length = (token_ids.shape[0] // self.tokens_per_block) * self.tokens_per_block

        aligned_token_ids = token_ids[:aligned_length]
        token_mask[aligned_length:] = False

        if aligned_length == 0 or not token_mask.any():
            transfer_graph = TransferOpGraph.create_empty_graph()
            return_mask = np.zeros_like(token_mask, dtype=np.bool_)
            callback = partial(self._transfer_callback, node_to_unlock={}, buffer_to_free={})
            return transfer_graph, return_mask, callback, {}, -1

        block_start_idx, block_end_idx = self._get_block_range(token_mask)
        # block_end_idx is the block just past the LAST True in token_mask. On the
        # plain path the caller marks every non-resident token up to the aligned
        # end, so this equals aligned_length // tokens_per_block. On the SWA-aware
        # path (swa_aware=True) _get_impl_* clamps the window to usable = min(full,
        # swa) after matching, which can end before the aligned length. So the
        # invariant is <= (can never exceed the aligned length), not ==. Nothing
        # below uses aligned_length; all downstream sizing keys off block_end_idx.
        assert block_end_idx <= aligned_length // self.tokens_per_block
        gpu_block_ids = self.slot_mapping_to_block_ids(slot_mapping,
                                                       self.tokens_per_block)[:block_end_idx-block_start_idx]

        sequence_meta = SequenceMeta(token_ids=aligned_token_ids,
                                     tokens_per_block=self.cache_config.tokens_per_block,
                                     namespace=namespace)

        temp_cache_strategy = resolve_get_cache_strategy(
            self.use_mooncake_store_backend, temp_cache_strategy)

        # 分派：远端不可用或本次策略忽略远端 -> 本地路径（CPU/SSD + peer）；
        # 否则走全局路径，把 REMOTE 也纳入匹配。两条路径产出同一种 GetTransferPlan。
        if not self.cache_config.enable_remote or temp_cache_strategy.ignore_remote:
            # from this entrance, we will also handle the case of peer_cpu and peer_ssd
            plan = self._get_impl_local(
                request_id,
                sequence_meta,
                block_start_idx,
                block_end_idx,
                gpu_block_ids,
                temp_cache_strategy,
                dp_client_id,
                swa_aware=swa_aware,
            )
        else:
            #TODO pcfs will be supported later
            plan = self._get_impl_global(
                request_id,
                sequence_meta,
                block_start_idx,
                block_end_idx,
                gpu_block_ids,
                temp_cache_strategy,
                dp_client_id,
                swa_aware=swa_aware,
            )

        # 把所有 finished_ops 汇聚成一个虚拟 op，作为整张图的完成标记
        transfer_graph, task_end_op_id = add_virtual_op_for_multiple_finished_ops(
            plan.transfer_graph,
            plan.finished_ops_ids,
            dp_client_id,
            )

        return_mask = np.zeros_like(token_mask, dtype=np.bool_)
        if temp_cache_strategy.ignore_gpu and temp_cache_strategy.ignore_gds:
            # 预取路径：只统计 Full REMOTE2H 搬下来的块（SWA 的 op 在另一个
            # slot 空间，不能混进来），并把 mask 对齐到远端片段的起点。
            # Prefetch return_mask covers Full REMOTE2H tokens only (A: planned
            # remote pull). SWA REMOTE2H ops live in a separate slot space and
            # must not be summed into prefetch_blocks. Place the True span at
            # the remote fragment start (f12), not block_start_idx — otherwise
            # a non-zero CPU/SSD prefix shifts the mask onto local blocks.
            prefetch_blocks = 0
            for op in transfer_graph._op_map.values():
                if op.transfer_type == TransferType.REMOTE2H and not op.is_swa:
                    prefetch_blocks += len(op.src_block_ids)
            if prefetch_blocks > 0:
                remote_start_block = block_start_idx
                for pending in plan.deferred_inserts:
                    if pending.device_type == DeviceType.CPU:
                        remote_start_block = pending.remote_start_block
                        break
                return_mask[remote_start_block * self.tokens_per_block:
                            (remote_start_block + prefetch_blocks) * self.tokens_per_block] = True
        else:
            return_mask[block_start_idx* self.tokens_per_block:
                    (block_start_idx + plan.num_gpu_blocks_to_transfer) * self.tokens_per_block] = True

        # if layer_num // layer_granularity != 1:
        #     transfer_graph, finished_ops_ids = convert_read_graph_to_layer_wise_graph(transfer_graph=transfer_graph,
        #                                                                         finished_ops_ids=finished_ops_ids,
        #                                                                         layer_num=layer_num,
        #                                                                         layer_granularity=layer_granularity)

        # 规划期锁住各 tier 上命中/插入的节点，防止数据在搬运途中被淘汰；
        # 解锁与置 ready 都推迟到完成回调（_transfer_callback）里做。
        for device_type in plan.node_to_unlock:
            self.cache_engines[device_type].lock_node(plan.node_to_unlock[device_type][0])

        callback = TransferPlanHandle(
            complete=partial(self._transfer_callback,
                             node_to_unlock=plan.node_to_unlock,
                             buffer_to_free=plan.buffer_to_free,
                             deferred_inserts=plan.deferred_inserts),
            abort=partial(self._abort_transfer_plan,
                          node_to_unlock=plan.node_to_unlock,
                          buffer_to_free=plan.buffer_to_free,
                          deferred_inserts=plan.deferred_inserts,
                          swa_reservation=plan.swa_reservation),
        )

        op_callback_dict = plan.op_callback_dict

        # Update mempool metrics after GET operation
        if self._metrics_collector is not None:
            self._update_mempool_metrics()

        return transfer_graph, return_mask, callback, op_callback_dict, task_end_op_id

    def _build_op_callback_dict(self, op_node_to_ready: Dict) -> Dict[int, Callable]:
        """把 {op_id: (device_type, node, ready_length)} 编译成 {op_id: 回调}。

        语义：某个 op 完成 == 它写往的那批 block 数据已落盘/落内存，
        于是把对应 radix 节点标记为 ready，从此可被后续请求命中。
        """
        op_callback_dict = {}
        for op_id, (device_type, node_to_ready, ready_length) in op_node_to_ready.items():
            op_callback_dict[op_id] = partial(self._op_callback,
                                              device_type=device_type,
                                              node_to_ready=node_to_ready,
                                              ready_length=ready_length)
        return op_callback_dict

    @staticmethod
    def _append_op_callback(op_callback_dict: Dict[int, Callable],
                            op_id: int,
                            callback: Callable) -> None:
        """Append ``callback`` without overwriting another completion action."""
        previous = op_callback_dict.get(op_id)
        if previous is None:
            op_callback_dict[op_id] = callback
            return

        def combined_callback() -> None:
            previous()
            callback()

        op_callback_dict[op_id] = combined_callback

    # SWA 槽位的"延迟挂载"：槽号必须在建图前就定下来（数据面要用它寻址），
    # 但要等本 tier 的传输完成回调才真正挂到 radix 节点上。这样 Full-KV 与
    # SWA 的发布互不依赖 —— 谁先写完都不影响，也不会暴露未写完的 SWA 字节。
    @_synchronized_cache_tree
    def _publish_swa_put_slot(self,
                              device_type: DeviceType,
                              node,
                              slot: int) -> None:
        """Make a reserved PUT slot readable after its tier transfer completes.

        The slot id is allocated before graph construction so the data plane can
        address it, but it is deliberately not mounted on the radix node until
        this callback.  Full-KV and SWA publication therefore remain independent:
        either transfer may finish first without exposing unfilled SWA bytes.
        """
        assert node is not None
        assert slot >= 0
        engine = self.cache_engines[device_type]
        engine.index.set_swa(node, int(slot))

    def _fail_put_before_insert(
            self,
            request_id: int,
            reason: str,
            cpu_blocks: np.ndarray,
            cpu_swa_slot: int = -1,
            ssd_blocks: Optional[np.ndarray] = None,
            ssd_swa_slot: int = -1,
            remote_blocks: Optional[np.ndarray] = None,
            remote_swa_slot: int = -1) -> PutTransferPlan:
        """PUT 在 insert 之前失败的统一收尾：归还已申请的 block 与 SWA 槽位。

        典型触发原因是 SWA 槽位分配失败。此时还没有任何节点上树，
        所有资源都还属于本次请求，可以安全全量回收后返回空计划。
        """
        flexkv_logger.warning(
            "[FlexKV-SWA] PUT request failed before radix insert; "
            f"request_id={request_id}, reason={reason}, "
            f"cpu_blocks={len(cpu_blocks)}, ssd_blocks={0 if ssd_blocks is None else len(ssd_blocks)}, "
            f"remote_blocks={0 if remote_blocks is None else len(remote_blocks)}, "
            f"cpu_swa_slot={cpu_swa_slot}, ssd_swa_slot={ssd_swa_slot}, "
            f"remote_swa_slot={remote_swa_slot}"
        )
        if cpu_swa_slot >= 0:
            self.cpu_cache_engine._free_swa_slot(cpu_swa_slot)
        if ssd_swa_slot >= 0:
            self.ssd_cache_engine._free_swa_slot(ssd_swa_slot)
        if remote_swa_slot >= 0:
            self.remote_cache_engine._free_swa_slot(remote_swa_slot)
        self.cpu_cache_engine.recycle(cpu_blocks)
        if ssd_blocks is not None:
            self.ssd_cache_engine.recycle(ssd_blocks)
        if remote_blocks is not None:
            self.remote_cache_engine.recycle(remote_blocks)
        return self._empty_put_return(request_id)

    # ------------------------------------------------------------------
    # 全局（分布式 / 含远端）GET 路径：CPU / SSD / REMOTE 三级一起参与匹配
    #
    # 决策思路（对照下面的 transfer pattern 图）：
    #   1. 三级各自做一次前缀匹配，得到三个"命中长度"；
    #   2. 因为缓存是分层的、命中必然是前缀，三级命中长度天然可比：
    #      CPU 命中最短的是 fragment1，SSD 比 CPU 多出来的那段是 fragment2，
    #      REMOTE 比 CPU/SSD 并集多出来的那段是 fragment3；
    #   3. fragment1 已在 CPU 内存里，直接 H2D；
    #      fragment2 需要 DISK2H（SSD -> CPU）再 H2D；
    #      fragment3 需要 REMOTE2H（远端 -> CPU）再 H2D；
    #   4. 从远端拉回来的数据顺手 H2DISK 回填 SSD（下次就不用再走远端）；
    #   5. 所有 fragment 在 CPU 内存中拼成一段连续区间，最后统一一次 H2D 上 GPU。
    #
    # 与 _get_impl_local 的区别：
    #   本路径多一层 REMOTE 参与，且会把远端数据回填 SSD；适合开了
    #   enable_remote 且本次策略未忽略远端的场景（多为预取任务）。
    # ------------------------------------------------------------------
    def _get_impl_global(self,
            request_id: int,
            sequence_meta: SequenceMeta,
            block_mask_start: int,
            block_mask_end: int,
            gpu_block_ids: np.ndarray,
            temp_cache_strategy: CacheStrategy,
            dp_client_id: int,
            swa_aware: bool = False) \
                 -> GetTransferPlan:
        """
        transfer pattern:

        GPU: (gpu cached) | fragment1 | fragment2      | fragment3      | (need compute)
                               ↑          ↑               ↑
        CPU:     ...      | fragment1 | fragment2(new) | fragment3(new) ← (from REMOTE)
                                          ↑               ↓
        SSD:     ...      | fragment1 | fragment2      | fragment3(new)

        """
        enable_gpu = not temp_cache_strategy.ignore_gpu
        enable_cpu = self.cache_config.enable_cpu
        enable_ssd = self.cache_config.enable_ssd
        enable_remote = self.cache_config.enable_remote and not temp_cache_strategy.ignore_remote
        assert enable_cpu and enable_remote
        assert self.cpu_cache_engine is not None
        assert self.remote_cache_engine is not None
        if self.index_accel:
            cpu_matched_result, ssd_matched_result, remote_matched_result = self.match_all_accel(sequence_meta)
        else:
            cpu_matched_result, ssd_matched_result, remote_matched_result = self.match_all(sequence_meta)
        transfer_graph = TransferOpGraph()
        swa_reservation: Optional[SWAReadReservation] = None
        swa_read_source: SWAReadSource = SWAReadSource()
        if swa_aware:
            block_mask_end, swa_read_source = self._select_swa_read_source(
                block_mask_start,
                block_mask_end,
                {DeviceType.CPU: cpu_matched_result,
                 DeviceType.SSD: ssd_matched_result,
                 DeviceType.REMOTE: remote_matched_result},
                sequence_meta=sequence_meta,
            )
            protected_cpu_node = (
                cpu_matched_result.last_ready_node
                if cpu_matched_result.num_ready_matched_blocks > block_mask_start
                else None
            )
            if enable_gpu:
                swa_reservation = self._reserve_swa_read_source(
                    transfer_graph, swa_read_source, protected_cpu_node, dp_client_id)
            # Compute path: SWA reservation failure -> no Full-only restore.
            # Prefetch path (not enable_gpu): reservation is intentionally
            # skipped; the SWA REMOTE2H is planned later in the joint block,
            # and the commit-time guard enforces the tree invariant.
            if enable_gpu and swa_read_source.found and swa_reservation is None:
                block_mask_end = block_mask_start
            if (enable_gpu and swa_read_source.found and swa_reservation is None
                    and self._metrics_collector is not None):
                self._metrics_collector.record_allocation_failure("global")
        cpu_matched_blocks = cpu_matched_result.physical_blocks[
            :cpu_matched_result.num_ready_matched_blocks][block_mask_start:block_mask_end]
        ssd_matched_blocks = ssd_matched_result.physical_blocks[
            :ssd_matched_result.num_ready_matched_blocks][block_mask_start:block_mask_end]
        remote_matched_blocks = remote_matched_result.physical_blocks[
            :remote_matched_result.num_ready_matched_blocks][block_mask_start:block_mask_end]
        shared_pcfs_read = (self.cache_config.enable_kv_sharing and self.index_accel
                            and not self.use_mooncake_store_backend)
        remote_file_nodeids = None
        if shared_pcfs_read:
            remote_file_nodeids = remote_matched_result.block_node_ids
        fragment123_num_blocks = max(len(cpu_matched_blocks), len(ssd_matched_blocks), len(remote_matched_blocks))
        #early return if no blocks to transfer
        if fragment123_num_blocks == 0:
            self._release_swa_read_reservation(swa_reservation)
            # All cache levels missed - record miss for all requested blocks
            if self._metrics_collector is not None:
                total_query_blocks = block_mask_end - block_mask_start
                if total_query_blocks > 0:
                    self._metrics_collector.record_cache_miss(total_query_blocks)
            return self._empty_get_return(request_id)
        assert fragment123_num_blocks <= len(gpu_block_ids)

        finished_ops_ids = []

        # 三级命中长度的"并集切分"：因为命中都是前缀，短的必然被长的包含，
        # 所以可以直接用"更深层命中数 - 浅层命中数"得到各 fragment 的长度。
        fragment1_num_blocks = len(cpu_matched_blocks)
        fragment2_num_blocks = max(len(ssd_matched_blocks) - len(cpu_matched_blocks), 0)
        fragment12_num_blocks = max(len(cpu_matched_blocks), len(ssd_matched_blocks))
        fragment3_num_blocks = max(len(remote_matched_blocks) - fragment12_num_blocks, 0)
        fragment23_num_blocks = fragment2_num_blocks + fragment3_num_blocks
        defer_mooncake_commit = (
            self.use_mooncake_store_backend and fragment3_num_blocks > 0)

        fragment123_gpu_blocks = gpu_block_ids[:fragment123_num_blocks]
        fragment123_cpu_blocks = cpu_matched_blocks
        fragment2_ssd_blocks = ssd_matched_blocks[-fragment2_num_blocks:]
        fragment3_remote_blocks = remote_matched_blocks[-fragment3_num_blocks:]
        fragment3_remote_file_nodeids = None
        if shared_pcfs_read:
            fragment3_remote_file_nodeids = remote_file_nodeids[-fragment3_num_blocks:]
        cpu_node_to_unlock = cpu_matched_result.last_ready_node
        ssd_node_to_unlock = ssd_matched_result.last_ready_node
        remote_node_to_unlock = remote_matched_result.last_ready_node
        cpu_blocks_to_free = np.array([], dtype=np.int64)
        cpu_node_to_ready = None
        ssd_node_to_ready = None

        if fragment23_num_blocks > 0:
            # CPU 内存是唯一的中转层：SSD 和远端的数据都要先落到这里，
            # 所以先按 fragment2 + fragment3 的总量申请 CPU block（可能触发淘汰）
            num_extra_required_blocks = fragment23_num_blocks
            try:
                fragment23_cpu_blocks = self.cpu_cache_engine.take(
                    num_required_blocks=num_extra_required_blocks,
                    protected_node=cpu_matched_result.last_node,
                    strict=True
                )
            except RuntimeError:
                self._release_swa_read_reservation(swa_reservation)
                if self._metrics_collector is not None:
                    self._metrics_collector.record_allocation_failure("global")
                return self._empty_get_return(request_id)
            if len(fragment23_cpu_blocks) < num_extra_required_blocks:
                self.cpu_cache_engine.recycle(fragment23_cpu_blocks)
                self._release_swa_read_reservation(swa_reservation)
                # Record allocation failure (resource unavailable, not cache miss)
                if self._metrics_collector is not None:
                    self._metrics_collector.record_allocation_failure("global")
                return self._empty_get_return(request_id)
            fragment123_cpu_blocks = np.concatenate([fragment123_cpu_blocks, fragment23_cpu_blocks])
            # Mooncake can partially fail after match. Keep its staging blocks
            # detached until graph completion; the callback will fresh-rematch
            # and publish only the successfully loaded prefix.
            # Non-mooncake still inserts now (is_ready=False) and publishes via
            # cpu_node_to_ready after the host-stage virtual join completes.
            if not defer_mooncake_commit:
                if (cpu_matched_result.num_ready_matched_blocks >= block_mask_start and
                        cpu_matched_result.num_ready_matched_blocks ==
                        cpu_matched_result.num_matched_blocks):
                    cpu_node_to_unlock = self.cpu_cache_engine.insert(
                        sequence_meta,
                        fragment23_cpu_blocks,
                        num_insert_blocks=fragment123_num_blocks + block_mask_start,
                        is_ready=False,
                        match_result=cpu_matched_result,
                    )
                    cpu_node_to_ready = cpu_node_to_unlock
                else:
                    cpu_blocks_to_free = fragment23_cpu_blocks

        # Record cache hit/miss metrics after confirming successful allocation
        if self._metrics_collector is not None:
            total_query_blocks = block_mask_end - block_mask_start
            # CPU hit blocks (directly from CPU cache)
            self._metrics_collector.record_cache_hit("cpu", fragment1_num_blocks)
            # SSD hit blocks (blocks loaded from SSD)
            self._metrics_collector.record_cache_hit("ssd", fragment2_num_blocks)
            # Remote hit blocks (blocks loaded from remote)
            self._metrics_collector.record_cache_hit("remote", fragment3_num_blocks)
            # Miss blocks (not in any cache)
            miss_blocks = total_query_blocks - fragment123_num_blocks
            if miss_blocks > 0:
                self._metrics_collector.record_cache_miss(miss_blocks)

        op_disk2h = None
        if fragment2_num_blocks > 0:
            op_disk2h = TransferOp(
                graph_id = transfer_graph.graph_id,
                transfer_type = TransferType.DISK2H,
                src_block_ids = fragment2_ssd_blocks,
                dst_block_ids = fragment123_cpu_blocks[fragment1_num_blocks:fragment12_num_blocks],
                dp_client_id = dp_client_id,
            )
            transfer_graph.add_transfer_op(op_disk2h)

        op_remote2h = None
        if fragment3_num_blocks > 0:
            # mooncake 是"按 key 寻址"的远端存储：地址只有 block hash 的尾值，
            # 没有 radix 节点 / host slot，所以额外带上 hash 列表作为远端句柄
            mooncake_block_hashes = None
            if self.use_mooncake_store_backend:
                mooncake_block_hashes = sequence_meta.block_hashes[
                    block_mask_start + fragment12_num_blocks:
                    block_mask_start + fragment12_num_blocks + fragment3_num_blocks
                ]
            op_remote2h = TransferOp(
                graph_id = transfer_graph.graph_id,
                transfer_type = TransferType.REMOTE2H,
                src_block_ids = fragment3_remote_blocks,
                dst_block_ids = fragment123_cpu_blocks[-fragment3_num_blocks:],
                src_block_node_ids = fragment3_remote_file_nodeids,
                dp_client_id = dp_client_id,
                mooncake_store_block_hashes = mooncake_block_hashes,
            )
            transfer_graph.add_transfer_op(op_remote2h)

        # 回填 SSD：把从远端拉回来的 fragment3 顺手 H2DISK 写进本地 SSD，
        # 下次同样前缀就不用再走远端。条件很苛刻 —— 必须 SSD 上已有的缓存
        # 正好构成本次命中的连续前缀，否则接上去会破坏 radix 树的前缀连续性。
        # prepare ssd blocks to transfer
        write_ssd_blocks_from_remote = False
        if (enable_ssd and
            op_remote2h is not None and
            ssd_matched_result.num_ready_matched_blocks >= block_mask_start and
            ssd_matched_result.num_ready_matched_blocks == ssd_matched_result.num_matched_blocks and
            ssd_matched_result.num_matched_blocks == block_mask_start + fragment12_num_blocks):
            # only when the above all are satisfied, we load data back from cpu to ssd
            write_ssd_blocks_from_remote = True
            fragment3_ssd_blocks = self.ssd_cache_engine.take(
                num_required_blocks=fragment3_num_blocks,
                protected_node=ssd_matched_result.last_node,
                strict=False
            )
            if len(fragment3_ssd_blocks) < fragment3_num_blocks:
                self.ssd_cache_engine.recycle(fragment3_ssd_blocks)
                write_ssd_blocks_from_remote = False
            if write_ssd_blocks_from_remote:
                op_h2disk = TransferOp(
                    graph_id = transfer_graph.graph_id,
                    transfer_type = TransferType.H2DISK,
                    src_block_ids = fragment123_cpu_blocks[-fragment3_num_blocks:],
                    dst_block_ids = fragment3_ssd_blocks,
                    dp_client_id = dp_client_id,
                )
                transfer_graph.add_transfer_op(op_h2disk)
                transfer_graph.add_dependency(op_h2disk.op_id, op_remote2h.op_id)

                if not defer_mooncake_commit:
                    ssd_node_to_unlock = self.ssd_cache_engine.insert(
                        sequence_meta,
                        fragment3_ssd_blocks,
                        num_insert_blocks=fragment123_num_blocks + block_mask_start,
                        is_ready=False,
                        match_result=ssd_matched_result,
                    )
                    ssd_node_to_ready = ssd_node_to_unlock

        # A prefetch has no H2D op, so its terminal op must be the host-stage
        # transfer itself.  The global path also inserts REMOTE2H/DISK2H
        # destinations into the CPU radix as ``is_ready=False``.  Publish that
        # node only after every host-stage fragment is complete; otherwise the
        # unready nodes accumulate, cannot be evicted, and eventually exhaust
        # the CPU pool.  A virtual join preserves parallel SSD and remote IO.
        # Mooncake insert-after skips plan-time insert, so cpu/ssd_node_to_ready
        # stay None and these set_ready callbacks are not registered.
        # 预取任务没有 H2D，所以它的"终止 op"只能是 host 侧的传输本身。
        # 这里给 DISK2H / REMOTE2H 加一个虚拟汇点，等 host 侧全部落盘后再统一
        # 把规划期 insert 的 is_ready=False 节点置为 ready —— 否则未 ready 节点
        # 会不断堆积、既不能被命中也不能被淘汰，最终耗尽 CPU 池。
        # 用虚拟汇点而非串行依赖，是为了保留 SSD 与远端 IO 的并行度。
        # Mooncake 走延迟上树（规划期不 insert），故 cpu/ssd_node_to_ready 为 None，
        # 这些 set_ready 回调也就不会注册。
        op_callback_dict = {}
        host_finished_ops_ids = [
            op.op_id for op in (op_disk2h, op_remote2h) if op is not None
        ]
        host_ready_op_id = -1
        if host_finished_ops_ids:
            transfer_graph, host_ready_op_id = add_virtual_op_for_multiple_finished_ops(
                transfer_graph, host_finished_ops_ids, dp_client_id
            )
            if not enable_gpu:
                finished_ops_ids.append(host_ready_op_id)
        if cpu_node_to_ready is not None:
            assert host_ready_op_id >= 0
            self._append_op_callback(
                op_callback_dict,
                host_ready_op_id,
                partial(
                    self._op_callback,
                    device_type=DeviceType.CPU,
                    node_to_ready=cpu_node_to_ready,
                    ready_length=cpu_node_to_ready.size(),
                ),
            )
        if ssd_node_to_ready is not None:
            assert op_h2disk is not None
            self._append_op_callback(
                op_callback_dict,
                op_h2disk.op_id,
                partial(
                    self._op_callback,
                    device_type=DeviceType.SSD,
                    node_to_ready=ssd_node_to_ready,
                    ready_length=ssd_node_to_ready.size(),
                ),
            )
        if enable_gpu:
            # 统一 H2D：CPU 内存里已经拼成连续区间，一次搬上 GPU。
            # 依赖边确保 H2D 一定在 DISK2H / REMOTE2H 之后执行。
            op_h2d = TransferOp(
                graph_id = transfer_graph.graph_id,
                transfer_type = TransferType.H2D,
                src_block_ids = fragment123_cpu_blocks,
                dst_block_ids = fragment123_gpu_blocks,
                dp_client_id = dp_client_id,
            )
            transfer_graph.add_transfer_op(op_h2d)
            if op_disk2h is not None:
                transfer_graph.add_dependency(op_h2d.op_id, op_disk2h.op_id)
            if op_remote2h is not None:
                transfer_graph.add_dependency(op_h2d.op_id, op_remote2h.op_id)
            finished_ops_ids.append(op_h2d.op_id)

        node_to_unlock = {}
        if cpu_node_to_unlock is not None:
            node_to_unlock[DeviceType.CPU] = (
                cpu_node_to_unlock,
                0 if defer_mooncake_commit else cpu_node_to_unlock.size(),
            )
        if ssd_node_to_unlock is not None:
            node_to_unlock[DeviceType.SSD] = (
                ssd_node_to_unlock,
                0 if defer_mooncake_commit else ssd_node_to_unlock.size(),
            )
        if remote_node_to_unlock is not None:
            node_to_unlock[DeviceType.REMOTE] = (remote_node_to_unlock, remote_node_to_unlock.size())

        buffer_to_free = {DeviceType.CPU: cpu_blocks_to_free}
        num_gpu_blocks_to_transfer = len(fragment123_gpu_blocks) if enable_gpu else 0
        deferred_inserts: List[DeferredCacheInsert] = []

        # construct the SWA op for joint prefetch
        joint_prefetch_swa_slot = -1
        joint_prefetch_swa_anchor = -1
        joint_prefetch_swa_load_result: Optional[MooncakeLoadResult] = None
        if (defer_mooncake_commit
                and swa_aware
                and not enable_gpu
                and swa_reservation is None
                and swa_read_source.found
                and swa_read_source.is_mooncake
                and self.swa_op_constructor.enabled):
            joint_prefetch_swa_slot = self.cpu_cache_engine._alloc_swa_slot(
                protected_node=cpu_matched_result.last_ready_node)
            if joint_prefetch_swa_slot >= 0:
                joint_prefetch_swa_anchor = swa_read_source.hit_blocks - 1
                swa_op_id = self.swa_op_constructor.build_swa_op(
                    transfer_graph,
                    TransferType.REMOTE2H,
                    src_slot_ids=np.array([0], dtype=np.int64),
                    dst_slot_ids=np.array(
                        [joint_prefetch_swa_slot], dtype=np.int64),
                    dp_client_id=dp_client_id,
                    mooncake_tail_hashes=[swa_read_source.mooncake_tail_hash],
                )
                if swa_op_id is None:
                    # SWA transfer gated off after all; return the slot.
                    self.cpu_cache_engine._free_swa_slot(joint_prefetch_swa_slot)
                    joint_prefetch_swa_slot = -1
                    joint_prefetch_swa_anchor = -1
                else:
                    joint_prefetch_swa_load_result = MooncakeLoadResult()
                    op_callback_dict[swa_op_id] = CompletionAwareCallback(
                        joint_prefetch_swa_load_result.record)
                    # Report the SWA REMOTE2H as a finished op so the graph
                    # cannot complete before its mask lands on the pending.
                    finished_ops_ids.append(swa_op_id)
            else:
                flexkv_logger.warning(
                    "[FlexKV-SWA] Joint prefetch SWA slot allocation failed; "
                    f"falling back to Full-only prefetch, request_id={request_id}"
                )
                if self._metrics_collector is not None:
                    self._metrics_collector.record_joint_prefetch_swa_slot_alloc_failure()

        if defer_mooncake_commit:
            assert op_remote2h is not None
            load_result = MooncakeLoadResult()
            # CPU publication drives prefetch return_mask; SSD commit is
            # independent and must not overwrite this tracker.
            publish_result = DeferredPublishResult()
            op_callback_dict[op_remote2h.op_id] = CompletionAwareCallback(
                load_result.record)
            remote_start_block = block_mask_start + fragment12_num_blocks
            requested_end_block = block_mask_start + fragment123_num_blocks
            deferred_inserts.append(DeferredCacheInsert(
                device_type=DeviceType.CPU,
                sequence_meta=sequence_meta,
                physical_blocks=fragment23_cpu_blocks,
                staged_start_block=block_mask_start + fragment1_num_blocks,
                remote_start_block=remote_start_block,
                requested_end_block=requested_end_block,
                load_result=load_result,
                swa_slot=joint_prefetch_swa_slot,
                swa_anchor_block=joint_prefetch_swa_anchor,
                swa_load_result=joint_prefetch_swa_load_result,
                publish_result=publish_result,
            ))
            if write_ssd_blocks_from_remote:
                deferred_inserts.append(DeferredCacheInsert(
                    device_type=DeviceType.SSD,
                    sequence_meta=sequence_meta,
                    physical_blocks=fragment3_ssd_blocks,
                    staged_start_block=remote_start_block,
                    remote_start_block=remote_start_block,
                    requested_end_block=requested_end_block,
                    load_result=load_result,
                ))
        elif joint_prefetch_swa_slot >= 0:
            # SWA op was planned but Full mooncake path was skipped
            # (defer_mooncake_commit False): no CPU pending will consume the
            # slot, so free it and drop the op-side callback wiring rather
            # than leak a slot into an untracked graph.
            self.cpu_cache_engine._free_swa_slot(joint_prefetch_swa_slot)
            joint_prefetch_swa_slot = -1
        if swa_reservation is not None:
            assert num_gpu_blocks_to_transfer > 0
            finished_ops_ids.append(swa_reservation.h2d_id)
            op_callback_dict[swa_reservation.h2d_id] = partial(
                self._swa_release_load_lock,
                node=swa_reservation.source.node,
                staging_slot=swa_reservation.staging_slot,
                engine=swa_reservation.source.engine,
            )

        return GetTransferPlan(
            transfer_graph=transfer_graph,
            finished_ops_ids=finished_ops_ids,
            node_to_unlock=node_to_unlock,
            op_callback_dict=op_callback_dict,
            buffer_to_free=buffer_to_free,
            num_gpu_blocks_to_transfer=num_gpu_blocks_to_transfer,
            deferred_inserts=deferred_inserts,
            swa_reservation=swa_reservation,
        )

    # ------------------------------------------------------------------
    # 本地 GET 路径：只在 CPU 内存 / 本地 SSD（以及 peer 节点）里匹配，不查远端
    #
    # 决策流程（对照下面的 transfer pattern 图）：
    #   1. CPU 与 SSD 各做一次前缀匹配，取"已 ready"的部分并裁剪到本次 mask 区间；
    #   2. 切分 fragment：
    #        fragment1 = CPU 命中的块（已在内存，可直接 H2D）
    #        fragment2 = CPU 没命中、SSD 命中的块（需先搬到内存）
    #      两者构成一段连续前缀 fragment12 = fragment1 + fragment2；
    #   3. 申请 CPU 中转 block（只在真的需要经内存中转时才申请 —— GDS 可以
    #      SSD 直通 GPU，就一分 CPU block 都不用）；
    #   4. 建图：
    #        GDS 开启 ：fragment2 走 DISK2D（SSD -> GPU 直通）
    #        否则     ：fragment2 走 DISK2H（SSD -> CPU），再统一 H2D
    #        peer 命中：CPU 走 PEERH2H、SSD 走 PEERSSD2H（从别的节点搬）
    #   5. 把 fragment2 的中转 block insert 进 CPU radix 树（is_ready=False），
    #      并登记 op -> node 的回调，等 DISK2H 完成后 set_ready；
    #   6. 最后一条 H2D 把整段搬上 GPU，并作为本请求的 finished op。
    #
    # 与 _get_impl_global 的区别：
    #   本路径不查远端、没有 fragment3、也不会把数据回填 SSD；peer 节点
    #   （HierarchyLRCacheEngine）也算在本地路径里，通过 matched_pos=="remote"
    #   区分"命中落在别的节点上"。适用场景：未开启远端，或本次策略忽略了远端。
    #
    # 一个容易踩的坑：中转 block 只有在满足"插入后仍是连续前缀"且"命中部分
    # 全部 ready"时才 insert 上树，否则只能标记为 buffer_to_free，传输完成后
    # 直接还给 mempool —— 不能把断裂的前缀挂到树上。
    # ------------------------------------------------------------------
    def _get_impl_local(self,
                        request_id: int,
                        sequence_meta: SequenceMeta,
                        block_mask_start: int,
                        block_mask_end: int,
                        gpu_block_ids: np.ndarray,
                        temp_cache_strategy: CacheStrategy,
                        dp_client_id: int,
                        swa_aware: bool = False) \
                            -> GetTransferPlan:
        """
        transfer pattern:

        GPU          : (gpu cached) | fragment1 | fragment2      | (need compute)
                               ↑          ↑
        CPU(+peerCPU):     ...      | fragment1 | fragment2(new) | (uncached)
                                          ↑
        SSD(+peerSSD):     ...      | fragment1 | fragment2      | (uncached)

        """
        nvtx_range = nvtx.start_range(message=f"CacheEngine.get_impl_local[{request_id}]", color="cyan")
        enable_gpu = not temp_cache_strategy.ignore_gpu
        enable_cpu = self.cache_config.enable_cpu
        enable_ssd = self.cache_config.enable_ssd and not temp_cache_strategy.ignore_ssd
        enable_gds = self.cache_config.enable_gds and not temp_cache_strategy.ignore_gds
        assert enable_cpu
        assert self.cpu_cache_engine is not None

        if self.index_accel:
            cpu_matched_result, ssd_matched_result = self.match_local_accel(
                sequence_meta, temp_cache_strategy, is_put=False, gpu_matched_blocks=block_mask_start)
        else:
            cpu_matched_result, ssd_matched_result = self.match_local(sequence_meta, temp_cache_strategy)

        transfer_graph = TransferOpGraph()
        swa_reservation: Optional[SWAReadReservation] = None
        swa_read_source: SWAReadSource = SWAReadSource()
        if swa_aware:
            block_mask_end, swa_read_source = self._select_swa_read_source(
                block_mask_start,
                block_mask_end,
                {DeviceType.CPU: cpu_matched_result,
                 DeviceType.SSD: ssd_matched_result},
                sequence_meta=sequence_meta,
            )
            protected_cpu_node = (
                cpu_matched_result.last_ready_node
                if cpu_matched_result.num_ready_matched_blocks > block_mask_start
                else None
            )
            if enable_gpu:
                swa_reservation = self._reserve_swa_read_source(
                    transfer_graph, swa_read_source, protected_cpu_node, dp_client_id)
            # Align with _get_impl_global: only refuse Full-only restore when
            # compute needs GPU SWA but reservation failed. Prefetch
            # (not enable_gpu) leaves Full intact; SWA is staged separately.
            # (When no SWA source exists, _select_swa_read_source already
            # returned block_mask_end == block_mask_start.)
            if enable_gpu and swa_read_source.found and swa_reservation is None:
                block_mask_end = block_mask_start
            if (enable_gpu and swa_read_source.found and swa_reservation is None
                    and self._metrics_collector is not None):
                self._metrics_collector.record_allocation_failure("local")

        # DEBUG: Log GET operation with hash info
        #if len(sequence_meta.block_hashes) > 0:
        #    print(f"[GET {request_id}] hash[0]={sequence_meta.block_hashes[0]}, "
        #          f"CPU={cpu_matched_result.num_matched_blocks}/{cpu_matched_result.num_ready_matched_blocks}, "
        #          f"SSD={ssd_matched_result.num_matched_blocks}/{ssd_matched_result.num_ready_matched_blocks}, "
        #          f"pos_CPU={cpu_matched_result.matched_pos}, pos_SSD={ssd_matched_result.matched_pos}")

        # tailor the blocks to assure:
        # the blocks are needed by the mask & the blocks are ready
        cpu_matched_blocks = cpu_matched_result.physical_blocks[:cpu_matched_result.num_ready_matched_blocks]
        cpu_matched_blocks = cpu_matched_blocks[block_mask_start:block_mask_end]
        # if ssd disabled, len(ssd_physical_blocks) is 0
        ssd_matched_blocks = ssd_matched_result.physical_blocks[:ssd_matched_result.num_ready_matched_blocks]
        ssd_matched_blocks = ssd_matched_blocks[block_mask_start:block_mask_end]

        # TODO: is this possible?
        if len(cpu_matched_blocks) > len(ssd_matched_blocks):
            ssd_matched_blocks = np.array([], dtype=np.int64)

        fragment12_num_blocks = max(len(cpu_matched_blocks), len(ssd_matched_blocks))
        fragment1_num_blocks = len(cpu_matched_blocks)
        # 命中必然是前缀，所以 SSD 比 CPU 多出来的那一段就是"只有 SSD 有"的部分
        fragment2_num_blocks = max(len(ssd_matched_blocks) - len(cpu_matched_blocks), 0)
        #early return if no blocks to transfer
        if fragment12_num_blocks == 0:
            self._release_swa_read_reservation(swa_reservation)
            # All cache levels missed - record miss for all requested blocks
            if self._metrics_collector is not None:
                total_query_blocks = block_mask_end - block_mask_start
                if total_query_blocks > 0:
                    self._metrics_collector.record_cache_miss(total_query_blocks)
            nvtx.end_range(nvtx_range)
            return self._empty_get_return(request_id)
        assert fragment12_num_blocks <= len(gpu_block_ids)

        finished_ops_ids = []
        op_node_to_ready = {}

        fragment12_gpu_blocks = gpu_block_ids[:fragment12_num_blocks]
        fragment2_ssd_blocks = ssd_matched_blocks[-fragment2_num_blocks:]
        fragment1_cpu_blocks = cpu_matched_blocks[:fragment1_num_blocks]

        cpu_node_to_unlock = cpu_matched_result.last_ready_node
        ssd_node_to_unlock = ssd_matched_result.last_ready_node

        # prepare cpu blocks to transfer
        cpu_blocks_to_free = np.array([], dtype=np.int64)
        op_disk2h = None
        op_gds_transfer = None
        fragment2_cpu_blocks = None

        # Allocate CPU blocks only for paths that actually stage data through
        # host memory. GDS moves fragment2 directly from SSD to GPU.
        allocated_cpu_block_num = 0 if enable_gds else fragment2_num_blocks
        # Remote CPU hits still need local CPU blocks for PEERH2H staging,
        # regardless of whether those blocks are inserted into the local index.
        if cpu_matched_result.matched_pos == "remote" and fragment1_num_blocks > 0:
            allocated_cpu_block_num += fragment1_num_blocks
        if allocated_cpu_block_num > 0:
            nvtx.push_range(f"take {allocated_cpu_block_num} cpu blocks", color="green")
            allocated_cpu_blocks = self.cpu_cache_engine.take(
                num_required_blocks=allocated_cpu_block_num,
                protected_node=cpu_matched_result.last_node,
                strict=False
            )
            nvtx.pop_range()
        else:
            # take(0) may still trigger proactive eviction at high utilization.
            allocated_cpu_blocks = np.empty(0, dtype=np.int64)
        # NOTE: not enough space to allocate, skip the request
        # there might be a better way to handle this
        if len(allocated_cpu_blocks) < allocated_cpu_block_num:
            self.cpu_cache_engine.recycle(allocated_cpu_blocks)
            self._release_swa_read_reservation(swa_reservation)
            # Record allocation failure (resource unavailable, not cache miss)
            if self._metrics_collector is not None:
                self._metrics_collector.record_allocation_failure("local")
            nvtx.end_range(nvtx_range)
            return self._empty_get_return(request_id)

        # Record cache hit/miss metrics after confirming successful allocation
        if self._metrics_collector is not None:
            total_query_blocks = block_mask_end - block_mask_start
            # CPU hit blocks (directly from CPU cache)
            self._metrics_collector.record_cache_hit("cpu", fragment1_num_blocks)
            # SSD hit blocks (loaded directly to GPU with GDS, otherwise via CPU)
            self._metrics_collector.record_cache_hit("ssd", fragment2_num_blocks)
            # Miss blocks (not in any cache)
            miss_blocks = total_query_blocks - fragment12_num_blocks
            if miss_blocks > 0:
                self._metrics_collector.record_cache_miss(miss_blocks)

        if cpu_matched_result.matched_pos == "remote" and fragment1_num_blocks > 0:
            fragment1_cpu_blocks_local = allocated_cpu_blocks[-fragment1_num_blocks:]
            op_peerh2h = TransferOp(
                graph_id = transfer_graph.graph_id,
                transfer_type = TransferType.PEERH2H,
                src_block_ids = fragment1_cpu_blocks,
                dst_block_ids = fragment1_cpu_blocks_local,
                remote_node_ids = cpu_matched_result.matched_node_ids,
                src_block_node_ids = cpu_matched_result.matched_node_ids,  # Add this for worker
                dp_client_id = dp_client_id,
            )
            transfer_graph.add_transfer_op(op_peerh2h)
            # TODO here we dont combine peer cpu or local cpu match results,
            # so we can safely add remote results to local cpu
            #TODO here assume all matched blocks are ready blocks for peer cpu
            if (cpu_matched_result.insert_to_local_cpu_index and
                cpu_matched_result.num_ready_matched_blocks >= block_mask_start and
                cpu_matched_result.num_ready_matched_blocks == cpu_matched_result.num_matched_blocks):
                cpu_node_to_unlock = self.cpu_cache_engine.insert(sequence_meta,
                                                                  fragment1_cpu_blocks_local,
                                                                  is_ready=False)
                op_node_to_ready[op_peerh2h.op_id] = (DeviceType.CPU, cpu_node_to_unlock, cpu_node_to_unlock.size())
            else:
                cpu_blocks_to_free = np.concatenate([cpu_blocks_to_free, fragment1_cpu_blocks_local])

        if fragment2_num_blocks > 0:
            if enable_gds:
                # GDS：GPU 直接从 SSD 读，跳过 CPU 内存中转，省一次拷贝
                # For GDS, transfer directly from SSD to GPU using GDS transfer path (DISK2D)
                op_gds_transfer = TransferOp(
                    graph_id = transfer_graph.graph_id,
                    transfer_type = TransferType.DISK2D,
                    src_block_ids = fragment2_ssd_blocks,
                    dst_block_ids = fragment12_gpu_blocks[-fragment2_num_blocks:],
                    dp_client_id = dp_client_id,
                )
                transfer_graph.add_transfer_op(op_gds_transfer)
                finished_ops_ids.append(op_gds_transfer.op_id)
                op_node_to_ready[op_gds_transfer.op_id] = (DeviceType.SSD,
                                                           ssd_node_to_unlock,
                                                           ssd_node_to_unlock.size())
            else:
                fragment2_cpu_blocks = allocated_cpu_blocks[:fragment2_num_blocks]

                op_disk2h = TransferOp(
                    graph_id = transfer_graph.graph_id,
                    transfer_type = TransferType.PEERSSD2H
                        if ssd_matched_result.matched_pos == "remote" else TransferType.DISK2H,
                    src_block_ids = fragment2_ssd_blocks,
                    dst_block_ids = fragment2_cpu_blocks,
                    remote_node_ids = ssd_matched_result.matched_node_ids
                        if ssd_matched_result.matched_pos == "remote" else None,
                    src_block_node_ids = ssd_matched_result.matched_node_ids
                        if ssd_matched_result.matched_pos == "remote" else None,
                    dp_client_id = dp_client_id,
                )
                transfer_graph.add_transfer_op(op_disk2h)
                # 中转 block 只有在满足下面两点时才上树（否则只是一次性 buffer）：
                #   1. 命中落在本地 CPU（peer 命中的话，本地树接不上）；
                #   2. 命中的块全部 ready，且命中数已经覆盖本次请求的起点，
                #      保证插入后树里仍是连续前缀。
                # we only insert the buffer blocks to cpu cache engine only:
                # 1. the cpu cache engine satisfies prefix cache after insertion
                # 2. the sequence is all ready blocks
                # TODO: for simplicity, if we use peer cpu results,
                # we dont insert the buffer ssd blocks to local cpu any more
                if (cpu_matched_result.matched_pos == "local" and
                    cpu_matched_result.num_ready_matched_blocks >= block_mask_start and
                    cpu_matched_result.num_ready_matched_blocks == cpu_matched_result.num_matched_blocks):
                    cpu_node_to_unlock = self.cpu_cache_engine.insert(sequence_meta,
                                                                    fragment2_cpu_blocks,
                                                                    num_insert_blocks=fragment12_num_blocks + \
                                                                        block_mask_start,
                                                                    is_ready=False,
                                                                    match_result=cpu_matched_result)
                    op_node_to_ready[op_disk2h.op_id] = (DeviceType.CPU, cpu_node_to_unlock, cpu_node_to_unlock.size())
                else:
                    cpu_blocks_to_free = np.concatenate([cpu_blocks_to_free, fragment2_cpu_blocks])
        if self.cache_config.enable_p2p_cpu and cpu_matched_result.matched_pos == "remote" and fragment1_num_blocks > 0:
            fragment1_cpu_blocks = fragment1_cpu_blocks_local

        if fragment2_cpu_blocks is not None:
            fragment12_cpu_blocks = np.concatenate([fragment1_cpu_blocks, fragment2_cpu_blocks])
        else:
            fragment12_cpu_blocks = fragment1_cpu_blocks

        if enable_gpu:
            op_h2d = TransferOp(
                graph_id = transfer_graph.graph_id,
                transfer_type = TransferType.H2D,
                src_block_ids = fragment12_cpu_blocks if not enable_gds else fragment1_cpu_blocks,
                dst_block_ids = fragment12_gpu_blocks if not enable_gds \
                    else fragment12_gpu_blocks[:fragment1_num_blocks],
                dp_client_id = dp_client_id,
            )
            transfer_graph.add_transfer_op(op_h2d)
            if op_disk2h is not None:
                transfer_graph.add_dependency(op_h2d.op_id, op_disk2h.op_id)
            if cpu_matched_result.matched_pos == "remote" and fragment1_num_blocks > 0:
                transfer_graph.add_dependency(op_h2d.op_id, op_peerh2h.op_id)
            finished_ops_ids.append(op_h2d.op_id)

        node_to_unlock = {}
        if cpu_node_to_unlock is not None:
            node_to_unlock[DeviceType.CPU] = (cpu_node_to_unlock, cpu_node_to_unlock.size())
        if ssd_node_to_unlock is not None:
            node_to_unlock[DeviceType.SSD] = (ssd_node_to_unlock, ssd_node_to_unlock.size())
        buffer_to_free = {DeviceType.CPU: cpu_blocks_to_free}
        num_gpu_blocks_to_transfer = len(fragment12_gpu_blocks) if enable_gpu else 0
        op_callback_dict = self._build_op_callback_dict(op_node_to_ready)

        if swa_reservation is not None:
            assert num_gpu_blocks_to_transfer > 0
            finished_ops_ids.append(swa_reservation.h2d_id)
            op_callback_dict[swa_reservation.h2d_id] = partial(
                self._swa_release_load_lock,
                node=swa_reservation.source.node,
                staging_slot=swa_reservation.staging_slot,
                engine=swa_reservation.source.engine,
            )
        nvtx.end_range(nvtx_range)
        return GetTransferPlan(
            transfer_graph=transfer_graph,
            finished_ops_ids=finished_ops_ids,
            node_to_unlock=node_to_unlock,
            op_callback_dict=op_callback_dict,
            buffer_to_free=buffer_to_free,
            num_gpu_blocks_to_transfer=num_gpu_blocks_to_transfer,
            swa_reservation=swa_reservation,
        )

    @_synchronized_cache_tree
    def put(self,
            request_id: int,
            token_ids: np.ndarray,
            token_mask: np.ndarray,
            slot_mapping: np.ndarray,
            dp_client_id: int,
            temp_cache_strategy: CacheStrategy = DEFAULT_CACHE_STRATEGY,
            namespace: Optional[List[str]] = None) \
                -> Tuple[TransferOpGraph, np.ndarray, Callable, Dict, int]:
        """PUT 主入口：把 GPU 上刚算出来的 KV 下沉到各级缓存。

        流程与 get() 对称：
            1. 对齐到 block 粒度，并由 token_mask 得到待下沉的 block 区间
               （PUT 的 mask 必须是从 0 开始的前缀，故 block_start_idx == 0）；
            2. 分派到 _put_impl_local / _put_impl_global 建图：
               D2H（GPU -> CPU 内存）是最基本的一条边，SSD 与 REMOTE 都从
               CPU 内存再往下写（H2DISK / H2REMOTE）；
            3. 加虚拟汇点得到 task_end_op_id；
            4. 计算 return_mask：GPU 上已缓存的头部 block（skipped_gpu_blocks）
               不需要再搬，mask 从它之后开始；
            5. 锁住规划期涉及的节点，打包完成 / 回滚回调。

        Returns:
            与 get() 同构的五元组
        """
        self._check_input(token_ids, token_mask, slot_mapping)
        # ignore the last incomplete block
        aligned_length = (token_ids.shape[0] // self.tokens_per_block) * self.tokens_per_block
        aligned_token_ids = token_ids[:aligned_length]
        token_mask[aligned_length:] = False
        block_start_idx, block_end_idx = self._get_block_range(token_mask)

        # the mask should has a prefix of True
        assert block_start_idx == 0

        gpu_block_ids = self.slot_mapping_to_block_ids(slot_mapping,
                                                       self.tokens_per_block)[:block_end_idx-block_start_idx]

        sequence_meta = SequenceMeta(token_ids=aligned_token_ids,
                                     tokens_per_block=self.cache_config.tokens_per_block,
                                     namespace=namespace)

        assert not temp_cache_strategy.ignore_gpu
        if not self.cache_config.enable_remote or temp_cache_strategy.ignore_remote:
            plan = self._put_impl_local(
                request_id,
                sequence_meta,
                block_start_idx,
                block_end_idx,
                gpu_block_ids,
                temp_cache_strategy,
                dp_client_id,
            )
        else:
            plan = self._put_impl_global(
                request_id,
                sequence_meta,
                block_start_idx,
                block_end_idx,
                gpu_block_ids,
                temp_cache_strategy,
                dp_client_id,
            )

        transfer_graph, task_end_op_id = add_virtual_op_for_multiple_finished_ops(
            plan.transfer_graph,
            plan.finished_ops_ids,
            dp_client_id,
        )
        # return_mask 要从 skipped_gpu_blocks 之后开始：前面那些 GPU 上已经有了，
        # 不算本次搬运的成果。
        return_mask = np.zeros_like(token_mask, dtype=np.bool_)
        mask_lo = (block_start_idx + plan.skipped_gpu_blocks) * self.tokens_per_block
        mask_hi = (block_start_idx + plan.skipped_gpu_blocks
                   + plan.num_gpu_blocks_to_transfer) * self.tokens_per_block
        return_mask[mask_lo:mask_hi] = True

        for device_type in plan.node_to_unlock:
            self.cache_engines[device_type].lock_node(plan.node_to_unlock[device_type][0])

        callback = TransferPlanHandle(
            complete=partial(self._transfer_callback,
                             node_to_unlock=plan.node_to_unlock,
                             buffer_to_free=plan.buffer_to_free,
                             deferred_inserts=plan.deferred_inserts,
                             is_put=True),
            abort=partial(self._abort_transfer_plan,
                          node_to_unlock=plan.node_to_unlock,
                          buffer_to_free=plan.buffer_to_free,
                          deferred_inserts=plan.deferred_inserts,
                          swa_slots_to_free=plan.swa_slots_to_free),
        )

        op_callback_dict = plan.op_callback_dict

        # Update mempool metrics after PUT operation
        if self._metrics_collector is not None:
            self._update_mempool_metrics()

        return transfer_graph, return_mask, callback, op_callback_dict, task_end_op_id

    # ------------------------------------------------------------------
    # 全局 PUT 路径：GPU -> CPU 内存 -> SSD -> REMOTE 逐级下沉
    #
    # 决策思路（对照下面的 transfer pattern 图）：
    #   1. 三级各做一次前缀匹配，已经在缓存里的部分不需要重复写：
    #        CPU 已缓存 num_skipped_blocks 个 -> 这些 GPU block 直接跳过（D2H 只搬剩余部分）
    #        SSD 已缓存的更多 -> H2DISK 只写 fragment2（SSD 比 CPU 多出的那段）
    #        REMOTE 已缓存 -> H2REMOTE 只写 fragment3
    #   2. 建图：D2H 是源头；H2DISK / H2REMOTE 都挂在 D2H 之后（add_dependency），
    #      因为写 SSD / 远端的数据来源正是刚落到 CPU 内存的那些 block；
    #   3. 每级都先把目标节点 insert 上树（is_ready=False），由对应 op 的回调
    #      置 ready —— 与 GET 路径完全对称；
    #   4. mooncake 远端启用时改为"延迟上树"（defer_put_commit）：规划期完全不
    #      insert，只登记 DeferredCacheInsert，等图完成后再统一提交。
    #      额外地，PUT 的源地址必须是本进程注册的 host buffer，所以此时要用
    #      match_local 重新匹配（不能选到 peer 节点上）。
    # ------------------------------------------------------------------
    def _put_impl_global(self,
            request_id: int,
            sequence_meta: SequenceMeta,
            block_mask_start: int,
            block_mask_end: int,
            gpu_block_ids: np.ndarray,
            temp_cache_strategy: CacheStrategy,
            dp_client_id: int) \
                -> PutTransferPlan:
        """
        transfer pattern:

        GPU:   (skipped)  | fragment1      | fragment2      | (uncompleted block)
                               ↓                ↓
        CPU: (cpu cached) | fragment1(new) | fragment2(new) |
                                                ↓
        SSD:          (ssd cached)         | fragment2(new) |

        CPU:            ...           |     fragment3      |
                                               ↓ (from cpu)
        REMOTE:     (remote cached)   |   fragment3(new)   |

        """
        enable_gpu = not temp_cache_strategy.ignore_gpu
        enable_cpu = self.cache_config.enable_cpu
        enable_ssd = self.cache_config.enable_ssd and not temp_cache_strategy.ignore_ssd
        enable_remote = self.cache_config.enable_remote and not temp_cache_strategy.ignore_remote
        assert enable_gpu
        assert enable_cpu
        assert enable_remote
        assert self.cpu_cache_engine is not None
        assert self.remote_cache_engine is not None

        if self.index_accel:
            cpu_matched_result, ssd_matched_result, remote_matched_result = self.match_all_accel(sequence_meta,
                                                                                               temp_cache_strategy=temp_cache_strategy,
                                                                                               is_get=False)
        else:
            cpu_matched_result, ssd_matched_result, remote_matched_result = self.match_all(sequence_meta,
                                                                                           temp_cache_strategy=temp_cache_strategy)
        defer_put_commit = self.use_mooncake_store_backend
        if defer_put_commit:
            # PUT sources must be addresses in this process's registered host
            # buffers. A hierarchical match may select a peer node, which is a
            # valid GET source but cannot back local H2DISK/H2REMOTE writes.
            match_cpu_local = getattr(self.cpu_cache_engine, "match_local", None)
            if callable(match_cpu_local):
                cpu_matched_result = match_cpu_local(sequence_meta)
            if enable_ssd:
                match_ssd_local = getattr(self.ssd_cache_engine, "match_local", None)
                if callable(match_ssd_local):
                    ssd_matched_result = match_ssd_local(sequence_meta)
        cpu_matched_count = (
            cpu_matched_result.num_ready_matched_blocks
            if defer_put_commit else cpu_matched_result.num_matched_blocks)
        ssd_matched_count = (
            ssd_matched_result.num_ready_matched_blocks
            if defer_put_commit else ssd_matched_result.num_matched_blocks)
        cpu_matched_blocks = cpu_matched_result.physical_blocks[
            :cpu_matched_count][block_mask_start:block_mask_end]
        ssd_matched_blocks = ssd_matched_result.physical_blocks[
            :ssd_matched_count][block_mask_start:block_mask_end]
        remote_matched_blocks = remote_matched_result.physical_blocks[
            :remote_matched_result.num_matched_blocks][block_mask_start:block_mask_end]

        num_skipped_blocks = len(cpu_matched_blocks)
        fragment12_num_blocks = len(gpu_block_ids) - num_skipped_blocks
        if fragment12_num_blocks == 0 and not defer_put_commit:
            return self._empty_put_return(request_id)
        fragment2_num_blocks = len(gpu_block_ids) - len(ssd_matched_blocks)
        if not enable_ssd:
            fragment2_num_blocks = 0

        # NOTE: to avoid full kv repeating write in mooncake store.
        if self.use_mooncake_store_backend:
            kv_hit = int(getattr(remote_matched_result, "kv_matched_blocks", 0)
                         or remote_matched_result.num_matched_blocks)
            remote_put_hit_blocks = max(0, min(len(gpu_block_ids), kv_hit - block_mask_start))
            fragment3_num_blocks = len(gpu_block_ids) - remote_put_hit_blocks
        else:
            remote_put_hit_blocks = len(remote_matched_blocks)
            fragment3_num_blocks = len(gpu_block_ids) - len(remote_matched_blocks)

        if (fragment12_num_blocks == 0
                and fragment2_num_blocks == 0
                and fragment3_num_blocks == 0):
            return self._empty_put_return(request_id)

        # GPU 上已缓存的头部（CPU 已命中部分）不必重复下沉，从 num_skipped_blocks 之后开始
        fragment12_gpu_blocks = gpu_block_ids[num_skipped_blocks:]

        fragment12_cpu_blocks = self.cpu_cache_engine.take(
            num_required_blocks=fragment12_num_blocks,
            protected_node=(cpu_matched_result.last_ready_node
                            if defer_put_commit else cpu_matched_result.last_node),
            strict=False
        )
        if len(fragment12_cpu_blocks) < fragment12_num_blocks:
            self.cpu_cache_engine.recycle(fragment12_cpu_blocks)
            return self._empty_put_return(request_id)
        put_to_ssd = False
        if enable_ssd and fragment2_num_blocks > 0:
            fragment2_ssd_blocks = self.ssd_cache_engine.take(
                num_required_blocks=fragment2_num_blocks,
                protected_node=(ssd_matched_result.last_ready_node
                                if defer_put_commit else ssd_matched_result.last_node),
                strict=False
            )
            if len(fragment2_ssd_blocks) == fragment2_num_blocks:
                put_to_ssd = True
            else:
                self.ssd_cache_engine.recycle(fragment2_ssd_blocks)
        else:
            fragment2_ssd_blocks = np.array([], dtype=np.int64)
        put_to_remote = False
        if fragment3_num_blocks > 0:
            fragment3_remote_blocks = self.remote_cache_engine.take(
                num_required_blocks=fragment3_num_blocks,
                protected_node = remote_matched_result.last_node,
                strict=False
            )
            if len(fragment3_remote_blocks) == fragment3_num_blocks:
                put_to_remote = True
            else:
                self.remote_cache_engine.recycle(fragment3_remote_blocks)
        else:
            fragment3_remote_blocks = np.array([], dtype=np.int64)

        cpu_swa_slot = -1
        ssd_swa_slot = -1
        remote_swa_slot = -1
        mooncake_swa_tail_hash: Optional[str] = None

        if self.swa_op_constructor.enabled:
            cpu_swa_slot = self.cpu_cache_engine._alloc_swa_slot(
                cpu_matched_result.last_ready_node
                if defer_put_commit else cpu_matched_result.last_node)
            if cpu_swa_slot >= 0 and put_to_ssd:
                ssd_swa_slot = self.ssd_cache_engine._alloc_swa_slot(
                    ssd_matched_result.last_ready_node
                    if defer_put_commit else ssd_matched_result.last_node)
            if (cpu_swa_slot >= 0 and
                    (not put_to_ssd or ssd_swa_slot >= 0) and
                    put_to_remote):
                if self.use_mooncake_store_backend:
                    # Key-addressed store: no remote slot to reserve / mount.
                    # SWA snapshot keyed by the tail hash of the written prefix.
                    tail_idx = block_mask_start + len(gpu_block_ids) - 1
                    mooncake_swa_tail_hash = str(
                        sequence_meta.block_hashes[tail_idx])
                else:
                    remote_swa_slot = self.remote_cache_engine._alloc_swa_slot(
                        remote_matched_result.last_node)
            if (cpu_swa_slot < 0 or
                    (put_to_ssd and ssd_swa_slot < 0) or
                    (put_to_remote and remote_swa_slot < 0
                     and mooncake_swa_tail_hash is None)):
                return self._fail_put_before_insert(
                    request_id=request_id,
                    reason="swa_slot_alloc_failed",
                    cpu_blocks=fragment12_cpu_blocks,
                    cpu_swa_slot=cpu_swa_slot,
                    ssd_blocks=fragment2_ssd_blocks if put_to_ssd else None,
                    ssd_swa_slot=ssd_swa_slot,
                    remote_blocks=fragment3_remote_blocks if put_to_remote else None,
                    remote_swa_slot=remote_swa_slot,
                )

        transfer_graph = TransferOpGraph()
        finished_ops_ids = []
        op_node_to_ready = {}
        op_d2h = None
        if fragment12_num_blocks > 0:
            op_d2h = TransferOp(
                graph_id=transfer_graph.graph_id,
                transfer_type=TransferType.D2H,
                src_block_ids=fragment12_gpu_blocks,
                dst_block_ids=fragment12_cpu_blocks,
                dp_client_id=dp_client_id,
            )
            flexkv_logger.info(
                "[FlexKV-SEGV-DEBUG] cache_engine create D2H op (global_put) "
                f"request_id={request_id}, op_id={op_d2h.op_id}, "
                f"graph_id={transfer_graph.graph_id}, dp_client_id={dp_client_id}, "
                f"fragment12_num_blocks={fragment12_num_blocks}, "
                f"fragment2_num_blocks={fragment2_num_blocks}, "
                f"fragment3_num_blocks={fragment3_num_blocks}, "
                f"{summarize_id_tensor('gpu_src', fragment12_gpu_blocks)}, "
                f"{summarize_id_tensor('cpu_dst', fragment12_cpu_blocks)}"
            )
            transfer_graph.add_transfer_op(op_d2h)
            finished_ops_ids.append(op_d2h.op_id)

        op_h2disk = None
        if put_to_ssd:
            if len(fragment12_cpu_blocks) < fragment2_num_blocks:
                num_needed_from_cpu_matched = fragment2_num_blocks - len(fragment12_cpu_blocks)
                fragment2_cpu_blocks = np.concatenate([cpu_matched_blocks[-num_needed_from_cpu_matched:], \
                    fragment12_cpu_blocks])
            else:
                fragment2_cpu_blocks = fragment12_cpu_blocks[-fragment2_num_blocks:]
            op_h2disk = TransferOp(
                graph_id = transfer_graph.graph_id,
                transfer_type = TransferType.H2DISK,
                src_block_ids = fragment2_cpu_blocks,
                dst_block_ids = fragment2_ssd_blocks,
                dp_client_id = dp_client_id,
            )
            transfer_graph.add_transfer_op(op_h2disk)

            if op_d2h is not None:
                transfer_graph.add_dependency(op_h2disk.op_id, op_d2h.op_id)

        op_h2remote = None
        if put_to_remote:
            if fragment3_num_blocks > fragment12_num_blocks:
                extra_num_cpu_blocks = fragment3_num_blocks - fragment12_num_blocks
                fragment3_cpu_blocks = np.concatenate([cpu_matched_blocks[-extra_num_cpu_blocks:],
                                                       fragment12_cpu_blocks])
            else:
                fragment3_cpu_blocks = fragment12_cpu_blocks[-fragment3_num_blocks:]
            mooncake_block_hashes = None
            if self.use_mooncake_store_backend:
                mooncake_block_hashes = sequence_meta.block_hashes[
                    block_mask_start + remote_put_hit_blocks:
                    block_mask_start + remote_put_hit_blocks + fragment3_num_blocks
                ]
            op_h2remote = TransferOp(
                graph_id = transfer_graph.graph_id,
                transfer_type = TransferType.H2REMOTE,
                src_block_ids = fragment3_cpu_blocks,
                dst_block_ids = fragment3_remote_blocks,
                dp_client_id = dp_client_id,
                mooncake_store_block_hashes = mooncake_block_hashes,
            )
            transfer_graph.add_transfer_op(op_h2remote)
            # 写远端的数据源就是刚落到 CPU 内存的 block，故必须排在 D2H 之后
            if op_d2h is not None:
                transfer_graph.add_dependency(op_h2remote.op_id, op_d2h.op_id)

        if op_d2h is None:
            assert defer_put_commit
            if op_h2disk is not None:
                finished_ops_ids.append(op_h2disk.op_id)
            if op_h2remote is not None:
                finished_ops_ids.append(op_h2remote.op_id)

        if cpu_swa_slot >= 0:
            empty = np.array([], dtype=np.int64)
            put_remote_via_mooncake = (
                mooncake_swa_tail_hash is not None and cpu_swa_slot >= 0)
            if remote_swa_slot >= 0:
                remote_slot_ids = np.array([remote_swa_slot], dtype=np.int64)
            elif put_remote_via_mooncake:
                remote_slot_ids = np.array([0], dtype=np.int64) # slot 0 will not be used in mooncake store backend.
            else:
                remote_slot_ids = empty
            swa_ops = self.swa_op_constructor.build_put_chain(
                transfer_graph,
                gpu_slot_ids=self._SWA_GPU_PLACEHOLDER.copy(),
                cpu_slot_ids=np.array([cpu_swa_slot], dtype=np.int64),
                ssd_slot_ids=(np.array([ssd_swa_slot], dtype=np.int64)
                              if ssd_swa_slot >= 0 else empty),
                remote_slot_ids=remote_slot_ids,
                dp_client_id=dp_client_id,
                return_op_ids=True,
                mooncake_tail_hashes=(
                    [mooncake_swa_tail_hash] if put_remote_via_mooncake else None),
            )
            assert swa_ops.d2h_id is not None
            if put_to_ssd:
                assert swa_ops.h2disk_id is not None
            if put_to_remote and (remote_swa_slot >= 0 or put_remote_via_mooncake):
                assert swa_ops.h2remote_id is not None
            finished_ops_ids.append(swa_ops.d2h_id)

        if defer_put_commit:
            deferred_inserts: List[DeferredCacheInsert] = []
            node_to_unlock = {}
            requested_end = block_mask_start + len(gpu_block_ids)
            if fragment12_num_blocks > 0 or cpu_swa_slot >= 0:
                deferred_inserts.append(DeferredCacheInsert(
                    device_type=DeviceType.CPU,
                    sequence_meta=sequence_meta,
                    physical_blocks=fragment12_cpu_blocks,
                    staged_start_block=block_mask_start + num_skipped_blocks,
                    remote_start_block=block_mask_start + num_skipped_blocks,
                    requested_end_block=requested_end,
                    swa_slot=cpu_swa_slot,
                    publish_to_peer=self.cache_config.enable_p2p_cpu,
                ))
            if put_to_ssd:
                deferred_inserts.append(DeferredCacheInsert(
                    device_type=DeviceType.SSD,
                    sequence_meta=sequence_meta,
                    physical_blocks=fragment2_ssd_blocks,
                    staged_start_block=block_mask_start + len(ssd_matched_blocks),
                    remote_start_block=block_mask_start + len(ssd_matched_blocks),
                    requested_end_block=requested_end,
                    swa_slot=ssd_swa_slot,
                    publish_to_peer=self.cache_config.enable_p2p_ssd,
                ))

            # Existing ready CPU blocks may feed H2DISK/H2REMOTE while the graph
            # is running.  Pin their deepest node until every consumer finishes.
            cpu_anchor = cpu_matched_result.last_ready_node
            if len(cpu_matched_blocks) > 0 and cpu_anchor is not None:
                node_to_unlock[DeviceType.CPU] = (cpu_anchor, 0)
            ssd_anchor = ssd_matched_result.last_ready_node
            if (put_to_ssd and len(ssd_matched_blocks) > 0
                    and ssd_anchor is not None):
                node_to_unlock[DeviceType.SSD] = (ssd_anchor, 0)
            skipped_gpu_blocks = len(cpu_matched_blocks)
            return PutTransferPlan(
                transfer_graph=transfer_graph,
                finished_ops_ids=finished_ops_ids,
                node_to_unlock=node_to_unlock,
                op_callback_dict={},
                buffer_to_free={},
                num_gpu_blocks_to_transfer=len(fragment12_gpu_blocks),
                skipped_gpu_blocks=skipped_gpu_blocks,
                deferred_inserts=deferred_inserts,
            )

        # 非延迟路径：规划期就把三级的目标节点 insert 上树（is_ready=False），
        # 并把 op_id -> (device, node, len) 登记进 op_node_to_ready，
        # 由各 op 的完成回调置 ready（_op_callback）。
        assert op_d2h is not None
        cpu_node_to_unlock = self.cpu_cache_engine.insert(
            sequence_meta,
            fragment12_cpu_blocks,
            is_ready=False,
            match_result=cpu_matched_result,
        )
        op_node_to_ready[op_d2h.op_id] = (
            DeviceType.CPU, cpu_node_to_unlock, cpu_node_to_unlock.size())
        ssd_node_to_unlock = None
        if put_to_ssd:
            ssd_node_to_unlock = self.ssd_cache_engine.insert(
                sequence_meta,
                fragment2_ssd_blocks,
                is_ready=False,
                match_result=ssd_matched_result,
            )
            op_node_to_ready[op_h2disk.op_id] = (
                DeviceType.SSD, ssd_node_to_unlock,
                ssd_node_to_unlock.size())
        remote_node_to_unlock = None
        if put_to_remote:
            remote_node_to_unlock = self.remote_cache_engine.insert(
                sequence_meta,
                fragment3_remote_blocks,
                is_ready=False,
                match_result=remote_matched_result,
            )
            op_node_to_ready[op_h2remote.op_id] = (
                DeviceType.REMOTE,
                remote_node_to_unlock,
                remote_node_to_unlock.size(),
            )
        node_to_unlock = {}
        if cpu_node_to_unlock is not None:
            node_to_unlock[DeviceType.CPU] = (
                cpu_node_to_unlock, cpu_node_to_unlock.size())
        if ssd_node_to_unlock is not None:
            node_to_unlock[DeviceType.SSD] = (
                ssd_node_to_unlock, ssd_node_to_unlock.size())
        if remote_node_to_unlock is not None:
            node_to_unlock[DeviceType.REMOTE] = (
                remote_node_to_unlock, remote_node_to_unlock.size())

        op_callback_dict = self._build_op_callback_dict(op_node_to_ready)
        if cpu_swa_slot >= 0:
            self._append_op_callback(
                op_callback_dict,
                swa_ops.d2h_id,
                partial(self._publish_swa_put_slot,
                        DeviceType.CPU, cpu_node_to_unlock, cpu_swa_slot),
            )
        if ssd_swa_slot >= 0:
            self._append_op_callback(
                op_callback_dict,
                swa_ops.h2disk_id,
                partial(self._publish_swa_put_slot,
                        DeviceType.SSD, ssd_node_to_unlock, ssd_swa_slot),
            )
        if remote_swa_slot >= 0:
            self._append_op_callback(
                op_callback_dict,
                swa_ops.h2remote_id,
                partial(self._publish_swa_put_slot,
                        DeviceType.REMOTE, remote_node_to_unlock,
                        remote_swa_slot),
            )
        skipped_gpu_blocks = len(cpu_matched_blocks)
        swa_slots_to_free = [(device_type, slot) for device_type, slot in
                             ((DeviceType.CPU, cpu_swa_slot),
                              (DeviceType.SSD, ssd_swa_slot),
                              (DeviceType.REMOTE, remote_swa_slot)) if slot >= 0]
        return PutTransferPlan(
            transfer_graph=transfer_graph,
            finished_ops_ids=finished_ops_ids,
            node_to_unlock=node_to_unlock,
            op_callback_dict=op_callback_dict,
            buffer_to_free={},
            num_gpu_blocks_to_transfer=len(fragment12_gpu_blocks),
            skipped_gpu_blocks=skipped_gpu_blocks,
            swa_slots_to_free=swa_slots_to_free,
        )

    # ------------------------------------------------------------------
    # 本地 PUT 路径：GPU -> CPU 内存 -> SSD 两级下沉（不写远端）
    #
    # 与 _put_impl_global 的区别：没有 fragment3 / H2REMOTE，也没有
    # defer_put_commit 延迟上树分支 —— 一律在规划期 insert、回调置 ready。
    # 切分逻辑相同：CPU 已缓存的跳过，SSD 比 CPU 多出的那段才写 SSD。
    # ------------------------------------------------------------------
    def _put_impl_local(self,
            request_id: int,
            sequence_meta: SequenceMeta,
            block_mask_start: int,
            block_mask_end: int,
            gpu_block_ids: np.ndarray,
            temp_cache_strategy: CacheStrategy,
            dp_client_id: int) \
                -> PutTransferPlan:
        """
        transfer pattern:

        GPU:   (skipped)  | fragment1      | fragment2      | (uncompleted block)
                                ↓                ↓
        CPU: (cpu cached) | fragment1(new) | fragment2(new) |
                                                 ↓
        SSD:          (ssd cached)         | fragment2(new) |

        """
        enable_gpu = not temp_cache_strategy.ignore_gpu
        enable_cpu = self.cache_config.enable_cpu
        enable_ssd = self.cache_config.enable_ssd and not temp_cache_strategy.ignore_ssd
        enable_gds = self.cache_config.enable_gds and not temp_cache_strategy.ignore_gds
        assert enable_gpu
        assert enable_cpu
        assert self.cpu_cache_engine is not None

        if self.index_accel:
            cpu_matched_result, ssd_matched_result = self.match_local_accel(sequence_meta,
                                                                            temp_cache_strategy=temp_cache_strategy,
                                                                            is_put=True)
        else:
            cpu_matched_result, ssd_matched_result = self.match_local(sequence_meta,
                                                                      temp_cache_strategy=temp_cache_strategy,
                                                                      is_put=True)
        cpu_matched_blocks = cpu_matched_result.physical_blocks[
            :cpu_matched_result.num_matched_blocks][block_mask_start:block_mask_end]
        ssd_matched_blocks = ssd_matched_result.physical_blocks[
            :ssd_matched_result.num_matched_blocks][block_mask_start:block_mask_end]

        #if len(cpu_matched_blocks) > len(ssd_matched_blocks):
        #    print(f"[PUT_LOCAL] CPU matched blocks are greater than SSD matched blocks, skipping")
        #    return self._empty_put_return(request_id)


        num_skipped_blocks = len(cpu_matched_blocks)
        fragment12_num_blocks = len(gpu_block_ids) - num_skipped_blocks
        if fragment12_num_blocks == 0:
            return self._empty_put_return(request_id)
        fragment2_num_blocks = len(gpu_block_ids) - len(ssd_matched_blocks)
        if not enable_ssd:
            fragment2_num_blocks = 0

        fragment12_gpu_blocks = gpu_block_ids[num_skipped_blocks:]

        fragment12_cpu_blocks = self.cpu_cache_engine.take(
            num_required_blocks=fragment12_num_blocks,
            protected_node = cpu_matched_result.last_node,
            strict=False
        )

        if enable_ssd:
            fragment2_ssd_blocks = self.ssd_cache_engine.take(
                num_required_blocks=fragment2_num_blocks,
                protected_node = ssd_matched_result.last_node,
                strict=False
            )
        else:
            fragment2_ssd_blocks = np.array([], dtype=np.int64)

        if len(fragment12_cpu_blocks) < fragment12_num_blocks or \
            len(fragment2_ssd_blocks) < fragment2_num_blocks:
            print(f"[WARNING] PUT request {request_id} FAILED: "
                  f"CPU={len(fragment12_cpu_blocks)}/{fragment12_num_blocks}, "
                  f"SSD={len(fragment2_ssd_blocks)}/{fragment2_num_blocks}")
            self.cpu_cache_engine.recycle(fragment12_cpu_blocks)
            if enable_ssd:
                self.ssd_cache_engine.recycle(fragment2_ssd_blocks)
            return self._empty_put_return(request_id)

        cpu_swa_slot = -1
        ssd_swa_slot = -1

        if self.swa_op_constructor.enabled:
            cpu_swa_slot = self.cpu_cache_engine._alloc_swa_slot(
                cpu_matched_result.last_node)
            if cpu_swa_slot >= 0 and fragment2_num_blocks > 0:
                ssd_swa_slot = self.ssd_cache_engine._alloc_swa_slot(
                    ssd_matched_result.last_node)
            if (cpu_swa_slot < 0 or
                    (fragment2_num_blocks > 0 and ssd_swa_slot < 0)):
                return self._fail_put_before_insert(
                    request_id=request_id,
                    reason="swa_slot_alloc_failed",
                    cpu_blocks=fragment12_cpu_blocks,
                    cpu_swa_slot=cpu_swa_slot,
                    ssd_blocks=fragment2_ssd_blocks if enable_ssd else None,
                    ssd_swa_slot=ssd_swa_slot,
                )

        transfer_graph = TransferOpGraph()
        finished_ops_ids = []
        op_node_to_ready = {}

        op_d2h = TransferOp(
            graph_id = transfer_graph.graph_id,
            transfer_type = TransferType.D2H,
            src_block_ids = fragment12_gpu_blocks,
            dst_block_ids = fragment12_cpu_blocks,
            dp_client_id = dp_client_id,
        )
        flexkv_logger.info(
            "[FlexKV-SEGV-DEBUG] cache_engine create D2H op (local_put) "
            f"request_id={request_id}, op_id={op_d2h.op_id}, "
            f"graph_id={transfer_graph.graph_id}, dp_client_id={dp_client_id}, "
            f"fragment12_num_blocks={fragment12_num_blocks}, "
            f"fragment2_num_blocks={fragment2_num_blocks}, "
            f"{summarize_id_tensor('gpu_src', fragment12_gpu_blocks)}, "
            f"{summarize_id_tensor('cpu_dst', fragment12_cpu_blocks)}"
        )
        transfer_graph.add_transfer_op(op_d2h)
        finished_ops_ids.append(op_d2h.op_id)

        if fragment2_num_blocks > 0:
            if len(fragment12_cpu_blocks) < fragment2_num_blocks:
                flexkv_logger.warning(f"fragment12_cpu_blocks: {len(fragment12_cpu_blocks)}, "
                                      f"fragment2_num_blocks: {fragment2_num_blocks}, "
                                      f"cpu match blocks are bigger than SSD match blocks number. "
                                      f"This should not often happen if CPU cache size is smaller than SSD cache size.")
                num_needed_from_cpu_matched = fragment2_num_blocks - len(fragment12_cpu_blocks)
                fragment2_cpu_blocks = np.concatenate([cpu_matched_blocks[-num_needed_from_cpu_matched:], \
                    fragment12_cpu_blocks])
            else:
                fragment2_cpu_blocks = fragment12_cpu_blocks[-fragment2_num_blocks:]
            op_h2disk = TransferOp(
                graph_id = transfer_graph.graph_id,
                transfer_type = TransferType.H2DISK,
                src_block_ids = fragment2_cpu_blocks,
                dst_block_ids = fragment2_ssd_blocks,
                dp_client_id = dp_client_id,
            )
            transfer_graph.add_transfer_op(op_h2disk)

            transfer_graph.add_dependency(op_h2disk.op_id, op_d2h.op_id)

        if cpu_swa_slot >= 0:
            empty = np.array([], dtype=np.int64)
            swa_ops = self.swa_op_constructor.build_put_chain(
                transfer_graph,
                gpu_slot_ids=self._SWA_GPU_PLACEHOLDER.copy(),
                cpu_slot_ids=np.array([cpu_swa_slot], dtype=np.int64),
                ssd_slot_ids=(np.array([ssd_swa_slot], dtype=np.int64)
                              if ssd_swa_slot >= 0 else empty),
                remote_slot_ids=empty,
                dp_client_id=dp_client_id,
                return_op_ids=True,
            )
            assert swa_ops.d2h_id is not None
            if fragment2_num_blocks > 0:
                assert swa_ops.h2disk_id is not None
            finished_ops_ids.append(swa_ops.d2h_id)

        """insert and lock"""
        cpu_node_to_unlock = self.cpu_cache_engine.insert(
            sequence_meta,
            fragment12_cpu_blocks,
            is_ready=False,
            match_result=cpu_matched_result,
        )
        op_node_to_ready[op_d2h.op_id] = (DeviceType.CPU, cpu_node_to_unlock, cpu_node_to_unlock.size())
        ssd_node_to_unlock = None
        if len(fragment2_ssd_blocks) > 0:
            ssd_node_to_unlock = self.ssd_cache_engine.insert(
                sequence_meta,
                fragment2_ssd_blocks,
                is_ready=False,
                match_result=ssd_matched_result,
            )
            op_node_to_ready[op_h2disk.op_id] = (DeviceType.SSD, ssd_node_to_unlock, ssd_node_to_unlock.size())
        node_to_unlock = {}
        if cpu_node_to_unlock is not None:
            node_to_unlock[DeviceType.CPU] = (cpu_node_to_unlock, cpu_node_to_unlock.size())
        if ssd_node_to_unlock is not None:
            node_to_unlock[DeviceType.SSD] = (ssd_node_to_unlock, ssd_node_to_unlock.size())

        op_callback_dict = self._build_op_callback_dict(op_node_to_ready)
        if cpu_swa_slot >= 0:
            self._append_op_callback(
                op_callback_dict,
                swa_ops.d2h_id,
                partial(self._publish_swa_put_slot,
                        DeviceType.CPU, cpu_node_to_unlock, cpu_swa_slot),
            )
        if ssd_swa_slot >= 0:
            self._append_op_callback(
                op_callback_dict,
                swa_ops.h2disk_id,
                partial(self._publish_swa_put_slot,
                        DeviceType.SSD, ssd_node_to_unlock, ssd_swa_slot),
            )
        skipped_gpu_blocks = len(cpu_matched_blocks)
        swa_slots_to_free = [(device_type, slot) for device_type, slot in
                             ((DeviceType.CPU, cpu_swa_slot),
                              (DeviceType.SSD, ssd_swa_slot)) if slot >= 0]
        return PutTransferPlan(
            transfer_graph=transfer_graph,
            finished_ops_ids=finished_ops_ids,
            node_to_unlock=node_to_unlock,
            op_callback_dict=op_callback_dict,
            buffer_to_free={},
            num_gpu_blocks_to_transfer=len(fragment12_gpu_blocks),
            skipped_gpu_blocks=skipped_gpu_blocks,
            swa_slots_to_free=swa_slots_to_free,
        )

    # 判断"当前匹配是否正好停在某个完整节点的边界上" —— 只有落在这种节点上，
    # 才能安全地把 SWA 快照槽挂上去（否则节点会分裂，挂载就错位了）
    @staticmethod
    def _matched_boundary_node(current_match, matched_blocks: int):
        """Return the matched node only when ``matched_blocks`` ends on it."""
        if int(current_match.num_matched_blocks) != matched_blocks:
            return None
        node = current_match.last_node
        if node is None:
            return None
        if int(current_match.last_node_matched_length) != int(node.size()):
            return None
        return node

    @staticmethod
    def _release_pending_swa_slot(engine, pending: DeferredCacheInsert) -> None:
        if pending.swa_slot >= 0:
            engine._free_swa_slot(pending.swa_slot)

    # 丢弃一条延迟上树记录：block 还没交给树，所有权仍归本次请求，可以整体回收
    def _discard_deferred_insert(
            self, engine, pending: DeferredCacheInsert,
            physical_blocks: np.ndarray) -> None:
        """Release staging that is still wholly owned by this request."""
        engine.recycle(physical_blocks)
        self._release_pending_swa_slot(engine, pending)

    @staticmethod
    def _publish_pending_swa_slot(
            engine, pending: DeferredCacheInsert, node) -> None:
        if pending.swa_slot < 0:
            return
        if node is None or node.has_swa():
            engine._free_swa_slot(pending.swa_slot)
            return
        engine.index.set_swa(node, int(pending.swa_slot))
        engine._drain_unmounted_swa_slots()

    # 上报"真正挂到树上的远端 block 数"。预取任务的 return_mask 必须用它来收敛：
    # 传输成功不代表树上看得见（可能被并发写入抢先、或插入被拒绝）。
    @staticmethod
    def _record_deferred_publish(
            pending: DeferredCacheInsert,
            published_end_block: int,
            reason: str,
            failed: bool = False) -> None:
        """Report how many remote blocks became matchable after commit."""
        publish_result = pending.publish_result
        if publish_result is None:
            return
        published_remote = max(
            0, int(published_end_block) - int(pending.remote_start_block))
        publish_result.record(
            published_remote, reason=reason, failed=failed)

    # ------------------------------------------------------------------
    # 【延迟上树】—— 本文件最需要讲清楚的一处设计
    #
    # 为什么传输完成前不能把 block 挂到 radix tree 上？
    #   1. 远端（mooncake）读可能**部分失败**：整批 block 里只有前面一段真的
    #      读到手。规划时并不知道能成功多少，一旦提前上树，后续请求就会
    #      "命中"到根本没数据的 block，读到脏 KV —— 这是不可恢复的正确性问题。
    #   2. 从规划到完成这段时间内，树的状态可能已经被别的并发请求改变：
    #      别人可能已经把这段前缀写进去了（我们成了冗余），也可能正在写
    #      （树上存在未 ready 的节点，我们不能越过它去接一段数据）。
    #   3. 因此正确的顺序是：先搬，搬完再重新 match 一次，只把"当前树上真正
    #      缺失、且确实搬成功"的那段连续前缀插进去，插完立刻在同一把锁内
    #      set_ready。整段 rematch + insert + set_ready 是一个原子事务。
    #
    # 对比：非远端路径（本地 / SSD）在规划期就 insert(is_ready=False)，
    # 因为那些路径的成功是可预期的、由 op 完成回调保证。
    #
    # 本方法的返回值是被挂载的节点（或 None），并会通过 publish_result
    # 回写"真正上树了多少个远端 block"，供预取任务收敛 return_mask。
    # ------------------------------------------------------------------
    def _commit_deferred_insert(self, pending: DeferredCacheInsert):
        """Fresh-rematch and atomically publish one valid staging prefix.

        When ``pending.publish_result`` is set (CPU Mooncake loads), every
        normal return path records the published remote-block count so
        prefetch finalize can clamp ``return_mask`` to what the radix tree
        actually mounts — not only what REMOTE2H transferred.
        """
        engine = self.cache_engines[pending.device_type]
        physical_blocks = np.asarray(pending.physical_blocks, dtype=np.int64)
        staged_blocks = pending.requested_end_block - pending.staged_start_block
        remote_blocks = pending.requested_end_block - pending.remote_start_block
        if (staged_blocks != len(physical_blocks)
                or pending.staged_start_block > pending.remote_start_block
                or remote_blocks < 0):
            self._discard_deferred_insert(engine, pending, physical_blocks)
            flexkv_logger.error(
                "Invalid deferred cache insert range: "
                f"staged=[{pending.staged_start_block}, "
                f"{pending.requested_end_block}), "
                f"remote_start={pending.remote_start_block}, "
                f"physical_blocks={len(physical_blocks)}")
            self._record_deferred_publish(
                pending, pending.remote_start_block, "invalid_range", failed=True)
            return None

        # 只有远端（mooncake）读才带 load_result：它决定了"搬成功了多少"。
        # 没有它就说明是 PUT 暂存 —— 整段都算成功。
        if pending.load_result is None:
            publish_end = pending.requested_end_block
        else:
            # 关键：取最长**连续**成功前缀，而不是成功总数 ——
            # 中间断了一个 block，后面的内容在语义上就是不可用的
            successful_remote = pending.load_result.successful_prefix(remote_blocks)
            publish_end = pending.remote_start_block + successful_remote

        # Joint Full+SWA prefetch guard: SWA is mounted ONLY when Full commit
        # reaches the SWA anchor AND the SWA op reported success. On any partial
        # (Full short of the anchor, SWA REMOTE2H failed, or SWA result missing)
        # we free the SWA slot and fall through to publish only the Full prefix.
        # This preserves the tree invariant "node has SWA => Full is ready up to
        # this node" (SWA-I1) at prefetch-commit time.
        if pending.swa_slot >= 0 and pending.swa_anchor_block >= 0:
            swa_covered = publish_end >= (pending.swa_anchor_block + 1)
            swa_ok = (
                pending.swa_load_result is not None
                and pending.swa_load_result.block_results is not None
                and len(pending.swa_load_result.block_results) > 0
                and all(pending.swa_load_result.block_results)
            )
            if not (swa_covered and swa_ok):
                engine._free_swa_slot(pending.swa_slot)
                pending = replace(pending, swa_slot=-1)

        # 提交前必须重新匹配一次：规划到完成这段时间内树可能已被并发改写
        # Hierarchical engines must rematch their local tree. match() may choose
        # a distributed peer node, which is not a valid insertion anchor here.
        match_local = getattr(engine, "match_local", None)
        try:
            current_match = (
                match_local(pending.sequence_meta)
                if callable(match_local)
                else engine.match(pending.sequence_meta)
            )
        except Exception:
            # No radix mutation has started, so all staging is still ours.
            self._discard_deferred_insert(engine, pending, physical_blocks)
            self._record_deferred_publish(
                pending, pending.remote_start_block, "rematch_error", failed=True)
            raise
        current_blocks = int(current_match.num_matched_blocks)
        ready_blocks = int(current_match.num_ready_matched_blocks)

        # 两种"放弃"的情形：
        #   a) 树上存在别人正在写的未 ready 节点（current != ready）——
        #      我们不能借道它的未就绪数据去接自己的块；
        #   b) 树已经比我们这次的起点更长了（规划已过期）。
        # 这两种情况下暂存 block 仍完全属于本次请求，可以安全回收。
        # Never attach below, or mark ready through, another in-flight writer's
        # unready node. The staged allocation remains ours and is safe to recycle.
        if (current_blocks != ready_blocks
                or current_blocks < pending.staged_start_block):
            self._discard_deferred_insert(engine, pending, physical_blocks)
            self._record_deferred_publish(
                pending, pending.remote_start_block, "rematch_stale")
            return None

        if current_blocks >= publish_end:
            # 并发的请求已经把这段前缀写好了，我们搬回来的是冗余数据，
            # 回收即可；但这次传输本身是成功的，仍要按 publish_end 上报
            engine.recycle(physical_blocks)
            boundary_node = self._matched_boundary_node(
                current_match, pending.requested_end_block)
            if pending.swa_slot >= 0:
                if boundary_node is None or publish_end != pending.requested_end_block:
                    self._release_pending_swa_slot(engine, pending)
                else:
                    self._publish_pending_swa_slot(
                        engine, pending, boundary_node)
            # Tree already covers the transferred prefix — still usable.
            self._record_deferred_publish(
                pending, publish_end, "already_covered")
            return boundary_node

        successful_staged_blocks = publish_end - pending.staged_start_block
        if successful_staged_blocks <= 0:
            self._discard_deferred_insert(engine, pending, physical_blocks)
            self._record_deferred_publish(
                pending, pending.remote_start_block, "zero_prefix")
            return None

        skipped_blocks = current_blocks - pending.staged_start_block
        blocks_to_insert = physical_blocks[
            skipped_blocks:successful_staged_blocks]

        try:
            # 依旧是 is_ready=False 上树；紧接着的 set_ready 在同一个
            # _cache_tree_lock 临界区内完成，外部观察不到"未就绪"的中间态
            node = engine.insert(
                pending.sequence_meta,
                blocks_to_insert,
                num_insert_blocks=publish_end,
                is_ready=False,
                match_result=current_match,
            )
        except RuntimeError as error:
            # The radix guard raises before any mutation and deliberately leaves
            # block ownership with the caller. Other insert failures may happen
            # after mutation, so keep those fail-closed.
            if not str(error).startswith("radix insert conflict:"):
                self._release_pending_swa_slot(engine, pending)
                self._record_deferred_publish(
                    pending, pending.remote_start_block, "insert_error", failed=True)
                raise
            self._discard_deferred_insert(engine, pending, physical_blocks)
            self._record_deferred_publish(
                pending, pending.remote_start_block, "insert_conflict")
            return None
        except Exception:
            self._release_pending_swa_slot(engine, pending)
            self._record_deferred_publish(
                pending, pending.remote_start_block, "insert_error", failed=True)
            raise
        if node is None:
            self._discard_deferred_insert(engine, pending, physical_blocks)
            self._record_deferred_publish(
                pending, pending.remote_start_block, "insert_none")
            return None

        unused_blocks = np.concatenate((
            physical_blocks[:skipped_blocks],
            physical_blocks[successful_staged_blocks:],
        ))
        if len(unused_blocks) > 0:
            engine.recycle(unused_blocks)

        # Keep the inserted length explicit. set_ready supports split fragments,
        # while this transaction keeps rematch, insert, and readiness atomic.
        # 到这里才算真正"发布"：节点变 ready，后续请求的前缀匹配才能命中它
        try:
            engine.set_ready(node, True, len(blocks_to_insert))
        except Exception:
            # Inserted blocks now belong to the unready tree node and must not
            # be recycled, but the detached SWA slot is still ours.
            self._release_pending_swa_slot(engine, pending)
            self._record_deferred_publish(
                pending, pending.remote_start_block, "set_ready_error", failed=True)
            raise

        if pending.swa_slot >= 0:
            if publish_end == pending.requested_end_block:
                self._publish_pending_swa_slot(engine, pending, node)
            else:
                self._release_pending_swa_slot(engine, pending)

        if pending.publish_to_peer:
            engine.local_index.insert_and_publish(node)
        self._record_deferred_publish(pending, publish_end, "ok")
        return node

    # 整张图跑完后的统一收尾（由 TransferPlanHandle 调用），做三件事：
    #   1. 提交所有延迟上树记录（_commit_deferred_insert），并单独容错 ——
    #      一条失败不能影响其它，更不能把节点锁永久挂住；
    #   2. 逐级解锁规划期锁住的节点，并按 ready_length 置 ready
    #      （ready_length == 0 表示"节点是别人已 ready 的老节点"，只解锁不改状态）；
    #   3. 回收没能上树的中转 buffer（buffer_to_free）。
    # 注意 finally 的作用：即使提交抛异常，锁和 buffer 也必须被释放。
    @_synchronized_cache_tree
    def _transfer_callback(self,
                           node_to_unlock: Dict[DeviceType, Tuple[RadixNode, int]],
                           buffer_to_free: Optional[Dict[DeviceType, np.ndarray]] = None,
                           deferred_inserts: Optional[List[DeferredCacheInsert]] = None,
                           is_put: bool = False) -> None:
        try:
            for pending in deferred_inserts or []:
                try:
                    self._commit_deferred_insert(pending)
                except Exception:
                    # Never strand the request's pre-existing radix-node locks.
                    # The commit helper recycles blocks on every known pre-insert
                    # rejection; an unexpected post-insert exception has uncertain
                    # ownership and must not return those blocks to the mempool.
                    # Commit paths that raise after recording leave publish_result
                    # set; unrecorded failures still need a zero publish report.
                    publish_result = getattr(pending, "publish_result", None)
                    if (publish_result is not None
                            and publish_result.published_remote_blocks is None):
                        publish_result.record_failure("callback_error")
                    flexkv_logger.error(
                        "Deferred cache publication failed: "
                        f"device={pending.device_type.name}",
                        exc_info=True,
                    )
        finally:
            if DeviceType.CPU in node_to_unlock:
                assert self.cpu_cache_engine is not None
                cpu_node = node_to_unlock[DeviceType.CPU][0]
                self.cpu_cache_engine.unlock(cpu_node)
                ready_length = node_to_unlock[DeviceType.CPU][1]
                if ready_length > 0:
                    self.cpu_cache_engine.set_ready(cpu_node, True, ready_length)
                if (is_put and ready_length > 0
                        and self.cache_config.enable_p2p_cpu):
                    self.cpu_cache_engine.local_index.insert_and_publish(cpu_node)
            if DeviceType.SSD in node_to_unlock:
                assert self.ssd_cache_engine is not None
                ssd_node = node_to_unlock[DeviceType.SSD][0]
                self.ssd_cache_engine.unlock(ssd_node)
                ready_length = node_to_unlock[DeviceType.SSD][1]
                if ready_length > 0:
                    self.ssd_cache_engine.set_ready(ssd_node, True, ready_length)
                if (is_put and ready_length > 0
                        and self.cache_config.enable_p2p_ssd):
                    self.ssd_cache_engine.local_index.insert_and_publish(node_to_unlock[DeviceType.SSD][0])
            if DeviceType.REMOTE in node_to_unlock:
                assert self.remote_cache_engine is not None
                self.remote_cache_engine.unlock(node_to_unlock[DeviceType.REMOTE][0])
                ready_length = node_to_unlock[DeviceType.REMOTE][1]
                if ready_length > 0:
                    self.remote_cache_engine.set_ready(
                        node_to_unlock[DeviceType.REMOTE][0], True, ready_length)
                if is_put and self.enable_kv_sharing:
                    self.remote_cache_engine.insert_and_publish(node_to_unlock[DeviceType.REMOTE][0])
            if buffer_to_free is not None:
                if DeviceType.CPU in buffer_to_free:
                    assert self.cpu_cache_engine is not None
                    self.cpu_cache_engine.recycle(buffer_to_free[DeviceType.CPU])
                if DeviceType.SSD in buffer_to_free:
                    assert self.ssd_cache_engine is not None
                    self.ssd_cache_engine.recycle(buffer_to_free[DeviceType.SSD])
                if DeviceType.REMOTE in buffer_to_free:
                    assert self.remote_cache_engine is not None
                    self.remote_cache_engine.recycle(buffer_to_free[DeviceType.REMOTE])

    # 计划被取消（图根本没提交给数据面）时的回滚路径。
    # 与 _transfer_callback 的分工：那条是"成功收尾"，这条是"撤销"，
    # 二者由 TransferPlanHandle 保证只会执行其中一个、且只执行一次。
    @_synchronized_cache_tree
    def _abort_transfer_plan(self,
                             node_to_unlock: Dict[DeviceType, Tuple[RadixNode, int]],
                             buffer_to_free: Optional[Dict[DeviceType, np.ndarray]] = None,
                             deferred_inserts: Optional[List[DeferredCacheInsert]] = None,
                             swa_reservation: Optional[SWAReadReservation] = None,
                             swa_slots_to_free: Optional[List[Tuple[DeviceType, int]]] = None) -> None:
        """Roll back a planned get/put whose graph was never launched.

        The completion path (:meth:`_transfer_callback`) unlocks, marks nodes
        ready and recycles staging. On a cancelled plan no transfer ever ran,
        so marking ready would publish unfilled blocks as valid cache — instead
        every node is unlocked and any node this plan inserted unready is
        removed and its blocks recycled. Nodes that pre-existed (matched, hence
        ready) are only unlocked; ``rollback_unready_insert`` is a no-op for
        them, so node_to_unlock can be processed uniformly.
        """
        for device_type, (node, _ready_length) in node_to_unlock.items():
            engine = self.cache_engines[device_type]
            engine.unlock(node)
            engine.rollback_unready_insert(node)
        if buffer_to_free is not None:
            for device_type, blocks in buffer_to_free.items():
                if blocks is not None and len(blocks) > 0:
                    self.cache_engines[device_type].recycle(blocks)
        for pending in deferred_inserts or []:
            engine = self.cache_engines[pending.device_type]
            physical_blocks = np.asarray(
                pending.physical_blocks, dtype=np.int64)
            self._discard_deferred_insert(engine, pending, physical_blocks)
        if swa_reservation is not None:
            self._release_swa_read_reservation(swa_reservation)
        if swa_slots_to_free:
            for device_type, slot in swa_slots_to_free:
                if slot >= 0:
                    self.cache_engines[device_type]._free_swa_slot(slot)

    @_synchronized_cache_tree
    def _op_callback(self, device_type: DeviceType, node_to_ready: RadixNode, ready_length: int) -> None:
        """单个 op 完成回调：把该 op 写入的那段节点标记为 ready。

        这是"规划期上树占位 + 完成期打开可见性"这一设计的落点：
        节点在规划时就已 insert(is_ready=False) 并加锁，只有这里把它置 ready
        之后，后续请求的前缀匹配才可能命中它。
        """
        if device_type == DeviceType.CPU:
            assert self.cpu_cache_engine is not None
            self.cpu_cache_engine.set_ready(node_to_ready, True, ready_length)
        elif device_type == DeviceType.SSD:
            assert self.ssd_cache_engine is not None
            self.ssd_cache_engine.set_ready(node_to_ready, True, ready_length)
        elif device_type == DeviceType.REMOTE:
            assert self.remote_cache_engine is not None
            self.remote_cache_engine.set_ready(node_to_ready, True, ready_length)

    @nvtx.annotate("Match Prefix Accel", color="yellow")
    def match_local_accel(self,
                        sequence_meta: SequenceMeta,
                        temp_cache_strategy: CacheStrategy = DEFAULT_CACHE_STRATEGY,
                        is_put: bool = False,
                        gpu_matched_blocks: int = 0) \
                            -> Tuple[MatchResultAccel, MatchResultAccel]:
        """在 CPU / SSD 两级上做前缀匹配（index_accel 版本），不查远端。

        P2P 开启时走分布式引擎的 match_all / match_local：
            GET 用 match_all（可以选中 peer 节点上的命中，matched_pos=="remote"）
            PUT 用 match_local（只认本节点，因为写入源必须是本进程的 buffer）
        未命中任何引擎时返回空的 MatchResultAccel（命中数为 0），
        调用方按 num_ready_matched_blocks 判断即可。
        """
        #from flexkv.common.debug import flexkv_logger, summarize_id_tensor
        cpu_matched_result = MatchResultAccel()
        ssd_matched_result = MatchResultAccel()
        if self.cpu_cache_engine:
            if not self.cache_config.enable_p2p_cpu:
                cpu_matched_result = self.cpu_cache_engine.match(sequence_meta)
            else:
                #flexkv_logger.info(f"[MATCH DEBUG] CPU P2P enabled, calling match_all() instead of match_local()")
                if is_put:
                    cpu_matched_result = self.cpu_cache_engine.match_local(sequence_meta)
                else:
                    cpu_matched_result = self.cpu_cache_engine.match_all(sequence_meta, gpu_matched_blocks)
        if temp_cache_strategy.ignore_ssd:
            return cpu_matched_result, ssd_matched_result
        #TODO: we assume that ssd and gds are not enabled at the same time
        if self.ssd_cache_engine:
            if not self.cache_config.enable_p2p_ssd:
                ssd_matched_result = self.ssd_cache_engine.match(sequence_meta)
            else:
                #flexkv_logger.info(f"[MATCH DEBUG] SSD P2P enabled, calling match_all() instead of match_local()")
                if is_put:
                    ssd_matched_result = self.ssd_cache_engine.match_local(sequence_meta)
                else:
                    ssd_matched_result = self.ssd_cache_engine.match_all(sequence_meta, gpu_matched_blocks)

        return cpu_matched_result, ssd_matched_result

    def _is_mooncake_swa_tier(self, device_type: DeviceType) -> bool:
        """True for the key-addressed mooncake-store REMOTE tier: SWA hits are
        keyed by the hit block's tail hash instead of a node-mounted slot."""
        return (self.use_mooncake_store_backend
                and device_type == DeviceType.REMOTE)

    # SWA 读选源：在各 tier 的匹配结果里挑一个"最深但仍在本次请求窗口内"的
    # SWA 快照，并据此把 Full-KV 的可用终点 block_mask_end 收紧到 usable_end。
    # 注意 SWA 与 Full 必须同窗口，否则窗口注意力会读到错位的上下文。
    def _select_swa_read_source(
        self,
        block_mask_start: int,
        block_mask_end: int,
        tier_match_results: Dict[DeviceType, object],
        sequence_meta: Optional[SequenceMeta] = None,
    ) -> Tuple[int, SWAReadSource]:
        """Return the largest usable SWA-aware Full-KV end and its exact SWA source."""
        if not self.swa_op_constructor.enabled or not tier_match_results:
            return block_mask_start, SWAReadSource()

        candidates: List[Tuple[int, DeviceType, object]] = []
        for device_type, match_result in tier_match_results.items():
            if match_result is None:
                continue

            swa_hit = int(match_result.swa_hit_blocks)
            if swa_hit <= block_mask_start:
                continue

            if swa_hit > block_mask_end:
                # The radix match covers the complete token sequence, while the
                # request mask may stop earlier. A snapshot for a deeper trailing
                # window cannot serve this request window; try another tier.
                continue

            if not self._is_mooncake_swa_tier(device_type):
                assert match_result.last_swa_node is not None
            candidates.append((swa_hit, device_type, match_result))

        for usable_end, device_type, match_result in sorted(
            candidates,
            key=lambda item: item[0],
            reverse=True,
        ):
            engine = self.cache_engines.get(device_type)
            if engine is None or not getattr(engine, "swa_enabled", False):
                continue

            if self._is_mooncake_swa_tier(device_type):
                assert sequence_meta is not None, (
                    "mooncake SWA source selection requires sequence_meta "
                    "for the tail hash")
                tail_hash = str(sequence_meta.block_hashes[usable_end - 1])
                return usable_end, SWAReadSource(
                    hit_blocks=usable_end,
                    device_type=device_type,
                    mooncake_tail_hash=tail_hash,
                )

            source_node = match_result.last_swa_node
            source_slot = int(source_node.swa_host_slot)
            assert source_slot >= 0

            return usable_end, SWAReadSource(
                hit_blocks=usable_end,
                host_slot=source_slot,
                node=source_node,
                device_type=device_type,
                engine=engine,
            )

        return block_mask_start, SWAReadSource()

    # 为 SWA 读建立"源 pin + 可能的 CPU 暂存槽 + H2D op"这套资源。
    # 必须在承认 Full-KV 命中之前完成：SWA 感知的 GET 若拿不到 SWA 快照，
    # 就整条作废（返回 None），不能只还原 Full 部分 —— 否则模型拿到的
    # Full 与 SWA 不一致，结果就是错的。
    def _reserve_swa_read_source(
        self,
        graph: TransferOpGraph,
        source: SWAReadSource,
        protected_cpu_node,
        dp_client_id: int,
    ) -> Optional[SWAReadReservation]:
        """Pin a source and build its SWA load chain before committing a Full hit.

        Non-CPU sources need a transient CPU SWA staging slot. Allocation may
        evict through the CPU radix, so protect the CPU Full-KV node referenced by
        this GET. Returning ``None`` means the caller must report no cache hit;
        Full-only restore is invalid for an SWA-aware GET.

        Mooncake-store REMOTE sources are key-addressed: no pin / host slot;
        a placeholder remote slot id and ``mooncake_tail_hashes`` key the
        SWA ``REMOTE2H`` op.
        """
        assert self.cpu_cache_engine is not None
        if not source.found:
            return None

        is_mooncake_source = source.is_mooncake
        # Mooncake-store will skip pin_swa_node for remote source.
        if not is_mooncake_source:
            source.engine._pin_swa_node(source.node)

        staging_slot = -1
        cpu_swa_slots = np.array([source.host_slot], dtype=np.int64)
        ssd_swa_slots = np.array([], dtype=np.int64)
        remote_swa_slots = np.array([], dtype=np.int64)

        if source.device_type != DeviceType.CPU:
            staging_slot = self.cpu_cache_engine._alloc_swa_slot(
                protected_node=protected_cpu_node)
            if staging_slot < 0:
                if not is_mooncake_source:
                    self._swa_release_load_lock(
                        node=source.node, engine=source.engine)
                flexkv_logger.warning(
                    "[FlexKV-SWA] GET staging allocation failed; "
                    f"source={source.device_type}, hit_blocks={source.hit_blocks}"
                )
                return None
            cpu_swa_slots = np.array([staging_slot], dtype=np.int64)
            if is_mooncake_source:
                remote_swa_slots = np.array([0], dtype=np.int64)
            else:
                source_slots = np.array([source.host_slot], dtype=np.int64)
                if source.device_type == DeviceType.SSD:
                    ssd_swa_slots = source_slots
                else:
                    remote_swa_slots = source_slots

        h2d_id = self.swa_op_constructor.build_get_chain(
            graph,
            gpu_slot_ids=self._SWA_GPU_PLACEHOLDER.copy(),
            cpu_slot_ids=cpu_swa_slots,
            ssd_slot_ids=ssd_swa_slots,
            remote_slot_ids=remote_swa_slots,
            dp_client_id=dp_client_id,
            mooncake_tail_hashes=(
                [source.mooncake_tail_hash] if is_mooncake_source else None),
        )
        if h2d_id is None:
            if is_mooncake_source:
                self._swa_release_load_lock(node=None, staging_slot=staging_slot)
            else:
                self._swa_release_load_lock(
                    node=source.node,
                    staging_slot=staging_slot,
                    engine=source.engine,
                )
            return None

        return SWAReadReservation(
            source=source,
            staging_slot=staging_slot,
            h2d_id=h2d_id,
        )

    def _release_swa_read_reservation(
        self, reservation: Optional[SWAReadReservation]) -> None:
        if reservation is None:
            return
        self._swa_release_load_lock(
            node=reservation.source.node,
            staging_slot=reservation.staging_slot,
            engine=reservation.source.engine,
        )

    # The GPU-side SWA slot is a size-1 placeholder here (window == one page ==
    # one slot on DSv4). It is rebound late from the request's swa_slot_mapping
    # via TransferOpGraph.set_swa_gpu_blocks() in launch, mirroring the Full-KV
    # GPU late-bind.

    _SWA_GPU_PLACEHOLDER = np.array([0], dtype=np.int64)

    @_synchronized_cache_tree
    def _swa_release_load_lock(self, node, staging_slot: int = -1, engine=None) -> None:
        """SWA H2D completion callback: release the source pin and free any
        transient CPU staging slot.

        For a CPU-sourced load, ``node`` is the matched CPU SWA node and its pin
        is dropped with the plain dec (dec_swa_lock_ref, NOT dec_swa_lock_only):
        the loaded window stays cached for future reuse. For a staged
        (SSD/REMOTE) source, ``node`` is the source-tier node (same pin release)
        and ``staging_slot`` is the transient CPU SWA slot used as the DISK2H/
        REMOTE2H destination — it is unmounted (not a cached entry), so free it
        back to the CPU SWA pool. No-op on parts that are absent."""
        try:
            if node is not None and getattr(node, "swa_lock_ref", 0) > 0:
                node.dec_swa_lock_ref()
                if engine is not None:
                    engine.index.unlock(node)
                elif hasattr(node, "unlock"):
                    node.unlock()
                else:
                    node.lock_cnt -= 1
        except Exception:  # noqa: BLE001 — never let a callback crash the loop
            pass
        try:
            if staging_slot is not None and staging_slot >= 0:
                cpu_engine = self.cpu_cache_engine
                if cpu_engine is not None:
                    cpu_engine._free_swa_slot(int(staging_slot))
        except Exception:  # noqa: BLE001
            pass

    @nvtx.annotate("Match Prefix", color="yellow")
    def match_local(self,
                    sequence_meta: SequenceMeta,
                    temp_cache_strategy: CacheStrategy = DEFAULT_CACHE_STRATEGY,
                    is_put: bool = False) \
                        -> Tuple[MatchResult, MatchResult]:
        """在 CPU / SSD 两级上做前缀匹配（Python 索引版本），不查远端。

        与 match_local_accel 的差别：本版本不区分 P2P（P2P 只在 accel 分支里
        处理），直接对本节点的两个引擎各自 match。
        """
        cpu_matched_result = MatchResult()
        ssd_matched_result = MatchResult()
        if self.cpu_cache_engine:
            cpu_matched_result = self.cpu_cache_engine.match(sequence_meta)
        if self.ssd_cache_engine and not temp_cache_strategy.ignore_ssd:
            ssd_matched_result = self.ssd_cache_engine.match(sequence_meta)

        return cpu_matched_result, ssd_matched_result

    @nvtx.annotate("Match All Prefix accel", color="yellow")
    def match_all_accel(self,
                        sequence_meta: SequenceMeta,
                        temp_cache_strategy: CacheStrategy = DEFAULT_CACHE_STRATEGY,
                        is_get: bool = True) \
                            -> Tuple[MatchResultAccel, MatchResultAccel, MatchResultAccel]:
        """在 CPU / SSD / REMOTE 三级上各做一次前缀匹配（index_accel 版本）。

        三级各自独立匹配，返回三个 MatchResultAccel；由调用方
        （_get_impl_global / _put_impl_global）比较命中长度来切分 fragment。
        远端开启 kv_sharing 时，GET 用 match_all（可跨节点）、PUT 用 match_local。
        """
        cpu_matched_result = MatchResultAccel()
        ssd_matched_result = MatchResultAccel()
        remote_matched_result = MatchResultAccel()
        if self.cpu_cache_engine:
            cpu_matched_result = self.cpu_cache_engine.match(sequence_meta)
        if self.ssd_cache_engine and not temp_cache_strategy.ignore_ssd:
            ssd_matched_result = self.ssd_cache_engine.match(sequence_meta)
        if self.remote_cache_engine and not temp_cache_strategy.ignore_remote:
            if self.enable_kv_sharing:
                if is_get:
                    remote_matched_result = self.remote_cache_engine.match_all(sequence_meta)
                else:
                    remote_matched_result = self.remote_cache_engine.match_local(sequence_meta)
            else:
                remote_matched_result = self.remote_cache_engine.match(sequence_meta)

        return cpu_matched_result, ssd_matched_result, remote_matched_result

    @nvtx.annotate("Match All Prefix", color="yellow")
    def match_all(self,
                  sequence_meta: SequenceMeta,
                  temp_cache_strategy: CacheStrategy = DEFAULT_CACHE_STRATEGY) \
                      -> Tuple[MatchResult, MatchResult, MatchResult]:
        """在 CPU / SSD / REMOTE 三级上各做一次前缀匹配（Python 索引版本）。"""
        cpu_matched_result = MatchResult()
        ssd_matched_result = MatchResult()
        remote_matched_result = MatchResult()
        if self.cpu_cache_engine:
            cpu_matched_result = self.cpu_cache_engine.match(sequence_meta)
        if self.ssd_cache_engine and not temp_cache_strategy.ignore_ssd:
            ssd_matched_result = self.ssd_cache_engine.match(sequence_meta)
        if self.remote_cache_engine and not temp_cache_strategy.ignore_remote:
            remote_matched_result = self.remote_cache_engine.match(sequence_meta)

        return cpu_matched_result, ssd_matched_result, remote_matched_result

    def _check_input(self,
                      token_ids: np.ndarray,
                      token_mask: np.ndarray,
                      slot_mapping: np.ndarray) -> None:
        """入参形状 / dtype 校验。slot_mapping 的长度必须等于 mask 中 True 的个数，
        因为它采用"紧凑排列"：只为需要搬运的 token 提供槽位。"""
        assert token_ids.dtype == np.int64
        # assert token_mask.dtype == np.bool_, f"token_mask.dtype={token_mask.dtype}"
        assert slot_mapping.dtype == np.int64
        assert token_ids.ndim == 1
        assert token_mask.ndim == 1
        assert slot_mapping.ndim == 1
        assert token_ids.size == token_mask.size, f"token_ids.size={token_ids.size}, token_mask.size={token_mask.size}"
        assert slot_mapping.size == token_mask.sum(), \
            f"slot_mapping.size={slot_mapping.size}, token_mask.sum()={token_mask.sum()}"

    @staticmethod
    def slot_mapping_to_block_ids(slot_mapping: np.ndarray, tokens_per_block: int) -> np.ndarray:
        """把 GPU 的 slot_mapping 换算成 block 编号。

        slot 与 block 的关系：一个 block 装 tokens_per_block 个 token，
        所以每隔 tokens_per_block 个 slot 取一个，再除以该值即为 block id
        （同一个 block 内所有 slot 整除后都得到同一个 id）。
        """
        block_ids: np.ndarray = slot_mapping[::tokens_per_block] // tokens_per_block
        return block_ids

    def swa_slot_mapping_to_slot_ids(self, swa_slot_mapping: np.ndarray) -> np.ndarray:
        """Convert an SWA slot_mapping into page-granular SWA pool slot ids."""
        window = self.tokens_per_block
        sm = np.asarray(swa_slot_mapping, dtype=np.int64)
        return sm[::window] // window

    def _get_block_range(self,
                         token_mask: np.ndarray) -> Tuple[int, int]:
        """由 token_mask 求出需要处理的 block 区间 [start, end)。

        end 是"最后一个 True 所在 block + 1"，即右开区间；mask 中间的空洞
        不额外处理（缓存命中必然是前缀，中间空洞只能重算）。
        """
        mask_idx = np.where(token_mask)[0]
        if len(mask_idx) == 0:
            return 0, 0
        start_idx = mask_idx[0].item() // self.tokens_per_block
        end_idx = mask_idx[-1].item() // self.tokens_per_block
        return start_idx, end_idx + 1
