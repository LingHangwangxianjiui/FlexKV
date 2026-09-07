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
"""Python RadixTreeIndex with node-mounted SWA (mirror of csrc/radix_tree.cpp).

SWA (sliding-window-attention) KV snapshots are mounted directly on the Full-KV
radix nodes (hicache / sglang swa_radix_cache.py style) rather than living in a
separate hash-keyed index. This unifies SWA and Full eviction so the two pools
never drift apart (which would erode the reusable prefix computed as
``usable = min(full_hit, swa_hit)``).

Core invariants (enforced here):
* I0: each node holds at most one SWA, at its LAST page (its trailing window).
      Splitting a node keeps the SWA on the half that still owns the original
      last page (the suffix half).
* I1: SWA subset of Full. Freeing a node's Full KV always frees its SWA slot.
* I2: every leaf must have SWA unless its Full is locked (a leaf that lost its
      SWA and is not full-locked is meaningless and gets deleted).
* I3: full_lock_ref (== node.lock_cnt here) >= swa_lock_ref always.
* I4: a non-tombstone node is homogeneous (whole node SWA-mapped, or tombstone).

This module is the non-accel mirror; DSv4 uses the C++ CRadixTreeIndex. Kept
method-for-method equivalent so the pure-Python path can be unit-tested without
torch/GPU (see tests/test_swa_node_mount.py).
"""
# ==============================================================================
# flexkv/cache/radixtree.py —— 纯 Python 实现的前缀索引（Radix Tree）
# ------------------------------------------------------------------------------
# 本文件职责：维护「token 前缀哈希序列 -> 物理 block 编号」的映射，回答一个
#   问题：「一个新来的请求，和已经缓存的内容最长能复用多少个 block？」
#   这是 KV Cache 复用（prefix caching）的核心数据结构。
#
# 为什么用 Radix Tree 而不是哈希表：
#   KV Cache 可复用的前提是「从第一个 token 起的连续公共前缀」——Transformer
#   的注意力是因果的，第 k 个 token 的 K/V 依赖它前面所有 token，所以只有前缀
#   相同才可复用，中间相同不算。哈希表只能做全等查询（整条命中或不命中），
#   既表达不了「部分命中了多少个 block」，也无法在命中一半时把已有条目切成
#   两段。Radix Tree（压缩前缀树）天然提供三件哈希表给不了的东西：
#     1. 最长前缀匹配：沿树一次下降就得到最长公共前缀（见 match_prefix），
#        复杂度只与命中深度有关，与树上总条目数无关；
#     2. 按任意位置切分：新请求在第 k 个 block 分叉时，把原节点 split 成
#        [0,k) 与 [k,n) 两段共享前缀（见 RadixNode.split）；
#     3. 前缀共享：N 条同前缀序列只存一份公共前缀，既省内存，也让「按前缀
#        聚合淘汰」成为可能——删掉一个叶子就只释放它独占的那段后缀。
#   节点里存的不是 token 本身，而是前缀哈希（common/hash_utils.py 的
#   gen_hashes，XXH64，与 C++ 侧 csrc/hash.cpp 同一算法），因此比较一整段
#   只需比哈希；部分匹配的边用二分查找定位首个不同的 block（见 match_prefix
#   里的 left/right），不是逐块线性扫。
#
# 在系统链路中的位置（控制面 / 纯 CPU 元数据路径）：
#   KVManager -> KVTaskEngine -> GlobalCacheEngine
#     -> CacheEngine（cache/cache_engine.py:603，用【本文件】的 RadixTreeIndex）
#        或 CacheEngineAccel（cache_engine.py:272，用 C++ 版 CRadixTreeIndex）
#     -> Mempool（cache/mempool.py，只记账物理块空闲与否）
#   本文件不参与任何数据搬运：它只回答「哪些 block 可复用 / 可被淘汰」，
#   真正的拷贝由 TransferEngine 执行。
#
# 核心内容速查：
#   - MatchResult    : match_prefix 的返回值（命中块数、ready 块数、插入锚点
#                      节点、命中的 physical_blocks、SWA 命中）
#   - RadixNode      : 树节点，持有一段连续的 block_hashes 及其 physical_blocks，
#                      外加锁计数、热度（grace_time / hit_count）与 SWA 挂载态
#   - RadixTreeIndex : 树本体，五套语义：
#       match_prefix  : 最长前缀匹配（读路径，决定能复用多少 KV）
#       insert        : 把写好的 (hash 段 -> blocks) 挂上树（写路径）
#       evict         : 按策略（lru/lfu/slru/fifo/mru/filo）淘汰叶子回收 block
#       lock / unlock : 引用计数锁，保护正在使用的节点不被淘汰
#       set_ready     : 标记数据真正落位，此后才可被命中（延迟可见）
#     以及 SWA（滑动窗口注意力）一整套：set_swa / promote_swa / evict_swa /
#     inc_lock_ref / dec_lock_ref / drain_freed_swa_slots
#
# 阅读提示 / 常见坑：
#   1. 【为什么要有 ready 状态】写入是异步的：规划阶段就先 insert 上树占位
#      （is_ready=False），等传输完成回调再 set_ready。因此节点可能「可见但
#      不可读」。match 同时给出 num_matched_blocks（含未 ready）与
#      num_ready_matched_blocks（真能读），调用方一律按 ready 数量切分。
#   2. 【为什么需要 lock】从 match 到传输完成的整个窗口内，命中的节点可能
#      被别人触发的 evict 踢掉，导致读到脏块或写坏别人的块。lock_cnt 让
#      evictable() 为 False 从而保住它。注意本文件的 lock() 只锁单个节点，
#      而 inc_lock_ref 会沿 [node, root) 整条路径加锁——因为父节点被删会
#      连带删掉子树。
#   3. 【淘汰的分工】Mempool 决定「什么时候需要淘汰」（利用率超阈值 / 本次
#      需求大于空闲数），本文件的 evict 决定「淘汰谁」（按策略挑叶子），
#      最后由 Mempool.recycle_blocks 真正回收。两者必须严格配对，否则会
#      出现双份释放或 block 泄漏。
#   4. 【split 的隐含约定】split 出的新节点 lock_cnt 从 0 开始、不继承，
#      代码注释写明 "only lock near-leaf node"：锁只加在叶子附近。
#   5. 【与 C++ 版的关系】本文件是 csrc/radix_tree.cpp 的 Python 镜像，方法
#      逐一对齐，目的之一是让 SWA 逻辑能在没有 GPU/torch 的环境下跑单测
#      （tests/test_swa_node_mount.py）。生产路径 CacheEngineAccel 走 C++。
#   6. 本文件内部不加锁，并发安全由调用方 CacheEngine 的
#      _synchronized_cache_tree 装饰器保证。
# ==============================================================================

import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

from flexkv.common.block import SequenceMeta
from flexkv.common.hash_utils import HashType, Hasher
from flexkv.common.transfer import DeviceType


# match_prefix 的结果。字段分三类：
#   * 数量：num_matched_blocks（含未 ready 的全部命中）/ num_ready_matched_blocks
#     （真能立即读的前缀长度，见 match_prefix 里关于「前缀长度而非计数」的注释）
#   * 锚点：last_node + last_node_matched_length 描述匹配终止的位置，insert 直接
#     以它为起点挂新节点（省一次查找）；last_ready_node 是最后一个全命中的
#     ready 节点
#   * 数据：physical_blocks 是命中路径上各节点 physical_blocks 的拼接，调用方
#     拿它直接构造传输 DAG
@dataclass
class MatchResult:
    """一次前缀匹配的结果，同时作为 insert 的锚点（见 insert 的 match_result 参数）。

    在链路中的职责：由 CacheEngine.match 产出，交给 GlobalCacheEngine 决定
    「哪些 block 可以直接复用、哪些要重算 / 从下级存储拉」，再传给 insert 复用。
    """
    num_ready_matched_blocks: int = 0
    num_matched_blocks: int = 0
    matched_pos: str = "local"
    last_ready_node: Optional['RadixNode'] = None
    last_node: Optional['RadixNode'] = None
    last_node_matched_length: int = 0
    physical_blocks: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    # ===== SWA (node-mounted) =====
    # Deepest fully-matched, ready node that carries a live SWA slot, and the
    # ready-prefix block count ending at that node. This is the SWA hit: the
    # longest reusable trailing-SWA prefix (one forward pass, no backtracking).
    last_swa_node: Optional['RadixNode'] = None
    swa_hit_blocks: int = 0

    def __post_init__(self) -> None:
        assert self.physical_blocks.ndim == 1
        assert self.physical_blocks.dtype == np.int64

    def is_empty(self) -> bool:
        return self.num_matched_blocks == 0

# 树节点 = 一段连续的 block（边），而不是单个 block。
# 这是 radix tree「压缩」的含义：把一串只有单一孩子的节点合并成一条边，
# 节点内 block_hashes[i] 与 physical_blocks[i] 一一对应（__post_init__ 断言
# 两者等长）。分裂（split）/ 收缩（shrink）都是对这两个数组做切片。
# 对应 C++ 侧的 CRadixNode。
@dataclass
class RadixNode:
    """Radix Tree 上的一个节点（一段连续的 block），对应 C++ 的 CRadixNode。

    在链路中的职责：索引的最小单元。它只记录「这段前缀哈希对应哪些物理
    block」以及用于淘汰的元信息，不持有任何真实 KV 数据。

    关键设计：
        block_hashes / physical_blocks  : 两条等长的 int64 数组，按下标一一对应。
                                          前者是索引键（前缀哈希），后者是值
                                          （本 tier 的物理 block 编号）。
        children                        : 见下方字段注释——按「孩子首 block 的
                                          前缀哈希」索引的 dict，等价于 C++ 的
                                          unordered_map。
        is_ready / lock_cnt             : 可见性与引用计数，决定 evictable()。
        grace_time / hit_count / *_time : 淘汰策略的输入，由 _get_eviction_priority
                                          按策略读取。
        swa_*                           : SWA（滑动窗口注意力）快照的挂载状态，
                                          见文件头与 SWA 相关方法。
    """
    block_hashes: np.ndarray
    physical_blocks: np.ndarray

    is_ready: bool
    lock_cnt: int          # Full-KV lock ref (== sglang full_lock_ref)
    grace_time: float
    hit_count: int = 0
    creation_time: float = 0.0
    last_access_time: float = 0.0

    # 只有父指针：节点不知道自己在树的哪一层，也无法从叶子反查整条序列，
    # 因此「从某叶子往上走到根」是唯一的方向（inc_lock_ref / set_ready 都靠它）。
    parent: Optional['RadixNode'] = None
    # 孩子按「该孩子第一个 block 的前缀哈希」索引（等价于 C++ 侧的
    # unordered_map<HashType, CRadixNode*>）。用哈希而非 token 作键，查找是
    # O(1) 且无需在节点里保存原始 token。节点本身不含自己的首哈希，需要时
    # 由 head_hash() 现算 block_hashes[0]。
    children: Dict[Optional[HashType], 'RadixNode'] = field(default_factory=dict)

    # ===== SWA (node-mounted) state — mirrors CRadixNode =====
    swa_host_slot: int = -1            # CPU SWA host-pool slot id (-1 = no SWA)
    swa_tombstone: bool = True         # True = no live SWA on this node
    swa_lock_ref: int = 0              # SWA lock ref (invariant: <= lock_cnt)
    swa_last_access_time: float = 0.0  # SWA-LRU timestamp
    # intrusive SWA-LRU doubly-linked list pointers (independent of full LRU)
    swa_lru_prev: Optional['RadixNode'] = None
    swa_lru_next: Optional['RadixNode'] = None
    on_swa_lru: bool = False

    def __post_init__(self) -> None:
        assert self.block_hashes.ndim == 1
        assert self.physical_blocks.ndim == 1
        assert self.block_hashes.size == self.physical_blocks.size

    def __lt__(self, other: 'RadixNode') -> bool:
        """仅为 heapq 兜底比较：dataclass 默认不可比较，当 (priority, node) 的
        priority 相等时，heapq 会继续比 node，此时按 grace_time 定序。"""
        return self.grace_time < other.grace_time

    def size(self) -> int:
        return self.block_hashes.size

    def head_hash(self) -> HashType:
        """本节点第一个 block 的前缀哈希，即它在父节点 children 里的键。

        空节点（root）返回 0——root 不参与 children 索引，所以这是个安全兜底。
        """
        return HashType(int(self.block_hashes[0])) if self.size() > 0 else HashType(0)

    def num_children(self) -> int:
        return len(self.children)

    def is_leaf(self) -> bool:
        return self.num_children() == 0

    def is_root(self) -> bool:
        return self.parent is None

    def evictable(self) -> bool:
        """本节点是否可被淘汰：非 root、是叶子、且未被占用。

        只淘汰叶子是 radix tree 的基本约束——内部节点被所有子孙共享，删掉它
        会连带丢掉别人的前缀。所以 evict 删掉一个叶子后，要把因此变成叶子的
        父节点重新塞回候选堆（见 RadixTreeIndex.evict）。
        """
        # 只有叶子可淘汰：内部节点承载子孙共享的前缀，删了会连坐
        return not self.is_root() and self.is_leaf() and not self.in_use()

    def in_use(self) -> bool:
        """是否被占用（不可淘汰）：被 Full 锁 / 被 SWA 锁 / 尚未 ready。

        最后一项是「延迟上树」的另一半保障：正在写入、数据还没落位的节点绝不能
        被淘汰，否则它的 physical_blocks 会被回收给别人，传输却还在往里写。
        """
        # A node is in use (not full-evictable) if its Full KV is locked, its SWA
        # is locked (I3: swa_lock_ref>0 implies it must stay), or it is not ready.
        return self.lock_cnt > 0 or self.swa_lock_ref > 0 or not self.is_ready

    def has_swa(self) -> bool:
        """True iff this node carries a live (non-tombstone) SWA slot."""
        return (not self.swa_tombstone) and self.swa_host_slot >= 0

    def inc_swa_lock_ref(self) -> None:
        """Pin the SWA against eviction (mirror of CRadixNode.inc_swa_lock_ref)."""
        self.swa_lock_ref += 1

    def dec_swa_lock_ref(self) -> None:
        """Release one SWA eviction pin (mirror of CRadixNode.dec_swa_lock_ref).

        Keeps the SWA slot cached (unlike dec_swa_lock_only, which frees a leaf's
        SWA); used to drop a load-time pin once the SWA H2D has read the slot."""
        assert self.swa_lock_ref > 0
        self.swa_lock_ref -= 1

    def lock(self) -> None:
        """加一次 Full-KV 引用锁（节点方法版本；树方法见 RadixTreeIndex.lock）。

        锁是引用计数而非布尔，允许多个请求同时保护同一节点；只有降到 0 才
        重新变得可淘汰。
        """
        assert self.lock_cnt >= 0
        self.lock_cnt += 1

    def unlock(self) -> None:
        """释放一次 Full-KV 引用锁。"""
        assert self.lock_cnt > 0
        self.lock_cnt -= 1

    def split(self, prefix_length: int) -> 'RadixNode':
        """在 prefix_length 处把本节点切成两段，返回前半段（新父节点）。

        Args:
            prefix_length: 切分点，必须满足 0 < prefix_length < size()（即真正
                的「部分匹配」；整段命中不需要切分）
        Returns:
            new_node：前缀半段 [0, prefix_length)，成为本节点的新父节点；
                      本节点 self 保留后缀半段 [prefix_length, size())
        Note:
            这是 radix tree 相对哈希表的关键能力：新请求在节点中间分叉时不必
            重建整条条目，切一刀就能让两段继续共享前缀。
            两个隐含约定：
              1. 新节点 lock_cnt 从 0 开始（锁只加在靠近叶子的节点上，内部
                 节点理论上不该被锁住）；
              2. SWA 留在 self 上（不变式 I0：SWA 属于节点最后一个 page，
                 切分后最后一页仍归后缀半段所有）。
            调用方（RadixTreeIndex.insert）在 split 后还需自行维护
            leaf_nodes 字典——本方法只调整树结构与 parent / children 指针。
        """
        assert prefix_length < self.size()
        assert prefix_length > 0
        assert self.parent is not None
        # assert not self.in_use()
        # new_node is the PREFIX half and becomes parent of self; self becomes
        # the SUFFIX half. SWA (I0) lives on the node's LAST page, which after the
        # split still belongs to self (the suffix). So SWA stays on self and
        # new_node has no SWA (default tombstone / slot=-1). This is the crux of
        # the node-mount design: split must PRESERVE the SWA, not free it.
        new_node = RadixNode(
            block_hashes=self.block_hashes[:prefix_length],
            physical_blocks=self.physical_blocks[:prefix_length],
            is_ready=self.is_ready,
            lock_cnt=0,  # Note: only lock near-leaf node
            grace_time=self.grace_time,
            hit_count=self.hit_count,
            creation_time=self.creation_time,
            last_access_time=self.last_access_time,
        )
        # new_node (prefix) inherits NO SWA — its trailing page is an interior
        # page of the original node, which never carried a window.
        assert new_node.swa_host_slot == -1 and new_node.swa_tombstone
        self.block_hashes = self.block_hashes[prefix_length:]
        self.physical_blocks = self.physical_blocks[prefix_length:]

        # 指针重接：把 new_node 插到 self 与原父节点之间
        # （原父 -> new_node -> self）。children 的键始终是孩子的 head_hash。
        self.parent.children[new_node.head_hash()] = new_node
        new_node.parent = self.parent
        self.parent = new_node
        new_node.children[self.head_hash()] = self
        # self keeps its SWA and its SWA-LRU membership unchanged (its last page
        # is unchanged), so no SWA-LRU surgery is needed here.
        return new_node

    def shrink(self, length: int) -> Tuple[np.ndarray, np.ndarray]:
        assert length < self.size()
        assert length > 0
        assert self.is_leaf()
        assert not self.in_use()
        remaining_length = self.size() - length
        physical_blocks = self.physical_blocks[remaining_length:]
        evicted_block_hashes = self.block_hashes[remaining_length:]
        self.block_hashes = self.block_hashes[:remaining_length]
        self.physical_blocks = self.physical_blocks[:remaining_length]
        return physical_blocks, evicted_block_hashes

    def merge_child(self) -> None:  # ignore status
        assert self.num_children() == 1
        child = list(self.children.values())[0]
        self.block_hashes = np.concatenate([self.block_hashes, child.block_hashes])
        self.physical_blocks = np.concatenate([self.physical_blocks, child.physical_blocks])
        self.grace_time = max(self.grace_time, child.grace_time)
        self.last_access_time = max(self.last_access_time, child.last_access_time)
        self.creation_time = min(self.creation_time, child.creation_time)
        self.hit_count = max(self.hit_count, child.hit_count)
        self.children.clear()
        # NOTE: SWA handoff (I0) is done by RadixTreeIndex.merge_child, which owns
        # freed_swa_slots + the SWA-LRU (this node has no back-ref to the tree).

class RadixTreeIndex:
    def __init__(self, tokens_per_block: int, max_num_blocks: int = 1000000,
                 hit_reward_seconds: int = 0, eviction_policy: str = "lru",
                 protected_threshold: int = 2):
        self.root_node: RadixNode = RadixNode(block_hashes=np.array([], dtype=np.int64),
                                              physical_blocks=np.array([], dtype=np.int64),
                                              is_ready=True,
                                              lock_cnt=0,
                                              grace_time=time.time())

        self.tokens_per_block = tokens_per_block

        self.leaf_nodes: Dict[HashType, RadixNode] = {}

        self.max_num_blocks = max_num_blocks

        self.hit_reward_seconds = hit_reward_seconds
        self.eviction_policy = eviction_policy
        self.protected_threshold = protected_threshold

        # ===== SWA-only LRU (intrusive doubly-linked list w/ sentinels) =====
        # head side = MRU, tail side = LRU. Nodes carrying a live SWA slot are on
        # this list; SWA-only eviction walks from the tail. Independent of the
        # Full-KV leaf_nodes / eviction so SWA can be reclaimed without touching
        # Full KV (multi-turn dialogue: evict interior-prefix SWA first).
        self._swa_lru_head = RadixNode(block_hashes=np.array([], dtype=np.int64),
                                       physical_blocks=np.array([], dtype=np.int64),
                                       is_ready=True, lock_cnt=0, grace_time=0.0)
        self._swa_lru_tail = RadixNode(block_hashes=np.array([], dtype=np.int64),
                                       physical_blocks=np.array([], dtype=np.int64),
                                       is_ready=True, lock_cnt=0, grace_time=0.0)
        self._swa_lru_head.swa_lru_next = self._swa_lru_tail
        self._swa_lru_tail.swa_lru_prev = self._swa_lru_head
        # SWA slots freed by structural changes (split / merge / evict / evict_swa),
        # drained by the cache engine to return them to the SWA host pool.
        self._freed_swa_slots: List[int] = []
        # True once any SWA slot has been mounted (set from set_swa). Gates the I2
        # tombstone-leaf cascade in the always-running full-KV evict(): a node's
        # swa_tombstone DEFAULTS to True, so in a non-SWA deployment EVERY leaf is
        # a tombstone and an unconditional cascade would over-evict valid
        # ancestors. evict_swa() needs no gate (it only runs when SWA is enabled).
        self._swa_enabled: bool = False

    def reset(self) -> None:
        self.root_node = RadixNode(block_hashes=np.array([], dtype=np.int64),
                                   physical_blocks=np.array([], dtype=np.int64),
                                   is_ready=True,
                                   lock_cnt=0,
                                   grace_time=time.time())
        self.leaf_nodes.clear()
        self._swa_lru_head.swa_lru_next = self._swa_lru_tail
        self._swa_lru_tail.swa_lru_prev = self._swa_lru_head
        self._freed_swa_slots.clear()
        # keep _swa_enabled sticky across reset (mirrors C++: the pool geometry /
        # SWA usage is a deployment property, not per-tree-generation state)

    def is_empty(self) -> bool:
        return len(self.leaf_nodes) == 0

    # ===== SWA-LRU intrusive list helpers =================================

    def _swa_lru_add_mru(self, node: RadixNode) -> None:
        """Insert (or move) ``node`` at the MRU side (right after head)."""
        if node.on_swa_lru:
            self._swa_lru_remove(node)
        nxt = self._swa_lru_head.swa_lru_next
        node.swa_lru_prev = self._swa_lru_head
        node.swa_lru_next = nxt
        self._swa_lru_head.swa_lru_next = node
        nxt.swa_lru_prev = node
        node.on_swa_lru = True

    def _swa_lru_remove(self, node: RadixNode) -> None:
        if not node.on_swa_lru:
            return
        node.swa_lru_prev.swa_lru_next = node.swa_lru_next
        node.swa_lru_next.swa_lru_prev = node.swa_lru_prev
        node.swa_lru_prev = None
        node.swa_lru_next = None
        node.on_swa_lru = False

    def _swa_lru_get_lru_unlocked(self) -> Optional[RadixNode]:
        """Least-recently-used SWA node with swa_lock_ref == 0 (not just leaves)."""
        x = self._swa_lru_tail.swa_lru_prev
        while x is not self._swa_lru_head and x.swa_lock_ref > 0:
            x = x.swa_lru_prev
        return x if x is not self._swa_lru_head else None

    def record_freed_swa_slot(self, node: RadixNode) -> None:
        """Release a node's SWA slot (if any) and clear its SWA state.

        Called from every path that deletes or invalidates a node's trailing
        page (split / merge / evict / evict_swa) so the slot is never leaked and
        the SWA-subset-of-Full invariant (I1) holds. The slot is buffered in
        ``_freed_swa_slots`` for the cache engine to drain back to the host pool.
        """
        if node.swa_host_slot >= 0:
            self._freed_swa_slots.append(node.swa_host_slot)
        node.swa_host_slot = -1
        node.swa_tombstone = True
        self._swa_lru_remove(node)

    def drain_freed_swa_slots(self) -> List[int]:
        """Return and clear SWA slots freed since the last call (engine → pool)."""
        out = self._freed_swa_slots
        self._freed_swa_slots = []
        return out

    def set_swa(self, node: RadixNode, slot: int) -> None:
        """Mount an SWA slot on ``node``'s trailing page (store side).

        The caller guarantees ``node`` is the node whose LAST page is the target
        window (split first if not — see insert / I0). A different existing slot
        must be explicitly unmounted first; remounting the same slot refreshes
        SWA-LRU recency.
        """
        assert node is not self.root_node
        assert node.swa_host_slot < 0 or node.swa_host_slot == slot
        node.swa_host_slot = slot
        node.swa_tombstone = False
        node.swa_last_access_time = time.time()
        self._swa_lru_add_mru(node)
        self._swa_enabled = True  # arm the I2 cascade in full-KV evict()

    def promote_swa(self, node: RadixNode) -> None:
        """Refresh a node's SWA recency on read-hit: splice it to the SWA-LRU MRU.

        SWA recency lives in the LRU *list position* (evict_swa walks the linked
        list via _swa_lru_get_lru_unlocked and never reads swa_last_access_time),
        so a hit that only bumped the timestamp would NOT survive eviction — we
        must actually move the node to MRU. Mirror of set_swa's LRU touch, minus
        the mount. No-op for root or a node with no live SWA (a tombstone is not
        on the SWA-LRU; re-adding it would corrupt the list)."""
        if node is self.root_node or not node.has_swa():
            return
        node.swa_last_access_time = time.time()
        self._swa_lru_add_mru(node)  # remove+reinsert = move to MRU (idempotent)

    def merge_child(self, node: RadixNode) -> None:
        """Merge ``node``'s only child into it, moving SWA to follow the child.

        Mirror of CRadixTreeIndex/CRadixNode::merge_child: after merging, the
        combined node's LAST page is the child's last page, so the SWA follows
        the child (I0). Free the parent's stale SWA, then remount the child's on
        the parent; the tree (not the node) owns freed_swa_slots + the SWA-LRU.
        """
        assert node.num_children() == 1
        child = list(node.children.values())[0]
        child_slot = child.swa_host_slot
        node.merge_child()  # block-level merge + children.clear()
        # SWA handoff (I0): release parent's own (now-stale) SWA, then move the
        # child's SWA onto the merged node.
        self.record_freed_swa_slot(node)          # frees parent's old slot (if any)
        if child_slot >= 0:
            self._swa_lru_remove(child)           # detach the doomed child
            child.swa_host_slot = -1
            child.swa_tombstone = True
            if node is not self.root_node:
                self.set_swa(node, child_slot)    # remount on the merged node
            else:
                # root can't carry SWA; free it rather than leak.
                self._freed_swa_slots.append(child_slot)

    def match_prefix(self,
                    sequence: SequenceMeta,
                    update_cache_info: bool = True) -> MatchResult:
        sequence.gen_hashes()
        current_node = self.root_node
        last_ready_node = self.root_node
        prefix_blocks_num = 0
        ready_prefix_blocks_num = 0
        # ``num_ready_matched_blocks`` is a prefix length, not a count of ready
        # blocks anywhere on the matched path.  A later child may become ready
        # before an in-flight parent (for example after a concurrent branch
        # split); once that happens, no descendant can extend the readable
        # prefix until the parent is ready too.
        ready_prefix_open = True
        last_node_matched_length = 0
        physical_blocks = np.array([], dtype=np.int64)
        # SWA: deepest fully-matched ready node carrying a live SWA slot.
        last_swa_node: Optional[RadixNode] = None
        swa_hit_blocks = 0
        while prefix_blocks_num < sequence.num_blocks:
            if update_cache_info:
                now = time.time()
                current_node.last_access_time = now
                if current_node.grace_time < now:
                    current_node.grace_time = now + self.hit_reward_seconds
                else:
                    current_node.grace_time += self.hit_reward_seconds
                # Python int is unbounded, so no overflow concern here.
                # For SLRU the value only matters via comparison against
                # protected_threshold; for LFU it preserves monotonic ordering.
                current_node.hit_count += 1
            child_hash = sequence.get_hash(prefix_blocks_num + current_node.size())
            if child_hash in current_node.children:
                if ready_prefix_open and current_node.is_ready:
                    last_ready_node = current_node
                    ready_prefix_blocks_num += current_node.size()
                    # current_node is FULLY matched (whole size consumed): if it
                    # carries a live SWA at its trailing page, it is the new
                    # deepest SWA hit (single forward pass, no backtracking).
                    if current_node.has_swa():
                        last_swa_node = current_node
                        swa_hit_blocks = ready_prefix_blocks_num
                elif current_node.size() > 0:
                    ready_prefix_open = False
                prefix_blocks_num += current_node.size()
                physical_blocks = np.concatenate([physical_blocks, current_node.physical_blocks])
                current_node = current_node.children[child_hash]
            else:
                if not current_node.is_root():
                    cmp_length = min(current_node.size(), sequence.num_blocks - prefix_blocks_num)
                    left = 0
                    right = cmp_length
                    while left < right:
                        mid = (left + right) // 2
                        if current_node.block_hashes[mid] == sequence.get_hash(prefix_blocks_num+mid):
                            left = mid + 1
                        else:
                            right = mid
                    matched_length = left
                    physical_blocks = np.concatenate([physical_blocks, current_node.physical_blocks[:matched_length]])
                else:
                    matched_length = 0
                if ready_prefix_open and current_node.is_ready:
                    last_ready_node = current_node
                    ready_prefix_blocks_num += matched_length
                    # Only a FULL node match (matched_length == size) exposes the
                    # trailing page, so only then can this node's SWA be reused.
                    if matched_length == current_node.size() and current_node.has_swa():
                        last_swa_node = current_node
                        swa_hit_blocks = ready_prefix_blocks_num
                elif matched_length > 0:
                    ready_prefix_open = False
                last_node_matched_length = matched_length
                prefix_blocks_num += matched_length
                break
        # Read-hit heat update: on a real match (update_cache_info=True), promote
        # the matched SWA node to its SWA-LRU MRU — the SWA peer of the per-node
        # Full-KV heat bump above, so a reused SWA copy survives eviction over a
        # never-reused one. last_swa_node is the deepest fully-matched ready node
        # with a live SWA slot (or None); promote_swa no-ops on root / no-SWA.
        # A probe (update_cache_info=False) must NOT touch the SWA-LRU.
        if update_cache_info and last_swa_node is not None:
            self.promote_swa(last_swa_node)
        return MatchResult(num_matched_blocks=prefix_blocks_num,
                           num_ready_matched_blocks=ready_prefix_blocks_num,
                           last_ready_node=last_ready_node,
                           last_node=current_node,
                           last_node_matched_length=last_node_matched_length,
                           physical_blocks=physical_blocks,
                           last_swa_node=last_swa_node,
                           swa_hit_blocks=swa_hit_blocks)

    def num_matched_blocks(self,
                    sequence: SequenceMeta) -> int:
        match_result = self.match_prefix(sequence)
        return match_result.num_matched_blocks

    def insert(self,
               sequence_meta: SequenceMeta,
               physical_block_ids: np.ndarray,
               num_insert_blocks: int = -1,
               is_ready: bool = True,
               match_result: Optional[MatchResult] = None) -> Optional[RadixNode]:
        if num_insert_blocks == -1:
            num_insert_blocks = sequence_meta.num_blocks
        assert 0 <= num_insert_blocks <= sequence_meta.num_blocks

        assert physical_block_ids.ndim == 1
        assert physical_block_ids.dtype == np.int64

        sequence_meta.gen_hashes()
        if match_result is None:
            match_result = self.match_prefix(sequence_meta)
        num_matched_blocks = match_result.num_matched_blocks
        last_node = match_result.last_node
        assert last_node is not None
        last_node_matched_length = match_result.last_node_matched_length

        assert len(physical_block_ids) == num_insert_blocks - num_matched_blocks, \
            f"num_insert_blocks = {num_insert_blocks}, " \
            f"num_matched_blocks = {num_matched_blocks}, " \
            f"len(physical_block_ids) = {len(physical_block_ids)}"

        if num_matched_blocks >= num_insert_blocks:
            # not insert any new blocks
            return None

        self._validate_insert_target(
            sequence_meta,
            last_node,
            num_matched_blocks,
            last_node_matched_length,
        )

        now = time.time()
        new_node = RadixNode(
            block_hashes=sequence_meta.block_hashes[num_matched_blocks:num_insert_blocks],
            physical_blocks=physical_block_ids,
            is_ready=is_ready,
            lock_cnt=0,
            grace_time=now,
            creation_time=now,
            last_access_time=now,
        )

        last_node_leaf = last_node.is_leaf() and not last_node.is_root()
        if last_node_leaf:
            self.leaf_nodes.pop(last_node.head_hash(), None)

        if last_node_matched_length < last_node.size():
            last_node.split(last_node_matched_length)
            if last_node_leaf:
                self.leaf_nodes[last_node.head_hash()] = last_node
            last_node = last_node.parent
            assert last_node is not None

        new_node.parent = last_node
        last_node.children[new_node.head_hash()] = new_node
        self.leaf_nodes[new_node.head_hash()] = new_node

        return new_node

    def _validate_insert_target(
        self,
        sequence_meta: SequenceMeta,
        last_node: RadixNode,
        num_matched_blocks: int,
        last_node_matched_length: int,
    ) -> None:
        """Reject a stale insertion anchor before mutating the tree.

        ``match_prefix`` results may outlive another insertion when this low-level
        API is used directly.  In that case assigning ``children[head_hash]``
        would silently orphan the existing subtree and its physical blocks.
        Raising leaves both the radix tree and ``physical_block_ids`` untouched;
        the caller still owns those blocks and may recycle them or rematch/retry.
        """
        conflict = "radix insert conflict: stale match result; rematch before retrying"
        node_size = last_node.size()
        anchor_root = last_node
        while anchor_root.parent is not None:
            anchor_root = anchor_root.parent
        if anchor_root is not self.root_node:
            raise RuntimeError(conflict)

        if last_node.is_root():
            if last_node_matched_length != 0 or num_matched_blocks != 0:
                raise RuntimeError(conflict)
        else:
            parent = last_node.parent
            if (parent is None
                    or parent.children.get(last_node.head_hash()) is not last_node
                    or last_node_matched_length <= 0
                    or last_node_matched_length > node_size):
                raise RuntimeError(conflict)

            matched_start = num_matched_blocks - last_node_matched_length
            if (matched_start < 0
                    or not np.array_equal(
                        last_node.block_hashes[:last_node_matched_length],
                        sequence_meta.block_hashes[
                            matched_start:num_matched_blocks])):
                raise RuntimeError(conflict)

        target_hash = sequence_meta.get_hash(num_matched_blocks)
        assert target_hash is not None
        if last_node_matched_length < node_size:
            # A valid partial match diverges at the split point.  Equality means
            # that the supplied match result no longer describes this node.
            if last_node.block_hashes[last_node_matched_length] == target_hash:
                raise RuntimeError(conflict)
        elif target_hash in last_node.children:
            raise RuntimeError(
                "radix insert conflict: target child already exists; "
                "rematch before retrying")

    def evict(self, num_evicted: int) -> Tuple[np.ndarray, np.ndarray]:
        candidates = []
        for node in self.leaf_nodes.values():
            if node.evictable():
                priority = self._get_eviction_priority(node)
                candidates.append((priority, node))
        heapq.heapify(candidates)
        evicted_blocks = np.array([], dtype=np.int64)
        evicted_block_hashes = np.array([], dtype=np.int64)
        while len(evicted_blocks) < num_evicted and candidates:
            priority, node = heapq.heappop(candidates)
            if node.size() > num_evicted - len(evicted_blocks):
                physical_blocks, _block_hashes = node.shrink(num_evicted - len(evicted_blocks))
                # SWA: node survives but its trailing page changed, so its old
                # window no longer covers the last page — release it (I1).
                self.record_freed_swa_slot(node)
            else:
                parent = node.parent
                assert parent is not None  # node is not root
                parent.children.pop(node.head_hash())
                self.leaf_nodes.pop(node.head_hash(), None)
                physical_blocks = node.physical_blocks
                _block_hashes = node.block_hashes
                # SWA: node is deleted; release its slot so it is not leaked (I1).
                self.record_freed_swa_slot(node)
                node.parent = None
                if parent.is_leaf() and not parent.is_root():
                    self.leaf_nodes[parent.head_hash()] = parent
                if self._swa_enabled:
                    # I2: a parent that just became a tombstone leaf (Full but no
                    # SWA, not locked) is meaningless — cascade-delete it and its
                    # tombstone ancestors. Their blocks append beyond num_evicted.
                    cascade_blocks, cascade_hashes, parent = \
                        self._iteratively_delete_tombstone_leaf(parent)
                    if cascade_blocks.size:
                        physical_blocks = np.concatenate([physical_blocks, cascade_blocks])
                        _block_hashes = np.concatenate([_block_hashes, cascade_hashes])
                if parent is not None and parent.evictable():
                    priority = self._get_eviction_priority(parent)
                    heapq.heappush(candidates, (priority, parent))

            evicted_blocks = np.concatenate([evicted_blocks, physical_blocks])
            evicted_block_hashes = np.concatenate([evicted_block_hashes, _block_hashes])

        return evicted_blocks, evicted_block_hashes

    def evict_swa(self, num_swa_evicted: int) -> Tuple[np.ndarray, int]:
        """SWA-only eviction, watermark-driven, WITHOUT touching Full-KV where
        possible. Mirrors sglang swa_radix_cache.evict_swa.

        Walks the SWA-LRU from the least-recently-used end (any node, not just
        leaves) and frees SWA slots until ``num_swa_evicted`` are reclaimed:

        * internal node: free the SWA slot + tombstone it; Full KV is KEPT (this
          is the multi-turn optimization — interior-prefix SWA is dropped first).
        * leaf with Full locked: free the SWA slot + tombstone it; the leaf stays
          alive because its Full KV is still referenced.
        * leaf without Full lock: a leaf that would lose its SWA is meaningless
          (I2), so delete the whole node (Full + SWA) and iteratively delete any
          resulting tombstone leaves.

        Freed SWA slots are buffered in ``_freed_swa_slots`` (drain to the pool).
        Returns ``(evicted_full_blocks, num_swa_freed)`` — the full physical
        blocks freed by leaf/tombstone deletions (to recycle into the mempool).
        """
        evicted_full_blocks = np.array([], dtype=np.int64)
        num_swa_freed = 0
        while num_swa_freed < num_swa_evicted:
            x = self._swa_lru_get_lru_unlocked()
            if x is None:
                break
            assert x.has_swa(), "node on SWA-LRU must carry a live SWA slot"
            if not x.is_leaf():
                # Internal node: drop SWA only, keep Full KV.
                self.record_freed_swa_slot(x)
                num_swa_freed += 1
            elif x.lock_cnt > 0:
                # Leaf whose Full is still locked: drop SWA only.
                self.record_freed_swa_slot(x)
                num_swa_freed += 1
            else:
                # Leaf, Full unlocked: delete the whole node (Full + SWA).
                self.record_freed_swa_slot(x)
                num_swa_freed += 1
                parent = x.parent
                assert parent is not None
                parent.children.pop(x.head_hash())
                self.leaf_nodes.pop(x.head_hash(), None)
                evicted_full_blocks = np.concatenate([evicted_full_blocks, x.physical_blocks])
                x.parent = None
                if parent.is_leaf() and not parent.is_root():
                    self.leaf_nodes[parent.head_hash()] = parent
                # Cascade: a parent that just became a tombstone leaf is
                # meaningless (I2) — delete it and continue up.
                freed_full, _freed_hashes, _survivor = \
                    self._iteratively_delete_tombstone_leaf(parent)
                if freed_full.size:
                    evicted_full_blocks = np.concatenate([evicted_full_blocks, freed_full])
        return evicted_full_blocks, num_swa_freed

    def remove_unready_leaf(self, node: Optional[RadixNode]) -> np.ndarray:
        """Roll back a not-yet-ready inserted leaf, returning its blocks.

        Used when a planned transfer is aborted before launch: the node was
        inserted with ``is_ready=False`` and its completion callback (which
        would have marked it ready) will never run. Left alone it can never be
        evicted (``in_use`` is true for unready nodes) and it shadows future
        puts of the same prefix, so the blocks and the prefix are both lost.

        Only removes the node when doing so is unambiguously safe: it must
        still be an unready, unlocked, SWA-unlocked leaf. If another request
        split it or inserted below it in the create-to-cancel window, the node
        is left in place (pre-existing behavior) rather than risk detaching
        blocks another plan references. Returns the physical blocks to recycle
        (empty when nothing was removed).
        """
        if (node is None or node.is_root() or node.is_ready
                or node.lock_cnt > 0 or node.swa_lock_ref > 0
                or not node.is_leaf()):
            return np.array([], dtype=np.int64)
        parent = node.parent
        assert parent is not None
        parent.children.pop(node.head_hash(), None)
        self.leaf_nodes.pop(node.head_hash(), None)
        self.record_freed_swa_slot(node)
        freed = node.physical_blocks
        node.parent = None
        if parent.is_leaf() and not parent.is_root():
            self.leaf_nodes[parent.head_hash()] = parent
        if self._swa_enabled:
            cascade_blocks, _hashes, _survivor = \
                self._iteratively_delete_tombstone_leaf(parent)
            if cascade_blocks.size:
                freed = np.concatenate([freed, cascade_blocks])
        return freed

    def _iteratively_delete_tombstone_leaf(
            self, node: RadixNode) -> Tuple[np.ndarray, np.ndarray, Optional[RadixNode]]:
        """Delete ``node`` and its ancestors while they are tombstone leaves with
        no Full lock (a leaf without SWA is meaningless, I2). Returns the Full
        physical blocks freed, their block hashes, and the last surviving
        ancestor (a non-tombstone leaf, a locked/unready node, an internal node,
        or root) so the caller can reconsider it for eviction."""
        freed = np.array([], dtype=np.int64)
        freed_hashes = np.array([], dtype=np.int64)
        while (node is not None and not node.is_root() and node.is_leaf()
               and node.swa_tombstone and node.lock_cnt == 0 and node.is_ready):
            parent = node.parent
            assert parent is not None
            parent.children.pop(node.head_hash(), None)
            self.leaf_nodes.pop(node.head_hash(), None)
            # tombstone node has no SWA slot, but call for uniformity / LRU safety
            self.record_freed_swa_slot(node)
            freed = np.concatenate([freed, node.physical_blocks])
            freed_hashes = np.concatenate([freed_hashes, node.block_hashes])
            node.parent = None
            if parent.is_leaf() and not parent.is_root():
                self.leaf_nodes[parent.head_hash()] = parent
            node = parent
        return freed, freed_hashes, node

    # ===== SWA dual lock (mirror of sglang inc/dec_lock_ref) ==============
    # FlexKV's SWA window == one page == one node's trailing page, so sglang's
    # "accumulate to sliding_window_size" logic degenerates: locking a node's SWA
    # locks exactly its own single trailing window. We still keep full and SWA
    # lock refs separate to express "Full locked but SWA already released" (I3).

    def inc_lock_ref(self, node: RadixNode) -> Optional[RadixNode]:
        """Pin [node, root): full_lock on every node; swa_lock on the single
        deepest node carrying a live SWA. Returns that SWA boundary node (the
        ``swa_uuid`` analogue) to pass back to dec_lock_ref, or None.

        FlexKV's SWA window == one page == one node's trailing page, so exactly
        ONE node on the path (the deepest with SWA) is SWA-locked; dec must
        release exactly that node — see dec_lock_ref / dec_swa_lock_only.
        """
        swa_boundary: Optional[RadixNode] = None
        cur = node
        while cur is not None and not cur.is_root():
            cur.lock_cnt += 1
            if swa_boundary is None and cur.has_swa():
                cur.swa_lock_ref += 1
                assert cur.lock_cnt >= cur.swa_lock_ref  # I3
                swa_boundary = cur
            cur = cur.parent
        return swa_boundary

    def dec_lock_ref(self, node: RadixNode,
                     swa_boundary: Optional[RadixNode] = None,
                     skip_swa: bool = False) -> None:
        """Mirror of inc_lock_ref. Unlock full on [node, root); unlock SWA on the
        exact ``swa_boundary`` node inc_lock_ref locked, unless ``skip_swa`` (the
        SWA was already released early via dec_swa_lock_only)."""
        cur = node
        while cur is not None and not cur.is_root():
            assert cur.lock_cnt > 0
            cur.lock_cnt -= 1
            # Release the SWA lock only on the exact boundary node (symmetric
            # with inc_lock_ref, which locked only that one node).
            if not skip_swa and cur is swa_boundary and cur.swa_lock_ref > 0:
                cur.swa_lock_ref -= 1
            assert cur.lock_cnt >= cur.swa_lock_ref  # I3
            cur = cur.parent

    def dec_swa_lock_only(self, swa_boundary: Optional[RadixNode]) -> None:
        """Early-release ONLY the SWA lock on the boundary node (Full lock
        untouched). Leaf → free SWA + tombstone (Full kept alive by its full
        lock); internal → leave on SWA-LRU as evictable. Caller must later
        dec_lock_ref with skip_swa=True. Symmetric with inc_lock_ref: touches
        exactly the one boundary node.

        When ``swa_boundary`` is None (no SWA was locked), this is a no-op.
        """
        if swa_boundary is None:
            return
        cur = swa_boundary
        assert cur.swa_lock_ref > 0, "dec_swa_lock_only on an unlocked SWA node"
        cur.swa_lock_ref -= 1
        if cur.swa_lock_ref == 0 and cur.has_swa() and cur.is_leaf():
            # Leaf: free SWA now (Full stays until the full lock drops).
            # An internal node keeps its SWA, staying evictable on the SWA-LRU.
            self.record_freed_swa_slot(cur)


    def _get_eviction_priority(self, node: RadixNode):
        """Get the eviction priority for a node based on the configured policy.

        Lower priority values are evicted first (min-heap).
        """
        if self.eviction_policy == "lru":
            return node.grace_time
        elif self.eviction_policy == "lfu":
            return (node.hit_count, node.last_access_time)
        elif self.eviction_policy == "fifo":
            return node.creation_time
        elif self.eviction_policy == "mru":
            return -node.last_access_time
        elif self.eviction_policy == "slru":
            is_protected = 1 if node.hit_count >= self.protected_threshold else 0
            return (is_protected, node.last_access_time)
        elif self.eviction_policy == "filo":
            return -node.creation_time
        else:
            raise ValueError(
                f"Unknown eviction policy: {self.eviction_policy}. "
                f"Supported policies: 'lru', 'lfu', 'slru', 'fifo', 'mru', 'filo'."
            )

    def lock(self, node: RadixNode) -> None:
        assert node.lock_cnt >= 0
        node.lock_cnt += 1

    def unlock(self, node: RadixNode) -> None:
        assert node.lock_cnt > 0
        node.lock_cnt -= 1

    def set_ready(self, node: RadixNode, is_ready: bool = True, ready_length: int = -1) -> None:
        node.is_ready = is_ready
        if ready_length > 0:
            ready_length -= node.size()
            num_node = 1
            while ready_length > 0:
                assert node.parent is not None
                node = node.parent
                ready_length -= node.size()
                node.is_ready = True
                num_node += 1
            assert ready_length == 0

    def total_cached_blocks(self) -> int:
        total_cached_blocks = 0
        queue = [self.root_node]
        while queue:
            node = queue.pop(0)
            total_cached_blocks += node.size()
            queue.extend(node.children.values())
        return total_cached_blocks

    def total_node_num(self) -> int:  # include root node
        total_node_num = -1  # exclude root node
        queue = [self.root_node]
        while queue:
            node = queue.pop(0)
            total_node_num += 1
            queue.extend(node.children.values())
        return total_node_num

    def total_swa_slots(self) -> int:
        """Number of live SWA slots mounted across the tree (for accounting/tests)."""
        total = 0
        queue = [self.root_node]
        while queue:
            node = queue.pop(0)
            if node.has_swa():
                total += 1
            queue.extend(node.children.values())
        return total

    def total_ready_blocks(self) -> int:
        total_ready_blocks = 0
        queue = [self.root_node]
        while queue:
            node = queue.pop(0)
            if node.is_ready:
                total_ready_blocks += node.size()
            queue.extend(node.children.values())
        return total_ready_blocks

    def total_unready_blocks(self) -> int:
        return self.total_cached_blocks() - self.total_ready_blocks()

if __name__ == "__main__":
    tokens_per_block = 2
    index = RadixTreeIndex(tokens_per_block=tokens_per_block)
    print(f"init index, tokens_per_block = {tokens_per_block}")

    token_ids1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
    token_ids2 = np.array([1, 2, 3, 4, 15, 16, 17, 18], dtype=np.int64)

    seq1 = SequenceMeta(token_ids=token_ids1, tokens_per_block=tokens_per_block)
    seq2 = SequenceMeta(token_ids=token_ids2, tokens_per_block=tokens_per_block)

    index.insert(seq1, np.array([0, 1, 2, 3], dtype=np.int64), is_ready=True)
    print(f"insert seq1 = {seq1.token_ids}, "
          f"total cached blocks = {index.total_cached_blocks()}")
    seq2_matched_blocks = index.num_matched_blocks(seq2)
    assert seq2_matched_blocks == 2
    index.insert(seq2, np.array([8, 9], dtype=np.int64), is_ready=True)
    print(f"insert seq2 = {seq2.token_ids}, "
          f"total cached blocks = {index.total_cached_blocks()}")

    seq3 = SequenceMeta(token_ids=np.array([1,2,3,4,0,0], dtype=np.int64),
                        tokens_per_block=tokens_per_block)
    match_result = index.num_matched_blocks(seq3)
    print(f"match {seq3.token_ids}, num cached blocks: {match_result}")

    evicted_blocks = index.evict(3)
    print(f"evict {len(evicted_blocks)} blocks, "
          f"total cached blocks = {index.total_cached_blocks()}")

    match_result = index.num_matched_blocks(seq1)
    print(f"match {seq1.token_ids}, num cached blocks: {match_result}")
    match_result = index.num_matched_blocks(seq2)
    print(f"match {seq2.token_ids}, num cached blocks: {match_result}")

    evicted_blocks = index.evict(10)
    print(f"evict {len(evicted_blocks)} blocks, "
          f"total cached blocks = {index.total_cached_blocks()}")

    match_result = index.num_matched_blocks(seq1)
    print(f"match {seq1.token_ids}, num cached blocks: {match_result}")
    match_result = index.num_matched_blocks(seq2)
    print(f"match {seq2.token_ids}, num cached blocks: {match_result}")
