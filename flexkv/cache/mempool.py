# ==============================================================================
# flexkv/cache/mempool.py —— 单层存储的物理 block 账本（空间追踪 + 淘汰触发器）
# ------------------------------------------------------------------------------
# 本文件职责：回答"这一层（CPU 内存 / SSD / 远端）还有多少空闲 block"，
# 并提供 block id 的批量分配与批量回收。它是纯粹的计数/位图工具，
# 既不知道 block 里存了什么，也不认识 token。
#
# 在系统链路中的位置：
#   KVManager -> KVTaskEngine -> 【GlobalCacheEngine 持有本文件 Mempool】
#   -> (控制面产出) TransferOpGraph -> TransferEngine -> Worker
#   实际持有者：cache/cache_engine.py:304（GlobalCacheEngine）与
#              cache/hie_cache_engine.py:91（HierarchyLRCacheEngine，分层/分布式）
#
# 阅读提示（本文件最关键的协作关系）：
#   * 「谁可以被淘汰」由 radix tree 决定，「是否需要淘汰」由本文件决定。
#     radix tree（CRadixTreeIndex / LocalRadixTree）保存 token 前缀 -> block id
#     的映射与访问热度；本文件只保存"物理块是否空闲"这一份事实。
#   * 典型协作流程见 GlobalCacheEngine.take()（cache_engine.py:480）：
#       1. utilization = 1 - num_free_blocks / num_total_blocks   <- 本文件提供
#       2. should_evict = 利用率超阈值 或 本次需求大于空闲数      <- 淘汰触发点
#       3. index.evict(...) 由 radix tree 挑出受害者 block id     <- 树决定淘汰谁
#       4. mempool.recycle_blocks(被淘汰的 block id)              <- 本文件回收
#       5. allocate_blocks(新数据需要的块数)
#   * 因此两个结构必须严格配对：一个 block id 要么被树引用、要么空闲在池里。
#     重复回收会触发本文件 "already free" 的 ValueError —— 这是排查
#     "双份释放 / 悬空引用"类 bug 的第一道防线，不要把它改成静默返回。
# ==============================================================================
from collections import deque
from typing import List

import numpy as np


class Mempool:
    """某一层存储的物理 block 池：用位图追踪每个 block id 的空闲/占用状态。

    在链路中的职责：GlobalCacheEngine（以及分层版本的 HierarchyLRCacheEngine）
    的"空间账本"。它不参与任何数据搬运，只做三件事：记账、发号、回收。

    关键设计——两套数据、一份事实：
        _free_mask  : 长度 num_total_blocks 的 bool 数组，True 表示空闲。
                      它是唯一的事实来源，O(1) 判空、批量置位。
        _num_free   : 空闲总数，冗余缓存，避免每次 sum() 全扫。

    关键设计——_free_ids + _free_ids_offset 游标：
        nonzero() 是 O(num_total_blocks) 的全量扫描，对上百万块的池子太贵。
        所以只在初始化/重算时扫一次，把结果缓存在 _free_ids 里，
        之后用 _free_ids_offset 当"已发到哪"的游标往后切片消费；
        游标逼近末尾（剩余不足本次需求）时才 _update_free_ids() 重扫。
        注意：回收只会把 _free_mask 置回 True，不会同步更新 _free_ids，
        所以 _free_ids 是"某一时刻的空闲快照"，可能已被后续分配消耗掉——
        这正是需要 offset 游标、且分配前必须校验长度的原因。
    """

    def __init__(
        self,
        num_total_blocks: int,
    ):
        """构造一个容量固定的 block 池。

        Args:
            num_total_blocks: 该层总 block 数，由配置按显存/内存/磁盘容量折算；
                创建后不可变（改变容量需要重建 Mempool，见 reset 的语义）
        """
        assert num_total_blocks > 0
        self.num_total_blocks = num_total_blocks

        self._free_mask = np.ones(self.num_total_blocks, dtype=np.bool_)
        self._num_free = num_total_blocks
        self._free_ids = self._free_mask.nonzero()[0]
        self._free_ids_offset = 0

    def reset(self) -> None:
        """把整池重置为全空闲（cache reset 时调用）。

        注意它只重置账本，不会去动 radix tree；调用方（如 cache_engine.reset()）
        必须同时清掉树，否则树里残留的引用会指向被重新分配出去的 block，
        造成数据错乱。
        """
        self._free_mask.fill(True)
        self._num_free = self.num_total_blocks
        self._free_ids = self._free_mask.nonzero()[0]
        self._free_ids_offset = 0

    def allocate_blocks(self, num: int) -> np.ndarray:
        """借出 num 个空闲 block id。

        Args:
            num: 需要的块数。允许为 0（返回空数组），负数直接报错
        Returns:
            int64 数组，长度等于 num，元素为本次借出的 block id
        Raises:
            ValueError: 空闲不足。调用方应先走淘汰流程（见 CacheEngine.take）
                        再重试，而不是直接捕获后吞掉
        """
        if num < 0:
            raise ValueError(f"num must be greater than 0, but got {num}")
        if num > self._num_free:
            raise ValueError(f"Not enough free blocks, required: {num}, available: {self._num_free}")

        # 缓存的空闲快照不够发：此时池里可能还有空闲（回收进来的没进 _free_ids），
        # 所以必须重扫 _free_mask 再判断，不能在这里直接报"不足"
        if num > len(self._free_ids) - self._free_ids_offset:
            self._update_free_ids()

        # 注意返回的是 _free_ids 的**视图切片**；调用方若就地修改它，
        # 会污染池内缓存数组，需要长期持有时应自行 copy()
        free_ids = self._free_ids[self._free_ids_offset:self._free_ids_offset+num]
        self._free_ids_offset += num

        self._free_mask[free_ids] = False
        self._num_free -= num
        return free_ids

    def recycle_blocks(self, block_ids: np.ndarray) -> None:
        """归还 block id，使其可被再次分配。

        这是淘汰流程的"落地"动作：radix tree 决定淘汰哪些块后，
        CacheEngine 把它们的 id 交给本方法回收（cache_engine.py:522）。

        Args:
            block_ids: 1D int64 数组；允许乱序、允许为空，但不允许重复、
                       不允许越界、不允许重复归还（已空闲的块）
        Raises:
            ValueError: 形状/类型/范围不合法，或其中已有空闲块
                        （后一种通常意味着上层出现了双重释放或与树失同步）
        """
        if block_ids.ndim != 1 or block_ids.dtype != np.int64:
            raise ValueError("block_ids must be a 1D tensor of int64")
        if len(block_ids) == 0:
            return
        if np.any(block_ids < 0) or np.any(block_ids >= self.num_total_blocks):
            raise ValueError("block_ids must be within the range of [0, num_total_blocks)")
        # Remove duplicates first (same block ID appearing multiple times)
        block_ids = np.unique(block_ids)

        already_free = self._free_mask[block_ids]
        if already_free.any():
            raise ValueError(f"block_ids {block_ids[already_free]} are already free")
        self._free_mask[block_ids] = True
        # 只改 mask 与计数，不修 _free_ids：否则每次回收都要 O(N) 重建索引。
        # 代价是 _free_ids 逐渐过期，由 allocate_blocks 的游标逻辑兜住
        self._num_free += len(block_ids)

    def _update_free_ids(self) -> None:
        """按当前 _free_mask 重建空闲 id 快照并把游标归零（O(N) 全扫，慎用）。"""
        self._free_ids = self._free_mask.nonzero()[0]
        self._free_ids_offset = 0

    @property
    def num_free_blocks(self) -> int:
        """空闲块数。淘汰触发判定的直接输入（utilization 与需求比对都用它）。"""
        return self._num_free

    @property
    def num_used_blocks(self) -> int:
        """已占用块数（总数 - 空闲数）。用于监控与利用率统计。"""
        return self.num_total_blocks - self._num_free
