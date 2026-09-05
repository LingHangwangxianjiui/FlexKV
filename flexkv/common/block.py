# ==============================================================================
# flexkv/common/block.py —— block（KV Cache 最小管理单元）与其哈希表示
# ------------------------------------------------------------------------------
# 职责：把"一段 token 序列"切成 block，并为每个 block 算出前缀哈希。block 是
# FlexKV 里 KV Cache 的**最小管理单元**：分配、复用、淘汰、搬运全都是以 block
# 为粒度，token 只是 block 内部的填充物。
#
# block 与 token 的关系（理解全系统其他模块的前提）：
#
#      token_ids:  [t0 t1 t2 t3 | t4 t5 t6 t7 | t8 t9 t10 t11 | ...]
#                   \___ block0 __/ \___ block1 __/ \___ block2 __/
#                   └──────── tokens_per_block 个 token 装一个 block ────────┘
#
#   * tokens_per_block 来自 CacheConfig（vLLM 里就是它的 block_size，常见 16）。
#   * num_blocks = len(token_ids) // tokens_per_block —— **整除，向下取整**。
#     末尾凑不满一个 block 的残尾 token 不参与缓存，也不产生哈希；
#     cache_engine 侧会先把 token 对齐到 block 边界（aligned_token_ids）。
#   * block_hashes[i] = hash(token_ids[0 : (i+1) * tokens_per_block])，是前缀
#     哈希而非单块哈希。详见 common/hash_utils.py 文件头注释。
#
# 在主链路中的位置：
#   * 上游（构造方）：flexkv/cache/cache_engine.py、flexkv/cache/hie_cache_engine.py
#     在 get / put 前构造 SequenceMeta；flexkv/kvtask.py 用 hash_token 生成
#     prefetch 去重 key。
#   * 下游（消费方）：flexkv/cache/radixtree.py 的 match_prefix / insert 消费
#     SequenceMeta.block_hashes 做前缀匹配，是整个命中率逻辑的输入。
#   * 依赖的下游模块：flexkv/common/hash_utils.py（Hasher / gen_hashes /
#     get_hash_size），底层是 C++ 的 XXH64。
#
# 关键名字清单：
#   _get_namespace_hash_key —— 把 namespace 列表拼成字节数组，作为哈希前缀。
#   hash_token              —— 算整段 token（+ 可选 namespace）的单个哈希。
#   format_block_hash       —— 把哈希值格式化成定长十六进制，便于日志比对。
#   SequenceMeta            —— 本文件主角：token 序列 + 它的 block 划分 + 逐
#                              block 前缀哈希。
#
# 阅读提示 / 容易踩的坑：
#   1. SequenceMeta 在 __init__ 里就**急切地**算好了所有 block 哈希（gen_hashes），
#      后续 get_hash() 只是查表。所以构造它的成本是 O(len(token_ids))，不要在
#      热路径里反复重建同一个 SequenceMeta。
#   2. gen_hashes() 是幂等的（_has_hashes 标志位），重复调用不会重算；radixtree
#      里到处调用它只是为了"确保已算过"。
#   3. namespace 通过"先 update namespace 再 update token"的方式混入哈希，
#      因此不同 namespace 下内容相同的序列会算出**不同**哈希，实现缓存隔离。
#      namespace_id 就是"只 update 了 namespace"时的中间摘要。
#   4. 分隔符用空字符 '\x00' 而不是冒号，是为了避免 ["a:b","c"] 与 ["a","b:c"]
#      拼接后产生歧义（见 _get_namespace_hash_key 内注释）。
#   5. get_hash(block_id) 越界返回 None 而不是抛异常，调用方需自己判空。
# ==============================================================================

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, NewType, Optional

import numpy as np
import torch

from flexkv.common.hash_utils import HashType, Hasher, gen_hashes, get_hash_size


def _get_namespace_hash_key(namespace: Optional[List[str]]) -> Optional[np.ndarray]:
    """把 namespace 列表编码成 int64 数组，用作哈希的前缀（盐）。

    namespace 为空 / None 时返回 None，表示"不加前缀"，此时序列哈希就等于
    纯 token 序列的哈希。
    """
    if not namespace or len(namespace) == 0:
        return None
    
    # Use null character as delimiter to prevent ambiguity
    # e.g. ["a:b", "c"] → "a:b\x00c" != ["a", "b:c"] → "a\x00b:c"
    namespace_key = "\x00".join(namespace)
    namespace_bytes = namespace_key.encode('utf-8')
    namespace_array = np.frombuffer(namespace_bytes, dtype=np.uint8).astype(np.int64)
    return namespace_array


def hash_token(token_ids: np.ndarray, namespace: Optional[List[str]]) -> HashType:
    """算整段 token 序列的单个哈希（可选带 namespace 前缀）。

    与 SequenceMeta 的区别：这里不按 block 切分，只产出一个哈希值。
    kvtask.py 用它给 prefetch 请求生成去重 key，避免同一段 token 被重复预取。
    """
    hasher = Hasher()
    hasher.reset()

    if namespace:
        namespace_key = _get_namespace_hash_key(namespace)
        if namespace_key is not None:
            hasher.update(namespace_key)

    hasher.update(token_ids)

    return HashType(hasher.digest())


def format_block_hash(value: Optional[int]) -> str:
    """Format signed or unsigned 64-bit hashes as fixed-width hex.

    中文要点：哈希在 C++ 侧是 uint64，经过 numpy int64 转换后可能变成负数，
    直接打印会和 C++ 日志对不上。这里统一先掩成无符号再按 016x 输出，
    所以**调试比对哈希时一律走这个函数**。
    """
    if value is None:
        return "-"
    return f"0x{int(value) & ((1 << 64) - 1):016x}"


@dataclass
class SequenceMeta:
    """一段 token 序列的 block 视图：序列本身 + 切块规则 + 逐 block 前缀哈希。

    这是 radix tree（cache/radixtree.py）做前缀匹配的**唯一输入**。构造时就
    会急切算出全部 block 哈希并缓存在 block_hashes 里。

    字段说明：
      token_ids        —— 一维 token 序列。
      tokens_per_block —— 每个 block 装多少 token（来自 CacheConfig）。
      block_hashes     —— 长度 = num_blocks 的哈希数组，block_hashes[i] 是
                          token_ids[0:(i+1)*tokens_per_block] 的哈希。
      _has_hashes      —— 是否已算过哈希（让 gen_hashes() 幂等）。
      namespace_id     —— 仅 namespace 部分的中间摘要；无 namespace 时为 None。
      _namespace       —— 在 __init__ 里额外挂上的私有字段（不在 dataclass
                          声明里），用于后续重算哈希时找回 namespace。

    注意：num_blocks = len(token_ids) // tokens_per_block 是**整除**，
    末尾不足一个 block 的残尾 token 不生成哈希、也不参与缓存。
    """

    token_ids: np.ndarray

    tokens_per_block: int

    block_hashes: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))

    _has_hashes: bool = False

    namespace_id: Optional[int] = None

    def __init__(self, token_ids: np.ndarray, tokens_per_block: int, namespace: Optional[List[str]] = None):
        assert token_ids.ndim == 1
        assert tokens_per_block > 0
        
        self.token_ids = token_ids
        self.tokens_per_block = tokens_per_block
        self.namespace_id = None
        self.block_hashes = np.array([], dtype=np.int64)
        self._has_hashes = False

        self._namespace = namespace

        self.gen_hashes()

    @property
    def num_blocks(self) -> int:
        """完整 block 的个数（向下取整，残尾 token 不计）。"""
        return len(self.token_ids) // self.tokens_per_block

    @property
    def length(self) -> int:
        """token 总数（不一定能被 tokens_per_block 整除）。"""
        return len(self.token_ids)

    def has_hashes(self) -> bool:
        return self._has_hashes

    def _create_initialized_hasher(self) -> Hasher:
        """新建一个"已经喂过 namespace"的哈希器。

        顺带把 namespace 的中间摘要记到 self.namespace_id 上；没有 namespace
        时 namespace_id 保持 None，哈希器就是干净初始态。
        """
        hasher = Hasher()
        hasher.reset()
        if self._namespace and len(self._namespace) > 0:
            namespace_key = _get_namespace_hash_key(self._namespace)
            if namespace_key is not None:
                hasher.update(namespace_key)
                self.namespace_id = int(hasher.digest())
        return hasher

    def get_hash(self, block_id: int) -> Optional[HashType]:
        """取第 block_id 个 block 的前缀哈希；越界返回 None（不抛异常）。

        调用方（radixtree.match_prefix）正是靠返回 None 来判断"序列走到底了"。
        """
        if block_id >= self.num_blocks:
            return None
        assert self._has_hashes, "Hashes should be generated during initialization"
        return HashType(int(self.block_hashes[block_id].item()))

    def gen_hashes(self) -> None:
        if self._has_hashes:
            return
        assert self.token_ids.ndim == 1

        hasher = self._create_initialized_hasher()

        self.block_hashes = gen_hashes(self.token_ids, self.tokens_per_block, hasher)

        assert self.block_hashes.ndim == 1
        assert self.block_hashes.size == self.num_blocks
        assert self.block_hashes.itemsize == get_hash_size()
        self._has_hashes = True
