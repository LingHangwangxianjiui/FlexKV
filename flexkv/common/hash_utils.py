# ==============================================================================
# flexkv/common/hash_utils.py —— 前缀哈希（radix tree 做前缀匹配的基础）
# ------------------------------------------------------------------------------
# 职责：把 token 序列 / block id 序列映射成 64 位整数哈希。真正的哈希实现在
# C++ 扩展里（csrc/hash.cpp，基于 xxhash 的 XXH64），本文件只是一层很薄的
# Python 封装。
#
# 在主链路中的位置：
#   * 上游调用方：
#       - flexkv/common/block.py：SequenceMeta.gen_hashes 用 gen_hashes 算出
#         每个 block 的前缀哈希，供 radix tree 匹配；
#         hash_token 用 Hasher 直接算整段 token 的哈希，作为 prefetch 的 key。
#       - flexkv/cache/radixtree.py、flexkv/cache/cache_engine.py：消费
#         SequenceMeta.block_hashes 做前缀匹配与插入。
#       - flexkv/common/ring_buffer.py：SharedOpPool 用 hash_array /
#         hash_array_with_prefix 给"一组 block id"生成去重 key，从而复用共享
#         内存槽位。
#   * 依赖的下游模块：
#       - flexkv/c_ext（C++ 扩展，提供 Hasher / gen_hashes / get_hash_size）
#       - torch（C++ 侧只接收 torch.Tensor，所以这里要把 numpy 转成 tensor）
#
# 关键名字清单：
#   HashType            —— 哈希值的类型别名，本质就是 Python int（底层是 64 位）。
#   get_hash_size()     —— 查询单个哈希值占多少字节（由 C++ 侧决定，通常 8）。
#   Hasher              —— 有状态的流式哈希器：reset -> update(多次) -> digest。
#   hash_array()        —— 无状态快捷函数，一次算完整个数组的哈希。
#   hash_array_with_prefix() —— 先把 prefix（如设备类型）混进哈希再算数组，
#                               用于隔离不同"命名空间"下同形的数据。
#   gen_hashes()        —— 一次遍历 token 序列，产出逐 block 的前缀哈希数组。
#
# 阅读提示 / 容易踩的坑：
#   1. 【为什么必须是前缀哈希】radix tree 是按 block 逐层往下走的：第 k 个
#      block 的哈希要同时代表"它自己 + 它前面所有 block"。两个请求只要前缀
#      相同，前 k 个 block 哈希就必须完全一致，否则命中不了。csrc/hash.cpp 的
#      实现正是一个 XXH64 流式状态从头滚到尾，每吃满 tokens_per_block 个 token
#      就 digest 一次，因此复杂度是 O(N) 而不是 O(N * tokens_per_block)。
#   2. 【为什么按 block 粒度】KV Cache 的最小管理单元是 block（见
#      common/block.py），一个 block 装 tokens_per_block 个 token。哈希必须与
#      缓存粒度对齐，才能做到"以 block 为单位的复用与部分命中"，也才能让树
#      节点按 block 切分（CRadixNode::split）。
#   3. 【Hasher 是有状态的】reset 之后可以 update 多次再 digest。SequenceMeta
#      正是靠"先 update namespace、再 update token_ids"把命名空间混进后续所有
#      block 哈希里，实现不同 namespace 之间的缓存隔离。
#   4. 【hash_array / hash_array_with_prefix 不是线程安全的】它们共用模块级
#      单例 _HASHER，且没有加锁。目前唯一的调用方 SharedOpPool 自己持锁保护，
#      所以没出问题；新增并发调用方时要格外小心。
#   5. 【有符号 / 无符号】gen_hashes 返回的数组是 uint64，而下游 radix tree 侧
#      多以 int64 语义存取比较；打印调试请用 block.py 的 format_block_hash，
#      它统一按无符号 64 位十六进制输出。
#   6. 64 位哈希理论上存在碰撞，项目接受这个风险，没有做二次校验。
# ==============================================================================

import time
from typing import NewType, Optional

import numpy as np
import torch

from flexkv import c_ext


HashType = NewType('HashType', int)

def get_hash_size() -> int:
    """单个哈希值占用的字节数（由 C++ 侧决定，一般固定为 8）。

    SequenceMeta.gen_hashes 用它校验 block_hashes 数组的 itemsize。
    """
    return int(c_ext.get_hash_size())

class Hasher:
    """有状态的流式哈希器（对 c_ext.Hasher / XXH64 的薄封装）。

    用法固定为三步：reset() -> update(...) 可多次 -> digest()。
    多次 update 的内容会按顺序累积进同一个哈希状态，这正是"前缀哈希"和
    "把 namespace 混进哈希"两种用法的实现基础。

    注意：实例本身不线程安全，一个 Hasher 同时只能服务一条哈希流水线。
    """
    def __init__(self) -> None:
        self.hasher = c_ext.Hasher()

    def reset(self) -> None:
        self.hasher.reset()

    def update(self, array: np.ndarray) -> None:
        self.hasher.update(torch.from_numpy(array))

    def digest(self) -> HashType:
        return HashType(self.hasher.digest())

_HASHER = Hasher()
# 模块级共享实例：hash_array / hash_array_with_prefix 复用它以避免反复构造
# C++ 对象。代价是这两个函数不可并发调用（见文件头注释第 4 条）。

def hash_array(array: np.ndarray) -> HashType:
    """一次算完整个数组的哈希（对调用方而言是无状态的）。

    注意它内部复用了全局 _HASHER，所以不是线程安全的。
    """
    _HASHER.reset()
    _HASHER.update(array)
    return HashType(_HASHER.digest())

def hash_array_with_prefix(array: np.ndarray, prefix: int) -> HashType:
    """Hash array with a prefix value (e.g., device type) to avoid collisions.

    中文要点：与 hash_array 的区别是「先把 prefix 作为首个元素喂进哈希状态」，
    相当于给数组加了一维命名空间。典型用途是区分不同设备上编号相同的 block
    集合——例如 CPU block [0,1] 与 SSD block [0,1] 内容不同却同形，不加 prefix
    就会算出同一个 key 而错误复用共享内存槽位。
    """
    _HASHER.reset()
    # Add prefix as a single-element array to the hash
    _HASHER.update(np.array([prefix], dtype=np.int64))
    _HASHER.update(array)
    return HashType(_HASHER.digest())

def gen_hashes(token_ids: np.ndarray, tokens_per_block: int, hasher: Optional[Hasher] = None) -> np.ndarray:
    """按 block 粒度生成前缀哈希数组（radix tree 匹配的输入）。

    语义：返回的第 i 个值是 token_ids[0 : (i+1) * tokens_per_block] 的哈希，
    即"包含自身在内的整段前缀"的哈希，而不是第 i 个 block 单独内容的哈希。
    C++ 侧（csrc/hash.cpp）用一个 XXH64 流式状态顺序扫描，每吃满
    tokens_per_block 个 token 就 digest 一次，因此是单次遍历 O(N)。

    参数：
      token_ids        —— 一维 int64 token 序列（末尾不足一个 block 的部分被
                          忽略，因为返回数组长度是 size // tokens_per_block）。
      tokens_per_block —— 每个 block 装多少 token（来自 CacheConfig）。
      hasher           —— 外部传入的哈希器；传 None 则新建一个。想让 namespace
                          参与哈希时，调用方会先 update 过 namespace 再传进来。

    返回：dtype=uint64、长度 = num_blocks 的一维数组。
    """
    block_hashes = np.zeros(token_ids.size // tokens_per_block, dtype=np.uint64)
    if hasher is None:
        hasher = Hasher()
    c_ext.gen_hashes(hasher.hasher, torch.from_numpy(token_ids), tokens_per_block, torch.from_numpy(block_hashes))
    return block_hashes

if __name__ == "__main__":
    np.random.seed(0)
    token_ids = np.random.randint(0, 10000, (1000, ), dtype=np.int64)
    print(f"token ids length: {token_ids.shape[0]}")
    result = hash_array(token_ids)
    start = time.time()
    for i in range(1):
        result = hash_array(token_ids)
    end = time.time()
    print(f"array hash: {result}, average time: {(end - start)*1000/5}ms")
    # start = time.time()
    # result2 = gen_hashes(token_ids, 16)
    # end = time.time()
    # print(f"block hashes: {result2}, time: {(end - start)*1000}ms")
