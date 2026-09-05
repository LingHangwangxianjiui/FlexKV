# ==============================================================================
# flexkv/common/ring_buffer.py —— 跨进程共享的 block id 槽位池（SharedOpPool）
# ------------------------------------------------------------------------------
# 注意：文件名叫 ring_buffer，但本文件**并不含环形缓冲**，内容只有一个
# SharedOpPool。这是命名上的历史遗留，搜 ring buffer 找到这里会扑空，
# 真正的结构见下面"SharedOpPool 是什么"一节。
#
# SharedOpPool 解决什么问题：
#   TransferOp（见 common/transfer.py）携带 src_block_ids / dst_block_ids 两个
#   numpy 数组。主进程要把 op 派发给 Worker 子进程，若每次都序列化这两个数组，
#   会随着 op 数量线性膨胀（一次调度成百上千个 op）。这里的做法是"用共享内存
#   存数组本体，op 里只带一个 int 槽位号"：
#
#       主进程                               共享内存 buffer (max_op_num x max_block_num)
#     ┌──────────────┐   写 block_ids     ┌────────────────────────────────────────┐
#     │ TransferOp   │ ─────────────────→ │ slot 0: [12, 7, 33, ... ]              │
#     │ src_slot_id=0│                    │ slot 1: [ 5, 9, ...  ]                 │
#     │ dst_slot_id=1│ ←─ 只回传 int ──── │ slot 2: (空闲)                          │
#     └──────────────┘                    └────────────────────────────────────────┘
#            │                                        ▲
#            │ op 被 pickle 到 Worker（只有 int）      │ Worker 直接读共享内存
#            └────────────────────────────────────────┘
#
#   同一个 slot_hash（同一组 block id）会被多个 op 复用，靠引用计数决定何时归还，
#   这样"同一批 block 被多个 op 引用"只需存一份。
#
# 在架构中的位置：
#   * 唯一使用者是 flexkv/transfer/transfer_engine.py：register_op_to_buffer()
#     在派发 op 前调 allocate_slot 拿到 src/dst slot_id 写回 op；free_op_from_buffer()
#     在 op 完成后调 free_slot 归还。
#   * 依赖 flexkv/common/hash_utils.py 的 hash_array / hash_array_with_prefix 生成
#     槽位去重 key。
#
# 关键名字清单：
#   buffer / buffer_o  —— 共享内存本体（torch int64 二维张量）+ 它的原始引用。
#   free_slots         —— 空闲槽位下标队列（deque）。
#   slot_map           —— {slot_hash: slot_id}，内容相同的 block id 复用同一槽。
#   slot_ref_count     —— 每槽引用计数，归零才真正归还。
#   slot_hashes        —— {slot_id: slot_hash} 的 list 形式，free 时反查用。
#
# 阅读提示 / 容易踩的坑：
#   1. 【buffer 的写入在锁外】allocate_slot 第 71 行的拷贝刻意在 with self.lock
#      之外（避免持锁做 memcpy）。由此产生一个真实的竞态：槽位首次分配时，
#      线程 A 刚出锁、还没写数据，线程 B 用相同 slot_hash 进来会命中 slot_map
#      走 reuse 路径并**立刻返回 slot_id**，此时 buffer 该行可能尚未写完。
#      当前调用方是单线程调度，所以没触发；若改成并发派发 op，必须先修这里
#      （例如把写入挪进锁内，或给槽位加"就绪"标志）。
#   2. 【哈希计算也在锁外】hash_array / hash_array_with_prefix 共用 hash_utils
#      的模块级单例 _HASHER，**不是线程安全的**。hash_utils 文件头注释第 4 条说
#      "唯一调用方 SharedOpPool 自己持锁保护"，与现状不符——这里的哈希调用发生
#      在进入 with self.lock **之前**。并发调用 allocate_slot 会相互破坏哈希状态。
#   3. 【device_type_prefix 不是可选的优化】不同介质上 block 编号会从 0 重新开始，
#      不加前缀的话 CPU block [0,1] 和 SSD block [0,1] 会算出同一个 hash 而错误
#      复用槽位。transfer_engine 里那张 TransferType -> (src, dst) 前缀表就是为此。
#   4. 【分配失败返回 -1 而不是抛异常】num_blocks == 0、超过 max_block_num、
#      或池子满时都返回 -1。调用方必须检查：free_op_from_buffer 正是靠
#      slot_id != -1 来决定要不要归还，把 -1 传进 free_slot 会引发越界/误判。
#   5. 【double free 检测不可靠】free_slot 用 slot_hashes[slot_id] 反查 slot_map，
#      若该槽位已被回收并重新分配给别的 hash，检测会漏掉真正的 double free。
#      引用计数正常时不会发生，但一旦发生就是静默的内存管理错乱。
#   6. 【槽位行尾是脏数据】只写前 num_blocks 个元素，后面残留旧值；消费方必须
#      按 op 自带的 block 数量读取，不能读满整行。
#
# 建议阅读顺序：
#   __init__（看四个状态容器）-> allocate_slot -> free_slot -> status
# ==============================================================================

import torch
import threading
import time
import random

from collections import OrderedDict,deque
import numpy as np
from flexkv.common.transfer import TransferOp
from flexkv.common.debug import flexkv_logger
from flexkv.common.hash_utils import hash_array, hash_array_with_prefix


class SharedOpPool:
    def __init__(self, max_op_num: int, max_block_num: int, dtype = np.int64):
        """建池：一次性预分配整块共享内存，之后只做槽位借还，运行期不再分配。

        参数：
          max_op_num    —— 槽位总数，即"同时在飞的 op 数组"上限。
          max_block_num —— 单个槽位能装多少个 block id，即单个 op 的 block 数上限。
          dtype         —— 只登记不使用（buffer 固定 int64），为将来扩展保留。

        为什么用 torch 而不是纯 numpy：Worker 子进程通过 fork 继承，torch 的
        share_memory_() 会把张量挪进 System V 共享内存段，子进程无需拷贝即可访问
        同一片内存。buffer_o 保留原始引用，避免共享段被提前回收（share_memory_()
        返回的就是 self，两个名字指向同一对象）。

        容量**固定**：池满时 allocate_slot 返回 -1，不会自动扩容。transfer_engine
        里按 SharedOpPool(2048, num_cpu_blocks) 建池，配置时要留足余量。
        """
        self.max_op_num = max_op_num
        self.max_block_num = max_block_num
        self.dtype = dtype
        # create the buffer tensor
        # 定长二维表：一行 = 一个槽位 = 一个 op 的 block id 数组。不用的尾部是脏数据。
        self.buffer_o = torch.empty((self.max_op_num, self.max_block_num), dtype = torch.int64)
        # move tensor to share memory
        self.buffer = self.buffer_o.share_memory_()

        self.free_slots = deque(range(max_op_num))  # 空闲槽位下标，popleft 取 / append 还
        self.slot_map = dict() # {slot_hash: slot_id}

        # 引用计数：同一组 block id 被多个 op 引用时只存一份，归零才真正归还槽位
        self.slot_ref_count = np.zeros(max_op_num, dtype=np.int32)
        self.slot_hashes = [0]*max_op_num  # slot_id -> slot_hash，free 时反查用

        self.lock = threading.Lock()

    def allocate_slot(self, block_ids: np.ndarray, device_type_prefix: int = 0):
        """
        Allocating a slot for the given block ids
        Params:
            block_ids: the block ids of src address or dst address
            device_type_prefix: optional prefix to distinguish different device types
        Returns:
            slot_id: the slot which is assigned to the given block ids, -1 if failed

        中文要点：为这组 block id 分配（或复用）一个共享内存槽位，把 block id
        写进去，只把 int 型的 slot_id 回传给调用方塞进 TransferOp。

        三条返回 -1 的路径都必须由调用方处理：
          * num_blocks == 0    —— 空 op，没有东西要存；
          * num_blocks > max_block_num —— 超过单槽容量，装不下；
          * free_slots 为空    —— 池满（只打一条 info 日志，不抛异常）。

        复用语义：slot_hash 相同的 block id 组会命中同一个槽位（reuse=True），
        此时**不再重复拷贝**数据（内容本来就一样），只把引用计数 +1。

        坑（详见文件头第 1、2 条）：哈希计算与 buffer 拷贝都在 with self.lock
        **之外**。前者让非线程安全的全局 _HASHER 暴露在并发下，后者可能让走
        reuse 路径的调用方读到尚未写完的槽位。当前是单线程调度所以安全，改成
        并发派发 op 前必须先修这两处。
        """
        # firstly, determine whether the length of block ids exceeds the limit
        num_blocks = block_ids.size
        if num_blocks > self.max_block_num or num_blocks == 0:
            return -1

        # Use prefix to avoid hash collisions between different device types
        if device_type_prefix != 0:
            slot_hash = hash_array_with_prefix(block_ids, device_type_prefix)
        else:
            slot_hash = hash_array(block_ids)
        reuse = False

        # get the slot of empty buffer
        # 锁内只动"簿记"状态（free_slots / slot_map / ref_count），不碰 buffer 本体，
        # 目的是把慢速的 memcpy 挡在临界区外。
        with self.lock:
            if slot_hash in self.slot_map:
                # 命中复用：内容必然相同，跳过拷贝。但注意首次分配的拷贝在锁外，
                # 并发下这里可能取到"槽位已登记、数据还没写完"的 slot_id。
                slot_id = self.slot_map[slot_hash]
                reuse = True
            else:
                if not self.free_slots:
                    flexkv_logger.info("No empty slot in SharedOpPool")
                    return -1

                slot_id = self.free_slots.popleft()
                self.slot_map[slot_hash] = slot_id
            # update status managers
            self.slot_ref_count[slot_id] += 1
            self.slot_hashes[slot_id] = slot_hash
        

        # do copy
        # 只写前 num_blocks 个元素，行尾残留旧值（消费方按 op 自带的 block 数读取）。
        # reuse 时内容必然相同，跳过这次拷贝。
        if not reuse:
            self.buffer[slot_id, :num_blocks] = torch.from_numpy(block_ids).to(torch.int64)

        return slot_id

    def free_slot(self, slot_id: int):
        """
        Free the relevant resources of corresponding op, called when op transfer completed.
        Input:
            op_id: the index of current op
        Output:
            None
        """
        with self.lock:
            slot_hash = self.slot_hashes[slot_id]
            if slot_hash not in self.slot_map:
                raise RuntimeError(f"Slot {slot_id} is not in use, double free detected!")
            self.slot_ref_count[slot_id] -= 1
            assert self.slot_ref_count[slot_id] >= 0, f"Slot {slot_id} ref count is negative"
            if self.slot_ref_count[slot_id] == 0:
                self.free_slots.append(slot_id)
                del self.slot_map[slot_hash]

    def get_buffer(self):
        return self.buffer

    def get_buffer_size(self):
        return self.max_op_num, self.max_block_num

    def status(self):
        """
        Current status logger
        """
        with self.lock:
            used = len(self.slot_map)
            free = self.max_op_num - used
            return {"used_slots": used,
                    "free_slots": free,
                    "capacity": self.max_op_num}


if __name__ == "__main__":
    manager = SharedOpPool(4, 10)
