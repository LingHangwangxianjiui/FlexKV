# ==============================================================================
# flexkv/transfer/worker_op.py —— Worker 侧的操作封装（跨进程传输的"线格式"）
# ------------------------------------------------------------------------------
# 本文件职责：把控制面的 TransferOp / LayerwiseTransferOp 翻译成要发给 Worker
# 子进程的消息体。它只包含"搬字节所需的最小信息"，是主进程 -> Worker 进程之间
# IPC（mp.Connection，pickle 序列化）的载荷定义。
#
# 在系统链路中的位置：
#   KVTaskEngine -> GlobalCacheEngine(控制面) -> TransferEngine -> WorkerHandle
#   -> 【本文件 Worker*Op，经管道序列化】-> Worker(worker.py) -> c_ext
#
# 核心内容速查：
#   - WorkerTransferOp         : 普通传输 op 的 worker 视图（对应 TransferOp）
#   - WorkerLayerwiseTransferOp: 逐层传输 op 的 worker 视图（对应 LayerwiseTransferOp）
#   - WorkerTransferResult     : 回程消息，worker -> 引擎的完成回报
#
# 与 common/transfer.py 中 TransferOp 的关系（本文件最需要理解的一点）：
#   * TransferOp 是**控制面内部**的 DAG 节点：带 predecessors/successors/status/
#     pending_count 等调度元数据，由 TransferScheduler 消费，只活在主进程里。
#   * Worker*Op 是同一件事的**跨进程视图**：Worker 进程不需要知道自己在 DAG 里的
#     位置（依赖由主进程调度好了才派发下来），所以这些字段全部剥掉，
#     只留下 transfer_type / 两端 slot 或 block id / block 数 这些执行必需项。
#   * 生成入口是 WorkerHandle.submit_transfer（worker.py:828）：按 op 类型二选一
#     构造，然后 conn.send() 发给 worker。
#   * 因此修改本文件的字段时，必须同时确认 worker.py 里对应的读取处；
#     少传一个字段 = worker 侧 AttributeError，多传无用字段 = 白白增加 pickle 开销。
#
# 字段裁剪的一个具体优化（见 WorkerTransferOp.__init__）：
#   当两端 slot_id 都有效时，block id 不随消息发送——worker 可以凭 slot_id 从
#   共享的 op_buffer_tensor（worker.py:594/601）自己取，省掉大数组的序列化。
# ==============================================================================
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from flexkv.common.transfer import TransferOp, TransferType, LayerwiseTransferOp


@dataclass(frozen=True)
class WorkerTransferResult:
    """Worker-to-scheduler completion with optional per-block outcomes.

    中文补充：worker 回传给 TransferEngine 的完成消息（回程方向）。
    frozen=True —— 一旦构造不可修改，因为它是跨进程流转的结果凭证。

    为什么需要 block_results：支持"部分成功"的传输（例如远端只读回一部分
    block）。它为 None 表示整体成功/失败、无逐块明细；非 None 时逐块给出
    成败，TransferEngine 据此做部分重试或把失败块标记为缺失
    （见 transfer_engine.py:1180 附近对 payload 的类型分派）。
    """

    transfer_op_id: int
    block_results: Optional[Tuple[bool, ...]] = None


@dataclass
class WorkerTransferOp:
    """普通传输 op 的 worker 视图：一次"从 src 到 dst 搬 N 个 block"的执行指令。

    对应 common/transfer.py 的 TransferOp，但剥掉了 DAG 调度元数据
    （predecessors / successors / status / pending_count / is_swa 路由标记等），
    只留下 worker 真正搬字节需要的东西。

    字段里两组"寻址方式"二选一，这是理解本类的关键：
      * slot 寻址：src_slot_id / dst_slot_id 有效（非 -1）时，worker 凭 slot_id
        从共享的 op_buffer_tensor 中取 block id（worker.py:585-601），
        本消息里的 src/dst_block_ids 会是空数组 —— 省掉大数组的序列化开销。
      * 数组寻址：slot_id 为 -1（典型是晚绑定的 GPU block）或 mooncake 远端
        场景时，block id 必须随消息原样带过去。

    注意：本类虽被 @dataclass 装饰，但类体里手写了 __init__。dataclass 对
    cls.__dict__ 中已存在的属性不会覆盖，所以生效的是手写的这个 __init__，
    字段声明仅用于生成 __repr__/__eq__ 与提供类型信息。
    """
    transfer_op_id: int
    transfer_graph_id: int
    transfer_type: TransferType
    src_slot_id: int
    dst_slot_id: int
    valid_block_num: int
    src_block_ids: np.ndarray
    dst_block_ids: np.ndarray
    src_block_node_ids: Optional[np.ndarray]
    mooncake_store_block_hashes: Optional[np.ndarray] = None
    mooncake_store_swa_block_hashes: Optional[list] = None
    prof_submitted_ns: int = 0

    def __init__(self, transfer_op: TransferOp):
        """从控制面的 TransferOp 构造 worker 侧指令。

        Args:
            transfer_op: 已由调度器放行（RUNNING）的 op；本构造只读它的字段，
                         不做任何校验，也不改变 op 的状态
        """
        self.transfer_op_id = transfer_op.op_id
        self.transfer_graph_id = transfer_op.graph_id
        self.transfer_type = transfer_op.transfer_type
        self.src_slot_id = transfer_op.src_slot_id
        self.dst_slot_id = transfer_op.dst_slot_id
        self.valid_block_num = transfer_op.valid_block_num
        # Always preserve optional src_block_node_ids from TransferOp
        self.src_block_node_ids = transfer_op.src_block_node_ids
        self.mooncake_store_block_hashes = transfer_op.mooncake_store_block_hashes
        self.mooncake_store_swa_block_hashes = transfer_op.mooncake_store_swa_block_hashes

        # slot_id 为 -1 表示"这一端没有共享槽位"（典型是 GPU 侧的晚绑定 block），
        # worker 无法自行推导，只能把数组随消息发过去
        if self.src_slot_id == -1 or self.dst_slot_id == -1:
            self.src_block_ids = transfer_op.src_block_ids
            self.dst_block_ids = transfer_op.dst_block_ids
        elif (transfer_op.mooncake_store_block_hashes is not None
              or transfer_op.mooncake_store_swa_block_hashes is not None):
            # Mooncake ops need block ids even when slot ids are set.
            self.src_block_ids = transfer_op.src_block_ids
            self.dst_block_ids = transfer_op.dst_block_ids
        else:
            self.src_block_ids = np.empty(0)
            self.dst_block_ids = np.empty(0)


@dataclass
class WorkerLayerwiseTransferOp:
    """逐层（layerwise）传输 op 的 worker 视图，对应 LayerwiseTransferOp。

    与 WorkerTransferOp 不同，它**必须**携带全部 block id 数组：一个 op 里同时
    打包了 disk->cpu 与 cpu->gpu 两条通路（外加 SWA 的两条），共八组数组，
    因为逐层执行要求它们在同一次 launch 内协同推进，不能拆成多个 op。

    执行语义（详见 transfer/layerwise.py 的 LayerwiseWorker）：
      每搬完一层 KV 就通过 eventfd 通知一次，counter_id 指定用哪一组计数器
      （三缓冲复用），推理引擎据此判断"第 N 层已就位，可以开始算"，
      从而把 H2D 传输与 prefill 计算重叠起来。
    """
    transfer_op_id: int
    transfer_graph_id: int
    transfer_type: TransferType
    src_block_ids_h2d: np.ndarray
    dst_block_ids_h2d: np.ndarray
    src_block_ids_disk2h: np.ndarray
    dst_block_ids_disk2h: np.ndarray
    # Always non-None: LayerwiseTransferOp normalizes missing SWA ids to empty
    # np.int64 arrays. Empty arrays signal cpp that this transfer carries no SWA.
    swa_src_block_ids_h2d: np.ndarray
    swa_dst_block_ids_h2d: np.ndarray
    swa_src_block_ids_disk2h: np.ndarray
    swa_dst_block_ids_disk2h: np.ndarray
    counter_id: int  # Counter set index for triple buffering eventfd notification
    prof_submitted_ns: int = 0

    def __init__(self, transfer_op: LayerwiseTransferOp):
        """从 LayerwiseTransferOp 构造 worker 侧指令（纯字段拷贝）。

        Args:
            transfer_op: LAYERWISE 类型的 op；这里显式断言类型，
                         防止调用方误把普通 TransferOp 传进来
        """
        self.transfer_op_id = transfer_op.op_id
        self.transfer_graph_id = transfer_op.graph_id
        # 防御性断言：worker 侧按 LAYERWISE 协议解析，类型不对会导致静默错乱
        assert transfer_op.transfer_type == TransferType.LAYERWISE
        self.transfer_type = transfer_op.transfer_type
        self.src_block_ids_h2d = transfer_op.src_block_ids_h2d
        self.dst_block_ids_h2d = transfer_op.dst_block_ids_h2d
        self.src_block_ids_disk2h = transfer_op.src_block_ids_disk2h
        self.dst_block_ids_disk2h = transfer_op.dst_block_ids_disk2h
        self.swa_src_block_ids_h2d = transfer_op.swa_src_block_ids_h2d
        self.swa_dst_block_ids_h2d = transfer_op.swa_dst_block_ids_h2d
        self.swa_src_block_ids_disk2h = transfer_op.swa_src_block_ids_disk2h
        self.swa_dst_block_ids_disk2h = transfer_op.swa_dst_block_ids_disk2h
        self.counter_id = transfer_op.counter_id
