# ==============================================================================
# flexkv/transfer/utils.py —— 传输层（分布式远端通路）的工具函数与消息结构
# ------------------------------------------------------------------------------
# 本文件职责：两块内容
#   1. block 列表的"分组 / 切段"工具：把一个 op 里的 block 按远端节点拆开，
#      并把地址连续的 block 合并成段，以便用尽量少、尽量大的 RDMA/mooncake
#      请求完成传输（请求数与握手开销是远端通路的主要成本）。
#   2. 跨节点传输的消息结构：远端 SSD->H 的元信息、节点元信息、RDMA 任务描述。
#      它们都要经 Redis / ZMQ 序列化后在节点间传递，所以都配了 to_dict/from_dict。
#
# 在系统链路中的位置：
#   Worker(worker.py 的远端/分布式分支) -> 【本文件】-> mooncake / zmqHelper / c_ext
#   消费方：transfer/worker.py（远端 worker 的 op 解析）、
#           transfer/zmqHelper.py:106（send_meta_info）、
#           mooncakeEngineWrapper.py（RDMATaskInfo）
#
# 核心内容速查：
#   - group_blocks_by_node            : 按远端 node_id 分组（保序，不切段）
#   - split_contiguous_blocks         : 按 src/dst 同时连续切成子段
#   - group_blocks_by_node_and_segment: 先分组、组内排序后切段（最常用）
#   - RemoteSSD2HMetaInfo : 对端把本地 SSD 数据推回来所需的全部元信息
#   - NodeMetaInfo        : 一个节点的地址与 buffer 基址（从 Redis 同步）
#   - RDMATaskInfo        : 一次 RDMA 批量传输的任务描述
#
# 阅读提示：
#   本文件只做"形状变换"和"数据搬运的元信息承载"，不含任何实际 I/O。
#   block id 在这里只是整数编号，真正的内存/显存地址由 worker 用
#   NodeMetaInfo 里的 *_bufer_base_ptr 加上 block id 偏移算出。
#   （注意拼写 bufer 是源码里的历史拼写，未改。）
# ==============================================================================
from collections import defaultdict
from typing import Tuple, List, Dict, Optional, Any
import torch


def group_blocks_by_node(
    src_block_ids: torch.Tensor,
    dst_block_ids: torch.Tensor,
    remote_block_node_ids: List[int]
) -> Dict[int, Dict[str, List[int]]]:
    """按远端节点把 block 对分组：发给同一个节点的归到一起。

    分布式部署下，一个 op 的 block 可能散落在多个远端节点上（见 TransferOp 的
    src_block_node_ids），而一次 RDMA 请求只能对单节点发起，所以先按 node 拆开。

    Args:
        src_block_ids: 源端 block id（远端节点上的编号）
        dst_block_ids: 目的端 block id（本节点上的编号），与 src 逐元素对应
        remote_block_node_ids: 每个 block 对所属的远端 node_id
    Returns:
        {node_id: {"src": [...], "dst": [...]}}，组内保持原始顺序
    """
    groups = defaultdict(lambda: {"src": [], "dst": []})
    for src, dst, node_id in zip(src_block_ids.tolist(), dst_block_ids.tolist(), remote_block_node_ids):
        groups[node_id]["src"].append(src)
        groups[node_id]["dst"].append(dst)
    return dict(groups)

def split_contiguous_blocks(
    src_list: List[int], dst_list: List[int]
)-> List[Dict[str, List[int]]]:
    """把一对等长的 block id 列表切成若干"双端都连续"的子段。

    合并的意义：连续 block 在物理上地址相邻，可以合成一次大块传输
    （见 worker.py:3733 把结果展开成 src_ptr/dst_ptr/data_len 三组列表），
    从而把 N 次小请求压成 K 次大请求（K << N），显著降低远端通路的请求开销。

    切段条件必须是 **src 与 dst 同时连续**：只有两端都连续，才能用
    (起始地址, 长度) 一次描述整段；单端连续而另一端跳变时也必须断开。

    Args:
        src_list: 源 block id 列表
        dst_list: 目的 block id 列表，与 src_list 等长、逐元素对应
    Returns:
        [{"src": [...], "dst": [...]}, ...]；空输入返回空列表
    """
    if not src_list:
        return []

    result = []
    current_src = [src_list[0]]
    current_dst = [dst_list[0]]

    for i in range(1, len(src_list)):
        src_cont = src_list[i] == src_list[i - 1] + 1
        dst_cont = dst_list[i] == dst_list[i - 1] + 1

        if src_cont and dst_cont:
            current_src.append(src_list[i])
            current_dst.append(dst_list[i])
        else:
            result.append({"src": current_src, "dst": current_dst})
            current_src = [src_list[i]]
            current_dst = [dst_list[i]]

    result.append({"src": current_src, "dst": current_dst})
    return result
def group_blocks_by_node_and_segment(
    src_block_ids: torch.Tensor,
    dst_block_ids: torch.Tensor,
    remote_block_node_ids: List[int],
) -> Dict[int, List[Dict[str, List[int]]]]:
    '''
    Group by node_id and divide blocks with consecutive source/dst into subsegments.
    Parameters:
        src_block_ids (torch.Tensor): source block ids
        dst_block_ids (torch.Tensor): target block ids
        remote_block_node_ids (List[int]): the remote node ids for each block
    Returns:
        Dict[node_id, List[Dict[str, List[int]]]]:
            {
                node_id: [
                    {"src": [...], "dst": [...]},
                    ...
                ]
            }

    中文补充：这是远端通路最常用的一个函数——先按 node 分组（跨节点必须拆开），
    再在组内按 (src, dst) 排序后切连续段（把零碎 block 合并成大请求）。
    与 group_blocks_by_node 的差异就在"排序 + 切段"这两步：
    它不保序（组内被排序过），但能显著减少请求数；需要严格保序的场景
    （如 PEERSSD2H，见 worker.py:3591）改用 group_blocks_by_node。
    '''
    groups = defaultdict(list)
    tmp = defaultdict(lambda: {"src": [], "dst": []})
    for src, dst, node_id in zip(src_block_ids.tolist(), dst_block_ids.tolist(), remote_block_node_ids):
        tmp[node_id]["src"].append(src)
        tmp[node_id]["dst"].append(dst)

    for node_id, pair in tmp.items():
        # 排序是切段的前提：只有按 src 升序排好后，地址相邻的 block 才会挨在一起，
        # 否则零散的顺序会让每块都自成一"段"，合并优化完全失效
        sorted_pairs = sorted(zip(pair["src"], pair["dst"]), key=lambda x: (x[0], x[1]))

        current_src_segment = []
        current_dst_segment = []

        last_src = None
        last_dst = None

        for src, dst in sorted_pairs:
            if last_src is not None and last_dst is not None:
                # 双端同时 +1 才续段；任一端跳变就必须断开，否则无法用
                # (基址, 长度) 一次描述整段
                if src == last_src + 1 and dst == last_dst + 1:
                    # src and dst are continuous
                    current_src_segment.append(src)
                    current_dst_segment.append(dst)
                else:
                    # Disconnect and save the current segment
                    groups[node_id].append({"src": current_src_segment, "dst": current_dst_segment})
                    current_src_segment = [src]
                    current_dst_segment = [dst]
            else:
                current_src_segment.append(src)
                current_dst_segment.append(dst)

            last_src = src
            last_dst = dst

        # Save the last segment
        if current_src_segment:
            groups[node_id].append({"src": current_src_segment, "dst": current_dst_segment})

    return dict(groups)

class RemoteSSD2HMetaInfo:
    """远端 SSD -> 本节点 CPU 的传输元信息（跨节点拉取 KV 的请求/回执载体）。

    使用场景：本节点要读的数据在**对端节点的本地 SSD** 上，本地无法直接访问。
    于是把这份元信息经 ZMQ 发给对端（zmqHelper.py:106 的 send_meta_info），
    由对端把 SSD 数据读出来、再 RDMA 写回本节点的 CPU buffer。

    方向容易搞混，这里明确一下（"peer" 指发起请求的一方，即本节点）：
      * ssd_block_ids —— **对端** SSD 上的 block id（数据实际所在处）
      * cpu_block_ids —— **本节点** CPU buffer 的 block id（数据要落到的位置）
      * peer_engine_addr / peer_cpu_base_ptr —— 本节点的 mooncake 地址与 CPU
        buffer 基址，对端据此计算 RDMA 写回的目的地址
      * peer_zmq_status_addr —— 本节点的 ZMQ 状态地址，对端搬完后回执用
    """

    task_id: int
    cpu_block_ids: List[int]
    ssd_block_ids: List[int]
    peer_engine_addr: str  # mooncake engine addr of the node that initiates the transfer,
                           # used for write data back to the node.
    peer_cpu_base_ptr: int # the cpu buffer base ptr of the peer node, used for calculating the dst ptrs.
    peer_zmq_status_addr: str
    data_size: int

    def __init__(self, task_id, cpu_block_ids, ssd_block_ids, peer_engine_addr, peer_cpu_base_ptr, peer_zmq_status_addr, data_size):
        self.task_id = task_id
        self.cpu_block_ids = cpu_block_ids
        self.ssd_block_ids = ssd_block_ids
        self.peer_engine_addr = peer_engine_addr
        self.peer_cpu_base_ptr = peer_cpu_base_ptr
        self.peer_zmq_status_addr = peer_zmq_status_addr
        self.data_size = data_size

    @classmethod
    def from_dict(self, data: dict) -> "RemoteSSD2HMetaInfo":
        """从 ZMQ 收到的 dict 还原。注意这里第一个参数写作 self 而非 cls，
        但它是 classmethod，所以 self 实际绑定的是类本身，写法异于惯例但可用。

        所有字段用 data.get() 取值，缺失时静默变成 None/空——对端版本不一致
        时不会报错，而是把 None 传到 RDMA 层，排查时要留意。
        """
        return RemoteSSD2HMetaInfo(
            task_id = data.get("task_id"),
            cpu_block_ids=data.get("cpu_block_ids"),
            ssd_block_ids=data.get("ssd_block_ids"),
            peer_engine_addr=data.get("peer_engine_addr"),
            peer_cpu_base_ptr = data.get("peer_cpu_base_ptr"),
            peer_zmq_status_addr = data.get("peer_zmq_status_addr"),
            data_size=data.get("data_size"),
        )
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "cpu_block_ids": self.cpu_block_ids,
            "ssd_block_ids": self.ssd_block_ids,
            "peer_engine_addr": self.peer_engine_addr,
            "peer_cpu_base_ptr": self.peer_cpu_base_ptr,
            "peer_zmq_status_addr": self.peer_zmq_status_addr,
            "data_size": self.data_size,
        }

class NodeMetaInfo:
    """Node information for flexkv sub-nodes

    中文补充：一个参与分布式缓存的节点自身发布的"名片"，经 Redis 同步给其他节点。
    其他节点拿到 engine_addr 建立 mooncake 连接、拿到 zmq_addr 收发控制消息，
    并靠两个 *_bufer_base_ptr 把 block id 换算成真实地址（远端 RDMA 需要物理
    地址，而 block id 只是本节点内部的逻辑编号）。

    worker 侧会把它按 node_id 缓存起来（worker.py:3192 的 node_metas），
    避免每次传输都回查 Redis。
    """

    def __init__(
        self,
        node_id: int,
        engine_addr: Optional[str] = None,
        zmq_addr: Optional[str] = None,
        cpu_bufer_base_ptr: Optional[int] = None,
        ssd_bufer_base_ptr: Optional[int] = None,
    ):
        self.node_id = node_id
        self.engine_addr = engine_addr
        self.zmq_addr = zmq_addr
        self.cpu_bufer_base_ptr = cpu_bufer_base_ptr
        self.ssd_bufer_base_ptr = ssd_bufer_base_ptr

    def to_dict(self) -> Dict[str, Any]:
        """序列化到 Redis。注意键名与字段名不一致：engine_addr -> "addr"、
        cpu_bufer_base_ptr -> "cpu_buffer_ptr"（后者还修正了拼写），
        from_dict 里按同样的键名反解，两边必须成对修改。
        """
        result = {
            "node_id": self.node_id,
            "addr": self.engine_addr,
            "zmq_addr": self.zmq_addr,
            "cpu_buffer_ptr": self.cpu_bufer_base_ptr,
            "ssd_buffer_ptr": self.ssd_bufer_base_ptr,
        }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeMetaInfo":
        return cls(
            node_id=data.get("node_id"),
            engine_addr=data.get("addr"),
            zmq_addr=data.get("zmq_addr"),
            cpu_bufer_base_ptr=data.get("cpu_buffer_ptr"),
            ssd_bufer_base_ptr=data.get("ssd_buffer_ptr"),
        )

class RDMATaskInfo:
    """一次 RDMA 批量传输的完整任务描述（交给 mooncake 引擎执行）。

    与 RemoteSSD2HMetaInfo 的区别：那份是"请对端帮我搬"的请求元信息，
    这份是"已经算好地址、直接搬"的任务——它携带的是**成对的真实地址**，
    而不只是 block id。

    三组等长列表是核心（下标一一对应，第 i 段用这三个列表的第 i 项）：
      * src_ptrs / dst_ptrs —— 每段的源/目的起始地址（由 block id 与 buffer
        基址算出，跨进程/跨节点后 block id 本身已无意义）
      * data_lens           —— 每段字节数，来自 split_contiguous_blocks 的切段
        结果：连续段越长，列表越短，RDMA 请求数越少
      * src/dst_block_ids   —— 保留 block id 用于回执与统计
    data_size 是总字节数，便于一次性校验与打点。
    """

    def __init__(
        self, task_id: int, local_engine_addr: str, peer_engine_addr: str,
        peer_zmq_addr: str, src_ptrs: List[int], dst_ptrs: List[int],
        src_block_ids: List[int], dst_block_ids: List[int], data_lens: List[int], data_size: int
    ):
        self.task_id = task_id
        self.local_engine_addr = local_engine_addr ## the mooncake engine address of local node
        self.peer_engine_addr = peer_engine_addr ## thre mooncake engine address of remote node
        self.src_ptrs = src_ptrs
        self.dst_ptrs = dst_ptrs
        self.peer_zmq_addr = peer_zmq_addr
        self.src_block_ids = src_block_ids
        self.dst_block_ids = dst_block_ids
        self.data_size = data_size
        self.data_lens = data_lens
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "local_engine_addr": self.local_engine_addr,
            "peer_engine_addr": self.peer_engine_addr,
            "peer_zmq_addr": self.peer_zmq_addr,
            "src_ptrs": self.src_ptrs,
            "dst_ptrs": self.dst_ptrs,
            "src_block_ids": self.src_block_ids,
            "dst_block_ids": self.dst_block_ids,
            "data_lens": self.data_lens,
            "data_size": self.data_size,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RDMATaskInfo":
        return cls(
            task_id=data.get("task_id", 0),
            local_engine_addr=data.get("local_engine_addr", ""),
            peer_engine_addr=data.get("peer_engine_addr", ""),
            peer_zmq_addr=data.get("peer_zmq_addr", ""),
            src_ptrs=data.get("src_ptrs", []),
            dst_ptrs=data.get("dst_ptrs", []),
            src_block_ids=data.get("src_block_ids"),
            dst_block_ids=data.get("dst_block_ids"),
            data_lens=data.get("data_lens", []),
            data_size=int(data.get("data_size", 0)),
        )
