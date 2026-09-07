# ==============================================================================
# flexkv/transfer/scheduler.py —— 数据面的 DAG 增量调度器
# ------------------------------------------------------------------------------
# 本文件职责：维护"在飞行中的传输 DAG"，每轮回答两个问题：
#   1. 哪些 op 现在可以执行（前驱已全部完成）？
#   2. 哪些图已经整体完成，可以回报给上层？
# 它只做调度决策，不搬数据、不建图、不感知存储层级。
#
# 在系统链路中的位置：
#   KVManager -> KVTaskEngine -> GlobalCacheEngine(控制面，产出 TransferOpGraph)
#   -> TransferEngine(数据面) -> 【本文件 TransferScheduler】-> Worker(真正搬字节) -> c_ext
#
# 核心内容速查：
#   - TransferScheduler.add_transfer_graph : 新图登记入册，并立刻标记为"脏"
#   - TransferScheduler.fail_graph         : 图作废，停止派发其剩余 op
#   - TransferScheduler.schedule           : 唯一入口，吃"已完成的 op"，吐"新就绪的 op"
#
# 阅读提示：
#   * 图的语义（ready 集合、predecessors、mark_completed）定义在
#     common/transfer.py 的 TransferOpGraph，读本文件前必须先读它。
#   * 调用方是 TransferEngine._scheduler_loop（transfer_engine.py:1115 / 1245），
#     它把 worker 回报的完成事件喂进来，拿 next_ops 派发给 worker。
#   * 本文件最重要的概念是"增量"：见 schedule() 与 _dirty_graph_ids 的注释。
# ==============================================================================
from dataclasses import dataclass
from typing import OrderedDict, List, Set, Tuple

from flexkv.common.transfer import TransferOp, TransferOpGraph, TransferType


class TransferScheduler:
    """纯 DAG 调度器：管理所有在飞行中的 TransferOpGraph，按依赖逐轮放行 op。

    在链路中的职责：TransferEngine 的"大脑"。它不接触任何存储设备，只知道
    (graph_id, op_id) 与依赖关系；真正的字节搬运由 Worker 完成。

    关键设计——增量调度（incremental scheduling）：
        朴素做法是每轮遍历所有在飞行中的图，逐个 take_ready_ops()，
        代价 O(在飞行的图数)。本实现改为只遍历"脏图"（_dirty_graph_ids），
        代价 O(本轮发生变化的图数)。正确性依赖两条前提：
          1. 依赖不会跨图。一张图只有在**它自己的** op 完成后才可能冒出新的
             就绪 op，所以没被标脏的图本轮不可能有新东西可调度。
          2. 任何改变图就绪状态的路径都必须把图标脏（add_transfer_graph 与
             schedule 的完成回传是仅有的两处）。漏标脏 = 该图永久不被访问，
             而旧的全量扫描会把这种遗漏悄悄吞掉——这是本次改造最大的风险点。
    """

    def __init__(self) -> None:
        """初始化两张表：全部在飞行中的图，以及本轮需要重新访问的脏图 id。"""
        # Store all transfer graphs
        self._transfer_graphs: OrderedDict[int, TransferOpGraph] = OrderedDict()
        # Graph ids whose op state changed since the last schedule() call, in
        # the order they became dirty. Dependencies never cross graphs, so a
        # graph can only expose new ready ops after one of its OWN ops
        # completes; visiting just these is equivalent to sweeping every
        # in-flight graph, at O(changed) instead of O(in-flight) per call.
        #
        # Standing requirement: any path that completes an op or otherwise
        # changes a graph's readiness must dirty that graph here. The old full
        # sweep tolerated such a path silently; this one will never revisit the
        # graph. TransferOpGraph.trigger_op() is one such path (no callers).
        # 中文要点：_dirty_graph_ids 是"拿有序字典当队列用"——value 恒为 None，
        # 只是借它的插入顺序实现 FIFO；schedule() 从队首取，必要时重新排到队尾。
        # 上文的 TransferOpGraph.trigger_op() 正是"改了就绪状态却不标脏"的隐患
        # 路径，目前因为没有调用方才没有暴露问题，二次开发时要留意。
        self._dirty_graph_ids: OrderedDict[int, None] = OrderedDict()

    def add_transfer_graph(self, graph: TransferOpGraph) -> None:
        """Add a new transfer graph to the scheduler

        中文补充：新图入册后立刻标脏，保证下一次 schedule() 会取走它的首批
        ready op。调用方是 TransferEngine 在提交图时（控制面产出的 DAG 到达
        数据面的第一站）。注意脏标记用 dict 赋值实现，重复添加同一 graph_id
        是幂等的——不会把它挪到队尾，但下一轮本来就会访问它。
        """
        self._transfer_graphs[graph.graph_id] = graph
        self._dirty_graph_ids[graph.graph_id] = None

    def fail_graph(self, graph_id: int) -> None:
        """Drop a graph whose transfer failed: none of its remaining ops may be
        dispatched. Ops of this graph already running on workers are allowed to
        drain; schedule() already ignores finished ops whose graph is gone.
        Idempotent, and a no-op for graphs that already completed.

        中文补充：作废一整张图。只从 _transfer_graphs 摘除，不清理
        _dirty_graph_ids —— 残留的脏 id 会在 schedule() 里被 get() 判空后跳过，
        这正是"图没了但完成事件还在路上"这种竞态的安全出口。
        """
        self._transfer_graphs.pop(graph_id, None)

    def schedule(self,
                finished_ops: List[TransferOp]
               ) -> Tuple[List[int], List[TransferOp]]:
        """
        Schedule transfer operations

        Args:
            finished_ops: Dictionary of completed transfer operations and their graph IDs

        Returns:
            Tuple[List[int], List[TransferOp]]:
                - List of completed transfer graph IDs
                - List of next executable transfer operations

        中文补充：本函数是"增量调度"的本体，一轮调用做三件事：
          1. 消化 finished_ops：把它们标为 COMPLETED，并把所属图标脏；
          2. 排空脏图队列：对每张脏图调 take_ready_ops() 取新就绪的 op；
          3. 顺带判定整图完成（all_transfer_ops_completed），汇报给上层。
        就绪判定不在本文件：common/transfer.py 的 take_ready_ops() 负责把
        "处于 PENDING 且 predecessors 已清空"的 op 提升为 RUNNING，本文件只是
        消费它的结果。
        本方法不阻塞、不抛给调用方（异常由 TransferEngine 的调度循环捕获后
        继续下一轮），返回值里的 next_ops 由调用方派发给 Worker。
        """
        # Mark completed operations. Dirty the graph before completing the op:
        # mark_completed() clears the op from its successors' predecessor sets
        # in a loop, so a raise partway through leaves the graph half-advanced
        # and needing a revisit. (Its leading assert fires before any mutation,
        # so that case needs no recovery either way -- this ordering is free
        # insurance for the partial-mutation one, not for the assert.)
        for op in finished_ops:
            # 图已经被 fail_graph() 摘掉（失败或已完成）时直接忽略其迟到事件
            if op.graph_id in self._transfer_graphs:
                self._dirty_graph_ids[op.graph_id] = None
                self._transfer_graphs[op.graph_id].mark_completed(op.op_id)

        # Drain the dirty set. Peek at the head and drop the dirty bit only once
        # the graph has been fully processed, so a raise below leaves the id
        # dirty and the graph gets revisited -- the caller logs and keeps
        # looping (TransferEngine._scheduler_loop), so a dropped dirty bit would
        # strand that graph for good. This recovers the dirty BIT, not the work:
        # ops already collected into next_ops are discarded along with the
        # exception, exactly as they were under the full sweep.
        next_ops = []
        completed_graph_ids = []
        # 增量主循环：只处理脏图，而不是 _transfer_graphs 里的每一张图。
        # 循环体内可能重新标脏（VIRTUAL op 自完成），所以用 while 而非 for。
        while self._dirty_graph_ids:
            graph_id = next(iter(self._dirty_graph_ids))
            # Defensive: every id reaching here should resolve, since the only
            # writers pair the two dicts. Skipping beats a KeyError, which the
            # caller would swallow and then retry forever on the same id.
            graph = self._transfer_graphs.get(graph_id)
            revisit = False
            if graph is not None:
                # take_ready_ops() 返回的 op 已被置为 RUNNING，不会重复取出
                for op_id in graph.take_ready_ops():
                    op = graph._op_map[op_id]
                    # VIRTUAL op 不搬字节，只作同步/汇合点：立刻自行完成即可，
                    # 但它可能解锁后继，所以本图需要再过一轮（revisit）
                    if op.transfer_type == TransferType.VIRTUAL:
                        # Self-completing, and that can unblock successors, so
                        # the graph needs another pass before this call returns.
                        graph.mark_completed(op_id)
                        revisit = True
                    next_ops.append(op)
                # 整图完成：从在飞行集合中摘除并上报；此后不会再有它的脏标记
                if graph.all_transfer_ops_completed():
                    completed_graph_ids.append(graph_id)
                    del self._transfer_graphs[graph_id]
                    # A finished graph needs no further pass. This is the common
                    # case -- a terminal virtual sink both sets revisit and
                    # completes the graph -- so skipping the re-queue keeps it
                    # off the hot path.
                    revisit = False
            del self._dirty_graph_ids[graph_id]
            if revisit:
                self._dirty_graph_ids[graph_id] = None  # re-queue at the tail

        return completed_graph_ids, next_ops
