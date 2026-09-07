# ==============================================================================
# 本文件职责：任务编排层。把一次 get / put / prefetch 请求包装成一个 KVTask，
#   管理它从「创建」到「完成或取消」的完整生命周期，并在存在多个 transfer handle
#   时聚合各路的完成情况（N 路归并）。本层不搬运任何真实数据，只做状态记账、
#   回调分发与资源回收。
#
# 在系统链路中的位置（控制面的最上层，夹在用户 API 与执行层之间）：
#   KVManager(kvmanager.py)
#     -> 【本文件 KVTaskEngine / KVTaskManager】 任务生命周期 + 完成聚合
#       -> GlobalCacheEngine(cache/cache_engine.py) 产出 TransferOpGraph（含延迟上树计划）
#         -> TransferManagerHandle -> TransferEngine(transfer/transfer_engine.py) 数据面调度
#           -> Worker(transfer/worker.py) -> c_ext
#
# 核心内容速查：
#   - TaskStatus / TaskType : 任务状态与类型枚举
#   - KVTask                : 单个任务的全部运行时状态（描述符 + 参数 + 引擎返回值）
#   - KVTaskManager         : 父类，引擎无关的「任务簿记」：创建、提交、完成聚合、取消
#   - KVTaskEngine          : 子类，面向用户的 API：get/put/prefetch、wait、批处理
#   - _launch_task          : 把图交给所有 transfer handle
#   - _update_tasks         : 轮询完成消息，驱动状态迁移与回调
#   - _mark_completed       : 落终态、跑图级回调（延迟上树）、释放重资源
#   - _get_completed_ops    : N 路 handle 的完成计数与聚合
#   - _merge_completed_op   : 多路 block 位图的「与」合并规则
#
# 任务状态机（本文件的核心）：
#   UNREADY --(set_slot_mappings 绑定真实 GPU 槽位)--> READY
#   READY   --(_launch_task / check_task_ready)-----> RUNNING
#   RUNNING --(整图在所有 handle 上到达终态 + 回调跑完)--> COMPLETED
#   RUNNING --(任一 handle 报告图失败)---------------> FAILED
#   任意态  --(cancel_tasks / reset_cache)-----------> CANCELLED
#   说明：除 UNREADY->READY 外没有回退边；COMPLETED / CANCELLED / FAILED 三个
#   终态由 KVTask.is_completed() 统一判定，终态任务会被回收出 self.tasks。
#
# 阅读提示：
#   1. 两阶段调用（get_match 之后再 launch）建出的任务处于 UNREADY —— 匹配时 GPU
#      槽位还没定，图里的 GPU block id 是假的；一阶段调用（get_async）直接给了真实
#      slot_mapping，建出来就是 READY 并立即提交。
#   2. 「传输完成」不等于「任务完成」：先跑 op 级回调，再在 _mark_completed 里跑
#      图级回调（其中就包含延迟上树 _commit_deferred_insert），最后才落终态。
#   3. 本文件的方法基本都在调用方线程上跑，靠 wait/try_wait 里的轮询推进状态，
#      所以 tasks 字典本身没有加锁；只有 task_id 自增用了 task_id_lock。
# ==============================================================================
import logging
import time
from typing import Dict, Optional, List, Union, Tuple
import threading
from enum import Enum
from dataclasses import dataclass, field, replace
from typing import Callable
import multiprocessing as mp
import copy
from expiring_dict import ExpiringDict
import nvtx
import numpy as np

from flexkv.common.config import CacheConfig, ModelConfig, GLOBAL_CONFIG_FROM_ENV
from flexkv.common.debug import flexkv_logger
from flexkv.common.block import hash_token
from flexkv.common.transfer import (
    CompletedOp,
    DeviceType,
    TransferOpGraph,
    TransferType,
    get_nvtx_default_color,
    invoke_op_callback,
    merge_to_batch_graph,
)
from flexkv.common.tracer import FlexKVTracer
from flexkv.cache.cache_engine import (
    GlobalCacheEngine,
    CacheStrategy,
    DEFAULT_CACHE_STRATEGY,
    CPUONLY_CACHE_STRATEGY,
)
from flexkv.transfer_manager import TransferManagerHandle, TransferManagerOnRemote
from flexkv.common.request import KVResponseStatus, KVResponse
from flexkv.cache.redis_meta import RedisMeta
from flexkv.integration.dynamo.collector import KVEventCollector
from flexkv.metrics.collector import get_global_collector
from flexkv.transfer_manager import TransferManagerMultiNodeHandle

class TaskStatus(Enum):
    """任务生命周期状态。迁移条件见文件头的状态机说明。

    UNREADY -> READY    : 两阶段调用在 launch 前补齐真实 GPU 槽位（_set_slot_mapping_impl）
    READY   -> RUNNING  : check_task_ready，图已交给数据面
    RUNNING -> COMPLETED: 所有 handle 的图级终态到达且回调执行完毕
    RUNNING -> FAILED   : 任一 handle 报告图失败，走 _fail_task 回滚
    任意态  -> CANCELLED: cancel_tasks / reset_cache
    """
    # slot mapping is not ready
    UNREADY = "unready"
    # waiting for the task to be launched
    READY = "ready"
    # in transfer
    RUNNING = "running"
    # transfer completed
    COMPLETED = "completed"
    # transfer cancelled
    CANCELLED = "cancelled"
    # transfer failed
    FAILED = "failed"

class TaskType(Enum):
    """任务类型。BATCH_* 是 merge_to_batch_kvtask 把多个同类任务融合后的批任务。"""
    GET = "get"
    PUT = "put"
    PREFETCH = "prefetch"
    BATCH_GET = "batch_get"
    BATCH_PUT = "batch_put"

@dataclass
class KVTask:
    """一次请求的全部运行时状态，按来源分三组字段。

    1) task descriptor：任务自身的状态（task_id / task_type / status 及终态判定辅助位）
    2) params：调用方传进来的原始参数
    3) cache engine return：GlobalCacheEngine 匹配后产出的图、掩码与回调

    生命周期末尾由 shed_heavy_resources() 主动丢掉大对象（图、token_ids 等），
    只保留 status 和 return_mask，这样已经被 wait 取走过响应的任务可以安全释放。
    """
    # task descriptor
    task_id: int
    task_type: TaskType
    # 图中代表「数据路径已搬完」的那个 op（如 H2D 的最后一段）。它比图级终态更早
    # 到达，用于 early-return：check_completed(completely=False) 见到它就认为可以
    # 提前返回成功，而不必等整图（写穿 SSD/远端、延迟上树等尾部动作）跑完。
    task_end_op_id: int
    task_end_op_finished: bool
    status: TaskStatus

    # params
    token_ids: np.ndarray
    slot_mapping: np.ndarray
    token_mask: Optional[np.ndarray]

    # cache engine return
    graph: TransferOpGraph
    # 命中情况的布尔掩码：单任务是 ndarray，批任务是每个子任务一个 ndarray 的 list。
    # prefetch 场景下它会在完成时被 _finalize_prefetch_return_mask 按实际搬到的
    # 长度重写，所以「规划时的 mask」和「最终回报的 mask」可能不同。
    return_mask: Union[np.ndarray, list[np.ndarray]]
    # 图级回调（cache engine 的 partial）：负责延迟上树 / 节点 set_ready。
    # 批任务同样退化成 list，每个子任务一个。
    callback: Optional[Union[Callable, List[Callable]]]
    # op_id -> op 级回调，单个 op 完成时按 id 精确派发
    op_callback_dict: Dict[int, Callable]
    transfer_failed: bool = False

    # SWA GPU slot_mapping (SWA-pool token index space), bound LATE at launch —
    # the SWA counterpart to slot_mapping. None when the request has no SWA ops.
    swa_slot_mapping: Optional[np.ndarray] = None
    created_ns: int = field(default_factory=time.perf_counter_ns)

    # True after wait()/try_wait() has produced a response for this task.
    request_returned: bool = False

    # Prefetch mooncake outcomes captured from CompletedOp (not from
    # task.graph ops — transfer engines may mutate a deepcopy).
    # Finalize return_mask (A) is the Full REMOTE2H success prefix clamped by
    # deferred publish; for joint SWA prefetch it is further gated so only
    # Full+SWA both succeeding reports a non-zero mask (else 0). SWA bitmaps
    # + ``prefetch_has_swa_remote`` also feed the joint outcome METRIC —
    # SWA lives in a separate slot space and is not summed into return_mask.
    prefetch_full_block_results: Optional[Tuple[bool, ...]] = None
    prefetch_swa_block_results: Optional[Tuple[bool, ...]] = None
    prefetch_has_swa_remote: bool = False
    prefetch_namespace: Optional[List[str]] = None
    prefetch_swa_aware: bool = False

    def is_completed(self) -> bool:
        """是否处于三个终态之一（COMPLETED / CANCELLED / FAILED）。

        这是全文件判定「任务是否可以回收 / wait 是否可以结束等待」的唯一入口。
        注意 CANCELLED 也算 completed：取消之后 wait 必须能拿到 CANCELLED 响应返回。
        """
        return self.status in [TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.FAILED]

    def shed_heavy_resources(self) -> None:
        """丢弃图、token_ids 等大对象，只留 status + return_mask。

        终态任务往往还要在 self.tasks 里多活一会儿（等 wait 把 return_mask 取走），
        但图可能很大（上百个 op），先释放能显著降低内存占用。
        """
        # Keep status and return_mask so a task whose response has not been
        # returned can still be observed by wait().
        self.graph = None
        self.token_ids = None
        self.slot_mapping = None
        self.token_mask = None
        self.callback = None

# 任务状态 -> 对外响应状态。RUNNING 映射成 SUCCESS 是刻意为之：wait 在
# task_end_op 完成后允许「提前返回成功」（数据已经可用），此时任务其实还在跑
# 尾部动作（写穿 / 延迟上树），但对调用方来说结果已经可以消费了。
TASK_STATUS_TO_RESPONSE_STATUS = {
    TaskStatus.COMPLETED: KVResponseStatus.SUCCESS,
    TaskStatus.CANCELLED: KVResponseStatus.CANCELLED,
    TaskStatus.FAILED: KVResponseStatus.FAILED,
    TaskStatus.RUNNING: KVResponseStatus.SUCCESS, # for early return: still running, but success
}

def convert_to_response_status(task_status: TaskStatus) -> KVResponseStatus:
    return TASK_STATUS_TO_RESPONSE_STATUS[task_status]


def _longest_success_prefix(block_results: Tuple[bool, ...]) -> int:
    """Longest contiguous True prefix of per-block transfer results."""
    # 取「最长连续成功前缀」而不是「成功块总数」：前缀中间断一个 block，后面即使
    # 搬成功在语义上也接不上（前缀匹配要求连续），所以不能按成功数计数。
    prefix = 0
    for succeeded in block_results:
        if not succeeded:
            break
        prefix += 1
    return prefix


class KVTaskManager:
    """任务簿记层（KVTaskEngine 的父类）：与用户 API 无关的那一半职责。

    它负责的东西可以概括成「三张表 + 一组状态迁移」：
      - tasks            : task_id -> KVTask，所有活着的任务
      - graph_to_task    : graph_id -> task_id，完成消息按 graph_id 回来，靠它反查任务
      - uncompleted_ops / uncompleted_op_results / uncompleted_graphs
                         : N 路 handle 的完成计数与聚合中间态

    子类 KVTaskEngine 只加「面向用户的入口」（get/put/prefetch/wait/批处理）和
    tracer 埋点；所有真正改变任务状态的动作（创建、提交、完成、失败、取消、回收）
    都在本类里，这样测试可以只实例化父类、不依赖 tracer 与模型配置。

    不变量：
      1. 一个 graph_id 同一时刻只属于一个 task_id（graph_to_task 是单射，只在
         建任务时写入，在 _fail_task / _mark_completed / _release_task 时摘除）。
      2. 每个 handle 都会收到同一张图，因此「完成」必须按 handle 数计数到 N
         （required_completed_count）；任一 handle 失败则整图失败。
    """

    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 gpu_register_port: Optional[str] = None,
                 redis_meta: RedisMeta = None,
                 event_collector: Optional[KVEventCollector] = None
                 ):
        """建缓存引擎与 transfer handle，初始化任务表。

        Args:
            model_config: 模型/并行配置，决定 tp/pp/nnodes 以及是否走 TRT-LLM 子进程
            cache_config: 各级存储开关（enable_cpu/ssd/gds/remote/nixl...），此处先
                做一轮组合合法性校验，非法组合直接 fail fast，不要等到传输时才炸
            gpu_register_port: GPU 内存注册的通信端口，传给 transfer handle
            redis_meta: 分布式元数据存储句柄，多机场景用于同步 radix 索引
            event_collector: KV 事件采集器（dynamo 集成），用于对外发布缓存事件
        """
        if not cache_config.enable_cpu:
            raise ValueError("enable_cpu must be True")
        # Mooncake store is a remote backend that does not require local SSD.
        # Keep this aligned with CacheConfig validation in common/config.py.
        if (cache_config.enable_remote and not cache_config.enable_cpu):
            raise ValueError("enable_cpu must be True if enable_remote is True")
        if not cache_config.enable_cpu and not cache_config.enable_gds:
            raise ValueError("enable_gds must be True if enable_cpu is False")
        if cache_config.enable_gds and not cache_config.enable_ssd:
            raise ValueError("enable_ssd must be True if enable_gds is True")
        if cache_config.enable_kv_sharing and cache_config.enable_gds:
            raise ValueError("enable_kv_sharing and enable_gds cannot be used at the same time")
        if cache_config.enable_nixl and not cache_config.enable_gds:
            raise ValueError("enable_nixl requires enable_gds to be True")
        if cache_config.enable_nixl and model_config.effective_tp_size_per_node > 1:
            raise ValueError(
                "enable_nixl GPU-SSD path currently requires effective_tp_size_per_node==1 "
                "(no tpNixlTransferWorker)"
            )
        self.model_config = model_config
        self.cache_config = cache_config

        flexkv_logger.info(
            f"[KVTaskEngine] topology: {self.model_config}"
        )

        self.cache_engine = GlobalCacheEngine(cache_config, model_config, redis_meta, event_collector)

        # self.transfer_handles 是「同一张图要被投递几次」的来源：
        #   [0] 本机 handle（process 模式 = 拉子进程；remote 模式 = 连远端服务）
        #   [1] 多机场景下追加的跨机 handle（nnodes > 1）
        # 见 required_completed_count：它的长度就是完成聚合的 N。
        if not self.model_config.use_trtllm_subprocess:
            self.transfer_handles = [TransferManagerHandle(
                model_config,
                cache_config,
                mode="process",
                gpu_register_port=gpu_register_port
            )]
        else:
            # When using FlexKV with TensorRT-LLM, we use remote mode to transfer data
            #  to avoid the way we launch subprocess in FlexKV
            #  conflict with TensorRT-LLM's MPI initialization.
            # 中文补充：TensorRT-LLM 自己要初始化 MPI，而 FlexKV 默认用 fork 起传输
            # 子进程，两者冲突（MPI 不允许 fork 后再用 MPI 调用）。所以这里改为
            # 先把传输服务当独立进程拉起来（TransferManagerOnRemote.create_process），
            # 本进程再用 remote 模式的 handle 连过去，绕开 fork。
            sub_host = self.model_config.trtllm_subprocess_host
            sub_ports = self.model_config.trtllm_subprocess_ports
            self.remote_process = TransferManagerOnRemote.create_process(
                master_host=sub_host,
                master_ports=sub_ports,
            )
            self.transfer_handles = [
                TransferManagerHandle(
                    model_config,
                    cache_config,
                    mode="remote",
                    gpu_register_port=gpu_register_port,
                    master_host=sub_host,
                    master_ports=sub_ports,
                )
            ]
            self.transfer_handles[0]._handle.send_config_to_remotes()

        if self.model_config.nnodes > 1:
            self.transfer_handles.append(TransferManagerHandle(
                model_config,
                cache_config,
                mode="remote",
                gpu_register_port=gpu_register_port,
                master_host=self.model_config.master_host,
                master_ports=self.model_config.master_ports,
            ))
            self.transfer_handles[-1]._handle.send_config_to_remotes()

        # 任务表用带 TTL 的字典：调用方忘了 wait 的任务不会永久泄漏，最多活 30 分钟
        self.tasks: ExpiringDict[int, KVTask] = ExpiringDict(max_age_seconds=1800, max_len=100000) # 30 minutes

        # hash(token_ids) -> task_id
        # 预取去重表：同一段 token 重复 prefetch 时，用它可以找到已在飞的预取任务
        self.prefetch_tasks: ExpiringDict[int, int] = ExpiringDict(max_age_seconds=1800, max_len=100000) # 30 minutes
        self._gen_prefetch_key = lambda token_ids, namespace: hash_token(token_ids, namespace)

        self.graph_to_task: Dict[int, int] = {}

        # (graph_id, op_id) -> completed_count; graph-keyed so a failed
        # graph's stale per-op counters can be purged.
        # 同一个 op 会被 N 个 handle 各报告一次，计数到 N 才算真的完成。
        self.uncompleted_ops: Dict[Tuple[int, int], int] = {}
        # 计数未到 N 之前，把已到的那几路合并结果暂存在这里
        self.uncompleted_op_results: Dict[Tuple[int, int], CompletedOp] = {}
        # graph_id -> (terminal_count, any_failed) across the N handles.
        self.uncompleted_graphs: Dict[int, Tuple[int, bool]] = {}
        # 完成所需的 handle 份数：每个 handle 都会收到同一张图
        self.required_completed_count: int = len(self.transfer_handles)

        self.task_id_counter = 0
        self.task_id_lock = threading.Lock()

        self.running_tasks: int = 0

    def start(self) -> None:
        """启动所有 transfer handle（拉起传输进程 / 连上远端），之后才 is_ready。"""
        for transfer_handle in self.transfer_handles:
            transfer_handle.start()

    def is_ready(self) -> bool:
        """所有 handle 都就绪才算就绪：少一路，N 路聚合就永远等不到 N。"""
        return all(transfer_handle.is_ready() for transfer_handle in self.transfer_handles)

    def __del__(self) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        """关闭所有 handle；TRT-LLM 模式下还要回收自己拉起的远端传输进程。

        用 hasattr 兜底：__del__ 可能在 __init__ 抛异常的半初始化对象上被调用。
        """
        if hasattr(self, "transfer_handles") and self.transfer_handles is not None:
            for transfer_handle in self.transfer_handles:
                transfer_handle.shutdown()
        if hasattr(self, "remote_process") and self.remote_process is not None:
            assert self.remote_process.is_alive()
            self.remote_process.terminate()
            self.remote_process.join()
            self.remote_process.close()
            self.remote_process = None

    @staticmethod
    def _operation_name(task_type: TaskType) -> str:
        """把任务类型归一成日志/指标里的操作名（批任务归并到 get/put）。"""
        if task_type in (TaskType.GET, TaskType.BATCH_GET):
            return "get"
        if task_type in (TaskType.PUT, TaskType.BATCH_PUT):
            return "put"
        return "prefetch"

    def _log_task_created(self, task: KVTask) -> None:
        """任务创建埋点。先判日志级别再拼字符串，避免热路径上的无谓开销。"""
        if not flexkv_logger.is_enabled_for(logging.DEBUG):
            return
        graph_id = task.graph.graph_id if task.graph is not None else -1
        tokens = len(task.token_ids) if task.token_ids is not None else 0
        flexkv_logger.debug(
            "[FlexKV-IO] operation=%s act=create status=%s blocks=%d "
            "flexkv_task_id=%d graph_id=%d tokens=%d graph_ops=%d",
            self._operation_name(task.task_type),
            task.status.value,
            tokens // self.cache_config.tokens_per_block,
            task.task_id,
            graph_id,
            tokens,
            task.graph.num_ops if task.graph is not None else 0,
        )

    def _log_task_terminal(self, task: KVTask, status: TaskStatus) -> None:
        """任务终态埋点，顺带打印从创建到终态的总耗时。

        空图（num_ops == 0，即完全未命中）的成功只打 DEBUG：这种情况在高命中率
        下会刷屏，却没什么信息量；非成功终态一律 WARNING。
        """
        graph_ops = task.graph.num_ops if task.graph is not None else 0
        if status == TaskStatus.COMPLETED:
            level = logging.INFO if graph_ops else logging.DEBUG
        else:
            level = logging.WARNING
        if not flexkv_logger.is_enabled_for(level):
            return
        graph_id = task.graph.graph_id if task.graph is not None else -1
        duration_s = (time.perf_counter_ns() - task.created_ns) / 1e9
        if level == logging.INFO:
            log = flexkv_logger.info
        elif level == logging.DEBUG:
            log = flexkv_logger.debug
        else:
            log = flexkv_logger.warning
        log(
            "[FlexKV-IO] operation=%s act=complete status=%s "
            "flexkv_task_id=%d graph_id=%d task_time=%.4fs",
            self._operation_name(task.task_type),
            convert_to_response_status(status).value,
            task.task_id,
            graph_id,
            duration_s,
        )

    def create_get_task(self,
                        task_id: int,
                        token_ids: np.ndarray,
                        slot_mapping: np.ndarray,
                        dp_client_id: int,
                        token_mask: Optional[np.ndarray] = None,
                        is_fake_slot_mapping: bool = False,
                        temp_cache_strategy: CacheStrategy = DEFAULT_CACHE_STRATEGY,
                        namespace: Optional[List[str]] = None,
                        swa_aware: bool = False,
                        ) -> None:
        """创建 GET 任务：查 radix tree 匹配前缀 -> 产出传输图 -> 落进 tasks 表。

        Args:
            token_ids: 请求 token 序列，用于前缀匹配
            slot_mapping: GPU 侧槽位映射；两阶段调用传的是全 0 的假映射
            dp_client_id: DP 分组 id，决定去哪一份 GPU 块表里取块
            token_mask: 标记哪些 token 参与匹配
            is_fake_slot_mapping: True 表示 slot_mapping 是假的，任务建为 UNREADY，
                等 launch 前由 set_slot_mappings 回填真实块 id
            temp_cache_strategy: 临时覆盖缓存策略（如 cpu_only 只读内存、prefetch 只到 CPU）
            namespace / swa_aware: 命名空间隔离 / 是否按 SWA 窗口裁剪 Full-KV 传输

        副作用：写 self.tasks[task_id] 与 self.graph_to_task[graph.graph_id]。
        注意：本方法只「建」不「提交」，提交要等 _launch_task（两阶段下中间还夹着
        槽位回填）。
        """
        if task_id in self.tasks:
            raise ValueError(f"Task ID {task_id} already exists")
        graph, return_mask, callback, op_callback_dict, task_end_op_id = self.cache_engine.get(
            request_id=task_id,
            token_ids=token_ids,
            token_mask=token_mask,
            slot_mapping=slot_mapping,
            dp_client_id=dp_client_id,
            temp_cache_strategy=temp_cache_strategy,
            namespace=namespace,
            swa_aware=swa_aware)
        self.tasks[task_id] = KVTask(
            task_id=task_id,
            task_type=TaskType.GET,
            task_end_op_id=task_end_op_id,
            task_end_op_finished=False,
            status=TaskStatus.UNREADY if is_fake_slot_mapping else TaskStatus.READY,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            token_mask=token_mask,
            graph=graph,
            return_mask=return_mask,
            callback=callback,
            op_callback_dict=op_callback_dict)

        self.graph_to_task[graph.graph_id] = task_id
        self._log_task_created(self.tasks[task_id])

    def create_put_task(self,
                        task_id: int,
                        token_ids: np.ndarray,
                        slot_mapping: np.ndarray,
                        dp_client_id: int,
                        token_mask: Optional[np.ndarray] = None,
                        is_fake_slot_mapping: bool = False,
                        namespace: Optional[List[str]] = None,
                        ) -> None:
        """创建 PUT 任务：把 GPU 上算好的 KV 卸载到 CPU/SSD/远端。

        PUT 的 return_mask 语义与 GET 相反：标记的是「还不在外部存储里、真正需要
        写回」的 token，调用方据此决定要搬多少数据。其余流程与 create_get_task 同构。
        """
        if task_id in self.tasks:
            raise ValueError(f"Task ID {task_id} already exists")
        graph, return_mask, callback, op_callback_dict, task_end_op_id = self.cache_engine.put(
            request_id=task_id,
            token_ids=token_ids,
            token_mask=token_mask,
            slot_mapping=slot_mapping,
            dp_client_id=dp_client_id,
            namespace=namespace)
        self.tasks[task_id] = KVTask(
            task_id=task_id,
            task_type=TaskType.PUT,
            task_end_op_id=task_end_op_id,
            task_end_op_finished=False,
            status=TaskStatus.UNREADY if is_fake_slot_mapping else TaskStatus.READY,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            token_mask=token_mask,
            graph=graph,
            return_mask=return_mask,
            callback=callback,
            op_callback_dict=op_callback_dict)
        self.graph_to_task[graph.graph_id] = task_id
        self._log_task_created(self.tasks[task_id])

    def create_prefetch_task(self,
                            task_id: int,
                            token_ids: np.ndarray,
                            dp_client_id: int,
                            namespace: Optional[List[str]] = None,
                            swa_aware: bool = False,
                            ) -> None:
        """创建 PREFETCH 任务：把远端/SSD 的数据提前搬到 CPU，不进 GPU。

        预取不需要 GPU 槽位，所以 slot_mapping / token_mask 都是假的，任务直接是
        READY；缓存策略也被强制改成「只到 CPU」（ignore_gpu + ignore_gds）。
        真正搬到的长度不在这里确定 —— 完成时由 _finalize_prefetch_return_mask 按
        实际成功前缀重写 return_mask。
        """
        if task_id in self.tasks:
            raise ValueError(f"Task ID {task_id} already exists")
        fake_slot_mapping = np.zeros_like(token_ids)
        fake_token_mask = np.ones_like(token_ids)
        temp_cache_strategy = copy.deepcopy(DEFAULT_CACHE_STRATEGY)
        # 预取只到 CPU：GPU 还没有给这段请求分配槽位（而且可能永远不会）
        temp_cache_strategy.ignore_gpu = True  # upload to CPU only
        temp_cache_strategy.ignore_gds = True
        graph, return_mask, callback, op_callback_dict, task_end_op_id = self.cache_engine.get(
            request_id=task_id,
            token_ids=token_ids,
            token_mask=fake_token_mask,
            slot_mapping=fake_slot_mapping,
            dp_client_id=dp_client_id,
            temp_cache_strategy=temp_cache_strategy,
            namespace=namespace,
            swa_aware=swa_aware)
        prefetch_has_swa_remote = any(
            op.transfer_type == TransferType.REMOTE2H and getattr(op, "is_swa", False)
            for op in graph._op_map.values()
        )
        self.tasks[task_id] = KVTask(
            task_id=task_id,
            task_type=TaskType.PREFETCH,
            task_end_op_id=task_end_op_id,
            task_end_op_finished=False,
            status=TaskStatus.READY,  # gpu slots are not needed for prefetch
            token_ids=token_ids,
            slot_mapping=fake_slot_mapping,  # ignore slot_mapping for prefetch
            token_mask=fake_token_mask,  # ignore token_mask for prefetch
            graph=graph,
            return_mask=return_mask,
            callback=callback,
            op_callback_dict=op_callback_dict,
            prefetch_has_swa_remote=prefetch_has_swa_remote,
            prefetch_namespace=namespace,
            prefetch_swa_aware=swa_aware)

        self.prefetch_tasks[self._gen_prefetch_key(token_ids, namespace)] = task_id

        self.graph_to_task[graph.graph_id] = task_id
        self._log_task_created(self.tasks[task_id])

    def _launch_task(self, task_id: int) -> None:
        """把任务对应的图投递给所有 transfer handle（READY -> RUNNING）。

        check_task_ready 内部完成状态迁移并做前置校验：终态任务直接返回 None（不
        重复提交），非 READY 状态（UNREADY/RUNNING）直接抛错——UNREADY 说明调用方
        还没给 GPU 槽位就急着提交，属于使用错误。

        Args:
            task_id: 要提交的任务 id
        """
        transfer_graph = self.check_task_ready(task_id)
        if transfer_graph is None:
            return
        nvtx.mark(f"launch task: task_id={task_id}, graph_id={transfer_graph.graph_id}")
        # 空图不用提交：没有任何 op 就没有数据要搬（已在建任务时被 _process_empty_graph 收尾）
        if transfer_graph.num_ops > 0:
            for transfer_handle in self.transfer_handles:
                # For remote handles: deepcopy graph and clear GPU blocks when
                # it's a cross-machine PP handle (different PP stages have
                # different GPU block_ids).  Cross-machine TP handles share
                # the same slot_mapping, so no clear is needed.
                if isinstance(transfer_handle._handle, TransferManagerMultiNodeHandle):
                    if self.model_config.nnodes > 1 and self.model_config.pp_size > 1:
                        # Cross-machine PP: each PP rank has different GPU blocks
                        graph_copy = copy.deepcopy(transfer_graph)
                        graph_copy.clear_gpu_blocks()
                        transfer_handle.submit(graph_copy, task_end_op_id=self.tasks[task_id].task_end_op_id)
                    else:
                        # Cross-machine TP: same slot_mapping across TP ranks
                        transfer_handle.submit(transfer_graph, task_end_op_id=self.tasks[task_id].task_end_op_id)
                else:
                    transfer_handle.submit(transfer_graph, task_end_op_id=self.tasks[task_id].task_end_op_id)

    def _update_tasks(self, timeout: float = 0.001) -> None:
        """轮询一次完成消息并推进所有相关任务的状态（本文件的「心跳」）。

        每一条聚合后的 CompletedOp 在这里被分发：
          1. 图级失败 -> _fail_task（回滚计划，落 FAILED）
          2. 远端读（REMOTE2H）-> 按 block 位图判断是否部分失败 / 记 prefetch 结果
          3. 有 op 级回调 -> 按 op_id 精确派发（如某个 tier 的节点 set_ready）
          4. 图级完成 -> _mark_completed（跑图级回调 = 延迟上树，再落终态）
          5. 仅 task_end_op 完成 -> 置 task_end_op_finished，允许提前返回成功

        Args:
            timeout: 传给 handle.wait 的超时（秒）。0 表示非阻塞地捞一次就走
        """
        completed_ops = self._get_completed_ops(timeout)
        metrics_collector = get_global_collector()
        for completed_op in completed_ops:
            if completed_op.graph_id not in self.graph_to_task:
                # 迟到的消息：对应任务已经终止并被回收（或已 reset_cache），
                # 没有可推进的状态，直接丢弃
                continue
            task_id = self.graph_to_task[completed_op.graph_id]
            task = self.tasks[task_id]
            if completed_op.is_graph_failed():
                self._fail_task(task_id)
                continue
            # A failed pull invalidates the current request and must trigger
            # fallback. Mooncake uploads retain the existing asynchronous PUT
            # completion contract (the task-end D2H may precede H2REMOTE).
            graph_op = getattr(task.graph, "_op_map", {}).get(
                completed_op.op_id)
            is_swa_op = graph_op is not None and getattr(graph_op, "is_swa", False)
            is_remote_load = (
                completed_op.transfer_type == TransferType.REMOTE2H.value
                or (graph_op is not None
                    and graph_op.transfer_type == TransferType.REMOTE2H)
            )
            expects_block_results = (
                graph_op is not None
                and (graph_op.mooncake_store_block_hashes is not None
                     or graph_op.mooncake_store_swa_block_hashes is not None)
            ) ## only mooncake store related ops expect block results now
            missing_results = (
                completed_op.block_results is None and expects_block_results)
            failed_blocks = (
                completed_op.block_results is not None
                and not all(completed_op.block_results)
            )
            # REMOTE2H policy + mooncake bitmap snapshot (CompletionOp itself;
            # do not rely on TransferOp.block_results on task.graph — engines
            # may mutate a submitted deepcopy).
            if is_remote_load:
                if is_swa_op:
                    # Joint prefetch: SWA partial/missing MUST NOT fail the task.
                    # Commit-time joint guard uses the bitmap with Full's mask.
                    if task.task_type == TaskType.PREFETCH:
                        task.prefetch_has_swa_remote = True
                        if completed_op.block_results is not None:
                            task.prefetch_swa_block_results = tuple(
                                bool(x) for x in completed_op.block_results)
                else:
                    if missing_results:
                        task.transfer_failed = True
                    elif failed_blocks:
                        # Prefetch: L==0 fails eagerly; L>0 waits for graph
                        # completion so joint SWA can still shape commit.
                        if task.task_type == TaskType.PREFETCH:
                            assert completed_op.block_results is not None
                            if _longest_success_prefix(
                                    tuple(completed_op.block_results)) == 0:
                                task.transfer_failed = True
                        else:
                            task.transfer_failed = True
                    if (task.task_type == TaskType.PREFETCH
                            and completed_op.block_results is not None):
                        task.prefetch_full_block_results = tuple(
                            bool(x) for x in completed_op.block_results)
            # Record transfer metrics for completed ops (post-completion statistics)
            # All three counters (ops_total, blocks_total, bytes_total) are updated
            # here after transfer completion, providing accurate post-transfer metrics.
            if metrics_collector is not None and completed_op.transfer_type is not None:
                if task.task_type in (TaskType.GET, TaskType.PREFETCH, TaskType.BATCH_GET):
                    operation = "get"
                elif task.task_type == TaskType.PUT:
                    operation = "put"
                else:
                    operation = "unknown"
                metrics_collector.record_transfer_completed(
                    completed_op.transfer_type,
                    completed_op.num_blocks,
                    completed_op.num_bytes,
                    operation,
                )
            if task.status == TaskStatus.CANCELLED and task.callback is None:
                # Cache was reset while this task was in flight: reset_cache()
                # cleared its callbacks and freed the radix nodes / mempool blocks
                # 中文补充：此时回调闭包指向的 radix 节点 / mempool 块已被释放，
                # 再触发就会「解锁已删除的节点」或「重复 free 同一个块」，必须跳过
                flexkv_logger.warning(
                    f"task {task_id}: transfer op {completed_op.op_id} completed "
                    "after reset_cache(), callback no longer exists and will be skipped."
                )
                continue
            has_callback = completed_op.op_id in task.op_callback_dict
            if has_callback:
                try:
                    invoke_op_callback(
                        task.op_callback_dict[completed_op.op_id], completed_op)
                except Exception:
                    task.transfer_failed = True
                    flexkv_logger.error(
                        "Transfer op callback failed: "
                        f"graph_id={completed_op.graph_id}, "
                        f"op_id={completed_op.op_id}",
                        exc_info=True,
                    )
            if completed_op.is_graph_completed():
                # _mark_completed runs deferred commit first, then (for
                # prefetch) finalizes return_mask from the Full REMOTE2H
                # success bitmap — report "how much remote this task pulled".
                # 中文补充：整图到达终态意味着数据已经落地，此时才允许执行图级回调
                # —— 也就才允许「延迟上树」（_commit_deferred_insert）。
                self._mark_completed(task_id)
            elif completed_op.op_id == task.task_end_op_id:
                self.tasks[task_id].task_end_op_finished = True

    def _narrow_return_mask_to_prefix_blocks(
            self, task: "KVTask", num_success_blocks: int) -> None:
        """Rewrite prefetch return_mask to the first ``num_success_blocks``.

        Prefetch builds a contiguous True span for planned REMOTE2H blocks.
        After partial mooncake success, only ``[:L]`` is published/usable.
        """
        mask = task.return_mask
        if mask is None or isinstance(mask, list):
            return
        if num_success_blocks <= 0:
            task.return_mask = np.zeros_like(mask, dtype=np.bool_)
            return
        true_idx = np.flatnonzero(mask)
        if true_idx.size == 0:
            return
        start = int(true_idx[0])
        orig_end = int(true_idx[-1]) + 1
        tpb = self.cache_config.tokens_per_block
        end = min(start + num_success_blocks * tpb, orig_end, mask.shape[0])
        new_mask = np.zeros_like(mask, dtype=np.bool_)
        new_mask[start:end] = True
        task.return_mask = new_mask

    @staticmethod
    def _prefetch_published_remote_blocks(task: "KVTask") -> Optional[int]:
        """CPU deferred-publish remote block count after graph callback.

        Returns ``None`` when this task has no deferred-publish tracker
        (non-mooncake / legacy path). Returns ``0`` when commit discarded
        or failed to mount anything matchable.

        中文补充：它读的是「延迟上树」的结果（publish_result.published_remote_blocks），
        即真正挂到 radix tree 上的远端块数，可能少于传输成功的块数（比如提交前
        rematch 发现别人已经写过了）。多个回调取 min —— 保守起见按最少的那个算。
        """
        callbacks = task.callback
        if callbacks is None:
            return None
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        saw_tracker = False
        published: Optional[int] = None
        for callback in callbacks:
            keywords = getattr(callback, "keywords", None) or {}
            for pending in keywords.get("deferred_inserts") or []:
                if getattr(pending, "device_type", None) != DeviceType.CPU:
                    continue
                publish_result = getattr(pending, "publish_result", None)
                if publish_result is None:
                    continue
                saw_tracker = True
                blocks = publish_result.published_remote_blocks
                if blocks is None or publish_result.failed:
                    blocks = 0
                published = (
                    int(blocks) if published is None
                    else min(published, int(blocks)))
        if not saw_tracker:
            return None
        return 0 if published is None else published

    def _finalize_prefetch_return_mask(self, task: "KVTask") -> None:
        """Report reusable Full REMOTE2H tokens for this prefetch (A).

        ``sum(return_mask)`` is the storage/L3 accounting number consumed by
        sglang as ``storage_hit_length``. It must NOT include a pre-existing
        CPU prefix (f1) or DISK2H.

        Base length is the longest success prefix of this task's Full mooncake
        REMOTE2H, clamped by CPU deferred-publish length so transfer-success /
        commit-discard cannot over-report.

        Joint Full+SWA prefetch (``prefetch_has_swa_remote``): subsequent
        ``swa_aware`` GET uses ``min(full, swa)`` and commit only mounts SWA
        when Full covers the whole planned span. Reporting therefore requires
        **both** Full transfer/publish complete (L == planned) **and** SWA
        transfer success; otherwise the mask is cleared to 0 even if Full
        alone was mounted. SWA tokens are never summed into the mask.

        Opaque backends without ``block_results`` leave the plan-time mask
        unchanged when there is no SWA remote op.
        """
        # 中文要点：这里算出的长度是「本次真的从远端搬回来、且能被复用的 token 数」，
        # 不含本来就在 CPU 上的前缀（f1），也不含 DISK2H —— sglang 把 sum(return_mask)
        # 当作 storage_hit_length 来做存储层命中统计。
        full_results = task.prefetch_full_block_results
        if full_results is not None:
            mounted_full_len = _longest_success_prefix(full_results)
        else:
            mounted_full_len = None

        published_remote = self._prefetch_published_remote_blocks(task)
        if published_remote is not None:
            if mounted_full_len is None:
                mounted_full_len = published_remote
            else:
                mounted_full_len = min(mounted_full_len, published_remote)

        # Length reported to callers (may be zeroed by the joint SWA gate).
        report_len = mounted_full_len
        if task.prefetch_has_swa_remote:
            swa_results = task.prefetch_swa_block_results
            swa_ok = (
                swa_results is not None
                and len(swa_results) > 0
                and all(swa_results)
            )
            if full_results is None:
                # Joint path without Full bitmaps cannot prove both succeeded.
                report_len = 0
            else:
                planned_full_len = len(full_results)
                if (report_len is None
                        or report_len != planned_full_len
                        or not swa_ok):
                    report_len = 0

        if report_len is not None:
            self._narrow_return_mask_to_prefix_blocks(task, report_len)

        # Joint / full-only outcome metric (mooncake bitmaps).
        # Classified from mounted Full length + SWA bitmap — not from the
        # caller-facing report_len gate — so full_only_swa_lost remains visible.
        if full_results is None:
            return

        assert mounted_full_len is not None
        if not task.prefetch_has_swa_remote:
            outcome = "full_only"
        else:
            planned_full_len = len(full_results)
            swa_results = task.prefetch_swa_block_results
            swa_ok = (
                swa_results is not None
                and len(swa_results) > 0
                and all(swa_results)
            )
            if mounted_full_len == 0:
                outcome = "all_failed"
            elif mounted_full_len == planned_full_len and swa_ok:
                outcome = "full_and_swa"
            elif mounted_full_len == planned_full_len and not swa_ok:
                outcome = "full_only_swa_lost"
            else:
                outcome = "partial_full"
        metrics_collector = get_global_collector()
        if metrics_collector is not None:
            metrics_collector.record_joint_prefetch_outcome(outcome)

    @staticmethod
    def _abort_task_plans(task: "KVTask") -> None:
        """Run the abort path of every plan handle the task carries (a batch
        task carries one per merged sub-task). Handles predating the abort API
        are skipped."""
        # 中文补充：这是失败/取消时归还规划期资源的唯一入口。plan handle 是 cache
        # engine 在规划时创建的，它持有「已锁住的 radix 节点」「已分配的 CPU 暂存块」
        # 等资源；abort 会把没真正用到的部分还回去。用 getattr 取 abort 属性，
        # 是为了兼容没有该方法的旧 handle。
        callbacks = task.callback if isinstance(task.callback, list) \
            else [task.callback]
        for callback in callbacks:
            abort = getattr(callback, "abort", None)
            if abort is not None:
                abort()

    def _fail_task(self, task_id: int) -> None:
        """A transfer op of this task's graph failed and the graph has fully
        drained. Roll the plan back instead of completing it: ops that did
        finish already ran their callbacks (their nodes are ready and their
        data is valid, so abort keeps them), while nodes whose transfer never
        ran are still unready and get removed with their blocks recycled.
        The task terminates as FAILED so wait() reports the failure instead
        of a misleading TIMEOUT."""
        # 中文补充（失败回滚的粒度）：_abort_task_plans 只回滚「没真正搬成功」的那
        # 部分 —— 已经跑完回调的 op，其节点已 ready、数据有效，保留；没跑到的 op
        # 节点仍是 unready，连块一起回收。既不留半截脏数据，也不白白扔掉好数据。
        if task_id not in self.tasks:
            return
        task = self.tasks[task_id]
        if task.is_completed():
            return
        flexkv_logger.error(f"[KVTaskEngine] task {task_id} FAILED: a transfer "
                            f"op of graph {task.graph.graph_id} failed")
        self._abort_task_plans(task)
        task.status = TaskStatus.FAILED
        task.task_end_op_finished = True
        self.graph_to_task.pop(task.graph.graph_id, None)
        task.shed_heavy_resources()
        if task.request_returned:
            self._release_task(task_id)

    def _cancel_task(self, task_id: int) -> None:
        """取消单个任务并回收它（cancel_tasks 的实现）。

        回滚策略按状态分两种（这是取消逻辑里最容易踩坑的地方）：
          - UNREADY / READY：图还没交出去（或刚交出去），完成回调可能永远不会来，
            必须由这里主动 _abort_task_plans 归还规划期拿到的锁与暂存块，否则这些
            资源会永久泄漏。
          - RUNNING：图已经在数据面飞了，完成回调照样会到，交给正常完成路径处理，
            这里只置状态，不 abort（避免和仍在写的引擎抢同一批块）。
        无论哪种，最后都调 _release_task 把任务从表里摘掉。
        """
        if task_id not in self.tasks:
            return
        task = self.tasks[task_id]
        if not task.is_completed():
            # A task whose graph never launched still holds everything its
            # plan acquired at create time: locked radix nodes, CPU staging
            # blocks, and is_ready=False index nodes that only a completion
            # callback could publish. Dropping the task without aborting leaks
            # all of it -- the staging blocks become unreachable (mempool
            # exhaustion) and the unready nodes are permanently unevictable
            # holes that also shadow future puts of the same prefix. Abort
            # rolls those back; RUNNING tasks keep the old behavior (their
            # graph is in flight and completion callbacks will still fire).
            if task.status in (TaskStatus.UNREADY, TaskStatus.READY):
                self._abort_task_plans(task)
            task.status = TaskStatus.CANCELLED
            self._log_task_terminal(task, TaskStatus.CANCELLED)
        self._release_task(task_id)

    def check_completed(self, task_id: int, completely: bool = False) -> bool:
        """判断任务是否「够格返回结果」，是 wait 循环的推进条件。

        completely=False（默认，快速返回）：只要 task_end_op 完成就算完成 —— 数据
        路径已经搬完，调用方可以消费了，尾部动作（写穿、延迟上树）还在后台跑。
        completely=True：必须等到整图终态（is_completed），适合真正要确认落盘/上树
        的场景。

        两个强制走 completely 的例外：
          - PREFETCH：它的价值就在于「数据已经进 CPU 树」，提前返回会对外宣称一个
            还不存在的命中。
          - transfer_failed：已经判定部分失败了，必须等图终态走完清理，好让调用方
            看到 FAILED 而不是「RUNNING 当成功」。
        """
        task = self.tasks[task_id]
        self._process_empty_graph(task_id)
        # Prefetch must wait for the graph terminal only. Joint Full+SWA graphs
        # may mark an early task_end (e.g. SWA REMOTE2H) while Full is still
        # in flight; SUCCESS before _finalize_prefetch_return_mask /
        # deferred commit would advertise a planned mask that is not yet ready.
        if task.task_type == TaskType.PREFETCH:
            completely = True
        if completely:
            return task.is_completed()
        # A partial-capable backend may finish the data-path sink after already
        # reporting failed blocks. Wait for graph completion so cleanup runs and
        # the caller observes FAILED instead of an early RUNNING-as-success result.
        if task.transfer_failed:
            return task.is_completed()
        # For tasks with callback (e.g., PUT tasks that need to call insert_and_publish),
        # we must wait until _mark_completed is called (i.e., is_completed() returns True)
        # to ensure the callback is executed before returning success.
        #if task.callback is not None:
        #    return task.is_completed()
        return task.is_completed() or task.task_end_op_finished

    def set_slot_mappings(self,
                          task_ids: List[int],
                          slot_mappings: List[np.ndarray],
                          swa_slot_mappings: Optional[List[Optional[np.ndarray]]] = None) -> None:
        """两阶段调用的「第二阶段前置动作」：批量回填真实 GPU 槽位。

        这是 UNREADY -> READY 的唯一入口。launch_tasks 会在提交前调它，把匹配阶段
        拿不到的 GPU block id 写进图里。每个任务还可选带一份 SWA 槽位映射。
        """
        if swa_slot_mappings is None:
            swa_slot_mappings = [None] * len(task_ids)
        for task_id, slot_mapping, swa_slot_mapping in zip(task_ids, slot_mappings, swa_slot_mappings):
            self._set_slot_mapping_impl(task_id, slot_mapping, swa_slot_mapping)

    def _set_slot_mapping_impl(self,
                               task_id: int,
                               slot_mapping: np.ndarray,
                               swa_slot_mapping: Optional[np.ndarray] = None) -> None:
        """把真实 slot_mapping 折叠成 GPU block id 写进图，并把任务置为 READY。

        只对 UNREADY 任务生效：已经是 READY/RUNNING 的说明槽位早就定了，重复设置
        会破坏图里已提交的块 id，所以静默返回。
        """
        task = self.tasks[task_id]
        if task.status != TaskStatus.UNREADY:
            return
        graph_ids = self.cache_engine.slot_mapping_to_block_ids(slot_mapping,
                                                                self.cache_config.tokens_per_block)
        # Late-bind the GPU-side SWA slots via the unified set_gpu_blocks(gpu,
        # swa_gpu) path (PR#191). SWA is page-granular, so the mapping folds by
        # the same stride as full-KV (slot_mapping_to_block_ids).
        # A None swa_slot_mapping leaves the graph's SWA ops at their built ids.
        swa_sm = swa_slot_mapping if swa_slot_mapping is not None else task.swa_slot_mapping
        swa_graph_ids = None
        if swa_sm is not None:
            swa_graph_ids = self.cache_engine.slot_mapping_to_block_ids(
                swa_sm, self.cache_config.tokens_per_block)
        task.graph.set_gpu_blocks(graph_ids, swa_graph_ids)
        task.slot_mapping = slot_mapping
        task.status = TaskStatus.READY

    def _gen_task_id(self) -> int:
        """生成全局自增 task_id。加锁是因为批任务 id 可能由不同线程发起。"""
        with self.task_id_lock:
            old_value = self.task_id_counter
            self.task_id_counter += 1
            return old_value

    def check_task_ready(self, task_id: int) -> TransferOpGraph:
        """校验任务可提交，并把它从 READY 推进到 RUNNING，返回待提交的图。

        Returns:
            待提交的 TransferOpGraph；任务已终态时返回 None（调用方据此跳过提交）。
        Raises:
            ValueError: 任务既非终态也非 READY（典型是 UNREADY 就提交，或重复提交）。
        Note:
            这里是 READY -> RUNNING 的唯一迁移点，提交动作必须走它，保证「已终态的
            图不会被二次投递」。
        """
        task = self.tasks[task_id]
        if task.is_completed():
            return None
        if task.status != TaskStatus.READY:
            raise ValueError(f"Task {task_id} status is {task.status}, cannot launch")
        task.status = TaskStatus.RUNNING
        if flexkv_logger.is_enabled_for(logging.DEBUG):
            flexkv_logger.debug(
                "[FlexKV-IO] operation=%s act=launch status=running "
                "flexkv_task_id=%d graph_id=%d",
                self._operation_name(task.task_type),
                task.task_id,
                task.graph.graph_id,
            )
        return task.graph

    def _release_task(self, task_id: int) -> None:
        """把任务从 tasks / graph_to_task 两张表里摘掉（真正删除任务的唯一入口）。

        调用时机很讲究：任务落终态后不能立刻删，因为 wait 还要从它身上取 return_mask
        和 status。所以 _mark_completed / _fail_task 里只在 request_returned 为真时
        才释放；wait 取完响应后自己也调一次。
        """
        if task_id not in self.tasks:
            return
        task = self.tasks[task_id]
        if task.graph is not None:
            self.graph_to_task.pop(task.graph.graph_id, None)
        self.tasks.pop(task_id, None)

    def _mark_completed(self, task_id: int) -> None:
        """整图完成：跑图级回调（含延迟上树）-> 落 COMPLETED/FAILED -> 收资源。

        顺序非常重要，不能颠倒：
          1. 先执行 task.callback —— 这就是 cache engine 的图级回调，其中
             _commit_deferred_insert 才把本次搬到的 block 挂上 radix tree。
          2. 再对 prefetch 收敛 return_mask（要读上一步的 publish_result，看看到底
             上树了多少），所以必须在回调之后。
          3. 最后按 transfer_failed 决定 COMPLETED 还是 FAILED，摘掉 graph_to_task
             映射、释放重资源；如果响应已被 wait 取走，直接回收任务。
        Note:
            回调抛异常不会中断流程，而是把任务标记为 transfer_failed（落到 FAILED），
            避免一个坏回调把整条等待链挂死。
        """
        task = self.tasks[task_id]
        if task.is_completed():
            return
        if task.callback:
            callbacks = (
                task.callback if isinstance(task.callback, list)
                else [task.callback]
            )
            for callback in callbacks:
                try:
                    callback()
                except Exception:
                    task.transfer_failed = True
                    flexkv_logger.error(
                        f"Transfer graph callback failed for task_id={task_id}",
                        exc_info=True,
                    )
        # Deferred commit (tree mount) has just run. Finalize return_mask from
        # the Full REMOTE2H success bitmap (how much remote this task pulled).
        # On fail we still finalize so outcome metrics (e.g. all_failed) land.
        if task.task_type == TaskType.PREFETCH:
            self._finalize_prefetch_return_mask(task)
        task.status = (
            TaskStatus.FAILED if task.transfer_failed else TaskStatus.COMPLETED)
        task.task_end_op_finished = True
        self._log_task_terminal(task, TaskStatus.COMPLETED)
        self.graph_to_task.pop(task.graph.graph_id, None)
        task.shed_heavy_resources()
        if task.request_returned:
            self._release_task(task_id)

    def _process_empty_graph(self, task_id: int) -> None:
        """空图（num_ops == 0）立即完成：没有任何 op，就不会有任何完成消息。

        空图意味着一点都没命中（GET）或一点都不用写（PUT），任务的生命周期在建完
        这一步就结束了，必须在这里直接推到终态，否则 wait 会一直空等到超时。
        """
        task = self.tasks[task_id]
        if task.graph is None:
            return
        if task.graph.num_ops == 0:
            self._mark_completed(task_id)

    def _get_completed_ops(self, timeout: Optional[float] = None) -> List[CompletedOp]:
        """N 路 handle 的完成聚合：把「每个 handle 各报一份」收敛成「一条」。

        为什么需要聚合：同一张图会被投递给 self.transfer_handles 里的每一个 handle
        （本机进程 + 可能的跨机 handle），所以同一件事会收到 N 份通知。只有第 N 份
        到达时，这条消息才被放进 results 交给上层，否则只累加计数并暂存合并结果。

        两类消息分别处理：
          - op_id == -1（图级终态）：按 graph_id 计数。达到 N 时，若任一 handle 失败
            就产出 failed_graph，并把该图残留的 per-op 计数器清掉（失败方不会把
            op 报全，留着只会泄漏；又因为每个 handle 的消息是 FIFO 的，第 N 个终态
            之后不会再有该图的消息，所以此刻清理是安全的）。
          - op_id >= 0（单个 op）：按 (graph_id, op_id) 计数，并用 _merge_completed_op
            逐路合并（block 位图做「与」）；达到 N 时产出合并结果。

        Args:
            timeout: 传给 handle.wait 的超时（秒）
        Returns:
            本轮真正「凑齐 N 份」的完成消息列表
        """
        results = []
        # Keep lightweight test/fallback managers created with ``__new__``
        # compatible with the pre-bitmap state shape.
        if not hasattr(self, "uncompleted_op_results"):
            self.uncompleted_op_results = {}
        for transfer_handle in self.transfer_handles:
            completed_ops = transfer_handle.wait(timeout)
            for completed_op in completed_ops:
                if completed_op.op_id == -1:
                    # Graph-level terminal message, completed OR failed. Every
                    # handle received the same graph, so the task terminates
                    # only after all of them have reported a terminal state --
                    # aborting on the first failure would recycle plan blocks
                    # a sibling engine is still writing into. Any failure
                    # among the N outcomes fails the graph.
                    graph_id = completed_op.graph_id
                    count, failed = self.uncompleted_graphs.get(graph_id, (0, False))
                    count += 1
                    failed = failed or completed_op.is_graph_failed()
                    if count == self.required_completed_count:
                        self.uncompleted_graphs.pop(graph_id, None)
                        if failed:
                            # A failed handle never finalizes some of the
                            # graph's ops, so their N-way per-op counters can
                            # never complete: purge them rather than leak.
                            # Safe because each handle's terminal message
                            # follows all its per-op messages (per-handle
                            # FIFO), so nothing can arrive for this graph
                            # after the Nth terminal and resurrect a counter.
                            stale = [key for key in self.uncompleted_ops
                                     if key[0] == graph_id]
                            for key in stale:
                                self.uncompleted_ops.pop(key, None)
                                self.uncompleted_op_results.pop(key, None)
                            results.append(CompletedOp.failed_graph(graph_id))
                        else:
                            results.append(completed_op)
                    else:
                        self.uncompleted_graphs[graph_id] = (count, failed)
                else:
                    op_key = (completed_op.graph_id, completed_op.op_id)
                    completed_count = self.uncompleted_ops.get(op_key, 0) + 1
                    aggregate = self._merge_completed_op(
                        self.uncompleted_op_results.get(op_key),
                        completed_op,
                    )
                    if completed_count == self.required_completed_count:
                        results.append(aggregate)
                        self.uncompleted_ops.pop(op_key, None)
                        self.uncompleted_op_results.pop(op_key, None)
                    else:
                        self.uncompleted_ops[op_key] = completed_count
                        self.uncompleted_op_results[op_key] = aggregate
        return results

    @staticmethod
    def _merge_completed_op(
        current: Optional[CompletedOp],
        incoming: CompletedOp,
    ) -> CompletedOp:
        """Combine multi-handle outcomes; a block succeeds only everywhere."""
        # 中文说明（多路合并规则）：
        #   - 第一次到达（current is None）：把 block_results 规范成与 num_blocks 等宽
        #     的元组，宽度对不上就整段判失败（fail-closed）。
        #   - 后续到达：两路的 block 位图逐块做「与」—— 只要有一路没搬成功，这个
        #     block 就算失败；位图缺失或宽度不一致的一路，整段按全 False 参与运算。
        #   - 标量字段：transfer_type 取先到的非空值，num_blocks / num_bytes 取最大值。
        # 为什么是「与」而不是「或」：上层把合并结果当作「数据已可用」的依据，
        # 只要有一路没写进去，这个 block 的内容就不可信。
        if current is None:
            if incoming.block_results is None:
                return incoming
            expected_blocks = incoming.num_blocks or len(incoming.block_results)
            block_results = (
                tuple(bool(result) for result in incoming.block_results)
                if len(incoming.block_results) == expected_blocks
                else (False,) * expected_blocks
            )
            return replace(
                incoming,
                num_blocks=expected_blocks,
                block_results=block_results,
            )

        block_results = None
        if (current.block_results is not None
                or incoming.block_results is not None):
            # Once one handle reports a bitmap, every handle must report the
            # same width. Missing or malformed data is a fail-closed result.
            expected_blocks = max(
                current.num_blocks,
                incoming.num_blocks,
                len(current.block_results or ()),
                len(incoming.block_results or ()),
            )

            def normalize(results: Optional[Tuple[bool, ...]]) -> Tuple[bool, ...]:
                if results is None or len(results) != expected_blocks:
                    return (False,) * expected_blocks
                return tuple(bool(result) for result in results)

            left = normalize(current.block_results)
            right = normalize(incoming.block_results)
            block_results = tuple(a and b for a, b in zip(left, right))
        return replace(
            current,
            transfer_type=current.transfer_type or incoming.transfer_type,
            num_blocks=max(
                current.num_blocks,
                incoming.num_blocks,
                len(block_results or ()),
            ),
            num_bytes=max(current.num_bytes, incoming.num_bytes),
            block_results=block_results,
        )

class KVTaskEngine(KVTaskManager):
    """面向用户的任务引擎（KVTaskManager 的子类），KVManager 直接持有的就是它。

    与父类的分工：
      - KVTaskManager（父类）：任务簿记。建任务、提交、完成聚合、失败/取消/回收，
        不关心「用户是怎么调进来的」。
      - KVTaskEngine（本类）：用户 API 层。把 KVManager 的 get/put/prefetch 翻译成
        任务生命周期动作，并提供 wait/try_wait/cancel/批处理；额外负责 tracer 埋点。
      - 这样切分的好处：状态机只有一份实现（父类），换接入框架时只需复用父类，
        不必连带 tracer 与模型配置。

    两种调用方式（本文件同时支撑）：
      1. 一阶段：get_async / put_async / prefetch_async —— 匹配完立即 _launch_task，
         任务建出来就是 READY（或 prefetch 的直接 READY），一行搞定。
      2. 两阶段：get_match / put_match 先只匹配、返回 (task_id, return_mask)，任务
         停在 UNREADY；调用方拿到 mask 决定调度后，再 launch_tasks(...) 回填真实
         GPU 槽位并提交。vLLM/sglang 的调度器需要「先知道命中多少再决定要不要这个
         请求」，就必须走两阶段。

    批处理：launch_tasks(as_batch=True) 会把多个同类任务的图融合成一张批图
    （merge_to_batch_kvtask），一次 IPC 提交，减少进程间开销；融合后子任务被摘掉，
    对外只剩一个 batch_id。
    """

    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 gpu_register_port: Optional[str] = None,
                 redis_meta: Optional[RedisMeta] = None,
                 event_collector: Optional[KVEventCollector] = None
                 ):
        super().__init__(model_config, cache_config, gpu_register_port, redis_meta, event_collector)
        # tracer 只做可观测性埋点（配置/请求/等待），不参与任何状态判定
        self.tracer = FlexKVTracer()
        self.tracer.trace_config(model_config, cache_config, gpu_layout=None)

    def get_async(self,
                  token_ids: np.ndarray,
                  slot_mapping: np.ndarray,
                  dp_client_id: int = 0,
                  token_mask: Optional[np.ndarray] = None,
                  task_id: int = -1,
                  namespace: Optional[List[str]] = None) -> Tuple[int, np.ndarray]:
        """一阶段 GET：匹配前缀 -> 立即提交传输 -> 返回 (task_id, return_mask)。

        与两阶段（get_match + launch_tasks）的区别：slot_mapping 是真实槽位，所以任务
        建出来就是 READY，这里紧接着 _launch_task 提交，没有等待槽位的中间态。适合
        「槽位已经分配好、不需要先知道命中量再决策」的调用方。

        Returns:
            (task_id, return_mask)：return_mask 是匹配结果（哪些 token 命中缓存），
            真实完成情况随后用 wait/try_wait 查询。
        """
        # self._sync_prefetch(token_ids, namespace)
        task_id, return_mask = self._get_match_impl(token_ids,
                                                    slot_mapping,
                                                    is_fake_slot_mapping=False,
                                                    token_mask=token_mask,
                                                    dp_client_id=dp_client_id,
                                                    task_id=task_id,
                                                    namespace=namespace)
        # trace get request
        self.tracer.trace_request(
            request_type="GET",
            request_id=task_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            token_mask=token_mask,
            dp_client_id=dp_client_id
        )
        self._launch_task(task_id)
        return task_id, return_mask

    def put_async(self,
                  token_ids: np.ndarray,
                  slot_mapping: np.ndarray,
                  dp_client_id: int = 0,
                  token_mask: Optional[np.ndarray] = None,
                  task_id: int = -1,
                  namespace: Optional[List[str]] = None) -> Tuple[int, np.ndarray]:
        """一阶段 PUT：匹配（算出哪些 token 需要写回）-> 立即提交 -> 返回 task_id。

        方向是 D2H（显存 -> 内存），后续的内存 -> SSD/远端写穿由数据面异步完成，
        所以 put 的延迟可以和后续计算重叠。
        """
        task_id, return_mask = self._put_match_impl(token_ids,
                                                    slot_mapping,
                                                    is_fake_slot_mapping=False,
                                                    token_mask=token_mask,
                                                    dp_client_id=dp_client_id,
                                                    task_id=task_id,
                                                    namespace=namespace)
        # trace put request
        self.tracer.trace_request(
            request_type="PUT",
            request_id=task_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            token_mask=token_mask,
            dp_client_id=dp_client_id
        )
        self._launch_task(task_id)
        return task_id, return_mask

    def _wait_impl(self,
                   task_ids: List[int],
                   timeout: float = 20.0,
                   completely: bool = False,
                   only_return_finished: bool = False,
                   ) -> Dict[int, KVResponse]:
        """wait / try_wait 的共同实现：轮询直到每个任务出结果。

        循环里对每个 task_id 依次判定（同一时刻只推进一个，先来的先返回）：
          不在 tasks 表 -> NOTFOUND（可能是 id 写错或已被回收）
          UNREADY       -> UNREADY（两阶段调用还没 launch，永远等不到）
          check_completed -> 组装 KVResponse，置 request_returned，终态则回收
          only_return_finished -> 直接跳出（try_wait 语义：没好就返回空）
          超时          -> TIMEOUT

        Args:
            timeout: 总超时（秒）。注意它是「所有任务共享的总预算」而不是每个任务一份
            completely: 是否必须等整图终态（见 check_completed）
            only_return_finished: True 表示非阻塞轮询一次即返回（try_wait）
        Note:
            每轮轮询都会 _update_tasks，也就是由调用方线程驱动整个状态机前进；
            没有任何后台线程在推进任务。
        """
        return_responses = {}
        start_time = time.time()
        is_timeout = timeout == 0.0

        self._update_tasks(timeout=0)

        for task_id in task_ids:
            nvtx_range = nvtx.start_range(message=f"KVTask.wait[{task_id}]", color="red")
            while True:
                if task_id not in self.tasks:
                    flexkv_logger.error(f"task_id {task_id} not submitted into flexKV")
                    return_responses[task_id] = KVResponse(
                        status=KVResponseStatus.NOTFOUND,
                        task_id=task_id,
                        return_mask=None
                    )
                    break
                elif self.tasks[task_id].status == TaskStatus.UNREADY:
                    flexkv_logger.warning(f"task_id {task_id} is unready")
                    return_responses[task_id] = KVResponse(
                        status=KVResponseStatus.UNREADY,
                        task_id=task_id,
                        return_mask=None
                    )
                    break
                elif self.check_completed(task_id, completely=completely):
                    task = self.tasks[task_id]
                    return_responses[task_id] = KVResponse(
                        status=convert_to_response_status(task.status),
                        task_id=task_id,
                        return_mask=task.return_mask
                    )
                    task.request_returned = True
                    if task.is_completed():
                        self._release_task(task_id)
                    break
                elif only_return_finished:
                    break
                elif time.time() - start_time > timeout:
                    is_timeout = True
                if is_timeout:
                    return_responses[task_id] = KVResponse(
                        status=KVResponseStatus.TIMEOUT,
                        task_id=task_id,
                        return_mask=None
                    )
                    break
                self._update_tasks(timeout=0.001)
            nvtx.end_range(nvtx_range)
        return return_responses

    def try_wait(self, task_ids: Union[int, List[int]]) -> Dict[int, KVResponse]:
        """非阻塞轮询：只返回「已经完成」的任务，没完成的直接从结果里缺席。

        适合推理主循环里每步顺手捞一次的场景（不阻塞调度）。
        """
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        nvtx.mark(f"try_wait task_ids: {task_ids}")
        # trace try_wait request
        self.tracer.trace_wait_request(
            wait_type="try_wait",
            task_ids=task_ids,
            timeout=None,  # try_wait doesn't have explicit timeout
            completely=False
        )
        return_responses = self._wait_impl(task_ids,
                                           completely=False,
                                           only_return_finished=True)
        return return_responses

    def wait(self,
             task_ids: Union[int, List[int]],
             timeout: float = 20.0,
             completely: bool = False) -> Dict[int, KVResponse]:
        """阻塞等待：直到所有任务出结果或超时，返回 task_id -> KVResponse。

        Args:
            completely: True 时要求整图终态（数据已落盘/上树）；False 时
                task_end_op 完成即可提前返回成功（数据已可用的快速返回）
        """
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        nvtx.push_range(f"wait task_ids: {task_ids}", color=get_nvtx_default_color())
        # trace wait request
        self.tracer.trace_wait_request(
            wait_type="wait",
            task_ids=task_ids,
            timeout=timeout,
            completely=completely
        )
        return_responses = self._wait_impl(task_ids, timeout, completely=completely)
        nvtx.pop_range()
        return return_responses

    def _sync_prefetch(self, token_ids: np.ndarray, namespace: Optional[List[str]] = None) -> None:
        """把「同一段 token 已在飞的预取任务」同步等完（当前默认未启用）。

        用途：get 之前如果发现这段前缀正在预取，就在这里等它上树，这样随后的匹配
        能直接命中。代价是阻塞，所以 get_async / get_match 里都把它注释掉了。
        """
        prefetch_task_id = self.prefetch_tasks.get(self._gen_prefetch_key(token_ids, namespace), None)
        if prefetch_task_id is not None:
            start_time = time.time()
            self.wait([prefetch_task_id], completely=True)
            end_time = time.time()
            flexkv_logger.debug(f"sync prefetch task {prefetch_task_id} cost {(end_time - start_time) * 1000} ms")

    def get_match(self,
                  token_ids: np.ndarray,
                  dp_client_id: int = 0,
                  token_mask: Optional[np.ndarray] = None,
                  cpu_only: bool = False,
                  task_id: int = -1,
                  namespace: Optional[List[str]] = None,
                  swa_aware: bool = False) -> Tuple[int, np.ndarray]:
        """Match a prefix and build the load graph; return (task_id, return_mask).

        With ``swa_aware=True`` the Full-KV transfer is clamped to the reusable
        SWA window (``usable = min(full_hit, swa_hit)``) from the same single radix
        match: past that window the Full-KV bytes would feed stale KV to the
        SWA-layer attention. The SWA window is the trailing block of the returned
        mask (page-granular), which the caller reads directly — there is no
        separate SWA mask. ``swa_aware=False`` (default) is the plain path,
        untouched.
        """
        # 中文补充（两阶段调用的第一阶段）：本方法只匹配、不提交。返回的 return_mask
        # 让调度器先知道「能命中多少 token」，据此决定要不要为这个请求分配 GPU 槽位；
        # 真正提交要等 launch_tasks —— 那时才回填真实 slot_mapping，任务从 UNREADY
        # 迁移到 READY 再到 RUNNING。这就是两阶段相比 get_async 的核心价值。
        nvtx.push_range(f"get match: task_id={task_id}", color=get_nvtx_default_color())
        # self._sync_prefetch(token_ids, namespace)
        # Flush pending D2H completions so set_ready callbacks run before
        # we check the radix tree.  Without this, blocks offloaded between
        # scheduler steps remain "not ready" until the next try_wait call,
        # which comes too late (after get_match).
        self._update_tasks(timeout=0)
        if token_mask is None:
            token_mask = np.ones_like(token_ids, dtype=bool)
        fake_slot_mapping = np.zeros_like(token_ids[token_mask])
        result_task_id, return_mask = self._get_match_impl(token_ids,
                                                           fake_slot_mapping,
                                                           is_fake_slot_mapping=True,
                                                           token_mask=token_mask,
                                                           dp_client_id=dp_client_id,
                                                           cpu_only=cpu_only,
                                                           task_id=task_id,
                                                           namespace=namespace,
                                                           swa_aware=swa_aware)
        # trace get match request
        self.tracer.trace_request(
            request_type="GET_MATCH",
            request_id=result_task_id,
            token_ids=token_ids,
            slot_mapping=fake_slot_mapping,
            token_mask=token_mask,
            dp_client_id=dp_client_id
        )
        nvtx.pop_range()
        return result_task_id, return_mask

    def _get_match_impl(self,
                  token_ids: np.ndarray,
                  slot_mapping: np.ndarray,
                  dp_client_id: int,
                  is_fake_slot_mapping: bool = False,
                  token_mask: Optional[np.ndarray] = None,
                  cpu_only: bool = False,
                  task_id: int = -1,
                  namespace: Optional[List[str]] = None,
                  swa_aware: bool = False) -> Tuple[int, np.ndarray]:
        """get_async 与 get_match 的共同实现：建 GET 任务并处理空图，返回 (id, mask)。

        Args:
            is_fake_slot_mapping: True 时任务建为 UNREADY（两阶段），等待后续回填槽位
            cpu_only: True 时把缓存策略换成 CPUONLY，只读内存不读 SSD/远端
        """
        if token_mask is None:
            token_mask = np.ones_like(token_ids)
        if task_id == -1:
            task_id = self._gen_task_id()
        temp_cache_strategy = DEFAULT_CACHE_STRATEGY
        if cpu_only:
            temp_cache_strategy = CPUONLY_CACHE_STRATEGY
        nvtx.push_range(f"get match: task_id={task_id}", color=get_nvtx_default_color())
        self.create_get_task(task_id=task_id,
                             token_ids=token_ids,
                             slot_mapping=slot_mapping,
                             dp_client_id=dp_client_id,
                             token_mask=token_mask,
                             is_fake_slot_mapping=is_fake_slot_mapping,
                             temp_cache_strategy=temp_cache_strategy,
                             namespace=namespace,
                             swa_aware=swa_aware)
        self._process_empty_graph(task_id)
        nvtx.pop_range()
        return task_id, self.tasks[task_id].return_mask

    def put_match(self,
                  token_ids: np.ndarray,
                  dp_client_id: int = 0,
                  token_mask: Optional[np.ndarray] = None,
                  task_id: int = -1,
                  namespace: Optional[List[str]] = None) -> Tuple[int, np.ndarray]:
        """PUT 的两阶段第一阶段：只匹配出「哪些 token 需要写回」，不提交传输。

        返回的 mask 标记尚不在外部存储中的 token；调用方据此决定调度后再 launch。
        """
        self._update_tasks(timeout=0)
        fake_slot_mapping = np.zeros_like(token_ids)
        result_task_id, return_mask = self._put_match_impl(token_ids,
                                                           fake_slot_mapping,
                                                           is_fake_slot_mapping=True,
                                                           token_mask=token_mask,
                                                           dp_client_id=dp_client_id,
                                                           task_id=task_id,
                                                           namespace=namespace)
        # trace put match request
        self.tracer.trace_request(
            request_type="PUT_MATCH",
            request_id=result_task_id,
            token_ids=token_ids,
            slot_mapping=fake_slot_mapping,
            token_mask=token_mask,
            dp_client_id=dp_client_id
        )
        return result_task_id, return_mask

    def _put_match_impl(self,
                        token_ids: np.ndarray,
                        slot_mapping: np.ndarray,
                        dp_client_id: int,
                        is_fake_slot_mapping: bool = False,
                        token_mask: Optional[np.ndarray] = None,
                        task_id: int = -1,
                        namespace: Optional[List[str]] = None) -> Tuple[int, np.ndarray]:
        """put_async 与 put_match 的共同实现：建 PUT 任务、处理空图，返回 (id, mask)。"""
        if token_mask is None:
            token_mask = np.ones_like(token_ids)
        if task_id == -1:
            task_id = self._gen_task_id()
        nvtx.push_range(f"put match: task_id={task_id}", color=get_nvtx_default_color())
        self.create_put_task(task_id=task_id,
                             token_ids=token_ids,
                             slot_mapping=slot_mapping,
                             dp_client_id=dp_client_id,
                             token_mask=token_mask,
                             is_fake_slot_mapping=is_fake_slot_mapping,
                             namespace=namespace)
        self._process_empty_graph(task_id)
        nvtx.pop_range()
        return task_id, self.tasks[task_id].return_mask

    def prefetch_async(self,
                       token_ids: np.ndarray,
                       dp_client_id: int = 0,
                       task_id: int = -1,
                       namespace: Optional[List[str]] = None,
                       swa_aware: bool = False) -> int:
        """Launch a prefetch task; return its task_id.

        The launch call is fire-and-forget: it publishes the plan and hands the
        graph to the transfer engine. Progress and usable-token accounting are
        polled later via ``try_wait``/``wait`` — ``KVResponse.return_mask`` is
        rewritten in ``_finalize_prefetch_return_mask`` to the reusable Full
        REMOTE2H length (clamped by deferred publish; for joint SWA only when
        Full+SWA both succeed, else 0). Callers that want that count should
        read ``sum(return_mask)`` on the response, not any launch-time value.
        Compute H2D length still comes from a subsequent local ``get_match``
        against the CPU tree.

        中文补充：预取是「发射后不管」的——本方法只负责把计划发布出去、把图交给数据
        面。它返回的 task_id 后面靠 try_wait/wait 轮询；真正可复用的长度要从响应里
        的 return_mask 读（完成时已被重写），不能相信发布时的值。
        """
        if task_id == -1:
            task_id = self._gen_task_id()
        nvtx.push_range(f"prefetch match: task_id={task_id}", color=get_nvtx_default_color())
        self.create_prefetch_task(task_id, token_ids, dp_client_id=dp_client_id, namespace=namespace, swa_aware=swa_aware)
        self._process_empty_graph(task_id)
        nvtx.pop_range()
        # trace prefetch async request
        self.tracer.trace_request(
            request_type="PREFETCH_ASYNC",
            request_id=task_id,
            token_ids=token_ids,
            slot_mapping=np.zeros_like(token_ids),
            token_mask=np.ones_like(token_ids),
            dp_client_id=dp_client_id
        )
        self._launch_task(task_id)
        return task_id

    def merge_to_batch_kvtask(self,
                              batch_id: int,
                              task_ids: List[int],
                              batch_task_type: TaskType,
                              layerwise_transfer: bool = False,
                              counter_id: int = 0) -> TransferOpGraph:
        """把多个同类任务的图融合成一张批图，并用一个批任务取代它们。

        融合的意义：N 个请求各自一张小图意味着 N 次进程间提交；合成一张大图后只有
        一次 IPC，摊薄了调度开销。融合由 common.transfer.merge_to_batch_graph 完成，
        它会把各子图的 op 合并、重新编号，并返回新的 task_end_op_id。

        副作用（调用方要留意）：子任务会被从 tasks 表里删掉，它们的 graph_id 映射也
        会摘除；对外只剩 batch_id 一个句柄，wait(batch_id) 就是等这一整批。
        """
        op_callback_dict = {}
        task_end_op_ids = []
        callbacks = []
        transfer_graphs = []
        return_masks = []
        expected_type = TaskType.GET if batch_task_type == TaskType.BATCH_GET else TaskType.PUT
        for task_id in task_ids:
            assert self.tasks[task_id].task_type == expected_type, \
                f"only {expected_type.value} task can be launched as {batch_task_type.value}"
            transfer_graph = self.check_task_ready(task_id)
            if transfer_graph is not None and transfer_graph.num_ops > 0:
                transfer_graphs.append(transfer_graph)
                op_callback_dict.update(self.tasks[task_id].op_callback_dict)
                task_end_op_ids.append(self.tasks[task_id].task_end_op_id)
                callbacks.append(self.tasks[task_id].callback)
                return_masks.append(self.tasks[task_id].return_mask)
        # When layerwise is on, SWA (+ optional C4 state sidecars) always folds
        # into the fused LAYERWISE op via launch_swa_h2d_layer_ /
        # launch_swa_mg_h2d_layer_.
        batch_task_graph, task_end_op_id, op_callback_dict = merge_to_batch_graph(
            batch_id,
            transfer_graphs,
            task_end_op_ids,
            op_callback_dict,
            layerwise_transfer,
            counter_id,
        )
        self.tasks[batch_id] = KVTask(
            task_id=batch_id,
            token_ids=np.concatenate([self.tasks[task_id].token_ids for task_id in task_ids]),
            slot_mapping=np.concatenate([self.tasks[task_id].slot_mapping for task_id in task_ids]),
            token_mask=np.concatenate([self.tasks[task_id].token_mask for task_id in task_ids]),
            task_type=batch_task_type,
            task_end_op_id=task_end_op_id,
            task_end_op_finished=False,
            status=TaskStatus.READY,
            graph=batch_task_graph,
            return_mask=return_masks,
            callback=callbacks,
            op_callback_dict=op_callback_dict,
        )
        self.graph_to_task[batch_task_graph.graph_id] = batch_id
        if flexkv_logger.is_enabled_for(logging.INFO):
            operation = self._operation_name(batch_task_type)
            flexkv_logger.info(
                "[FlexKV-IO] operation=%s act=merge status=ready direction=%s "
                "child_task_ids=%s flexkv_batch_task_id=%d graph_id=%d "
                "mode=%s graph_ops=%d",
                operation,
                "H2D" if operation == "get" else "D2H",
                ",".join(str(task_id) for task_id in task_ids),
                batch_id,
                batch_task_graph.graph_id,
                "layerwise" if layerwise_transfer else "no-layerwise",
                batch_task_graph.num_ops,
            )
        for task_id in task_ids:
            child_task = self.tasks[task_id]
            if child_task.graph is not None:
                self.graph_to_task.pop(child_task.graph.graph_id, None)
            self.tasks.pop(task_id, None)
        return batch_task_graph

    def launch_tasks(self,
                    task_ids: List[int],
                    slot_mappings: List[np.ndarray],
                    swa_slot_mappings: Optional[List[Optional[np.ndarray]]] = None,
                    as_batch: bool = False,
                    batch_id: int = -1,
                    layerwise_transfer: bool = False,
                    counter_id: int = 0) -> List[int]:
        """两阶段调用的提交入口：回填真实槽位 -> 可选融合成批图 -> 批量提交。

        Args:
            task_ids: 待提交的任务（通常来自 get_match / put_match）
            slot_mappings: 与 task_ids 一一对应的真实 GPU 槽位映射
            swa_slot_mappings: 可选的 SWA 槽位映射，无 SWA 时传 None
            as_batch: 是否融合成一张批图提交
            batch_id: 批任务 id，-1 则自动生成
            layerwise_transfer: 按层传输（需要环境开关打开，且只支持 GET）
            counter_id: 传给融合图的计数器 id（用于跨层同步）

        Returns:
            实际提交出去的 task_id 列表；成批时只含 batch_id 一项。
        Note:
            融合条件：任务数 > 1 或 layerwise，且 as_batch，且全部同类（全 GET 或
            全 PUT）。不满足就退化成逐个提交。
        """
        assert isinstance(slot_mappings[0], np.ndarray)
        # trace launch tasks
        self.tracer.trace_launch_tasks(task_ids, slot_mappings, as_batch)
        self.set_slot_mappings(task_ids, slot_mappings, swa_slot_mappings)

        # Batch optimization: collect all transfer graphs first
        nvtx_range = nvtx.start_range(message=f"KVTaskEngine.launch_tasks batch={len(task_ids)}", color="blue")

        all_get = all(self.tasks[tid].task_type == TaskType.GET for tid in task_ids)
        all_put = all(self.tasks[tid].task_type == TaskType.PUT for tid in task_ids)
        if (len(task_ids) > 1 or layerwise_transfer) and as_batch and (all_get or all_put):
            if batch_id == -1:
                batch_id = self._gen_task_id()
            if layerwise_transfer:
                if not GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer:
                    flexkv_logger.warning("layerwise transfer is not enabled")
                    layerwise_transfer = False
                elif not all_get:
                    flexkv_logger.warning("only support layerwise get")
                    layerwise_transfer = False
            batch_task_type = TaskType.BATCH_GET if all_get else TaskType.BATCH_PUT
            batch_task_graph = self.merge_to_batch_kvtask(
                batch_id, task_ids, batch_task_type, layerwise_transfer, counter_id
            )
            transfer_graphs = [batch_task_graph]
            self.tasks[batch_id].status = TaskStatus.RUNNING
            task_ids = [batch_id]
        else:
            transfer_graphs = []
            for task_id in task_ids:
                transfer_graph = self.check_task_ready(task_id)
                if transfer_graph is not None and transfer_graph.num_ops > 0:
                    transfer_graphs.append(transfer_graph)

        # Submit all graphs in batch to reduce IPC overhead
        if transfer_graphs:
            for transfer_handle in self.transfer_handles:
                transfer_handle.submit_batch(transfer_graphs)

        nvtx.end_range(nvtx_range)
        return task_ids

    def cancel_tasks(self, task_ids: Union[int, List[int]]) -> None:
        """取消一批任务（对外入口），逐个走 _cancel_task。

        典型场景：调度器决定放弃这个请求（抢占/重算），此时必须主动取消，否则
        UNREADY 任务持有的锁与暂存块会一直不释放。
        """
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        for task_id in task_ids:
            self._cancel_task(task_id)

    def _clear_cpu_cache(self) -> None:
        """只清 CPU 这一级的缓存（不动 SSD / 远端索引）。"""
        self.cache_engine.cpu_cache_engine.reset()

    def reset_cache(self) -> None:
        """Invalidate the cache across ALL tiers (CPU + SSD + remote): drop the
        whole prefix tree and return every block to the mempool.

        Used after a weight update (e.g. verl RL rollout) so that KV computed
        against stale weights is never reused.

        We do NOT drain or cancel in-flight transfers here. verl issues the
        reset at a rollout/weight-update boundary where no new generation
        requests are being served, so in practice no task is ongoing. If any
        task IS still in flight we only warn: resetting the radix tree +
        mempool is cheap and the stale-weight invalidation must not be blocked
        on transfer completion. This mirrors vLLM's own reset_encoder_cache /
        reset_mm_cache, which likewise only warn on has_unfinished_requests().

        中文补充（为什么在飞行中的任务被置为 CANCELLED 而不是等它跑完）：回调闭包
        直接持有 radix 节点指针和 mempool 块号。如果 reset 先释放了它们、而迟到的
        传输完成消息随后触发回调，就会出现「给已删除的节点解锁」或「把已释放的块
        再 free 一次」。所以这里先把在飞任务的 callback / op_callback_dict 清空、
        置 CANCELLED —— 它们在 _update_tasks 里会因为「CANCELLED 且无回调」被跳过。
        保留 graph_to_task 映射，让迟到的消息还能找到任务并打一条 warning。
        """
        ongoing = sum(1 for t in list(self.tasks.values()) if not t.is_completed())
        if ongoing:
            flexkv_logger.warning(
                f"reset_cache called while {ongoing} task(s) are still in flight; "
                f"resetting anyway. In-flight transfers may target blocks that are "
                f"being freed — ensure reset is issued at a quiesced boundary."
            )

        # Invalidate in-flight tasks' callbacks BEFORE dropping the cache. The
        # callback / op_callback_dict partials close over the exact radix nodes
        # and mempool blocks we are about to free; if a late transfer fired them
        # after reset they would unlock/set_ready a deleted node, or recycle a
        # block into the freshly-emptied mempool (which raises "already free").
        # Clear them here so the dispatch in _update_tasks has nothing to fire.
        # Note: reset_cache() runs on the same thread as the callback dispatch
        # (_update_tasks), so no lock is needed. We keep the graph_to_task
        # mapping so a late-completing op still resolves to its task and warns.
        for task_id, task in list(self.tasks.items()):
            if task.is_completed():
                continue  # already-fired callbacks are harmless
            task.callback = None
            task.op_callback_dict = {}
            task.status = TaskStatus.CANCELLED

        # Drop index (radix tree) + mempool on every tier. CRadixTreeIndex (C++)
        # and the pure-Python RadixTreeIndex both expose reset(); GlobalCacheEngine
        # fans out to whichever tier engines are enabled.
        self.cache_engine.reset()  # GlobalCacheEngine.reset()
