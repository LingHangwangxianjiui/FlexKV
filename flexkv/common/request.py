# ==============================================================================
# flexkv/common/request.py —— 控制面对外的"请求 / 响应"数据契约
# ------------------------------------------------------------------------------
# 职责：定义 FlexKV 与推理框架（vLLM / SGLang / TensorRT-LLM）之间交换的请求与
# 响应对象。本文件不含任何业务逻辑，纯粹是边界上的数据契约。
#
# 在主链路中的位置：
#
#   框架 adapter                FlexKV 控制面                      框架 adapter
#  (integration/*)  ──请求──▶  kvmanager / kvtask  ──响应──▶      (integration/*)
#                                     │
#                                     └── 本文件的 KVResponse / KVResponseStatus
#
#   * 上游（构造方）：flexkv/kvtask.py 在 _wait_impl 里构造 KVResponse；
#     flexkv/kvmanager.py 的 wait / try_wait 把它向上返回。
#   * 下游（消费方）：flexkv/server/client.py（server-client 模式下的远程返回）、
#     flexkv/server/request.py，以及 integration/vllm、integration/sglang、
#     integration/tensorrt_llm 三个 adapter（主要消费 KVResponseStatus）。
#   * 依赖的下游模块：numpy（mask 的存储格式）、torch（把 ndarray 转 Tensor）。
#
# 关键名字清单：
#   KVRequestType    —— 请求类型枚举：GET / PUT / SHUTDOWN。
#   KVRequest        —— 一次请求的入参包（token_ids / token_mask / slot_mapping）。
#   KVResponseStatus —— 任务终态枚举：SUCCESS / NOTFOUND / UNREADY / TIMEOUT /
#                       CANCELLED / FAILED。
#   KVResponse       —— 一次任务的返回结果，核心字段 return_mask 是逐 token
#                       的命中掩码。
#   KVResponse.get_mask —— 取第 idx 份 mask 并转成 torch.Tensor。
#
# 阅读提示 / 容易踩的坑：
#   1. 【重要】KVRequest 与 KVRequestType 目前在 flexkv/ 包内没有任何引用点，
#      真正被主链路使用的是 KVResponse 与 KVResponseStatus。另外
#      benchmarks/utils.py、benchmarks/dist_benchmark/utils.py 里存在**同名但
#      不同定义**的 KVRequest，不要与本文件的混淆。
#   2. KVResponse 用 task_id 而非 request_id 关联结果：一个上层请求可能被拆成
#      多个内部 task，返回时按 task_id 逐个给结果。
#   3. return_mask 有两种形态：单个 np.ndarray，或 List[np.ndarray]（多段/多
#      PP 阶段各一份）。get_mask(idx) 只支持后一种，传前者会断言失败。
#   4. return_mask 是逐 token 的布尔掩码，不是 block 掩码；上层通常按
#      sum(return_mask) 统计实际可用的 token 数。
# ==============================================================================

from dataclasses import dataclass
from enum import Enum
from typing import Union, List, Optional

import torch
import numpy as np


class KVRequestType(Enum):
    """请求类型枚举（读路径 / 写路径 / 关停指令）。

    GET      —— 把外部缓存里的 KV 取回 GPU（读路径）
    PUT      —— 把 GPU 上的 KV 写回外部缓存（写路径）
    SHUTDOWN —— 关停指令，用于通知控制面优雅退出

    注意：本枚举目前只在 benchmarks / tests 的回放脚本里出现，flexkv/ 主链路
    并没有按它做分支派发（读/写在 kvtask.py 里是 wait/put 两个独立入口）。
    """
    GET = "get"
    PUT = "put"
    SHUTDOWN = "shutdown"

@dataclass
class KVRequest:
    """一次 KV 请求的入参包。

    字段说明：
      request_type      —— 见 KVRequestType。
      request_id        —— 上层（框架 / benchmark）侧的请求编号。
      token_ids         —— 本次请求的 token 序列（一维）。
      token_mask        —— 与 token_ids 等长的掩码，标记哪些 token 参与本次操作。
      slot_mapping      —— 推理引擎分配的 GPU 显存槽位，即 token -> block slot
                           的映射；这是 CPU 缓存与 GPU 显存对齐的关键输入。
      layer_granularity —— 分层传输的粒度，即一次搬运多少层；-1 表示不按层切分
                           （整块一次性搬）。最终会透传给 C++ 侧
                           tp_transfer_thread_group 的传输核函数。
      dp_id             —— DP 分片编号，用于多 DP 实例下路由到正确的 task 引擎。
    """
    request_type: KVRequestType
    request_id: int
    token_ids: torch.Tensor
    token_mask: torch.Tensor
    slot_mapping: torch.Tensor
    layer_granularity: int = -1
    dp_id: int = 0


class KVResponseStatus(Enum):
    """任务终态枚举，由 kvtask.py 的 convert_to_response_status 从内部
    TaskStatus 转换而来。

    SUCCESS   —— 命中并搬运完成。
    NOTFOUND  —— task_id 不在当前引擎里（传错 id，或任务已被回收）。
    UNREADY   —— 任务还在排队，尚未进入可执行状态。
    TIMEOUT   —— 等待超时，任务仍在跑，并不代表失败。
    CANCELLED —— 被上层取消（如请求被抢占）。
    FAILED    —— 传输本身出错。

    坑：TIMEOUT 与 UNREADY 都属于"还没结果"，上层可以重试等待；只有 FAILED /
    CANCELLED / NOTFOUND 才是终态。
    """
    SUCCESS = "success"
    NOTFOUND = "not_found"
    UNREADY = "unready"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    FAILED = "failed"

@dataclass
class KVResponse:
    """一次任务的结果。

      status      —— 见 KVResponseStatus。
      task_id     —— 内部任务号（不是上层 request_id），与 submit 时返回的 id
                     对应。
      return_mask —— 逐 token 的命中掩码：为 True 的位置表示该 token 的 KV 已
                     经就绪可用。可能是单个 np.ndarray，也可能是
                     List[np.ndarray]（多段 / 多 PP 阶段各一份）。
                     非 SUCCESS 状态下恒为 None。
    """
    status: KVResponseStatus
    task_id: int
    return_mask: Optional[Union[np.ndarray, List[np.ndarray]]]

    def get_mask(self, idx: int) -> torch.Tensor:
        """取第 idx 份 mask 并转成 torch.Tensor。

        只支持 return_mask 为 List[np.ndarray] 的形态，单 ndarray 形态会直接
        断言失败——调用前先确认自己的 return_mask 是哪一种。
        """
        assert self.return_mask is not None and isinstance(self.return_mask, list), "return_mask must be a list of np.ndarray"
        assert idx < len(self.return_mask), "idx out of range"
        return torch.from_numpy(self.return_mask[idx])
