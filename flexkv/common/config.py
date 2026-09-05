# ==============================================================================
# flexkv/common/config.py —— 全项目的配置中心（模型 / 缓存 / 用户三套配置 + 加载入口）
# ------------------------------------------------------------------------------
# 职责：定义 FlexKV 所有可调参数的数据结构，并提供"环境变量 / JSON / YAML -> 配置对象"
# 的加载与换算入口。本文件不做任何数据搬运，也不持有运行时状态；唯一的例外是模块级
# 单例 GLOBAL_CONFIG_FROM_ENV，它在 import 本模块时就把 FLEXKV_* 环境变量读完了。
#
# 在架构中的位置（位于三层之外，被三层共同依赖）：
#
#     推理框架 adapter ──▶ UserConfig ──┐
#     环境变量 / 配置文件 ──▶ UserConfig ─┼─▶ update_default_config_from_user_config()
#                                        │                │
#                          ModelConfig ──┘                ▼
#                          RankInfo ────────────────▶ CacheConfig（GB 换算成 block 数）
#                                                          │
#        ┌────────────────┬───────────────────┬────────────┴────────┐
#        ▼                ▼                   ▼                     ▼
#   控制面 cache_engine  数据面 transfer   存储层 storage         C++ c_ext
#
# 三套配置的分工（名字很像但生命周期完全不同，务必分清）：
#   * ModelConfig —— 描述"模型长什么样 + 怎么并行"（层数、KV head、TP/PP/DP/CP、
#     节点数）。由框架 adapter 填，freeze() 之后不可再改。
#   * CacheConfig —— 描述"缓存池开多大、走哪条物理通路"，是本文件最该细读的部分。
#     enable_cpu / enable_ssd / enable_gds / enable_remote / enable_p2p_* 等开关
#     直接决定 cache_engine 造出的 DAG 里会出现哪些 TransferType。
#   * UserConfig  —— 面向用户的"人性化"入参（以 GB 为单位、以开关为单位）。
#     update_default_config_from_user_config() 负责换算并回填到 CacheConfig。
#   * RankInfo    —— 某个 rank 在 (tp, pp, dp, cp) 网格中的坐标，以及由它派生的
#     local_rank / dp_client_id 等路由标签。frozen dataclass。
#
# 关键设计取舍 / 容易踩的坑：
#   1. 【最重要】"Python 配置开关" 与 "C++ 编译宏" 是两回事，必须同时满足才真的
#      走某条通路。Python 侧开关（CacheConfig.enable_gds / enable_p2p_cpu /
#      enable_p2p_ssd / enable_3rd_remote 等）只决定"是否尝试走这条路"；真正的实现
#      在 C++ 扩展里，由编译期宏控制：FLEXKV_ENABLE_GDS / FLEXKV_ENABLE_P2P /
#      FLEXKV_ENABLE_CFS / FLEXKV_ENABLE_NVCOMP，**默认全部为 0（关闭）**，需编译
#      前 export FLEXKV_ENABLE_XXX=1（见 setup.py:300-304）。只开开关不开宏，通常
#      表现为 import 期 AttributeError 或运行期 "Unsupported transfer type"。
#   2. GLOBAL_CONFIG_FROM_ENV 是模块级单例，import 本模块时即固化所有环境变量。
#      import 之后再改 os.environ 是无效的；运行时要覆盖只能 setattr（
#      update_default_config_from_user_config 末尾的 override_ 机制正是这么做的）。
#   3. 单位陷阱：UserConfig 用 GB，CacheConfig 用 block 数。GB -> block 的换算依赖
#      block_size_in_bytes，而它又依赖 model_config.layer_groups。异构模型（DSv4）
#      的 layer_groups 常在 freeze() 之后才补齐，所以换算要延后到
#      recompute_cache_block_counts() 再算一遍（靠 _user_*_cache_gb 记住原始 GB）。
#   4. ModelConfig 有 freeze 机制：freeze() 后任何字段赋值都抛 AttributeError，
#      以此强制所有字段在 post_init_from_*() 里一次性填好。layer_groups 是唯一
#      例外（允许补登记，见 __setattr__）。
#   5. CacheConfig.__post_init__ 里的 enable_kv_sharing / enable_remote 是**派生**
#      字段，由 enable_p2p_* / enable_3rd_remote / use_mooncake_store_backend 推导。
#      不要在外面直接赋值，否则会被覆盖。
#
# 建议阅读顺序：
#   LayerGroupSpec / LayerMemberMap（异构 KV）-> ModelConfig（freeze 与派生拓扑）
#   -> RankInfo -> SWAPoolConfig -> CacheConfig（重点看各类 enable_* 开关）
#   -> GLOBAL_CONFIG_FROM_ENV -> UserConfig -> load_user_config_from_env / from_file
#   -> update_default_config_from_user_config（GB 换算回填的主入口）
# ==============================================================================

import os
import json
import yaml
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from functools import cached_property
from typing import Optional, List, Tuple, Union, Dict, Any
from argparse import Namespace
import copy

import torch

from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.debug import flexkv_logger


def assert_mooncake_prefetch_ready(cache_config, prefetch_enabled: bool) -> None:
    """M0 gate: mooncake REMOTE2H is prefetch-only.

    Serving stacks must enable prefetch (typically via enable_remote, which
    mooncake already implies) before compute GET. Call from connectors after
    computing ``prefetch_enabled``.

    中文要点：Mooncake store 后端目前只支持"预取"语义——REMOTE2H 必须在计算发起
    GET **之前**完成，不能在 GET 的同步路径上临时去远端拉数据。因此接入方在算完
    prefetch_enabled 之后要调用本函数做一次门禁校验，否则延迟会直接暴露在首 token
    上。这是 M0 阶段的临时约束，后续若 Mooncake 支持同步 GET 可移除。
    """
    if (getattr(cache_config, "use_mooncake_store_backend", False)
            and not prefetch_enabled):
        raise RuntimeError(
            "use_mooncake_store_backend requires prefetch. "
            "So REMOTE2H runs in prefetch before compute GET."
        )


@dataclass
class LayerGroupSpec:
    """One group of layers sharing the same KV cache shape.

    ``layer_indices[k]`` is the *original layer id* (index into the full
    ``model_config.num_layers`` range) that this group's k-th local layer
    (local_id = k) maps to.  Multiple groups MAY share the same original
    layer id — this expresses heterogeneous KV at one transformer block
    (e.g. DSv4: main KV bf16 and indexer uint8 both attached to the same
    layer).

    Invariants (enforced by ``ModelConfig._validate_layer_groups``):

    * ``num_layers == len(layer_indices)``
    * ``layer_indices`` has no internal duplicates
    * every element of ``layer_indices`` is in ``[0, model_config.num_layers)``
    * original layers may be omitted when they have no cached state, and may
      appear in multiple groups when they have multiple cache members

    中文要点：这是为**异构 KV** 准备的。同构模型（所有层形状一致）不需要它，
    ModelConfig.layer_groups 保持 None 即可，走的仍是传统的单组路径。
    典型场景是 DeepSeek V4：同一个 transformer block 上既挂 bf16 的主 KV，
    又挂 uint8 的 indexer，两者形状/dtype 不同，于是拆成两个 group，
    而 layer_indices 里出现相同的原始层号（一个原始层 -> 多个成员）。
    """
    num_layers: int
    num_kv_heads: int
    head_size: int
    layer_indices: List[int]
    # Per-group storage dtype. None = inherit ModelConfig.dtype.
    # Indexer groups use a different dtype (e.g. fp8/uint8) than main KV (bf16).
    dtype: Optional[torch.dtype] = None
    # Per-group KV compression along the tokens_per_block dimension. The CPU/SSD
    # block stores ``tokens_per_block // compress_ratio`` tokens worth of data
    # for this group (the GPU tensor sglang allocates is already compressed to
    # the same shrunk shape). ``1`` = uncompressed (legacy behavior). Used by
    # DSv4-style models where different layer roles compress at different ratios
    # (e.g. CSA at 4x, HCA at 128x). ``tokens_per_block % compress_ratio == 0``
    # is enforced when the KVCacheLayout is built.
    compress_ratio: int = 1


@dataclass(frozen=True)
class LayerMemberMap:
    """Dense mapping from original layer id -> tuple of (group_idx, local_layer_id).

    ``members[i]`` is the tuple of ``(group_idx, local_layer_id)`` pairs for
    original layer ``i``, ordered by ascending ``group_idx`` (main KV before
    auxiliary groups like indexer).

    Examples:

      Single group (uniform model): every layer has 1 member.
        members = (((0, 0),), ((0, 1),), ..., ((0, N-1),))

      DSv4 (main + indexer share every layer): every layer has 2 members.
        members = (((0, 0), (1, 0)), ((0, 1), (1, 1)), ...)

      Alternating partition: each layer belongs to exactly one group.
        members = (((0, 0),), ((1, 0),), ((0, 1),), ((1, 1),), ...)
    """
    members: Tuple[Tuple[Tuple[int, int], ...], ...]

    # 中文要点：这是"原始层 id -> 若干缓存成员"的稠密索引（CSR 风格），用来在分层
    # 传输时把一层展开成多次搬运。members[i] 为空表示该原始层不缓存（如 DSv4 的
    # 前两层），这是合法的——并不要求所有原始层都被覆盖。
    # 类被声明为 frozen=True，因为它是可哈希的缓存键，构造后不允许再改。

    @property
    def num_original_layers(self) -> int:
        return len(self.members)

    @property
    def total_members(self) -> int:
        return sum(len(m) for m in self.members)

    def members_of(self, original_layer_id: int) -> Tuple[Tuple[int, int], ...]:
        """Return ((group_idx, local_id), ...) for one original layer."""
        return self.members[original_layer_id]


def build_layer_member_map(
    layer_groups: List[LayerGroupSpec],
    num_original_layers: int,
) -> LayerMemberMap:
    """Construct a ``LayerMemberMap`` from ``layer_groups``.

    Members within one original layer are ordered by ascending ``group_idx``
    (i.e., the order in which groups appear in ``model_config.layer_groups``).
    By convention main KV should be group 0 so that on the stream it always
    fires before any auxiliary group (e.g., indexer).

    中文要点：同一原始层内成员的排序约定为 group_idx 升序，这不只是美观问题——
    分层传输时成员按这个顺序在同一条 CUDA stream 上依次发射，主 KV 排前面才能
    先落地、先可用。所以**主 KV 必须是 group 0**。
    """
    buckets: List[List[Tuple[int, int]]] = [[] for _ in range(num_original_layers)]
    for gi, g in enumerate(layer_groups):
        for local_id, orig in enumerate(g.layer_indices):
            buckets[orig].append((gi, local_id))
    for b in buckets:
        b.sort(key=lambda x: x[0])
    return LayerMemberMap(members=tuple(tuple(b) for b in buckets))


@dataclass
class ModelConfig:
    """模型与并行拓扑的静态描述：模型长什么样、用多少卡、卡怎么组织。

    生命周期（这点决定了你能在什么时候改它）：
      框架 adapter 构造 -> post_init_from_*() 里填齐所有字段 -> freeze() 加锁。
      freeze() 之后任何字段赋值都会抛 AttributeError（layer_groups 除外），
      以此保证运行时读到的拓扑是自洽且不可变的。

    字段大致分四组：
      * 模型形状    —— num_layers / num_kv_heads / head_size / kv_dim / dtype。
      * 并行度      —— tp_size / pp_size / dp_size / cp_size（全局值，所有 rank 相同）。
                      注意 tp_size 是"归一化后的 attention TP"，cp_size 单独表示。
      * 拓扑与部署  —— nnodes / master_host / master_ports / instance_num，以及
                      TRT-LLM 特有的子进程开关。
      * 异构 KV     —— layer_groups（None 表示同构模型）。

    派生值一律用 @property（total_gpus / gpus_per_node / nnodes_per_pp_rank /
    tp_size_per_node / effective_tp_size / token_size_in_bytes ...），**不是字段**，
    所以既不能被赋值，也不会被 dataclass 的 __eq__ / repr 带上。

    坑：
      1. 默认值全为 1 / bfloat16，是"能构造出一个合法对象"的占位值，**不代表任何
         真实模型**。真实值必须由 adapter 显式填入。
      2. cp_size 与 attn_cp_size 是同一语义的两份镜像（后者为兼容 SGLang 老调用方）。
         freeze() 会做双向对齐：一方为 1 时以另一方为准，两方非 1 且不相等则报错。
      3. token_size_in_bytes 与 bytes_per_token_per_layer 只在同构模型下有意义；
         设了 layer_groups 之后请一律用 token_size_in_bytes（它会按组求和）。
    """
    num_layers: int = 1
    num_kv_heads: int = 1
    head_size: int = 1
    kv_dim: int = 2
    dtype: torch.dtype = torch.bfloat16

    # ------------------------------------------------------------------
    # Parallelism sizes (global, identical for every rank)
    # ------------------------------------------------------------------
    # 中文：这四个是**全局**并行度，所有 rank 看到的值都一样。想知道"本节点有几张卡"
    # "一个 TP group 跨几个节点"，请用下面的 gpus_per_node / nnodes_per_pp_rank 等
    # 派生 property，不要在这里手动除以 nnodes（整除方向由 property 统一保证）。
    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1

    # ------------------------------------------------------------------
    # Attention-level parallel configs
    # ------------------------------------------------------------------
    # Compatibility flag retained for framework adapters. ``tp_size`` is the
    # normalized attention TP size and ``cp_size`` is represented separately.
    enable_dp_attention: bool = False
    # Compatibility mirror of cp_size for existing SGLang connector callers.
    attn_cp_size: int = 1
    # cp_size: context-parallel size (global), default 1.
    cp_size: int = 1
    # 中文：cp_size 与 attn_cp_size 是同一件事的两份镜像（后者兼容 SGLang 老调用方）。
    # 约定——两者都为 1 或相等最安全；一方为 1、另一方非 1 时 freeze() 会以非 1 的
    # 那个为准补齐；两者都非 1 且不相等则直接报错。见 ModelConfig.freeze()。

    # ------------------------------------------------------------------
    # Topology configs (global)
    # ------------------------------------------------------------------
    # nnodes: number of physical machines spanned by one replica
    nnodes: int = 1

    # Multi-node bootstrap: master node's IP for TransferManager rendezvous.
    # Bare host (no scheme); zmq sockets prepend ``tcp://`` at connect time.
    # Set this from the framework's own launch config (e.g. sglang's
    # ``--dist-init-addr``, TRT-LLM's launch script) so runtime layers
    # (TransferManager / KVTaskManager) never need to read env vars.
    master_host: str = "localhost"

    # Master endpoint ports (command, result, query). Set via integration
    # adapter when the launch script exposes a different port triple.
    master_ports: Tuple[str, str, str] = ("5556", "5557", "5558")

    # Whether KVTaskManager should run TransferManagerOnRemote in a
    # subprocess (currently only used by TRT-LLM to avoid MPI conflicts).
    # Set by the TRT-LLM adapter; default ``False`` for sglang/vllm.
    use_trtllm_subprocess: bool = False

    # Endpoint of that TRT-LLM subprocess TransferManagerOnRemote.
    # Only consulted when ``use_trtllm_subprocess`` is True.
    trtllm_subprocess_host: str = "localhost"
    trtllm_subprocess_ports: Tuple[str, str, str] = ("6667", "6668", "6669")

    # ------------------------------------------------------------------
    # Multi-instance deployment
    # ------------------------------------------------------------------
    instance_num: int = 1

    # ------------------------------------------------------------------
    # Heterogeneous KV cache layers (including Indexer-as-group)
    # ------------------------------------------------------------------
    # When None, all layers share the same (num_kv_heads, head_size, dtype).
    # When set, each group carries its own shape (and optionally its own dtype),
    # and token_size_in_bytes/num_cpu_blocks are computed by summing across groups.
    layer_groups: Optional[List[LayerGroupSpec]] = None

    # ------------------------------------------------------------------
    # Freeze mechanism: after post_init, ModelConfig must not be mutated
    # ------------------------------------------------------------------
    _frozen: bool = field(default=False, init=False, repr=False)

    def freeze(self) -> None:
        """Lock the config so that any subsequent __setattr__ raises an error.

        中文要点：这是"配置就绪"的**提交点**，而不是普通的 setter。调用它意味着
        "所有字段都填完了，从此刻起把它们当常量读"。运行时任何试图改 ModelConfig
        的代码都会在这里之后的第一次赋值上炸掉，从而把错误暴露在初始化阶段。

        它会顺带做三件事：cp_size/attn_cp_size 双向对齐、拓扑自洽校验、
        layer_groups 不变量校验。任何一项不通过都在启动时报 ValueError，
        不会等到跑推理时才出问题。

        唯一的后门：layer_groups 允许在 freeze() 之后补登记（见 __setattr__）。
        """
        # ---- CP 双镜像对齐：一方为 1 时以另一方为准，都不为 1 且不相等则报错 ----
        if self.cp_size == 1 and self.attn_cp_size != 1:
            self.cp_size = self.attn_cp_size
        elif self.attn_cp_size == 1 and self.cp_size != 1:
            self.attn_cp_size = self.cp_size
        elif self.cp_size != self.attn_cp_size:
            raise ValueError(
                f"[ModelConfig] cp_size={self.cp_size} and "
                f"attn_cp_size={self.attn_cp_size} disagree"
            )
        # ---- Topology validation ----
        # 中文：目前只支持"一个 PP stage 最多跨 2 个节点"的 TP（即 2-node TP）。
        # 这是 C++ 传输侧的硬约束（tp_transfer_thread_group 只按 2 路分段处理），
        # 与其让它在传输时静默算错偏移，不如在这里启动期就报错。
        if self.total_gpus % self.nnodes != 0:
            raise ValueError(
                f"[ModelConfig] cannot derive gpus_per_node: "
                f"total_gpus={self.total_gpus} not divisible by nnodes={self.nnodes}"
            )
        if self.nnodes_per_pp_rank > 2:
            raise ValueError(
                f"[ModelConfig] only support 2-nodes TP for now, but got "
                f"nnodes_per_pp_rank={self.nnodes_per_pp_rank} "
                f"(tp_size={self.tp_size}, gpus_per_node={self.gpus_per_node})"
            )
        if self.instance_num < 1:
            raise ValueError(
                f"[ModelConfig] instance_num must be >= 1, got {self.instance_num}"
            )

        # ---- LayerGroup invariants ----
        self._validate_layer_groups()

        object.__setattr__(self, '_frozen', True)

    def _validate_layer_groups(self) -> None:
        """Validate ``layer_groups`` against ``num_layers``.

        No-op when ``layer_groups`` is None (uniform model). Enforces:

        * ``g.num_layers == len(g.layer_indices)`` for every group.
        * No internal duplicate ``layer_indices`` within one group.
        * Every ``layer_indices`` entry lies in ``[0, num_layers)``.
        * ``g.compress_ratio >= 1`` for every group.

        Uncached layers (e.g. DSv4 layers 0/1 with ``compress_ratio == 0`` at
        the model level) simply do not appear in any group's ``layer_indices``
        and produce empty member lists in the resulting :class:`LayerMemberMap`;
        the union is therefore *not* required to cover every original layer.

        中文要点：校验的是"组内自洽"，不是"全局覆盖"。允许某些原始层不出现在任何
        group 里（不缓存的层），也允许同一个原始层出现在多个 group 里（一层多成员）。
        所以不要指望 layer_member_map 能覆盖 [0, num_layers) 的每一层。
        """
        if not self.layer_groups:
            return
        N = self.num_layers
        for gi, g in enumerate(self.layer_groups):
            if g.num_layers != len(g.layer_indices):
                raise ValueError(
                    f"[ModelConfig] layer_groups[{gi}].num_layers={g.num_layers} "
                    f"does not match len(layer_indices)={len(g.layer_indices)}"
                )
            if len(set(g.layer_indices)) != len(g.layer_indices):
                raise ValueError(
                    f"[ModelConfig] layer_groups[{gi}].layer_indices has duplicates: "
                    f"{g.layer_indices}"
                )
            if g.compress_ratio < 1:
                raise ValueError(
                    f"[ModelConfig] layer_groups[{gi}].compress_ratio must be >= 1, "
                    f"got {g.compress_ratio}"
                )
            for orig in g.layer_indices:
                if not 0 <= orig < N:
                    raise ValueError(
                        f"[ModelConfig] layer_groups[{gi}] has out-of-range "
                        f"layer index {orig} (must be in [0, {N}))"
                    )

    @cached_property
    def layer_member_map(self) -> Optional[LayerMemberMap]:
        """CSR mapping from original layer id -> [(group_idx, local_id), ...].

        Returns ``None`` when ``layer_groups`` is not set (uniform model — the
        layerwise transfer path then uses the legacy single-group code path).
        Computed once on first access and cached in ``__dict__`` via
        ``functools.cached_property`` (write bypasses the frozen
        ``__setattr__``).

        中文要点：返回 None 是**有意的**信号，表示"同构模型，走单组老路径"。调用方
        必须先判空再决定要不要按组展开搬运。
        这里用 cached_property 而不是普通 property，是因为 ModelConfig 已 frozen，
        普通 property 里的赋值会走 __setattr__ 被拦——cached_property 直接写
        __dict__，绕过冻结。
        """
        if not self.layer_groups:
            return None
        return build_layer_member_map(self.layer_groups, self.num_layers)

    def __setattr__(self, name: str, value) -> None:
        # 中文：冻结机制的实现点。三个放行条件，其余一律拦下：
        #   1) _frozen 自身（内部标志位，永远可写）
        #   2) layer_groups（后门，见下）
        #   3) 尚未 freeze
        if name == '_frozen':
            return object.__setattr__(self, name, value)
        # ``layer_groups`` is a derived field that is sometimes discovered late
        # (e.g. DeepSeek V4 sub-pool layout is only known once SGLang has built
        # the GPU KV pools, which happens after FlexKVConfig.from_env()/post_init
        # have already called freeze()). Allow late assignment of this field
        # specifically so multi-group registration paths work.
        if name == 'layer_groups':
            return object.__setattr__(self, name, value)
        if getattr(self, '_frozen', False):
            raise AttributeError(
                f"ModelConfig is frozen — cannot set '{name}'. "
                f"All primitive fields must be set during post_init_from_*(), "
                f"after which freeze() is called.  Derived fields (effective_tp_size, "
                f"tp_size_per_node, cp_size_per_node, nnodes_per_pp_rank) are @property "
                f"and cannot be set at all."
            )
        object.__setattr__(self, name, value)

    # ------------------------------------------------------------------
    # Derived topology properties
    # ------------------------------------------------------------------
    # 中文：下面全是 @property，**不是 dataclass 字段**——不参与 __init__ / __eq__ /
    # 序列化，也不能被赋值。它们把"全局并行度"翻译成"每个节点 / 每个 PP stage 上的
    # 实际布局"，是存储层算显存切片、传输层算 block 偏移的主要依据。
    # 记忆口诀：带 _per_node 后缀的 = 全局值 ÷ 该维度跨的节点数（至少为 1）。
    @property
    def total_gpus(self) -> int:
        """Total GPU worker registration slots across all nodes for one FlexKV instance.

        Unified formula: dp_size × tp_size × cp_size × pp_size.

        中文：一个 FlexKV 实例要注册的 GPU worker 总数，也是 TransferManager 等待
        的注册槽位数。它决定 zmq socket 数量、worker 进程数量，配错会表现为
        "等待注册超时"而非显式的数量不匹配。
        """
        return self.dp_size * self.tp_size * self.cp_size * self.pp_size

    @property
    def total_clients(self) -> int:
        """Total number of DPClient endpoints across all instances."""
        return self.instance_num * self.dp_size

    @property
    def gpus_per_node(self) -> int:
        """GPU worker registration slots on this node (across all DP shards, PP stages and TP groups)."""
        return self.total_gpus // self.nnodes

    @property
    def nnodes_per_pp_rank(self) -> int:
        """Number of nodes spanned by one PP stage.

        中文：一个 PP stage 横跨几个节点。它是 "TP 是否跨节点" 的判据——值为 1 表示
        一个 TP group 完全落在单节点内（走 NVLink，快）；大于 1 则是多节点 TP（走
        网卡，慢，且目前最多支持 2）。tp_size_per_node 就是靠它把 tp_size 拆到节点内。
        """
        return max(self.nnodes // self.pp_size, 1)

    @property
    def nnodes_per_tp_group(self) -> int:
        """Number of nodes spanned by one TP group."""
        return self.nnodes_per_pp_rank

    @property
    def tp_size_per_node(self) -> int:
        """Number of TP ranks on this node within one TP group."""
        return max(1, self.tp_size // self.nnodes_per_pp_rank)

    @property
    def attn_dp_size(self) -> int:
        """Attention-level DP size (= dp_size when enable_dp_attention else 1)."""
        return max(1, self.dp_size) if self.enable_dp_attention else 1

    @property
    def attn_tp_size(self) -> int:
        """Compatibility alias for the normalized attention TP size."""
        return max(1, self.tp_size)

    @property
    def attn_tp_size_per_node(self) -> int:
        """Attention-level TP size per node."""
        return self.tp_size_per_node

    @property
    def attn_cp_size_per_node(self) -> int:
        """Compatibility alias for per-node context parallel size."""
        return self.cp_size_per_node

    @property
    def cp_size_per_node(self) -> int:
        """CP size on this node for a single PP stage.

        Used for multi-node scenarios where the CP group spans multiple nodes.
        """
        return max(1, self.cp_size // self.nnodes_per_pp_rank)

    @property
    def effective_tp_size(self) -> int:
        """Number of CPU block slices = tp_size × cp_size.

        中文：这是**数据面**的分片数——一个逻辑 KV block 在 CPU/SSD 上被切成多少片
        （每个 (cp_rank, tp_rank) 各持一片）。注意它和 tp_size 不是一回事：TP 与 CP
        都会产生分片，所以 CPU 侧的容量规划和偏移计算一律用这个值。
        与之配对的是 RankInfo.effective_tp_rank（分片编号）。
        """
        return max(1, self.tp_size) * max(1, self.cp_size)

    @property
    def effective_tp_size_per_node(self) -> int:
        """Per-node counterpart of :pyattr:`effective_tp_size`."""
        return self.tp_size_per_node * self.cp_size_per_node

    @property
    def num_kv_heads_per_node(self) -> int:
        """Number of KV heads visible to a single node.

        中文：MLA（num_kv_heads == 1）是特例——KV 只有一份，节点上看到的就是 1，
        不做切分。其余情况按"节点内 TP 占比"缩放。
        """
        if self.num_kv_heads == 1:
            return self.num_kv_heads
        return self.num_kv_heads * self.tp_size_per_node // max(1, self.tp_size)

    @property
    def bytes_per_token_per_layer(self) -> int:
        """Raw byte footprint of a single (layer, token) KV slot.

        NOTE: assumes uniform (num_kv_heads, head_size, dtype) across layers.
        Not meaningful when ``layer_groups`` is set — callers in that path must
        use ``token_size_in_bytes`` instead.

        中文：单层单 token 的 KV 字节数 = num_kv_heads × head_size × kv_dim ×
        dtype.itemsize（kv_dim=2 即 K 和 V 两份）。**只在同构模型下成立**，
        异构模型（layer_groups 非 None）请改用 token_size_in_bytes。
        """
        return self.num_kv_heads * self.head_size * self.kv_dim * self.dtype.itemsize

    @property
    def token_size_in_bytes(self) -> int:
        """Whole-model per-token KV footprint (bytes) across all layers/groups.

        中文：整个模型一个 token 的 KV 总字节数（所有层/所有组求和）。
        它是"GB -> block 数"换算的分母来源之一，直接影响缓存能装多少 block。

        坑：layer_groups 里存的 num_kv_heads 是**单卡**的，所以要乘 tp_size 还原成
        全模型口径；且每层贡献要先整除 compress_ratio 再求和，因此这个整数 property
        只适合做聚合估算——真正分配 block 必须走 block_size_in_bytes_for_cache，
        它对 tokens_per_block 先做整除再乘，不会因为逐层取整而丢字节。
        """
        kv_dim = self.kv_dim
        if self.layer_groups:
            # layer_groups store per-GPU num_kv_heads; multiply by tp_size
            # to get full-model per-token size (matching CPU/SSD block sizing).
            # Each group may carry its own dtype (None = inherit ModelConfig.dtype),
            # so indexer-as-group (fp8/uint8) and main KV (bf16) sum correctly.
            # ``compress_ratio`` shrinks the per-token contribution. This
            # integer property is suitable for aggregate metrics; exact block
            # allocation must use block_size_in_bytes_for_cache so division is
            # applied to tokens_per_block before rounding.
            return sum(
                g.num_layers * g.num_kv_heads * g.head_size * kv_dim
                * (g.dtype or self.dtype).itemsize
                // g.compress_ratio
                for g in self.layer_groups
            ) * self.tp_size
        return self.num_layers * self.num_kv_heads * self.head_size * kv_dim * self.dtype.itemsize

    def __str__(self) -> str:
        layer_groups_str = (
            f", layer_groups={len(self.layer_groups)}groups"
            if self.layer_groups else ""
        )
        return (
            f"ModelConfig(num_layers={self.num_layers}, num_kv_heads={self.num_kv_heads}"
            f", head_size={self.head_size}, kv_dim={self.kv_dim}"
            f", dtype={self.dtype}"
            f", tp_size={self.tp_size}, pp_size={self.pp_size}, dp_size={self.dp_size}"
            f", cp_size={self.cp_size}"
            f", total_gpus={self.total_gpus}"
            f", nnodes={self.nnodes}, master_host={self.master_host!r}"
            f", instance_num={self.instance_num}"
            f"{layer_groups_str}"
        )


@dataclass(frozen=True)
class RankInfo:
    """某一个 rank 在 (instance, dp, pp, tp, cp) 网格里的"身份证"。

    它回答两类问题，对应两套编号，务必分清：
      * 我在哪 —— tp_rank / pp_rank / dp_rank / cp_rank / node_rank / instance_id，
        以及由它们派生的 local_rank（本节点内的 GPU 序号）等**控制面**坐标。
      * 数据怎么走 —— dp_client_id（跨实例的扁平路由标签）、intra_client_id
        （单个 DP client 内的 worker 号）、effective_tp_rank（数据面分片编号），
        这些是**数据面 / 传输层**用来寻址的键：worker 映射表、NVTX 标签、
        unix socket 路径、server 端 client 注册全都按 dp_client_id 索引。

    为什么是 frozen：RankInfo 会被当作字典键和跨进程传递的身份标识，
    可变会引入极难排查的路由错乱。派生几何量一律走 @property。

    坑：
      1. local_rank 默认为 -1，表示"未指定，请自动推导"。推导在 __post_init__ 里
         完成，且推导结果依赖 model_config 的拓扑；显式传非负值可覆盖（用于
         adapter 的 GPU 可见性重排与本工具算的不一致时）。
      2. cp_rank 与 attn_cp_rank 是 cp_size / attn_cp_size 的 rank 侧镜像，
         __post_init__ 里做同样的双向对齐，规则与 ModelConfig.freeze() 一致。
      3. pp_end_layer 默认 -1 表示"到模型最后一层"，num_layers_per_pp_stage 会
         把它替换成 num_layers。
    """
    model_config: ModelConfig
    tp_rank: int = 0
    pp_rank: int = 0
    dp_rank: int = 0
    cp_rank: int = 0
    # Compatibility mirror for origin/main's SGLang-facing API.
    attn_cp_rank: int = 0
    node_rank: int = 0
    instance_id: int = 0
    pp_start_layer: int = 0
    pp_end_layer: int = -1
    local_rank: int = -1

    def __post_init__(self) -> None:
        # 中文：cp_rank / attn_cp_rank 双向对齐，规则与 ModelConfig.freeze() 里
        # cp_size / attn_cp_size 的对齐完全一致（一方为 0 时以另一方为准）。
        if self.cp_rank == 0 and self.attn_cp_rank != 0:
            object.__setattr__(self, "cp_rank", self.attn_cp_rank)
        elif self.attn_cp_rank == 0 and self.cp_rank != 0:
            object.__setattr__(self, "attn_cp_rank", self.cp_rank)
        elif self.cp_rank != self.attn_cp_rank:
            raise ValueError(
                f"cp_rank={self.cp_rank} and attn_cp_rank={self.attn_cp_rank} disagree"
            )
        # 中文：local_rank < 0 表示"未指定，按拓扑自动推导"。
        # 推导顺序 = 从粗到细的嵌套：PP stage(节点内) -> CP -> TP，
        # 是否插入 DP 维度取决于 enable_dp_attention：
        #   * 开 DP attention 时，同一节点上不同 DP 分片共享同一组 GPU（注意力被
        #     DP 切分），所以 local_rank 里**不含** dp_rank_per_node。
        #   * 不开时，每个 DP 分片独占自己的 GPU，所以要先按 dp 分段再排 pp/cp/tp。
        if self.local_rank < 0:
            model_config = self.model_config
            tp_cp_rank = (
                self.cp_rank * model_config.tp_size_per_node
                + self.tp_rank_per_node
            )
            if model_config.enable_dp_attention:
                local_rank = (
                    self.pp_rank_per_node
                    * model_config.cp_size_per_node
                    * model_config.tp_size_per_node
                    + tp_cp_rank
                )
            else:
                local_rank = (
                    (
                        self.dp_rank_per_node * self.pp_size_per_node
                        + self.pp_rank_per_node
                    )
                    * model_config.cp_size_per_node
                    * model_config.tp_size_per_node
                    + tp_cp_rank
                )
            object.__setattr__(self, "local_rank", local_rank)

    @property
    def tp_rank_per_node(self) -> int:
        """TP rank index within the local node (within one TP group)."""
        return self.tp_rank % self.model_config.tp_size_per_node

    @property
    def dp_client_id(self) -> int:
        """Flat DP route label: unique int across all instances.

        Equals ``instance_id * dp_size + dp_rank``. All transfer-engine
        routing (worker maps, NVTX labels, socket paths, server-side
        client registration) keys on this single int so the legacy
        ``(instance_id, dp_rank)`` tuple (DPRoutingKey) is no longer
        needed.

        中文：这是**最常用**的路由标签，等于 instance_id * dp_size + dp_rank。
        多实例部署时它是全局唯一的，传输层所有按 DP 分流的映射表都以它为键。
        """
        return self.instance_id * self.model_config.dp_size + self.dp_rank

    @property
    def attn_tp_rank(self) -> int:
        """Compatibility alias for the normalized attention TP rank."""
        return self.tp_rank

    @property
    def effective_tp_rank(self) -> int:
        """Effective tp-rank in the *data-plane* segmentation space.

        Each ``(cp_rank, tp_rank)`` pair owns a unique slice in the combined
        CP×TP segmentation space.

        中文：与 ModelConfig.effective_tp_size 配对使用——前者是分片总数，
        这里是本 rank 的分片编号，两者构成 CPU/SSD block 的 (偏移, 总数) 寻址。
        """
        return self.cp_rank * max(1, self.model_config.tp_size) + self.tp_rank

    @property
    def intra_client_id(self) -> int:
        """Unique data-plane worker ID within one DP client.

        中文：pp_rank × effective_tp_size + effective_tp_rank，即"同一个 DP client
        内部"的扁平 worker 编号。一个 DP client 下挂 pp × (tp × cp) 个 GPU worker，
        传输引擎靠这个 id 在同一 client 内唯一定位某个 worker 进程。
        """
        return (
            self.pp_rank * self.model_config.effective_tp_size
            + self.effective_tp_rank
        )

    @property
    def pp_size_per_node(self) -> int:
        """Number of PP stages co-located on a single node."""
        model_config = self.model_config
        return max(model_config.pp_size // model_config.nnodes, 1)

    @property
    def pp_rank_per_node(self) -> int:
        """This rank's PP index *within* its node."""
        return self.pp_rank % self.pp_size_per_node

    @property
    def dp_size_per_node(self) -> int:
        """Number of DP replicas co-located on a single node."""
        model_config = self.model_config
        return max(
            1,
            model_config.gpus_per_node
            // (
                self.pp_size_per_node
                * model_config.tp_size_per_node
                * model_config.cp_size_per_node
            ),
        )

    @property
    def dp_rank_per_node(self) -> int:
        """This rank's DP index within its node."""
        return self.dp_rank % self.dp_size_per_node

    @property
    def num_layers_per_pp_stage(self) -> int:
        """Number of layers managed by this PP stage.

        中文：本 PP stage 负责的层数。pp_end_layer 为 -1 表示"一直到模型最后一层"，
        这是默认值——只有不均匀切分时才需要显式给 pp_start_layer / pp_end_layer。
        """
        end = self.pp_end_layer if self.pp_end_layer >= 0 else self.model_config.num_layers
        return end - self.pp_start_layer

    @property
    def token_size_in_bytes_per_pp_stage(self) -> int:
        """Per-pp-stage token footprint (bytes) — rank-exact."""
        return (self.num_layers_per_pp_stage
                * self.model_config.bytes_per_token_per_layer)

    def __str__(self) -> str:
        """Human-readable summary of this rank including derived quantities.

        Equivalent to the retired ``FlexKVContext.describe_rank`` output
        (kept stable so log-grep patterns keep working).
        """
        return (
            f"RankInfo(tp_rank={self.tp_rank}, pp_rank={self.pp_rank}"
            f", dp_rank={self.dp_rank}, cp_rank={self.cp_rank}"
            f", node_rank={self.node_rank}, instance_id={self.instance_id}"
            f", local_rank={self.local_rank}, effective_tp_rank={self.effective_tp_rank}"
        )

@dataclass
class SWAPoolConfig:
    """Configuration for SWA (Sliding Window Attention) host pool(s).

    SWA is managed at PAGE granularity: one pool slot stores exactly one
    ``tokens_per_block`` page of SWA KV, and all SWA IO moves a whole page
    (one slot) at a time.

    中文要点：SWA（Sliding Window Attention，滑窗注意力）层的 KV 是"滑窗"语义，
    主流水的 radix tree 复用逻辑对它不适用，所以单独用一块**页式**主机池管理：
    一个 slot 恰好装一个 tokens_per_block 页，SWA 的 IO 永远是整页进整页出，
    不做页内部分读写。默认关闭（enabled=False），DeepSeek V4 这类带 SWA 的模型
    才需要开。

    三级 SWA 池（CPU / SSD / REMOTE）共用同一份"页几何"配置，只是槽位数不同：
    num_slots 是 CPU 池大小，num_ssd_slots / num_remote_slots 为 0 表示该层不建
    SWA 池。三层各自的配置由 for_cache_tier(device_type) 派生。
    """
    enabled: bool = False
    num_slots: int = 1024              # Number of CPU SWA pool slots
    num_ssd_slots: int = 0             # Number of SSD SWA pool slots (0 = no SSD SWA tier)
    num_remote_slots: int = 0          # Number of REMOTE SWA pool slots (0 = no REMOTE SWA tier)
    num_swa_layers: int = 61           # Number of SWA layers (all 61 for DSv4)
    bytes_per_token_per_layer: int = 584  # nope_fp8(448) + rope_bf16(128) + scale(8)
    # True when the SWA page also carries heterogeneous sidecar groups (for
    # example DeepSeek-V4 attention/indexer compress states).
    multi_group: bool = False
    evict_ratio: float = 0.1           # Fraction of pool to evict when full
    pin_memory: bool = True            # Use pinned memory for async DMA
    # 中文：bytes_per_token_per_layer 的默认 584 = nope_fp8(448) + rope_bf16(128)
    # + scale(8)，是 DSv4 的经验值。**换模型必须重算**，算错会导致 SWA 页越界
    # 或页内留空洞，且这种错误只在特定序列长度下才暴露，极难排查。

    def for_ssd_tier(self) -> "SWAPoolConfig":
        """Derive the SSD-tier SWA config (same slot geometry, num_ssd_slots slots).

        SSD SWA slots are not pinned host memory; pin_memory is forced off.

        中文：SSD 层的 SWA 槽位不住在主机内存里，pin_memory 无意义，强制关掉。
        """
        return replace(self, num_slots=self.num_ssd_slots, pin_memory=False)

    def for_remote_tier(self) -> "SWAPoolConfig":
        """Derive the REMOTE-tier SWA config (same slot geometry, num_remote_slots).

        REMOTE SWA slots are not pinned host memory; pin_memory is forced off."""
        return replace(self, num_slots=self.num_remote_slots, pin_memory=False)

    def for_cache_tier(self, device_type) -> Optional["SWAPoolConfig"]:
        """Return the SWA config this cache tier should own, or None.

        CPU uses the primary pool config. SSD and REMOTE use their tier-specific
        slot counts and are disabled when that tier's slot count is zero.

        中文：这是存储层建 SWA 池时的统一入口——按 device_type 返回该层该用的配置，
        该层槽位数为 0（或未启用 SWA）时返回 None，调用方据此跳过建池。
        注意 device_type 是 DeviceType 枚举，这里用 getattr(name) 取名字而非直接
        import，是为了避免 common 包内部产生循环依赖。
        """
        if not self.enabled:
            return None
        device_name = getattr(device_type, "name", str(device_type))
        if device_name == "CPU":
            return self
        if device_name == "SSD" and self.num_ssd_slots > 0:
            return self.for_ssd_tier()
        if device_name == "REMOTE" and self.num_remote_slots > 0:
            return self.for_remote_tier()
        return None


@dataclass
class CacheConfig:
    """缓存池容量 + "走哪条物理通路"的开关集合——本文件最需要逐项读懂的类。

    它只决定两件事：各级缓存开多大（num_*_blocks），以及数据能在哪些设备之间流动
    （各类 enable_* 开关）。这两个决定会一路传导到 cache_engine 造出的 DAG 里。

    三级缓存与主要通路：

        GPU ──(H2D / D2H)── CPU ──(H2DISK / DISK2H)── SSD
                                    └──(H2REMOTE / REMOTE2H)── REMOTE
         └── 若 enable_gds：GPU ──(GDS)── SSD 直连，不经过 CPU 中转 ──┘
         └── 若 enable_p2p_*：本节点 CPU/SSD ──(PEER2H)── 对端节点 CPU/SSD ──┘

    关键开关（默认值 / 开启后走哪条路）：
      enable_cpu          True  —— CPU 一级缓存。当前**必须为 True**（
                          update_default_config_from_user_config 末尾有硬校验），
                          因为 SSD / REMOTE 都以 CPU 为中转，关掉整条链路就断了。
      enable_ssd          False —— 本地 SSD 二级缓存。由 ssd_cache_gb > 0 自动置位。
      enable_gds          False —— GPU<->SSD 直连（GPUDirect Storage），绕过 CPU
                          中转。依赖 enable_ssd=True（有校验），且需编译时开
                          FLEXKV_ENABLE_GDS=1。
      enable_nixl         False —— 在 enable_gds 之上，改用 NIXL（GDS_MT）后端
                          代替 cuFile GDS worker。
      enable_remote       False —— 第三方远端存储三级缓存。**派生字段**，由
                          enable_3rd_remote 或 use_mooncake_store_backend 推出。
                          注意它跟 p2p_cpu / p2p_ssd 没有任何关系。
      enable_p2p_cpu      False —— 跨节点复用**对端节点的 CPU 缓存**（PEER_CPU）。
      enable_p2p_ssd      False —— 跨节点复用**对端节点的 SSD 缓存**（PEER_SSD）。
      enable_3rd_remote   False —— 第三方远端（如 CFS）作为共享层。
      enable_kv_sharing   False —— **派生字段** = p2p_cpu or p2p_ssd or 3rd_remote。
                          与 enable_gds 互斥（有校验）。
      use_hugepage_cpu_buffer / use_hugepage_tmp_buffer
                          False —— 分别把"主 CPU 缓存"和"SSD->CPU 中转临时缓冲"
                          改用 Linux HugePage（mmap MAP_HUGETLB）分配，降低 TLB
                          miss。需宿主机预留大页（/proc/sys/vm/nr_hugepages）；
                          分配失败会**静默回退**到普通内存，所以开了不等于生效。
      enable_swa_transfer False —— SWA 的**数据面**开关。SWA 控制面（匹配/容量/锁）
                          与它无关。置 True 前必须确认专用 SWA transfer worker 已
                          注册，否则 SWA op 会被传输引擎判为
                          "Unsupported transfer type"。

    坑：
      1. 【配置开关 ≠ 编译宏】凡涉及 GDS / P2P / CFS / NVCOMP 的通路，除了这里置
         True，还必须在**编译 C++ 扩展前** export 对应宏（默认全为 0）：
         FLEXKV_ENABLE_GDS / FLEXKV_ENABLE_P2P / FLEXKV_ENABLE_CFS /
         FLEXKV_ENABLE_NVCOMP（见 setup.py:300-304）。只开开关不开宏，通常表现为
         import 期 AttributeError 或运行期 "Unsupported transfer type"。
      2. enable_kv_sharing / enable_remote 是派生字段，__post_init__ 会重新推导并
         覆盖外部赋值，不要直接写它们。
      3. num_ssd_blocks 会被向上取整到 ssd_cache_dir 个数的整数倍（多盘均分），
         实际容量可能略大于按 GB 算出的值。
      4. 单位：*_blocks 是 block 数不是字节；GB -> block 的换算见
         update_default_config_from_user_config()。
    """
    tokens_per_block: int = 16
    eviction_policy: str = "lru"
    # ==================================================================
    # 物理通路开关：下面这一组直接决定 DAG 里会出现哪些 TransferType。
    # 每个开关的语义、默认值与对应的 C++ 编译宏见类 docstring。
    # ==================================================================
    enable_cpu: bool = True
    enable_ssd: bool = False
    enable_gds: bool = False # Requires enable_ssd=True
    # When True with enable_gds, GPU<->SSD uses NIXL (GDS_MT) instead of cuFile GDS worker.
    enable_nixl: bool = False
    # Optional plugin dict for NixlAgentSession (see nixl README); only used if enable_nixl.
    nixl_extra_config: Optional[Dict[str, Any]] = None
    enable_remote: bool = False # used for indicating whether the 3rd-party remote storage is enabled
                                # has nothing to do with whether the p2p_cpu and p2p_ssd are supported
    enable_kv_sharing: bool = False # pcfs_sharing or p2p_cpu or p2p_ssd or p2p_3rd_remote
    enable_p2p_cpu: bool = False
    enable_p2p_ssd: bool = False
    enable_3rd_remote: bool = False

    # 中文：下面两个只在"分布式（p2p）"场景生效。
    # distributed_node_id 是本节点在分布式集群里的编号，-1 表示未初始化；
    # 它必须等 redis_meta_client 建好之后才能赋值，因为节点编号要与 Redis 里的
    # node:<id> 键对齐。
    distributed_node_id: int = -1 # only used when distributed cpu/ssd and only can be set when redis_meta_client initialized
    # 中文：p2p_ssd 场景下的 CPU 中转缓冲块数。对端 SSD 的数据不能直接进本地 SSD，
    # 必须先落到这块临时 CPU buffer，所以它是"开 p2p_ssd 就一定要付的内存代价"。
    num_tmp_cpu_blocks: int = 500 # only used when distributed ssd p2p, it controls the number blocks of temp cpu buffer which used for copy data from ssd to cpu
    # When True, the main CPU KV cache is allocated from Linux HugePages via
    # ``mmap(MAP_HUGETLB)`` instead of regular CPU memory. Requires pre-reserved
    # huge pages on the host (see ``/proc/sys/vm/nr_hugepages``). Falls back
    # silently if allocation fails.
    use_hugepage_cpu_buffer: bool = False
    # When True, the temporary SSD->CPU staging buffer (used by PEER2CPUTransferWorker
    # under enable_p2p_ssd) is allocated from Linux HugePages via ``mmap(MAP_HUGETLB)``
    # instead of a pinned ``torch.empty``. Requires pre-reserved huge pages on the host
    # (see ``/proc/sys/vm/nr_hugepages``). Falls back silently if allocation fails.
    use_hugepage_tmp_buffer: bool = False
    hugepage_size_bytes: int = 2 * 1024 * 1024  # 2 MiB by default; set to 1<<30 for 1GiB
    # Maximum size accepted by one Mooncake external memory registration.
    # Keep the historical 512 GiB default; deployments with a smaller NIC or
    # transport limit (for example ionic's 2 GiB limit) can override it.
    mooncake_max_mr_size_bytes: int = 512 * 1024 * 1024 * 1024

    # mempool capacity configs
    # 中文：各级缓存的容量，单位是 **block 数**（不是字节）。默认值只是占位，
    # 真实值由 update_default_config_from_user_config() 按 GB 换算后回填。
    # num_local_blocks 是 GPU 侧的 block 数（通常与推理引擎的显存池对齐）。
    num_cpu_blocks: int = 1000000
    num_ssd_blocks: int = 10000000
    num_remote_blocks: Optional[int] = None
    num_local_blocks: int = 1000000

    # ssd cache configs
    # 中文：支持多盘——传字符串会被 parse_path_list 按 ';' 切成列表，传列表则直接用。
    # 盘数会影响 num_ssd_blocks 的对齐（向上取整到盘数的整数倍，见
    # update_default_config_from_user_config 末尾）。
    ssd_cache_dir: Optional[Union[str, List[str]]] = None

    # remote cache configs for cfs
    # todo: remove this in the future
    # 中文：这一组是 CFS 远端存储的遗留配置，作者已标注待删除。两种容量口径由
    # remote_cache_size_mode 选择：
    #   "file_size" —— 给每个文件的字节数 remote_file_size + 文件数 remote_file_num，
    #                  由 update_default_config_from_user_config 反推出 num_remote_blocks；
    #   "block_num" —— 直接给 num_remote_blocks。
    # 只在 enable_remote 且未启用 mooncake_store 后端时才会被校验（见那处分支）。
    remote_cache_size_mode: str = "file_size"  # file_size or block_num
    remote_file_size: Optional[int] = None
    remote_file_num: Optional[int] = None
    remote_file_prefix: Optional[str] = None
    remote_cache_path: Optional[Union[str, List[str]]] = None
    remote_config_custom: Optional[Dict[str, Any]] = None

    # distributed zmq configs
    # 中文：传输管理面（TransferManager）的 zmq 端点。默认只监听回环地址，
    # 多节点部署必须改成真实网卡 IP，否则对端连不上。
    local_zmq_ip: str = "127.0.0.1"
    local_zmq_port: int = 5555
    # Redis configs (for KV sharing / metadata)
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    local_ip: str = "127.0.0.1"
    redis_password: Optional[str] = None
    # TTL (seconds) for node:<id> key in Redis. Active nodes renew via heartbeat.
    # If a process crashes, the key auto-expires after this period.
    node_ttl_seconds: int = 30

    # Mooncake transfer engine config path (serialized via pickle to survive spawn subprocesses)
    mooncake_config_path: Optional[str] = None

    # Mooncake-store distributed KV cache backend (key-addressed; ≠ Transfer Engine P2P)
    # 中文：注意区分两个都叫 Mooncake 的东西——
    #   * mooncake_config_path      -> Mooncake **Transfer Engine**（RDMA 传输引擎），
    #                                 只是给传输层用的底层通路配置。
    #   * use_mooncake_store_backend -> Mooncake **Store**（按 key 寻址的分布式 KV
    #                                 存储后端），是一条独立的 REMOTE 缓存层，
    #                                 与上面的 P2P 通路不是一回事。
    # config_path 要 pickle 序列化后才能安全传给 spawn 出来的子进程。
    use_mooncake_store_backend: bool = False
    mooncake_store_config_path: Optional[str] = None
    mooncake_store_pp_rank: int = 0
    mooncake_store_pp_size: int = 1
    mooncake_store_node_layer_start: int = 0
    mooncake_store_node_layer_end: int = 0
    mooncake_store_total_layers: int = 0

    # Stored for deferred recomputation when layer_groups become known
    # 中文：存下用户原始的 GB 预算，供 recompute_cache_block_counts() 延后重算。
    # 为什么需要延后：GB -> block 的换算依赖 block_size_in_bytes，而它依赖
    # model_config.layer_groups；异构模型（DSv4）的 layer_groups 常常在
    # update_default_config_from_user_config() **之后**才补齐，那时第一次换算的
    # 结果就偏了，必须靠这两个字段记住原始意图再算一遍。
    _user_cpu_cache_gb: float = 0
    _user_ssd_cache_gb: float = 0

    # SWA pool config (DeepSeek V4)
    # 中文：None 表示本实例不做 SWA 池（见 SWAPoolConfig）。
    swa: Optional['SWAPoolConfig'] = None

    # Gate for the SWA peer-op DATA-PLANE transfer (SWA_H2D/SWA_D2H ops built into
    # the transfer graph). Default False: the SWA control plane (node-mounted match /
    # capacity / lock) works regardless, but the actual async SWA byte transfer
    # requires the dedicated SWA transfer worker (data plane). Keep this False
    # until that worker is registered, otherwise SWA ops would hit "Unsupported
    # transfer type" in the transfer engine. Flip to True once the worker lands.
    enable_swa_transfer: bool = False

    def __post_init__(self):
        """推导派生开关并做启动期校验。

        中文要点：这里做的全是"派生 + 兜底"，不改用户意图：
          * enable_kv_sharing = p2p_cpu or p2p_ssd or 3rd_remote（纯派生）
          * use_mooncake_store_backend 会从环境变量兜底补一次（因为构造路径可能是
            default_factory，用户没显式传）
          * enable_remote = 3rd_remote or mooncake_store（纯派生）
        因此**不要**在外部直接给 enable_kv_sharing / enable_remote 赋值，会被覆盖。
        """
        self.enable_kv_sharing = self.enable_p2p_cpu or \
            self.enable_p2p_ssd or self.enable_3rd_remote
        # 中文：顺序很关键——必须**先**从环境变量补全 use_mooncake_store_backend，
        # **再**拿它去推 enable_remote。反过来的话，
        # FLEXKV_USE_MOONCAKE_STORE_BACKEND=1 会留下 enable_remote=False，
        # 从而被下面那句 "requires enable_remote" 的检查误伤（FlexKVConfig
        # 走 default_factory 构造 CacheConfig 时就是这个路径）。
        # Resolve mooncake flag from env BEFORE deriving enable_remote;
        # otherwise FLEXKV_USE_MOONCAKE_STORE_BACKEND=1 leaves enable_remote
        # False and trips the check below (FlexKVConfig default_factory path).
        self.use_mooncake_store_backend = self.use_mooncake_store_backend or bool(
            int(os.getenv('FLEXKV_USE_MOONCAKE_STORE_BACKEND', '0')))
        self.enable_remote = self.enable_3rd_remote or self.use_mooncake_store_backend
        if self.use_mooncake_store_backend and self.mooncake_store_config_path is None:
            self.mooncake_store_config_path = os.getenv(
                'FLEXKV_MOONCAKE_STORE_CONFIG_PATH', None)
            if self.mooncake_store_config_path is None:
                raise ValueError(
                    "Mooncake store config path not found; set "
                    "mooncake_store_config_path or FLEXKV_MOONCAKE_STORE_CONFIG_PATH")
      
        if self.use_mooncake_store_backend and not self.enable_remote:
            raise ValueError(
                "use_mooncake_store_backend requires enable_remote "
                "(prefetch REMOTE2H path before compute GET)"
            )

    def __str__(self) -> str:
        return (
            f"CacheConfig(tokens_per_block={self.tokens_per_block}"
            f", enable_cpu={self.enable_cpu}, enable_ssd={self.enable_ssd}"
            f", enable_gds={self.enable_gds}, enable_remote={self.enable_remote}"
            f", enable_kv_sharing={self.enable_kv_sharing}"
            f", enable_p2p_cpu={self.enable_p2p_cpu}"
            f", enable_p2p_ssd={self.enable_p2p_ssd}"
            f", enable_3rd_remote={self.enable_3rd_remote}"
            f", num_cpu_blocks={self.num_cpu_blocks}"
            f", num_ssd_blocks={self.num_ssd_blocks})"
        )

# ==============================================================================
# GLOBAL_CONFIG_FROM_ENV —— 模块级全局单例（Namespace），装的是"进程级"的运行时开关
# ------------------------------------------------------------------------------
# 与 CacheConfig 的分工：CacheConfig 是**每个实例**一份、会被 GB 换算回填的业务
# 配置；这里则是**整个进程**只读一份、且大多直接透传给 C++ 扩展或传输层的底层开关
# （传输并发度、io_uring 参数、metrics 端口、trace、关停超时等）。
#
# 【重要】这是一个 Namespace 而非 dataclass，且在所有 FLEXKV_* 环境变量的读取上
# 是**一次性的**：本模块被 import 时 os.getenv 就已经执行完了。此后
#   * 再改 os.environ 无效；
#   * 想运行时覆盖只能 setattr(GLOBAL_CONFIG_FROM_ENV, name, value)——
#     update_default_config_from_user_config() 末尾的 override_ 机制正是这么做的
#     （配置文件里写 override_xxx 即可覆盖这里的 xxx）。
#
# 命名约定：这里的属性名 xxx 对应环境变量 FLEXKV_XXX（大写）。覆盖时报错信息里的
# 变量名也是这么拼出来的。
#
# 另一半注意点：这里的值会被 fork/spawn 出的 Worker 子进程继承，所以凡是影响
# C++ 侧行为的项（如 ce_* / iouring_* / nvcomp_batch_size）在子进程里同样生效，
# 不需要每个 worker 再读一遍环境变量。
# ==============================================================================
GLOBAL_CONFIG_FROM_ENV: Namespace = Namespace(
    # Multi-instance configuration
    instance_num=int(os.getenv('FLEXKV_INSTANCE_NUM', 1)),
    instance_id=int(os.getenv('FLEXKV_INSTANCE_ID', 0)),

    # Metrics configuration
    ## Enable/disable metrics collection and HTTP server (shared by C++ and Python)
    ## 中文：C++ 侧与 Python 侧各起一个 HTTP 服务，端口不同（8081 / 8080），别混淆。
    enable_metrics=bool(int(os.getenv('FLEXKV_ENABLE_METRICS', 0))),
    ## Port for C++ metrics HTTP server (default: 8081)
    cpp_metrics_port=int(os.getenv('FLEXKV_CPP_METRICS_PORT', 8081)),
    ## Port for Python metrics HTTP server (default: 8080)
    py_metrics_port=int(os.getenv('FLEXKV_PY_METRICS_PORT', 8080)),

    use_mooncake_store_backend=bool(int(os.getenv('FLEXKV_USE_MOONCAKE_STORE_BACKEND', 0))),
    mooncake_store_config_path=os.getenv('FLEXKV_MOONCAKE_STORE_CONFIG_PATH', None),

    # Server-client mode configuration
    # 中文：server-client 模式 = FlexKV 跑在独立服务进程里，框架通过 zmq 与它通信。
    # server_recv_port 默认是 ipc:// 的 unix socket（同机场景比 TCP 快）；
    # server_launch_mode 为 'embedded' 时由框架进程内嵌拉起服务，否则需外部先启动。
    server_client_mode=bool(int(os.getenv('FLEXKV_SERVER_CLIENT_MODE', 0))),
    server_launch_mode=os.getenv('FLEXKV_SERVER_LAUNCH_MODE', 'embedded').lower(),
    server_recv_port=os.getenv('FLEXKV_SERVER_RECV_PORT', 'ipc:///tmp/flexkv_server'),

    # 中文：下面四个 *_layout_type 分别指定 CPU / SSD / REMOTE / GDS 各级缓存中 KV
    # 的**内存布局**（见 common/storage.py 的 KVCacheLayoutType：BLOCKFIRST /
    # LAYERFIRST 等）。布局直接决定 C++ 搬运核函数的访存模式——配错不会报错，
    # 只会让传输吞吐大幅掉档，所以必须与分配器一侧的选择保持一致。
    index_accel=bool(int(os.getenv('FLEXKV_INDEX_ACCEL', 1))),
    cpu_layout_type=KVCacheLayoutType(os.getenv('FLEXKV_CPU_LAYOUT', 'BLOCKFIRST').upper()),
    ssd_layout_type=KVCacheLayoutType(os.getenv('FLEXKV_SSD_LAYOUT', 'BLOCKFIRST').upper()),
    remote_layout_type=KVCacheLayoutType(os.getenv('FLEXKV_REMOTE_LAYOUT', 'BLOCKFIRST').upper()),
    gds_layout_type=KVCacheLayoutType(os.getenv('FLEXKV_GDS_LAYOUT', 'BLOCKFIRST').upper()),

    # 中文：按层传输（layerwise）——把一次整块搬运拆成逐层的细粒度 op，让前几层的
    # KV 尽早可用，从而把传输延迟**重叠**进计算里。默认关闭。
    # 代价：DAG 会膨胀成 num_layers 个节点，调度与通知开销上升；小 batch /
    # 短序列场景可能反而更慢。开启后还要关注 layerwise_notify_mode。
    enable_layerwise_transfer=bool(int(os.getenv('FLEXKV_ENABLE_LAYERWISE_TRANSFER', 0))),

    # 中文：CE = Copy Engine，GPU 上的 DMA 拷贝引擎。默认 H2D / D2H 走 SM（核函数）
    # 搬运，开启后改由 CE 代劳，把 SM 让给计算。num_cta 是搬运核函数的 CTA（线程块）
    # 并发数，是吞吐调优的主要旋钮。
    use_ce_transfer_h2d=bool(int(os.getenv('FLEXKV_USE_CE_TRANSFER_H2D', 0))),
    use_ce_transfer_d2h=bool(int(os.getenv('FLEXKV_USE_CE_TRANSFER_D2H', 0))),
    transfer_num_cta_h2d=int(os.getenv('FLEXKV_TRANSFER_NUM_CTA_H2D', 4)),
    transfer_num_cta_d2h=int(os.getenv('FLEXKV_TRANSFER_NUM_CTA_D2H', 4)),

    # 中文：这三个是传输层的启发式阈值与优化开关。
    # ce_segment_threshold —— 待搬段数少于该值时放弃 CE，退回普通路径（CE 每次提交
    #                         有固定开销，段太少不划算）；
    # ce_path_opt / ssd_io_opt —— 分别开启 CE 路径优化与 SSD IO 优化，默认都开。
    ce_segment_threshold=int(os.getenv('FLEXKV_CE_SEGMENT_THRESHOLD', 8)),
    ce_path_opt=bool(int(os.getenv('FLEXKV_CE_PATH_OPT', 1))),
    ssd_io_opt=bool(int(os.getenv('FLEXKV_SSD_IO_OPT', 1))),

    # 中文：CE 的 2D memcpy 与 gather 细节参数：gather 用 ce_gather_threads 个线程
    # 并行收集分散的 block；nt = non-temporal（流式写，绕过 L2，避免把计算用的
    # 缓存行挤出去）。CE 未开启时这几个值不生效。
    enable_ce_memcpy2d=bool(int(os.getenv('FLEXKV_ENABLE_CE_MEMCPY2D', 1))),
    ce_gather_threads=int(os.getenv('FLEXKV_CE_GATHER_THREADS', 4)),
    ce_gather_nt=bool(int(os.getenv('FLEXKV_CE_GATHER_NT', 1))),

    # 中文：SSD 走 io_uring 时的队列深度（entries）与 flags。深度太小限制并发 IO、
    # 跑不满 NVMe；太大则白占内存且对延迟无益。512 是折中默认值。
    iouring_entries=int(os.getenv('FLEXKV_IOURING_ENTRIES', 512)),
    iouring_flags=int(os.getenv('FLEXKV_IOURING_FLAGS', 0)),

    max_file_size_gb=float(os.getenv('FLEXKV_MAX_FILE_SIZE_GB', -1)),  # -1 means no limit

    # 中文：淘汰相关参数。evict_ratio=0 表示不按"比例"批量淘汰（交由别处决定批量）；
    # evict_start_threshold=1.0 表示缓存用满才触发淘汰；
    # hit_reward_seconds 与 slru_protected_threshold 是 SLRU 策略的参数
    # （命中后给多久的"保护期"、进保护区需要命中几次）。
    evict_ratio=float(os.getenv('FLEXKV_EVICT_RATIO', 0)),
    evict_start_threshold=float(os.getenv('FLEXKV_EVICT_START_THRESHOLD', 1.0)),
    hit_reward_seconds=int(os.getenv('FLEXKV_HIT_REWARD_SECONDS', 0)),
    eviction_policy=os.getenv('FLEXKV_EVICTION_POLICY', 'lru'),
    slru_protected_threshold=int(os.getenv('FLEXKV_SLRU_PROTECTED_THRESHOLD', 2)),

    # 中文：enable_mps —— 是否与 CUDA MPS（多进程服务）共存，影响传输核函数能否与
    # 其它进程共享同一个 GPU 上下文，默认开。
    enable_mps=bool(int(os.getenv('FLEXKV_ENABLE_MPS', 1))),

    # 中文：传输前是否做集合同步（让参与同一次传输的 rank 对齐起步），默认开。
    # 关掉可以省掉一次同步开销，但可能出现 rank 间进度参差。
    enable_collective_sync=bool(int(os.getenv('FLEXKV_ENABLE_COLLECTIVE_SYNC', 1))),

    # 中文：trace 是**低频诊断**用的落盘日志（与 metrics 不是一回事），默认关闭。
    # 按 trace_max_file_size_mb 滚动、最多留 trace_max_files 个文件，
    # 每 trace_flush_interval_ms 刷盘一次。常开会带来可观的 IO 与内存开销。
    enable_trace=bool(int(os.getenv('FLEXKV_ENABLE_TRACE', 0))),
    trace_file_path=os.getenv('FLEXKV_TRACE_FILE_PATH', './flexkv_trace.log'),
    trace_max_file_size_mb=int(os.getenv('FLEXKV_TRACE_MAX_FILE_SIZE_MB', 100)),
    trace_max_files=int(os.getenv('FLEXKV_TRACE_MAX_FILES', 5)),
    trace_flush_interval_ms=int(os.getenv('FLEXKV_TRACE_FLUSH_INTERVAL_MS', 1000)),

    enable_transfer_trace=bool(int(os.getenv('FLEXKV_TRANSFER_TRACE', 0))),

    # 中文：这一组是 radix tree（lt = lookup tree）与"租约（lease）"机制的调优参数：
    #   lt_pool_initial_capacity —— 树节点池初始容量（预分配，避免运行时扩容抖动）
    #   lease_ttl_ms / renew_lease_ms / safety_ttl_ms —— 缓存项租约的时长、续约间隔
    #     与安全余量；租约过期即视为可回收，安全余量用来吸收时钟与调度误差
    #   rebuild_interval_ms / refresh_batch_size / idle_sleep_ms —— 后台重建与空闲
    #     轮询的节奏控制：每多久重建一次、每批刷新多少项、空闲时睡多久（省 CPU）
    #   reset_barrier_timeout_ms / poll_ms —— 全局 reset 时的屏障等待与轮询间隔
    lt_pool_initial_capacity=int(os.getenv('FLEXKV_LT_POOL_INITIAL_CAPACITY', 10000000)),
    refresh_batch_size=int(os.getenv('FLEXKV_REFRESH_BATCH_SIZE', 256)),
    rebuild_interval_ms=int(os.getenv('FLEXKV_REBUILD_INTERVAL_MS', 2000)),
    idle_sleep_ms=int(os.getenv('FLEXKV_IDLE_SLEEP_MS', 10)),
    lease_ttl_ms=int(os.getenv('FLEXKV_LEASE_TTL_MS', 30000)),
    safety_ttl_ms=int(os.getenv('FLEXKV_SAFETY_TTL_MS', 100)),
    renew_lease_ms=int(os.getenv('FLEXKV_RENEW_LEASE_MS', 4000)),
    reset_barrier_timeout_ms=int(os.getenv('FLEXKV_RESET_BARRIER_TIMEOUT_MS', 60000)),
    reset_barrier_poll_ms=int(os.getenv('FLEXKV_RESET_BARRIER_POLL_MS', 50)),

    # 中文：NVComp 压缩的批大小，0 = 自动选择。注意这是**压缩**开关，需要编译时
    # 打开 FLEXKV_ENABLE_NVCOMP=1 才有实现（默认关闭），否则这里设了也不生效。
    nvcomp_batch_size=int(os.getenv('FLEXKV_NVCOMP_BATCH_SIZE', '0')),  # 0 = auto

    # 中文：多 rank 共享同一份 KV（如 MLA，num_kv_heads == 1）时的写入模式：
    #   'sharded'   = 各 rank 各写自己那片（默认，省空间，偏移按 rank 切分）
    #   'all_write' = 每个 rank 都写完整一份，于是一个逻辑 block 在 CPU/SSD 上要占
    #                 N 倍物理空间。选它必须注意：update_default_config_from_user_config()
    #                 会据此把逻辑容量除以 N（capacity_divisor），否则预算会超。
    kv_shared_across_ranks_mode=os.getenv('FLEXKV_KV_SHARED_ACROSS_RANKS_MODE', 'sharded'),

    # 中文：按层传输时的完成通知方式。'hostfunc' = 用 CUDA host callback 通知 CPU
    # 侧，无需在 GPU 上额外插同步点（对计算流的干扰最小）。
    layerwise_notify_mode=os.getenv('FLEXKV_LAYERWISE_NOTIFY_MODE', 'hostfunc'),

    # Graceful shutdown timeout hierarchy (each layer waits for the next inner
    # layer plus a small buffer to avoid mid-unpin SIGKILL):
    #   sglang tokenizer wait   (SGLANG_SCHEDULER_SHUTDOWN_WAIT_S)  = 1200s
    #     > TM parent-side wait (FLEXKV_TRANSFER_MANAGER_SHUTDOWN_TIMEOUT_S) = 900s
    #       > per-worker wait   (FLEXKV_WORKER_SHUTDOWN_TIMEOUT_S)  = 600s
    # If you tune one, keep the ordering: worker < TM < scheduler-wait.
    worker_shutdown_timeout_s=float(os.getenv('FLEXKV_WORKER_SHUTDOWN_TIMEOUT_S', 600)),
    # Parent wait for TransferManager subprocess after sending shutdown command.
    # Must cover parallel worker unregister + a buffer over per-worker timeout.
    transfer_manager_shutdown_timeout_s=float(
        os.getenv('FLEXKV_TRANSFER_MANAGER_SHUTDOWN_TIMEOUT_S', 900)
    ),
)

@dataclass
class UserConfig:
    """面向用户的"人性化"入参：以 GB 而不是 block 数来表达容量，以开关表达通路。

    与 CacheConfig 的关系：UserConfig 是**输入**，CacheConfig 是**产物**。
    update_default_config_from_user_config() 负责把这里的 GB 换算成 block 数并
    逐项回填到 CacheConfig；本类自己不参与任何运行时决策。

    两个构造入口（推荐用这两个，别手写字段）：
      * load_user_config_from_env()  —— 全量读 FLEXKV_* 环境变量。
      * load_user_config_from_file() —— 读 JSON / YAML 配置文件；
        文件里**未定义**的字段会退化成这里的默认值；文件里**多出来**的字段不会丢，
        而是以 override_<name> 的形式挂到实例上，用于覆盖 GLOBAL_CONFIG_FROM_ENV。

    坑：
      1. cpu_cache_gb 用 int，ssd_cache_gb 也用 int——所以**不支持小数 GB**，
         需要 0.5 GB 这类粒度时请直接改 CacheConfig 的 block 数。
      2. ssd_cache_gb 必须严格大于 cpu_cache_gb（__post_init__ 校验），
         因为缓存是分级的：SSD 层比 CPU 层小就毫无意义。0 表示不启用 SSD。
      3. 这里的字段凡是 Optional[...] = None 的（zmq / redis / kv_cache_dtype 等），
         None 一律表示"用户没指定，保留 CacheConfig 的默认值"，回填时会跳过
         （见 update_default_config_from_user_config 末尾那串 if xxx is not None）。
    """
    cpu_cache_gb: int = 16
    ssd_cache_gb: int = 0  # 0 means disable ssd
    ssd_cache_dir: Union[str, List[str]] = "./ssd_cache"
    enable_gds: bool = False
    enable_nixl: bool = False
    use_hugepage_cpu_buffer: bool = False
    use_hugepage_tmp_buffer: bool = False
    hugepage_size_bytes: int = 2 * 1024 * 1024
    mooncake_max_mr_size_bytes: int = 512 * 1024 * 1024 * 1024
    enable_p2p_cpu: bool = False
    enable_p2p_ssd: bool = False
    enable_3rd_remote: bool = False
    use_mooncake_store_backend: bool = False
    mooncake_store_config_path: Optional[str] = None
    mooncake_store_pp_rank: int = 0
    mooncake_store_pp_size: int = 1
    mooncake_store_node_layer_start: int = 0
    mooncake_store_node_layer_end: int = 0
    mooncake_store_total_layers: int = 0

    # distributed zmq configs
    local_zmq_ip: Optional[str] = None
    local_zmq_port: Optional[int] = None
    # Redis configs (for KV sharing / metadata)
    redis_host: Optional[str] = None
    redis_port: Optional[int] = None
    local_ip: Optional[str] = None
    redis_password: Optional[str] = None
    node_ttl_seconds: Optional[int] = None
    kv_cache_dtype: Optional[str] = None  # Override kv_cache_dtype when TRT config uses "auto". Supported values: "fp8", "float8", "e4m3", "fp16", "float16", "bf16", "bfloat16", "fp32", "float32", "nvfp4" (packed fp4+fp8-scale, stored as uint8)
    # DeepSeek-V4 SWA sidecar policy. None/True enables attention and indexer
    # compress-state I/O together with SWA; False keeps the legacy SWA-only
    # path. None is intentionally distinct from False so old configs default
    # to the correctness-preserving state restore path.
    swa_multi_group: Optional[bool] = None

    def __post_init__(self):
        """启动期校验：把明显不合理的值在构造时就拒掉，别等到分配器 OOM。"""

        # 中文：SSD 必须严格大于 CPU——分级缓存的语义要求下一级比上一级大，
        # 否则"CPU 放不下的才下沉到 SSD"这条规则永远不成立。
        if self.cpu_cache_gb <= 0:
            raise ValueError(f"Invalid cpu_cache_gb: {self.cpu_cache_gb}")
        if self.ssd_cache_gb < 0:
            raise ValueError(f"Invalid ssd_cache_gb: {self.ssd_cache_gb}")
        if self.ssd_cache_gb > 0 and self.ssd_cache_gb <= self.cpu_cache_gb:
            raise ValueError(f"Invalid ssd_cache_gb: {self.ssd_cache_gb}, "
                             f"must be greater than cpu_cache_gb: {self.cpu_cache_gb}.")
        if self.mooncake_max_mr_size_bytes <= 0:
            raise ValueError(
                "mooncake_max_mr_size_bytes must be positive, "
                f"got {self.mooncake_max_mr_size_bytes}"
            )
        if self.swa_multi_group is not None and not isinstance(
            self.swa_multi_group, bool
        ):
            raise ValueError(
                "swa_multi_group must be a boolean when configured, "
                f"got {self.swa_multi_group!r}"
            )

def parse_path_list(path_str: str) -> List[str]:
    """把分号分隔的路径串切成列表，用于 SSD / 远端存储的多盘配置。

    约定：分隔符是英文分号 ';'，不是冒号（冒号在路径里更常见，易歧义）；
    空段会被丢弃，所以 "a; ;b;" 得到 ["a", "b"]。

    坑：签名标注的是 str，但 CacheConfig.ssd_cache_dir / remote_cache_path 的类型
    都是 Union[str, List[str]]。load_user_config_from_file() 会无条件对配置里的
    ssd_cache_dir 调用本函数，因此在 YAML / JSON 里把它写成 YAML **列表**
    （而不是 "path1;path2" 字符串）会直接抛 AttributeError。
    """
    paths = [p.strip() for p in path_str.split(';') if p.strip()]
    return paths

def load_user_config_from_file(config_file: str) -> UserConfig:
    """从 JSON / YAML 配置文件构造 UserConfig——离线部署与多环境切换的推荐入口。

    处理规则（两条都很反直觉，值得记住）：
      1. 文件里**未出现**的字段 -> 用 UserConfig 的默认值，不会报错。
      2. 文件里**多出来的**字段（UserConfig 没有的）-> 不会丢弃，而是以
         ``override_<name>`` 的形式 setattr 到实例上。这些属性后面会被
         update_default_config_from_user_config() 识别，用来覆盖
         GLOBAL_CONFIG_FROM_ENV 里的同名项（相当于在配置文件里覆盖环境变量）。
         名字写错会在那里抛 "Unknown config name"，属**延迟到那一步才报错**。

    坑：ssd_cache_dir 会被无条件交给 parse_path_list() 切分，所以文件里必须写成
    "path1;path2" 字符串；写成 YAML 列表会抛 AttributeError（见该函数注释）。
    """
    # read json config file or yaml config file
    if config_file.endswith('.json'):
        with open(config_file) as f:
            config = json.load(f)
    elif config_file.endswith(('.yaml', '.yml')):
        with open(config_file) as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config file extension: {config_file}")

    if 'ssd_cache_dir' in config:
        config['ssd_cache_dir'] = parse_path_list(config['ssd_cache_dir'])

    # 中文：过滤分两类——known_config 进构造函数，extra_config 挂成 override_* 属性。
    # 这样配置文件就能覆盖那些"本该由环境变量决定"的 GLOBAL_CONFIG_FROM_ENV 项。
    defined_fields = {f.name for f in fields(UserConfig)}
    known_config = {k: v for k, v in config.items() if k in defined_fields}
    extra_config = {k: v for k, v in config.items() if k not in defined_fields}

    user_config = UserConfig(**known_config)

    for key, value in extra_config.items():
        setattr(user_config, f"override_{key}", value)

    return user_config

def load_user_config_from_env() -> UserConfig:
    """从 FLEXKV_* 环境变量构造 UserConfig——最常用、也最推荐的配置入口。

    注意它与 GLOBAL_CONFIG_FROM_ENV 的分工：
      * 本函数每次调用都**重新读**环境变量，所以可以在 import 之后、初始化之前
        动态调整（GLOBAL_CONFIG_FROM_ENV 在 import 时就固化了，做不到这点）。
      * 本函数只覆盖 UserConfig 里这些"人性化"项；底层传输/调试开关属于
        GLOBAL_CONFIG_FROM_ENV，由模块 import 时读取。

    坑：
      1. 开关类变量一律按 int 解析（'0'/'1'），写 'true' 会抛 ValueError。
      2. 未列出的项不会从环境读取——例如 enable_p2p_cpu / enable_p2p_ssd /
         enable_3rd_remote / zmq / redis 这些**没有**对应的环境变量，
         只能用配置文件（load_user_config_from_file）或直接构造 UserConfig。
      3. FLEXKV_SWA_MULTI_GROUP 的处理是三态的：变量不存在 -> None；
         存在则按 int 转 bool。None 与 False 语义不同，别混用
         （None = 走兼容性默认值，即恢复完整状态）。
    """
    swa_multi_group_env = os.getenv('FLEXKV_SWA_MULTI_GROUP')
    return UserConfig(
        cpu_cache_gb=int(os.getenv('FLEXKV_CPU_CACHE_GB', 16)),
        ssd_cache_gb=int(os.getenv('FLEXKV_SSD_CACHE_GB', 0)),
        ssd_cache_dir=parse_path_list(os.getenv('FLEXKV_SSD_CACHE_DIR', "./flexkv_ssd")),
        enable_gds=bool(int(os.getenv('FLEXKV_ENABLE_GDS', 0))),
        enable_nixl=bool(int(os.getenv('FLEXKV_ENABLE_NIXL', 0))),
        use_hugepage_cpu_buffer=bool(int(os.getenv('FLEXKV_USE_HUGEPAGE_CPU_BUFFER', 0))),
        use_hugepage_tmp_buffer=bool(int(os.getenv('FLEXKV_USE_HUGEPAGE_TMP_BUFFER', 0))),
        hugepage_size_bytes=int(os.getenv('FLEXKV_HUGEPAGE_SIZE_BYTES', 2 * 1024 * 1024)),
        mooncake_max_mr_size_bytes=int(os.getenv(
            'FLEXKV_MOONCAKE_MAX_MR_SIZE_BYTES', 512 * 1024 * 1024 * 1024
        )),
        use_mooncake_store_backend=bool(int(os.getenv('FLEXKV_USE_MOONCAKE_STORE_BACKEND', 0))),
        mooncake_store_config_path=os.getenv('FLEXKV_MOONCAKE_STORE_CONFIG_PATH', None),
        kv_cache_dtype=os.getenv('FLEXKV_KV_CACHE_DTYPE', None),
        swa_multi_group=(
            None
            if swa_multi_group_env is None
            else bool(int(swa_multi_group_env))
        ),
    )

def convert_to_block_num(size_in_GB: float, block_size_in_bytes: int) -> int:
    """把 GB 预算换算成 block 数——"用户说人话、系统按 block 分配"的换算桥。

    中文要点：
      * 这里的 GB 按 1024^3 字节算（严格说是 GiB，与厂商口径的 10^9 不同）。
      * int() 是**向下截断**，所以结果是"不超过预算的最大整数 block 数"，
        不会因四舍五入而超配内存。
      * block_size_in_bytes 必须与目标层（CPU / SSD）实际使用的 block 大小一致，
        否则算出来的容量会系统性偏差；异构模型下这个值由
        block_size_in_bytes_for_cache() 给出，且会在 layer_groups 补齐后重算。
    """
    return int(size_in_GB * 1024 * 1024 * 1024 / block_size_in_bytes)


def block_size_in_bytes_for_cache(
    model_config: ModelConfig,
    cache_config: CacheConfig,
    rank_info: Optional["RankInfo"] = None,
) -> int:
    """Bytes per CPU/SSD block for pool sizing.

    When ``layer_groups`` is set, sum exact per-group bytes after applying each
    group's block compression. Otherwise fall back to the uniform per-PP-stage
    estimate from ``rank_info``.

    中文要点：这是"一个 CPU/SSD block 到底多大"的**权威答案**，所有容量换算都必须
    走这里，不要自己拿 token_size_in_bytes × tokens_per_block 去算。

    为什么两者不等价：异构模型下每个 group 的 compress_ratio 不同，
    "先算每 token 再乘 tokens_per_block" 会对每个 group 的整除结果取整，
    压缩组（如 128x）几乎必然丢字节；正确做法是**先对 tokens_per_block 整除
    compress_ratio，再乘**，这里正是这么做的（并与 KVCacheLayout._compute_kv_shape
    保持逐位一致，否则分配出来的 block 装不下 GPU 侧的数据）。

    参数：layer_groups 为 None（同构）时必须给 rank_info，否则抛 ValueError；
    反之若给了 layer_groups，rank_info 会被忽略。
    """
    if model_config.layer_groups is not None:
        # Match KVCacheLayout._compute_kv_shape exactly.  Computing a rounded
        # per-token size first and multiplying it by tokens_per_block loses
        # bytes for compressed groups whenever the group contribution is not
        # divisible by compress_ratio.
        for gi, group in enumerate(model_config.layer_groups):
            if cache_config.tokens_per_block % group.compress_ratio != 0:
                raise ValueError(
                    f"layer_groups[{gi}].compress_ratio={group.compress_ratio} "
                    f"does not divide tokens_per_block="
                    f"{cache_config.tokens_per_block}"
                )
        return model_config.tp_size * sum(
            group.num_layers
            * model_config.kv_dim
            * (cache_config.tokens_per_block // group.compress_ratio)
            * group.num_kv_heads
            * group.head_size
            * (group.dtype or model_config.dtype).itemsize
            for group in model_config.layer_groups
        )
    if rank_info is None:
        raise ValueError(
            "rank_info is required when model_config.layer_groups is None")
    return rank_info.token_size_in_bytes_per_pp_stage * cache_config.tokens_per_block


def recompute_cache_block_counts(
    model_config: ModelConfig,
    cache_config: CacheConfig,
) -> bool:
    """Recompute ``num_cpu_blocks`` / ``num_ssd_blocks`` from stored GB budgets.

    No-op when ``layer_groups`` is unset (initial uniform estimate is final).
    Returns True if any block count changed.

    中文要点：**延后重算**的入口。调用时机是 layer_groups 刚刚补齐之后——
    异构模型（DSv4）的 layer_groups 常在 update_default_config_from_user_config()
    跑完之后才由 SGLang 建好 GPU KV 池时补登记，而那时第一次换算用的还是同构口径，
    结果是错的。这里靠 CacheConfig._user_cpu_cache_gb / _user_ssd_cache_gb
    记住的用户原始 GB 预算重算一遍。

    返回 True 表示 block 数真的变了——**调用方必须据此决定是否重建缓存池**；
    如果只是算完就丢掉这个返回值，分配器会继续用旧的（可能装不下的）容量。

    注意：SSD 的块数重算后会再次向上对齐到 ssd_cache_dir 的盘数整数倍，
    因此实际值可能大于按 GB 直接算出的值。
    """
    if model_config.layer_groups is None:
        return False

    block_size_in_bytes = block_size_in_bytes_for_cache(
        model_config, cache_config)
    capacity_divisor = 1
    if (model_config.num_kv_heads == 1
            and GLOBAL_CONFIG_FROM_ENV.kv_shared_across_ranks_mode == "all_write"):
        capacity_divisor = max(
            1, model_config.effective_tp_size_per_node)

    changed = False

    if cache_config._user_cpu_cache_gb > 0:
        old_cpu = cache_config.num_cpu_blocks
        new_cpu = (
            convert_to_block_num(
                cache_config._user_cpu_cache_gb, block_size_in_bytes)
            // capacity_divisor
        )
        if new_cpu != old_cpu:
            flexkv_logger.info(
                f"Recomputed num_cpu_blocks with layer_groups: "
                f"{old_cpu} -> {new_cpu} "
                f"(block_size={block_size_in_bytes} B)")
            cache_config.num_cpu_blocks = new_cpu
            changed = True

    if cache_config._user_ssd_cache_gb > 0:
        old_ssd = cache_config.num_ssd_blocks
        new_ssd = (
            convert_to_block_num(
                cache_config._user_ssd_cache_gb, block_size_in_bytes)
            // capacity_divisor
        )
        if new_ssd != old_ssd:
            flexkv_logger.info(
                f"Recomputed num_ssd_blocks with layer_groups: "
                f"{old_ssd} -> {new_ssd} "
                f"(block_size={block_size_in_bytes} B)")
            cache_config.num_ssd_blocks = new_ssd
            changed = True
            if (cache_config.num_ssd_blocks
                    % len(cache_config.ssd_cache_dir) != 0):
                cache_config.num_ssd_blocks = (
                    (cache_config.num_ssd_blocks
                     // len(cache_config.ssd_cache_dir) + 1)
                    * len(cache_config.ssd_cache_dir)
                )

    return changed


def update_default_config_from_user_config(rank_info: RankInfo,
                                           cache_config: CacheConfig,
                                           user_config: UserConfig) -> None:
    """把 UserConfig（GB + 开关）换算并回填到 CacheConfig——配置生效的**主入口**。

    整体流程（按文件中的顺序）：
      1. 算出 block_size_in_bytes（依赖 model_config 与 rank_info）。
      2. 处理 MLA all_write 的容量折损（capacity_divisor）。
      3. GB -> block 数，写入 num_cpu_blocks / num_ssd_blocks，
         并把原始 GB 存进 _user_*_cache_gb 供日后重算。
      4. 逐项回填 SSD / GDS / NIXL / HugePage / Mooncake / p2p 等开关。
      5. 重算派生开关 enable_kv_sharing / enable_remote。
      6. 一致性校验（CPU 必须开、GDS 依赖 SSD、kv_sharing 与 GDS 互斥等）。
      7. 补算 REMOTE 层的 remote_cache_path 与 num_remote_blocks。
      8. 回填 zmq / Redis 等分布式配置（用户没给的保留默认值）。
      9. 处理 override_* —— 用配置文件覆盖 GLOBAL_CONFIG_FROM_ENV。

    中文要点 / 坑：
      * 这是**就地修改** cache_config，没有返回值。cache_config 应当是已构造好的
        实例（其 __post_init__ 已经跑过一轮派生）。
      * 顺序敏感：第 5 步必须在第 4 步之后，因为 enable_kv_sharing / enable_remote
        是由 p2p_* / 3rd_remote / mooncake 派生的，改完底座就要重算。
      * 第 9 步的 override_ 是本文件里唯一"在运行时改 GLOBAL_CONFIG_FROM_ENV"的
        合法途径（该单例在 import 时就固化了，改 os.environ 无效）。
      * 本函数**不会**处理 layer_groups 补齐后的重算，那要另外调
        recompute_cache_block_counts()。
      * 若 use_mooncake_store_backend 为真，只会打一条日志提示 REMOTE2H 是
        prefetch-only；真正的门禁校验在 assert_mooncake_prefetch_ready()。
    """
    block_size_in_bytes = block_size_in_bytes_for_cache(
        rank_info.model_config, cache_config, rank_info)

    assert user_config.cpu_cache_gb > 0
    assert user_config.ssd_cache_gb >= 0

    # MLA all_write mode: each logical KV block occupies N× physical space
    # on CPU/SSD (N GPUs each write a complete KV copy to distinct block slots).
    # To keep the physical memory budget (cpu_cache_gb / ssd_cache_gb) unchanged,
    # the logical block capacity must be divided by N.
    # This mirrors the C++ offset logic in tp_transfer_thread_group.cpp where
    # GPU i writes to cpu_startoff = i * chunk_size, requiring N slots per logical block.
    # 中文补充：只在 MLA（num_kv_heads == 1）且模式为 all_write 时折损容量。
    # 折损的**方向**很重要——除的是"逻辑 block 数"而不是"物理内存预算"，
    # 这样用户填的 cpu_cache_gb 仍然是他实际要付的内存，不会因为模式切换而超配。
    model_config = rank_info.model_config
    kv_shared_across_ranks_mode = GLOBAL_CONFIG_FROM_ENV.kv_shared_across_ranks_mode
    capacity_divisor = 1
    if model_config.num_kv_heads == 1 and kv_shared_across_ranks_mode == "all_write":
        num_gpus_per_node = model_config.effective_tp_size_per_node
        if num_gpus_per_node > 1:
            capacity_divisor = num_gpus_per_node
            flexkv_logger.info(
                f"[config] KV shared across ranks all_write mode: logical cpu/ssd capacity "
                f"÷{num_gpus_per_node} (each block occupies {num_gpus_per_node}× "
                f"physical space, total memory budget unchanged)"
            )

    # Store original GB values for deferred recomputation (when layer_groups become known)
    cache_config._user_cpu_cache_gb = user_config.cpu_cache_gb
    cache_config._user_ssd_cache_gb = user_config.ssd_cache_gb

    cache_config.num_cpu_blocks = (
        convert_to_block_num(user_config.cpu_cache_gb, block_size_in_bytes)
        // capacity_divisor
    )
    cache_config.num_ssd_blocks = (
        convert_to_block_num(user_config.ssd_cache_gb, block_size_in_bytes)
        // capacity_divisor
    )

    flexkv_logger.info(
        f"[CacheConfig] GB->blocks conversion: "
        f"block_size={block_size_in_bytes} B; "
        f"cpu_cache_gb={user_config.cpu_cache_gb} -> num_cpu_blocks={cache_config.num_cpu_blocks}, "
        f"ssd_cache_gb={user_config.ssd_cache_gb} -> num_ssd_blocks={cache_config.num_ssd_blocks}"
    )

    cache_config.ssd_cache_dir = user_config.ssd_cache_dir
    cache_config.enable_ssd = user_config.ssd_cache_gb > 0
    cache_config.enable_gds = user_config.enable_gds
    cache_config.enable_nixl = user_config.enable_nixl
    cache_config.use_hugepage_cpu_buffer = user_config.use_hugepage_cpu_buffer
    cache_config.use_hugepage_tmp_buffer = user_config.use_hugepage_tmp_buffer
    cache_config.hugepage_size_bytes = user_config.hugepage_size_bytes
    cache_config.mooncake_max_mr_size_bytes = user_config.mooncake_max_mr_size_bytes
    cache_config.enable_p2p_cpu = user_config.enable_p2p_cpu
    cache_config.enable_p2p_ssd = user_config.enable_p2p_ssd
    cache_config.enable_3rd_remote = user_config.enable_3rd_remote
    cache_config.use_mooncake_store_backend = user_config.use_mooncake_store_backend
    cache_config.mooncake_store_config_path = user_config.mooncake_store_config_path
    cache_config.mooncake_store_pp_rank = int(rank_info.pp_rank)
    cache_config.mooncake_store_pp_size = int(rank_info.model_config.pp_size)
    cache_config.mooncake_store_total_layers = int(rank_info.model_config.num_layers)
    # 中文：Mooncake 是按 (层范围) 分片的——单节点时本节点负责全部层；多节点时
    # 这里**故意不改** start/end，留给上层按实际切分填（默认 0 表示未划分）。
    # 也就是说：只有 nnodes == 1 这个分支会真正写入层范围。
    if int(rank_info.model_config.nnodes) == 1:
        cache_config.mooncake_store_node_layer_start = 0
        cache_config.mooncake_store_node_layer_end = int(rank_info.model_config.num_layers)
    # Update derived flags after setting p2p and remote configs
    cache_config.enable_kv_sharing = (cache_config.enable_p2p_cpu or
                                      cache_config.enable_p2p_ssd or
                                      cache_config.enable_3rd_remote)
    cache_config.enable_remote = (cache_config.enable_3rd_remote or
                                  cache_config.use_mooncake_store_backend)

    if cache_config.use_mooncake_store_backend:
        flexkv_logger.info(
            "Mooncake store: REMOTE2H is prefetch-only; "
            "compute GET will force ignore_remote"
        )

    if cache_config.num_ssd_blocks % len(cache_config.ssd_cache_dir) != 0:
        cache_config.num_ssd_blocks = \
            (cache_config.num_ssd_blocks // len(cache_config.ssd_cache_dir) + 1) * len(cache_config.ssd_cache_dir)
        flexkv_logger.warning(f"num_ssd_blocks is not a multiple of num_ssd_devices, "
                              f"adjust num_ssd_blocks to {cache_config.num_ssd_blocks}")

    # 中文：CPU 是**强制**的一级缓存——SSD / REMOTE 都以它为中转，关掉整条链路就断了。
    if not cache_config.enable_cpu:
        raise ValueError("enable_cpu must be True")
    # SSD and REMOTE are peer cold tiers under CPU (H2DISK vs H2REMOTE);
    # enabling remote does not require a local SSD mid-tier.
    # 中文：三条互斥/依赖约束，全部在启动期拦下，避免运行期出现"配了但没走"的假象。
    # 注意：第一条的前半段 `not enable_cpu` 在上面已被拒，因此
    # "enable_gds must be True if enable_cpu is False" 这条分支**永远进不去**——
    # 它是为"未来允许关掉 CPU 层"预留的，目前是死代码，别依赖它做校验。
    if not cache_config.enable_cpu and not cache_config.enable_gds:
        raise ValueError("enable_gds must be True if enable_cpu is False")
    # GDS 是 GPU<->SSD 直连，没开 SSD 就没有直连对象。
    if cache_config.enable_gds and not cache_config.enable_ssd:
        raise ValueError("enable_ssd must be True if enable_gds is True")
    # kv_sharing（跨节点复用）与 GDS（本地直连）在传输路径上冲突，不能同时开。
    if cache_config.enable_kv_sharing and cache_config.enable_gds:
        raise ValueError(
            "enable_kv_sharing and enable_gds cannot be used at the same time"
        )

    # 中文：只有"第三方远端（CFS）"才走下面这套 remote_file_* / remote_cache_path
    # 的容量推导；Mooncake store 后端的容量由它自己的配置文件决定，所以被排除在外。
    if cache_config.enable_remote and not cache_config.use_mooncake_store_backend:
        if cache_config.remote_cache_path is None:
            if cache_config.remote_file_prefix is None:
                raise ValueError(
                    "remote_file_prefix must be provided when remote_cache_path is None"
                )
            if (cache_config.remote_file_num is None
                    or cache_config.remote_file_num <= 0):
                raise ValueError("remote_file_num must be a positive integer")
            cache_config.remote_cache_path = [
                f"{cache_config.remote_file_prefix}_{i}"
                for i in range(cache_config.remote_file_num)
            ]

        if cache_config.remote_cache_size_mode not in ("block_num", "file_size"):
            raise ValueError(
                f"remote_cache_size_mode must be 'block_num' or 'file_size', "
                f"got {cache_config.remote_cache_size_mode!r}"
            )

        if cache_config.remote_cache_size_mode == "file_size":
            if cache_config.remote_file_size is None:
                raise ValueError(
                    "remote_file_size must be set when remote_cache_size_mode == 'file_size'"
                )
            if (cache_config.remote_file_num is None
                    or cache_config.remote_file_num <= 0):
                raise ValueError("remote_file_num must be a positive integer")
            cache_config.num_remote_blocks = (
                cache_config.remote_file_size // block_size_in_bytes
                * cache_config.remote_file_num
            )
            flexkv_logger.info(
                f"num_remote_blocks derived from remote_file_size "
                f"(per-pp-stage, num_layers_per_pp_stage="
                f"{rank_info.num_layers_per_pp_stage}): "
                f"remote_file_size={cache_config.remote_file_size}, "
                f"remote_file_num={cache_config.remote_file_num}, "
                f"block_size_in_bytes={block_size_in_bytes} "
                f"-> num_remote_blocks={cache_config.num_remote_blocks}"
            )

        if (cache_config.num_remote_blocks is None
                or cache_config.num_remote_blocks <= 0):
            raise ValueError(
                "num_remote_blocks must be a positive integer "
                "(file_size mode: derived above from remote_file_size; "
                "block_num mode: set it explicitly)"
            )

    # Update distributed zmq and Redis configs if provided in user_config
    if user_config.local_zmq_ip is not None:
        cache_config.local_zmq_ip = user_config.local_zmq_ip
    if user_config.local_zmq_port is not None:
        cache_config.local_zmq_port = user_config.local_zmq_port
    if user_config.redis_host is not None:
        cache_config.redis_host = user_config.redis_host
    if user_config.redis_port is not None:
        cache_config.redis_port = user_config.redis_port
    if user_config.local_ip is not None:
        cache_config.local_ip = user_config.local_ip
    if user_config.redis_password is not None:
        cache_config.redis_password = user_config.redis_password
    if user_config.node_ttl_seconds is not None:
        cache_config.node_ttl_seconds = user_config.node_ttl_seconds

    # 中文：处理配置文件里的 override_<name> —— 这是**唯一**能在运行时改写
    # GLOBAL_CONFIG_FROM_ENV 的合法途径（该单例 import 时就固化了，改 os.environ
    # 是无效的）。转换规则是"以环境变量里已有的值的类型为目标类型"：
    # bool 兼容 'true'/'1'/'yes' 字符串与数值，枚举（KVCacheLayoutType）接受
    # 大小写不敏感的字符串，其余直接强转。名字对不上会抛错并列出所有可覆盖项。
    global_config_attrs = set(vars(GLOBAL_CONFIG_FROM_ENV).keys())
    for attr_name in dir(user_config):
        if attr_name.startswith('override_'):
            global_attr_name = attr_name[9:]  # len('override_') = 9
            if global_attr_name in global_config_attrs:
                attr_value = getattr(user_config, attr_name)
                original_value = getattr(GLOBAL_CONFIG_FROM_ENV, global_attr_name)

                original_type = type(original_value)

                try:
                    if original_type is bool:
                        if isinstance(attr_value, str):
                            attr_value = attr_value.lower() in ('true', '1', 'yes')
                        else:
                            attr_value = bool(int(attr_value))
                    elif issubclass(original_type, Enum):  # KVCacheLayoutType
                        if isinstance(attr_value, str):
                            attr_value = original_type(attr_value.upper())
                        elif not isinstance(attr_value, original_type):
                            attr_value = original_type(attr_value)
                    else:
                        attr_value = original_type(attr_value)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Cannot convert config value '{attr_value}' to type {original_type.__name__} "
                                    f"for config '{global_attr_name}': {e}") from e

                setattr(GLOBAL_CONFIG_FROM_ENV, global_attr_name, attr_value)
                flexkv_logger.info(f"Override environment variable: {'FLEXKV_' + global_attr_name.upper()} "
                                   f"to {attr_value} from config file.")
            else:
                raise ValueError(f"Unknown config name: {global_attr_name} in config file, "
                                 f"available config names: {global_config_attrs}")


@dataclass
class MooncakeTransferEngineConfig:
    """Mooncake **Transfer Engine**（RDMA 传输引擎）的连接参数。

    中文要点：这里配置的是 Mooncake 的**传输引擎**，即底层 RDMA 通路的连接信息，
    对应 CacheConfig.mooncake_config_path 指向的那个 JSON 文件。
    它**不是** Mooncake Store 分布式 KV 后端（后者由
    CacheConfig.use_mooncake_store_backend / mooncake_store_config_path 控制），
    两者虽然都叫 Mooncake，但一个是"路"，一个是"仓库"，别混。

    字段全部**无默认值**（必填），构造时少一个都会 TypeError；但通过
    from_dict() 加载时每个字段都有兜底默认值，所以真实用法通常是 from_file /
    load_from_env，而不是直接构造。

    典型字段：
      engine_ip / engine_port —— 本节点 RDMA 网卡监听地址。
      metadata_backend / metadata_server / metadata_server_auth —— 元数据服务
        （默认 redis://127.0.0.1:6380，注意默认密码是占位串 "yourpass"，
        生产环境必须换）。
      protocol —— 传输协议，默认 "rdma"。
      device_name —— 使用的 RDMA 设备名，空串表示由 Mooncake 自选。
    """
    engine_ip: str
    engine_port: int
    metadata_backend: Union[str, None]
    metadata_server: str
    metadata_server_auth: str
    protocol: str
    device_name: str
    # redis_server: str
    # redis_db: int
    # redis_auth: str


    @staticmethod
    def from_file(file_path: str) -> "MooncakeTransferEngineConfig":
        """Load the config from a JSON file."""
        with open(file_path) as fin:
            config = json.load(fin)
        return MooncakeTransferEngineConfig.from_dict(config)


    @staticmethod
    def load_from_env(env_name: str) -> "MooncakeTransferEngineConfig":
        """Load config from a file specified in the environment variable.

        中文：从 env_name 指定的**文件路径**加载（读的是文件，不是环境变量的值）。
        默认用于 MOONCAKE_CONFIG_PATH 这一支；变量未设置时抛 ValueError。

        坑：报错信息里写死了 "MOONCAKE_CONFIG_PATH"，即使你传的是别的 env_name
        也会显示那串固定文案——按变量名排查时别被误导。
        """
        config_file_path = os.getenv(env_name)
        if config_file_path is None:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set."
            )
        return MooncakeTransferEngineConfig.from_file(config_file_path)


    @staticmethod
    def from_dict(config: dict) -> "MooncakeTransferEngineConfig":
        """Load the config from a JSON file.

        中文：上一条英文 docstring 写的是 "from a JSON file"，实际入参是**已解析好的
        dict**（from_file 负责读文件再转给它）。这是文档笔误，行为以签名为准。
        缺失字段一律用默认值兜底，因此 JSON 里只写 engine_ip 这类关键项也能跑。
        """
        return MooncakeTransferEngineConfig(
            engine_ip=config.get("engine_ip", "127.0.0.1"),
            engine_port=config.get("engine_port", 5555),
            metadata_backend=config.get("metadata_backend", "redis"),
            metadata_server=config.get("metadata_server", "redis://127.0.0.1:6380"),
            metadata_server_auth=config.get("metadata_server_auth", "yourpass"),
            protocol=config.get("protocol", "rdma"),
            device_name=config.get("device_name", ""),
            # redis_server=config.get("redis_server", "redis://127.0.0.1:6379"),
            # redis_db=config.get("redis_db", 0),
            # redis_auth=config.get("redis_auth", "yourpass"),
        )
