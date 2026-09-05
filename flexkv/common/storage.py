# ==============================================================================
# flexkv/common/storage.py —— 存储层的公共类型层：KV Cache 布局 + 访问句柄
# ------------------------------------------------------------------------------
# 职责：只回答两个问题——"这段 KV Cache 在内存里长什么样"（KVCacheLayout）和
# "怎么拿到它"（StorageHandle）。本文件不分配内存、不搬任何字节、不含任何 IO
# 逻辑：真正的分配在 storage/allocator.py，编排在 storage/storage_engine.py，
# 真正的搬运在 transfer/worker.py。
#
# 在架构中的位置（本文件是最底层的公共依赖之一）：
#
#   控制面 cache/cache_engine.py ── 决策"搬哪些 block"，产出 TransferOpGraph
#        │
#   数据面 transfer/transfer_engine.py ── 调度 DAG，把 op 派给 Worker 子进程
#        │
#   存储层 storage/storage_engine.py ── 编排 GPU / CPU / SSD / 远端 三级缓存
#        │
#        ├── storage/allocator.py ── 按 KVCacheLayout 算容量并产出 StorageHandle ─┐
#        └── transfer/worker.py ── 按 KVCacheLayout 的 stride 定位 block 并读写 ──┤
#                                                                               └─ 共同依赖本文件
#
#   换句话说，layout 是控制面 / 数据面 / 存储层三方对齐的**坐标系**：分配器按它
#   算容量，worker 按它算偏移，两边必须严格一致，否则会静默写错位置而不报错。
#
# 为什么有三种 KVCacheLayoutType（注意：三种是"维度排列顺序"，不是三个框架）：
#   LAYERFIRST  [num_layer, kv_dim, num_block, tpb, num_head, head_size]
#               —— 层在最外、block 在内。vLLM <= 0.21 的非 MLA GPU KV cache 形状；
#                  逐层传输（layerwise）也用它，因为一次只搬一层时层必须是最外层。
#   BLOCKFIRST  [num_block, num_layer, kv_dim, tpb, num_head, head_size]
#               —— block 在最外，单个 block 的字节完全连续。CPU / SSD / 远端默认
#                  用它，因为落盘和"按 block 粒度搬运"都要求一个 block 是连续内存。
#   LAYERBLOCK  [num_layer, num_block, kv_dim, tpb, num_head, head_size]
#               —— 层在最外、block 次之。vLLM >= 0.23 的非 MLA GPU KV cache 形状。
#   此外还有"多组（layer_groups）"模式，它只支持 BLOCKFIRST，且此时 kv_shape 退化
#   成二维 [num_block, bytes_per_block]，第二维是**字节数**而不是元素数，详见
#   _compute_kv_shape 内注释。
#
# 关键名字清单：
#   AccessHandleType —— 句柄背后的"介质形态"：张量 / 文件 / 可跨进程共享的张量
#                       句柄 / GDS（GPU Direct Storage）管理器。
#   KVCacheLayout    —— 本文件主角：布局描述 + 各种 stride 推导（get_*_stride）。
#   StorageHandle    —— 分配器产出的"可访问句柄"，携带布局、dtype 与各介质私有
#                       元数据，是 allocator 与 worker 之间传递的包裹。
#
# 阅读提示 / 容易踩的坑：
#   1. 【stride 的单位会变】普通模式下 get_block_stride / get_layer_stride /
#      get_kv_stride / get_chunk_size / get_elements_per_block 返回的都是**元素
#      个数**；但多组模式下 get_block_stride 返回的是**字节数**（kv_shape 第二维
#      本身就是字节），而 get_layer_stride / get_kv_stride / get_chunk_size 直接
#      抛 ValueError。单位混淆是这类代码最典型的 bug。
#   2. 【div_layer / div_head 会丢多组配置】div_block 透传了 layer_groups 与
#      tp_size，而 div_layer 与 div_head 只透传到了 num_kv_heads。多组布局经它们
#      切分后会**静默退化成普通布局**（layer_groups 变成 None，于是 _validate_layout
#      也无从报错）。改动这块时先确认是否需要补传。
#   3. 【KVCacheLayout 不可哈希】类里手写了 __eq__ 却没有配套定义 __hash__，
#      Python 会因此把 __hash__ 置为 None，所以它不能当 dict key、也不能放进 set。
#   4. 【_kv_shape 是懒计算的缓存】它同时是一个 dataclass 字段（构造时要写成
#      _kv_shape=...），默认 None，首次访问 kv_shape / get_total_elements 时才由
#      _compute_kv_shape 填充；__post_init__ 里已主动算过一次，所以通常已就绪。
#   5. 【__eq__ 比的是 kv_shape 而不是 _kv_shape】这会触发懒计算，因此"比较两个
#      layout"这个动作本身也可能顺带把 _kv_shape 填上。
#
# 建议阅读顺序：
#   AccessHandleType -> KVCacheLayoutType -> KVCacheLayout（重点看 _compute_kv_shape）
#   -> get_layer_stride / get_block_stride / get_kv_stride / get_chunk_size
#   -> get_group_strides（多组模式，建议第二遍再读）-> div_block / div_layer / div_head
#   -> StorageHandle（及其各个 get_* 访问器）
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Union, List, Optional, Any, Dict, TYPE_CHECKING

import torch

from flexkv.common.memory_handle import TensorSharedHandle

if TYPE_CHECKING:
    from flexkv.common.config import LayerGroupSpec


class AccessHandleType(Enum):
    """StorageHandle.data 里装的东西是什么形态。

    它决定了该用 StorageHandle 里的哪个访问器去取数据，取错访问器会直接抛
    ValueError（见 StorageHandle.get_tensor / get_file_list 等）。

    四种形态的取舍：
      TENSOR        —— 进程内的普通 torch.Tensor（或 Tensor 列表）。最快，但
                       无法跨进程传递，Worker 子进程拿到的是拷贝。
      FILE          —— 落盘文件的路径列表（SSD 场景还可能是
                       {ssd_device_id: [path, ...]} 的字典，用于多盘条带）。
      TENSOR_HANDLE —— TensorSharedHandle 列表，内部封装 CUDA IPC handle，
                       专用于把 GPU 显存"句柄"跨进程交给 Worker 子进程，
                       由子进程在自己的上下文里重新映射出可用的 tensor。
      GDS_MANAGER   —— GPU Direct Storage 管理器，让 SSD 绕过 CPU 内存直接与
                       GPU 显存 DMA。注意 StorageHandle 目前**没有**为它提供
                       对应的访问器方法，属于预留形态。
    """
    TENSOR = auto()  # single tensor or tensor list
    FILE = auto()  # single file or file list
    TENSOR_HANDLE = auto()  # single tensor handle or tensor handle list
    GDS_MANAGER = auto()

# NOTE: GPU layout depends on the vLLM version's non-MLA KV cache shape:
#   vLLM <= 0.21: (kv, num_blocks, ...)  -> LAYERFIRST
#   vLLM >= 0.23: (num_blocks, kv, ...)  -> LAYERBLOCK
# CPU, SSD, remote layout should be the same, either LAYERFIRST or BLOCKFIRST.
class KVCacheLayoutType(Enum):
    """KV Cache 六个维度（层、K/V、block、token、头、头内维度）的排列顺序。

    为什么这很重要：同一份数据在不同介质上要求不同的连续性。GPU 侧要匹配
    vLLM 分配的形状（否则不能零拷贝对接），而 CPU / SSD / 远端要求"一个 block
    是一段连续内存"（否则无法按 block 落盘和搬运）。本枚举就是把这两种诉求
    显式化，让 stride 推导（get_block_stride 等）可以按类型分派。

    取值用字符串而非 auto()，是为了便于写进配置、日志和跨进程传递时可读。
    """
    LAYERFIRST = "LAYERFIRST"
    BLOCKFIRST = "BLOCKFIRST"
    LAYERBLOCK = "LAYERBLOCK"

@dataclass
class KVCacheLayout:
    """一份 KV Cache 的完整布局描述：维度顺序 + 各维长度 + 由它们推导出的 stride。

    什么时候用：分配器（storage/allocator.py）用它算出"要申请多少元素"，然后原样
    塞进 StorageHandle 交给数据面；Worker（transfer/worker.py）再用 get_*_stride
    把 block_id 换算成内存偏移。它本身不持有任何内存，只是一个**描述对象**。

    字段说明：
      type             —— 维度排列顺序，见 KVCacheLayoutType。
      num_layer        —— 层数。
      num_block        —— block 个数（KV Cache 的最小管理单元，见 common/block.py）。
      tokens_per_block —— 每个 block 装多少 token。
      num_head / head_size —— 注意力头的形状。注意这里是 num_head；多组模式下
                        真正参与计算的是每个 group 自己的 num_kv_heads / head_size。
      kv_dim           —— K 与 V 两个分量，固定为 2。
      num_kv_heads     —— 单卡（TP 切分后）的 KV 头数。均匀布局下等于 num_head
                        的 KV 侧值；多组模式下不参与 kv_shape 计算。
      _kv_shape        —— 布局对应的 torch 形状，**懒计算的缓存**。它同时是一个
                        dataclass 字段（构造时写作 _kv_shape=...），默认 None，
                        由 _compute_kv_shape 填充。外部一律走 kv_shape 属性。
      layer_groups     —— 多组（异构）布局：不同层组可以有不同的 num_kv_heads /
                        head_size / dtype / compress_ratio（典型场景是 DSv4 的
                        bf16 主 KV + uint8 indexer）。**非空时强制要求
                        type == BLOCKFIRST**，见 _validate_layout。
      tp_size          —— TP 并行度。多组模式下它会把 bytes_per_block 放大，
                        使一个 CPU/SSD block 能装下所有 TP rank 的数据
                        （因为 layer_groups 里存的是 TP 切分后的单卡 num_kv_heads）。

    坑：
      * 多组模式下 kv_shape 是 [num_block, bytes_per_block]，第二维是**字节**，
        不是元素数；get_*_stride 的单位也随之改变（见文件头第 1 条）。
      * 本类不可哈希（见文件头第 3 条）。
    """
    type: KVCacheLayoutType
    num_layer: int
    num_block: int
    tokens_per_block: int
    num_head: int
    head_size: int
    kv_dim: int = 2  # K 与 V 两个分量，永远为 2
    num_kv_heads: int = 1
    # 既是 dataclass 字段又是懒计算缓存：None 表示"尚未算过"，
    # 首次访问 kv_shape / get_total_elements 时由 _compute_kv_shape 填充。
    _kv_shape: Optional[torch.Size] = None
    # Multi-group support: when set, the layout represents a heterogeneous block
    # where different layer groups have different (num_kv_heads, head_size).
    # Requires type == BLOCKFIRST (enforced in __post_init__).
    layer_groups: Optional[List[LayerGroupSpec]] = None
    # TP size: when > 1 and layer_groups is set, elements_per_block is scaled
    # so that each CPU/SSD block can hold data for all TP ranks.
    tp_size: int = 1

    def __eq__(self, other: object) -> bool:
        """逐个字段比较布局是否等价（用于"两个 handle 的布局能否直接对接"的判断）。

        注意两处与 dataclass 默认行为不同的地方：
          1. 这里比的是 kv_shape（而非 _kv_shape），所以比较动作本身会触发懒计算；
          2. 手写了 __eq__ 却没有定义 __hash__，Python 会把 __hash__ 置为 None，
             因此 KVCacheLayout **不可哈希**，不能做 dict key 或塞进 set。

        与不同类型比较时返回 NotImplemented（而不是 False），让 Python 去尝试
        反射操作，找不到才回落为 False —— 这是 __eq__ 的标准写法。
        """
        if not isinstance(other, KVCacheLayout):
            return NotImplemented
        return (self.type == other.type and
                self.num_layer == other.num_layer and
                self.num_block == other.num_block and
                self.tokens_per_block == other.tokens_per_block and
                self.num_head == other.num_head and
                self.head_size == other.head_size and
                self.kv_dim == other.kv_dim and
                self.num_kv_heads == other.num_kv_heads and
                self.layer_groups == other.layer_groups and
                self.tp_size == other.tp_size and
                self.kv_shape == other.kv_shape)

    @property
    def kv_shape(self) -> torch.Size:
        """布局对应的 torch 形状，懒计算并缓存到 _kv_shape。

        多组模式下形状是二维 [num_block, bytes_per_block]，第二维单位是**字节**；
        其余模式是六维，单位都是元素数。分配器基本只关心它的 numel()。
        """
        if self._kv_shape is None:
            self._compute_kv_shape()
        assert self._kv_shape is not None
        return self._kv_shape

    def __post_init__(self) -> None:
        """构造后立即校验并算好 kv_shape。

        顺序很重要：必须先 _validate_layout 再 _compute_kv_shape。后者在多组分支
        里只有一句 assert（不是 ValueError）来要求 BLOCKFIRST，那道 assert 其实是
        "防御性"的——正常情况下非法组合早在 _validate_layout 就抛掉了。所以如果
        有人绕过 __post_init__ 直接调 _compute_kv_shape（比如手工改完字段后重算），
        就会直面 AssertionError 而不是友好的 ValueError。
        """
        self._validate_layout()
        self._compute_kv_shape()

    def _validate_layout(self) -> None:
        """Fail fast on unsupported layout combinations.

        中文要点：目前只拦一条规则——多组布局（layer_groups 非空）必须是
        BLOCKFIRST。原因是异构层组靠"把各组区域按字节平铺进一个 block"来实现，
        这种铺法只有在 block 连续（BLOCKFIRST）时才有定义；LAYERFIRST /
        LAYERBLOCK 的六维形状根本表达不了"不同层组 dtype 不同"这件事。
        """
        if self.layer_groups is not None and self.type != KVCacheLayoutType.BLOCKFIRST:
            raise ValueError(
                "Multi-group KVCacheLayout (layer_groups is set) only supports "
                f"BLOCKFIRST layout, got {self.type.value}. "
                "Heterogeneous layer groups pack per-group regions into a single "
                "block in byte-flat BLOCKFIRST order; LAYERFIRST/LAYERBLOCK are "
                "not defined for this mode."
            )

    def _compute_kv_shape(self) -> None:
        """按 type（以及是否多组）推导 kv_shape，结果缓存进 _kv_shape。

        两个分支的本质区别：
          * 多组分支：形状退化成二维 [num_block, bytes_per_block]。之所以用字节
            而不是元素数，是因为各组 dtype 可能不同（bf16 主 KV 配 uint8 indexer），
            无法提出一个统一的 itemsize 把元素数换算成字节，只能直接记字节。
          * 普通分支：六维，六种排列顺序按 KVCacheLayoutType 一一对应。

        幂等：_kv_shape 非 None 时整个方法直接空转，所以可以被反复调用。
        """
        if self._kv_shape is None:
            if self.layer_groups is not None:
                assert self.type == KVCacheLayoutType.BLOCKFIRST, (
                    "multi-group layout requires BLOCKFIRST; "
                    "call _validate_layout() before _compute_kv_shape()"
                )
                # Multi-group: kv_shape's second dim is BYTES per block, not
                # element count.  Groups may carry different dtypes (e.g. bf16
                # main KV + uint8 indexer in DSv4), so we cannot factor out a
                # single dtype_size.  Each group contributes
                #   num_layers * kv_dim * tokens_per_block * num_kv_heads *
                #   head_size * dtype.itemsize
                # bytes.  kv_dim and tokens_per_block are taken from the layout
                # top level (uniform across groups in supported configs).
                # layer_groups store per-GPU num_kv_heads (after TP split);
                # tp_size multiplies so each block holds data for ALL TP ranks.
                if any(g.dtype is None for g in self.layer_groups):
                    raise ValueError(
                        "Multi-group KVCacheLayout requires LayerGroupSpec.dtype "
                        "to be set on every group (resolve None to ModelConfig.dtype "
                        "before constructing the layout)."
                    )
                # Per-group tokens_per_block after compression. The CPU/SSD block
                # only stores ``tpb_g`` compressed tokens per group, matching the
                # shrunk shape sglang allocates on the GPU for that group.
                for gi, g in enumerate(self.layer_groups):
                    if self.tokens_per_block % g.compress_ratio != 0:
                        raise ValueError(
                            f"KVCacheLayout: layer_groups[{gi}].compress_ratio="
                            f"{g.compress_ratio} does not divide tokens_per_block="
                            f"{self.tokens_per_block}. Choose a page_size that is a "
                            f"multiple of every group's compress_ratio."
                        )
                bytes_per_block = self.tp_size * sum(
                    g.num_layers * self.kv_dim *
                    (self.tokens_per_block // g.compress_ratio) *
                    g.num_kv_heads * g.head_size * g.dtype.itemsize
                    for g in self.layer_groups
                )
                # 第二维单位是字节，不是元素数 —— 这是多组模式与普通模式最关键
                # 的语义差异，后面 get_block_stride 会原样返回它。
                self._kv_shape = torch.Size([self.num_block, bytes_per_block])
            elif self.type == KVCacheLayoutType.LAYERFIRST:  # for Layerwise transfer
                self._kv_shape = torch.Size([self.num_layer,
                                             self.kv_dim,
                                             self.num_block,
                                             self.tokens_per_block,
                                             self.num_head,
                                             self.head_size])
            elif self.type == KVCacheLayoutType.BLOCKFIRST:
                self._kv_shape = torch.Size([self.num_block,
                                             self.num_layer,
                                             self.kv_dim,
                                             self.tokens_per_block,
                                             self.num_head,
                                             self.head_size])
            elif self.type == KVCacheLayoutType.LAYERBLOCK:  # vLLM >= 0.23 non-MLA GPU layout
                self._kv_shape = torch.Size([self.num_layer,
                                             self.num_block,
                                             self.kv_dim,
                                             self.tokens_per_block,
                                             self.num_head,
                                             self.head_size])
            else:
                raise ValueError(f"Invalid KVCacheLayoutType: {self.type}")

    def div_block(self, num_chunks: int, padding: bool = False) -> KVCacheLayout:
        """把 block 维切成 num_chunks 份，返回"其中一份"的布局。

        用途：block 维并行（多路 CUDA stream、或多个 Worker 各搬一段）时，用派生
        出的子布局描述每人负责的那一段。子布局的 stride 与原布局完全一致，只有
        num_block 变小，因此偏移换算可以照抄原布局。

        参数 padding：
          False（默认）—— 要求 num_block 能被 num_chunks 整除，否则 assert 失败。
          True        —— 向上取整 (num_block + num_chunks - 1) // num_chunks，
                         最后一份会多出若干个"凑数"的 block。这些多出来的 block
                         业务上没有对应数据，读写时要小心不要越界。

        本方法透传了 layer_groups 与 tp_size，多组布局可以安全切分；
        但 div_layer / div_head 没透传这两项，见文件头第 2 条。
        """
        if padding:
            num_blocks = (self.num_block + num_chunks - 1) // num_chunks
        else:
            assert self.num_block % num_chunks == 0, \
                f"num_block {self.num_block} must be divisible by num_chunks {num_chunks}"
            num_blocks = self.num_block // num_chunks
        new_layout = KVCacheLayout(
            type=self.type,
            num_layer=self.num_layer,
            num_block=num_blocks,
            tokens_per_block=self.tokens_per_block,
            num_head=self.num_head,
            head_size=self.head_size,
            kv_dim=self.kv_dim,
            num_kv_heads=self.num_kv_heads,
            layer_groups=self.layer_groups,
            tp_size=self.tp_size,
        )
        return new_layout

    def div_layer(self, num_chunks: int) -> KVCacheLayout:
        """把层维切成 num_chunks 份，返回"其中一份"（若干连续层）的布局。

        用途：逐层传输（layerwise）与按层分片的场景，要求 num_layer 能被
        num_chunks 整除，否则 assert 失败。

        坑：构造新布局时**没有透传 layer_groups 与 tp_size**（对比 div_block 是
        透传的）。多组布局经本方法切分后会静默退化成普通布局，且因为 layer_groups
        变成 None，_validate_layout 也无从报错，最终算出一个"看起来对、实际错"的
        kv_shape。要让多组模式支持按层切分，必须先在这里补传。
        """
        assert self.num_layer % num_chunks == 0, \
            f"num_layer {self.num_layer} must be divisible by num_chunks {num_chunks}"
        new_layout = KVCacheLayout(
            type=self.type,
            num_layer=self.num_layer // num_chunks,
            num_block=self.num_block,
            tokens_per_block=self.tokens_per_block,
            num_head=self.num_head,
            head_size=self.head_size,
            kv_dim=self.kv_dim,
            num_kv_heads=self.num_kv_heads,
        )
        return new_layout

    def div_head(self, num_chunks: int) -> KVCacheLayout:
        """把注意力头维切成 num_chunks 份，返回"其中一份"的布局。

        用途：按 head 分片搬运（例如多卡各搬一部分 KV head）。要求 num_head 能被
        num_chunks 整除，否则 assert 失败。

        两个坑：
          1. 与 div_layer 一样，没有透传 layer_groups 与 tp_size，多组布局经此
             切分会静默退化成普通布局（见文件头第 2 条）。
          2. num_head 被除以 num_chunks，但 num_kv_heads **原样保留**。num_kv_heads
             在普通布局的 kv_shape 计算里并不参与（kv_shape 用的是 num_head），
             所以结果与 kv_shape 自洽；但如果下游拿 num_kv_heads 去做自己的容量
             推算，就会得到未缩放的旧值，需要留意。
        """
        assert self.num_head % num_chunks == 0, \
            f"num_head {self.num_head} must be divisible by num_chunks {num_chunks}"
        new_layout = KVCacheLayout(
            type=self.type,
            num_layer=self.num_layer,
            num_block=self.num_block,
            tokens_per_block=self.tokens_per_block,
            num_head=self.num_head // num_chunks,
            head_size=self.head_size,
            kv_dim=self.kv_dim,
            num_kv_heads=self.num_kv_heads,
        )
        return new_layout

    def get_chunk_size(self) -> int:
        """一个"K 分量"或"V 分量"的元素数 = tokens_per_block * num_head * head_size。

        这是搬运的最小连续单元：一次搬一个（K 或 V）chunk。它同时也是 get_kv_stride
        在 BLOCKFIRST / LAYERBLOCK 下的值（那里一个 block 内 K、V 紧挨着，间距
        恰好就是一个 chunk）。

        多组模式下没有统一的答案，直接抛 ValueError，请改用 get_group_strides 里
        每个组各自的 chunk_size。
        """
        if self.layer_groups is not None:
            raise ValueError("get_chunk_size() is not valid for multi-group layout; "
                             "use per-group strides instead")
        return self.tokens_per_block * self.num_head * self.head_size

    def get_layer_stride(self) -> int:
        """相邻两层之间的元素间距（层 stride），单位：元素数。

        写法统一是"取 kv_shape 中"层维之后的所有维"的 numel()，下标随 type 变化：

            LAYERFIRST [层, kv, block, tpb, head, hs] -> kv_shape[1:]
            BLOCKFIRST [block, 层, kv, tpb, head, hs] -> kv_shape[2:]
            LAYERBLOCK [层, block, kv, tpb, head, hs] -> kv_shape[1:]

        多组模式下"层"不再是独立维度（各组层数是可变的），无定义，抛 ValueError。
        """
        if self.layer_groups is not None:
            raise ValueError("get_layer_stride() is not valid for multi-group layout; "
                             "use per-group strides instead")
        if self.type == KVCacheLayoutType.LAYERFIRST:
            return self.kv_shape[1:].numel()
        elif self.type == KVCacheLayoutType.BLOCKFIRST:
            return self.kv_shape[2:].numel()
        elif self.type == KVCacheLayoutType.LAYERBLOCK:
            return self.kv_shape[1:].numel()
        else:
            raise ValueError(f"Invalid KVCacheLayoutType: {self.type}")

    def get_block_stride(self) -> int:
        """"一个 block 有多大"，即相邻两个 block 之间的间距（block stride）。

        这是数据面用得最多的一个 stride：block_id 到偏移的换算全靠它
        （offset = block_id * block_stride）。

        **单位陷阱（务必注意）**：
          * 多组模式 —— 返回 kv_shape[1]，那已经是**字节数**了，不要（也不能）
            再乘 dtype.itemsize；
          * 普通模式 —— 返回的是**元素数**，换算字节时才需要乘 dtype.itemsize。
        同一个方法在两条分支下单位不同，是这块最容易出错的地方。
        """
        if self.layer_groups is not None:
            # For multi-group BLOCKFIRST, kv_shape is [num_block, bytes_per_block]
            # — the value is already in BYTES, do not multiply by dtype.itemsize.
            return self.kv_shape[1]
        if self.type == KVCacheLayoutType.LAYERFIRST:
            return self.kv_shape[3:].numel()
        elif self.type == KVCacheLayoutType.BLOCKFIRST:
            return self.kv_shape[1:].numel()
        elif self.type == KVCacheLayoutType.LAYERBLOCK:
            return self.kv_shape[2:].numel()
        else:
            raise ValueError(f"Invalid KVCacheLayoutType: {self.type}")

    def get_kv_stride(self) -> int:
        """同一个 block 内、K 分量起点到 V 分量起点的元素间距（kv stride）。

        规率是"取 kv_dim 那一维之后的所有维"的 numel()，而 kv_dim 的下标因 type
        而异：LAYERFIRST 在 1（故取 [2:]），BLOCKFIRST 与 LAYERBLOCK 在 2（故取
        [3:]）。这也解释了为什么 LAYERFIRST 的 kv_stride 会比另两者大一个
        num_block 因子——它的 block 维排在 kv_dim 之后。

        多组模式下 K/V 的位置由各组自己决定，无统一定义，抛 ValueError。
        """
        if self.layer_groups is not None:
            raise ValueError("get_kv_stride() is not valid for multi-group layout; "
                             "use per-group strides instead")
        if self.type == KVCacheLayoutType.LAYERFIRST:
            return self.kv_shape[2:].numel()
        elif self.type == KVCacheLayoutType.BLOCKFIRST:
            return self.kv_shape[3:].numel()
        elif self.type == KVCacheLayoutType.LAYERBLOCK:
            return self.kv_shape[3:].numel()
        else:
            raise ValueError(f"Invalid KVCacheLayoutType: {self.type}")

    def get_group_strides(self) -> List[Dict[str, int]]:
        """Compute per-group stride info for multi-group BLOCKFIRST layout.

        Returns a list of dicts, one per group, each containing:
            - num_layers: number of layers in this group
            - offset_elements: element offset of this group within a block
            - layer_stride: elements per layer (kv_dim * tpb_g * num_kv_heads * head_size)
            - kv_stride: elements per KV half (tpb_g * num_kv_heads * head_size)
            - chunk_size: elements per K or V chunk (tpb_g * num_kv_heads * head_size)

        ``tpb_g = tokens_per_block // g.compress_ratio`` — compressed groups
        store only ``1/compress_ratio`` tokens per block in this group's region.

        中文要点与坑：
          * 这是多组模式下**唯一**能拿到各层组偏移 / stride 的途径，其余
            get_*_stride 在多组模式下都直接抛 ValueError。
          * 返回的 dict 额外带了 num_kv_heads / head_size / layer_indices，
            供数据面按"原始层号"（layer_indices）把数据写回 GPU 上对应的层。
          * 各组的 tokens 数是压缩后的 tpb_g，所以同一 block 内各组区域的长度
            **并不**与原始层数成比例，不能按比例推算偏移。
          * offset_elements 是累加的**元素数**，不含 dtype.itemsize。各组 dtype
            相同时它可直接当元素偏移用；一旦出现混合 dtype（bf16 主 KV + uint8
            indexer），它就不是真实字节偏移，换算字节时必须自己乘上该组
            dtype.itemsize。
        """
        if self.layer_groups is None:
            raise ValueError("get_group_strides() requires layer_groups to be set")
        if self.type != KVCacheLayoutType.BLOCKFIRST:
            raise ValueError("get_group_strides() only supports BLOCKFIRST layout")

        result = []
        offset = 0
        for g in self.layer_groups:
            tpb_g = self.tokens_per_block // g.compress_ratio
            chunk = tpb_g * g.num_kv_heads * g.head_size
            kv_stride = chunk  # tpb_g * num_kv_heads * head_size
            layer_stride = self.kv_dim * kv_stride
            group_elements = g.num_layers * layer_stride
            result.append({
                'num_layers': g.num_layers,
                'num_kv_heads': g.num_kv_heads,
                'head_size': g.head_size,
                'layer_indices': g.layer_indices,
                'offset_elements': offset,
                'layer_stride': layer_stride,
                'kv_stride': kv_stride,
                'chunk_size': chunk,
            })
            offset += group_elements
        return result

    def get_total_elements(self) -> int:
        """布局涵盖的元素总数 = kv_shape.numel()。

        分配器按这个数去申请一维扁平内存（torch.empty((total,), dtype=...)），
        多维 kv_shape 只是逻辑视图，物理上始终是一段连续内存。

        注意多组模式下 kv_shape 是 [num_block, bytes_per_block]，此时这个"元素
        总数"实际上是**字节总数**，分配器需要另行换算成 dtype 元素数。
        """
        return self.kv_shape.numel()

    def get_elements_per_block(self) -> int:
        """每个 block 占多少（普通模式是元素数，多组模式是字节数）。

        实现是 numel() // num_block，依赖"所有 block 等长"这一前提。多组模式下
        各组区域是平铺进**每一个** block 的，长度一致，所以前提成立。
        """
        return self.get_total_elements() // self.num_block


@dataclass
class StorageHandle:
    """一块已分配好的 KV Cache 存储的"可访问句柄"——分配器交给数据面的包裹。

    什么时候用：storage/allocator.py 的各类 Allocator（GPU / CPU / HugePage /
    SSD / 远端）分配完内存或文件后，把结果包成 StorageHandle 返回；
    storage/storage_engine.py 汇总各级 handle；最后 transfer/ 侧把它交给 Worker
    子进程，由 Worker 按 kv_layout 的 stride 去真正读写。

    为什么要有这一层抽象：GPU / CPU / SSD / 远端的"可访问形态"完全不同（显存
    指针 / 内存张量 / 文件路径 / 远端连接配置），数据面不应该关心这些差异。
    handle_type 声明形态，data 装具体内容，调用方用对应的 get_* 访问器取值，
    取错访问器会立刻抛 ValueError 而不是拿到错误类型的数据。

    字段说明：
      handle_type  —— data 的形态，见 AccessHandleType。
      data         —— 具体内容，类型由 handle_type 决定（见该字段的联合类型）。
      kv_layout    —— 这块存储的布局，是数据面算偏移的唯一依据。
      dtype        —— 元素类型。**多组布局下它只是"默认 dtype"**，各层组真正的
                      dtype 在 kv_layout.layer_groups 里，别拿它去换算字节。
      num_blocks_per_file —— SSD 场景下一个文件装多少个 block，用于把全局
                      block_id 拆成 (文件下标, 文件内 block 下标)。
      gpu_device_id —— GPU 张量所在的设备号。Worker 子进程在访问前需要先
                      torch.cuda.set_device 到这块卡，否则 IPC 映射会失败。
      remote_config_custom —— 远端后端的私有配置，原样透传给 Worker，本层不理解。
      worker_data  —— Worker 侧应优先使用的替代视图。典型场景是大页内存：
                      主进程持有普通 tensor，而 Worker 需要可跨进程共享的
                      hugepage handle，于是把它挂在 worker_data 上
                      （见 get_worker_tensor）。

    坑：
      * 实例会被 pickle 后传给 Worker 子进程，所以字段必须可序列化；GPU 显存
        不能直接序列化，才有了 TENSOR_HANDLE 这种"传 handle 不传数据"的形态。
      * data 是 list 时，元素必须"全是 tensor"或"全是 handle"，不允许混装；
        空 list 会让两个 all(...) 都为真从而绕过断言。
    """
    handle_type: AccessHandleType
    # The actual handle data
    data: Union[List[torch.Tensor],
                torch.Tensor,
                List[str],
                List[TensorSharedHandle],  # for shared gpu tensors
                Dict[int, List[str]]  # for ssd files: ssd_device_id -> file_paths
                ]
    kv_layout: KVCacheLayout
    dtype: torch.dtype
    # Optional metadata
    # 一个 SSD 文件装多少个 block：把全局 block_id 拆成
    # (file_index = block_id // num_blocks_per_file,
    #  offset_in_file = block_id % num_blocks_per_file)。
    num_blocks_per_file: Optional[int] = None
    gpu_device_id: Optional[int] = None  # Worker 侧访问前需 set_device 到这张卡
    remote_config_custom: Optional[Dict[str, Any]] = None  # 透传给远端后端，本层不解析
    worker_data: Optional[Any] = None  # Worker 侧优先使用的替代视图（如大页 handle）

    def get_tensor_list(self) -> List[torch.Tensor]:
        """取张量列表，用于"一块存储由多个张量组成"的场景（GPU 的 num_chunks
        切分、CPU 分片等）。

          TENSOR        —— data 本来就是 tensor 列表，直接返回。
          TENSOR_HANDLE —— 逐个调 get_tensor()，把跨进程共享 handle 还原成当前
                           进程可用的 tensor。这是 Worker 子进程访问 GPU 显存的
                           正常路径。

        坑：TENSOR 分支没有再校验元素类型就原样返回。若 data 里其实是
        TensorSharedHandle 而 handle_type 被标成了 TENSOR，这里会静默返回一个
        handle 列表（与返回类型标注不符），错误会推迟到真正拿它做 IO 时才暴露。
        """
        assert isinstance(self.data, list) and \
                (all(isinstance(x, torch.Tensor) for x in self.data) or \
                all(isinstance(x, TensorSharedHandle) for x in self.data)), \
                "handle data must be List[Tensor] or List[TensorWrapper]"
        if self.handle_type == AccessHandleType.TENSOR:
            return self.data  # type: ignore
        elif self.handle_type == AccessHandleType.TENSOR_HANDLE:
            assert all(isinstance(x, TensorSharedHandle) for x in self.data), \
                "All elements must be TensorSharedHandle for TENSOR_HANDLE type"
            return [x.get_tensor() for x in self.data]  # type: ignore
        else:
            raise ValueError(f"Invalid handle type: {self.handle_type}, expected TENSOR or TENSOR_HANDLE")

    def get_tensor(self) -> torch.Tensor:
        """取整块存储对应的**单个**张量，用于 CPU / HugePage 这类"一块内存就是
        一个扁平 tensor"的介质。

        与 get_tensor_list 的区别：这里要求 data 本身是 torch.Tensor（不接受
        列表），也只接受 TENSOR 形态。GPU 侧常按 num_chunks 切成多个张量，所以
        一般走 get_tensor_list 而不是这里。
        """
        assert isinstance(self.data, torch.Tensor), \
            "handle data must be torch.Tensor"
        if self.handle_type == AccessHandleType.TENSOR:
            return self.data
        else:
            raise ValueError(f"Invalid handle type: {self.handle_type}, expected TENSOR")

    def get_worker_tensor(self) -> Any:
        """取 Worker 子进程应该使用的张量视图：有 worker_data 就优先用它，
        否则回落到普通的 get_tensor()。

        为什么要有这个分叉：大页内存场景下，主进程持有的是普通 tensor，直接
        pickle 给子进程既慢又会丢掉大页属性；所以分配器（HugePageAllocator）在
        worker_data 上挂一个可跨进程共享的 hugepage handle，子进程走这条路才能
        拿到真正的大页内存。对绝大多数介质 worker_data 是 None，等价于
        get_tensor()。
        """
        if self.worker_data is not None:
            return self.worker_data
        return self.get_tensor()

    def get_file_list(self) -> Union[List[str], Dict[int, List[str]]]:
        """取 SSD 落盘文件的路径：单盘是 List[str]，多盘条带是
        {ssd_device_id: [路径, ...]}。

        拿到路径后由 Worker 自行 open / pread / pwrite（或走 GDS 直连 GPU）。
        注意本方法只校验 handle_type，**不校验文件是否真的存在**——文件是否创建
        成功是分配阶段（SSDAllocator.allocate）的事。
        """
        if self.handle_type == AccessHandleType.FILE:
            return self.data  # type: ignore
        else:
            raise ValueError(f"Invalid handle type: {self.handle_type}, expected FILE")

    def get_tensor_handle_list(self) -> List[TensorSharedHandle]:
        """取共享张量句柄列表，用于把 GPU 显存**跨进程**交给 Worker 子进程。

        两种入参都会被规整成 List[TensorSharedHandle]：
          TENSOR_HANDLE —— 本来就是，直接返回；
          TENSOR        —— 逐个包一层 TensorSharedHandle，让主进程侧也能统一成
                           同一种形态再往下传（例如交给需要 handle 的调度路径）。

        坑：TENSOR 分支会**新建** TensorSharedHandle，也就是真的去导出 CUDA IPC
        handle。这有实际开销、要求 CUDA 上下文可用，且每次调用都会重新导出一套
        新对象（与原 tensor 不是同一个对象），不要在热路径里反复调用。
        """
        assert isinstance(self.data, list) and \
                (all(isinstance(x, torch.Tensor) for x in self.data) or \
                all(isinstance(x, TensorSharedHandle) for x in self.data)), \
                "handle data must be List[Tensor] or List[TensorWrapper]"
        if self.handle_type == AccessHandleType.TENSOR_HANDLE:
            assert all(isinstance(x, TensorSharedHandle) for x in self.data), \
                "All elements must be TensorSharedHandle for TENSOR_HANDLE type"
            return self.data  # type: ignore
        elif self.handle_type == AccessHandleType.TENSOR:
            assert all(isinstance(x, torch.Tensor) for x in self.data), \
                "All elements must be torch.Tensor for TENSOR type"
            return [TensorSharedHandle(x) for x in self.data]  # type: ignore
        else:
            raise ValueError(f"Invalid handle type: {self.handle_type}, expected TENSOR_HANDLE or TENSOR")
