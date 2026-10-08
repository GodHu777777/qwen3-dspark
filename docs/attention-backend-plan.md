# Packed target 的下一步 attention backend

2026-10-09 只读检查现有 AMD 运行环境，未安装、import attention backend、JIT、
编译或执行 GPU kernel。已安装文件与源码可证明存在候选路径，不能证明 gfx1201
已经成功运行、数值正确或更快。原始 package 清单和相关源码 SHA256 位于 ignored
`output/attention-backend-feasibility-20261009/inventory.json`。

## 已有接口与证据

| 候选 | 本机静态证据 | 尚未验证 |
| --- | --- | --- |
| PyTorch native varlen | Torch 2.12.0+rocm7.2 有 `torch.nn.attention.varlen.varlen_attn`；支持 THD、cu_seq_q/cu_seq_k、GQA、causal window；AOTriton images 有 amd-gfx120x/flash/attn_fwd 等家族 | 该模型 shape/dtype 的 gfx1201 dispatch、GQA、非方形因果对齐、数值与资源 |
| FlashAttention AMD Triton | flash_attn 2.8.3 含 AMD varlen 实现；`is_rdna()` 显式列出 gfx1200/gfx1201；varlen kernel 按请求 cu_seqlens 限制自己的 K 范围 | 包实际 import、Triton 编译与目标芯片执行；小 query 的利用率 |
| FlexAttention | PyTorch flex API、Inductor Triton lowering、ROCm kernel options 和 HF adapter 均存在 | gfx1201 编译/执行、短 query 布局是否真有大量可跳过 blocks |
| AITER | amd-aiter 0.1.21.post2 含 Triton varlen wrappers；部分公共 RDNA 架构表含 gfx1201 | 具体 attention entry 的 dispatch，而不是其它算子的支持；部分 ASM/Gluon entry 明确仅 gfx1250/gfx942 |

环境同时保留 `triton 3.7.1+gitf0b55c07` 与 `triton-rocm 3.7.0` 的 distribution
metadata，而实际 `triton/__init__.py` 声明 `__version__ = "3.7.0"`。不能只根据
pip metadata 推断运行代码版本。未来最小执行探针应记录实际 import 文件与版本，
先不重装依赖或修改现有服务环境。

## 最小实现路径

优先做 native `varlen_attn` 的独立小 tensor GPU 门槛，必须另行协调空闲 GPU 窗口。
其 API 直接表示独立请求的 packed Q/K/V，不需要用一个大 Q×K dense mask 表达
请求隔离；源码走 ATen flash attention，已有 gfx120x AOTriton images 是可尝试的
依据。若该实际组合不支持，再验证已安装的 FlashAttention AMD Triton 路径。
不能通过 silently fallback 到 dense SDPA 把“接口调用成功”写成 varlen kernel 成功。

与 `PackedTarget` 接轨时保留原 dense 实现为独立 oracle，另建 attention backend：

1. 在一次 request batch 中准备有 query 请求的顺序、各 q_r、完整 k_r=old_r+q_r，
   int32 cumulative lengths，以及把当前共享物理 KV 聚集到该请求顺序的 index。
2. 保留 Qwen 层中的 QKV projection、QK normalization、显式本地 RoPE positions 和
   DynamicCache update；用 HF attention registry 中独立命名的 callback 接收这些
   已变换 Q/K/V，避免复制/重写 decoder layer。
3. Q 从 `[1,Hq,Q,D]` 转为 `[Q,Hq,D]`，现有 query 已按本轮请求连续排列。
   各层 K/V 用同一 gather index 转为 `[sum(k_r),Hkv,D]`。inactive resident 的
   KV 仍留在持久 cache 中，但不进入本轮 attention 的 transient gather。
4. 一次 backend call 处理所有请求。native API 使用 `enable_gqa=True` 和
   `window_size=(-1,0)`；cached query 对应 K 的尾部，必须验证 bottom-right causal
   对齐。输出转回 HF 要求的 `[1,Q,Hq,D]`，再由原 Qwen output projection 处理。
5. 后续 target crop/exit/readd 继续用 PackedTarget 的 request marker/local-position
   状态 oracle。首版允许 K/V gather 拷贝，但计入耗时/字节；再评估 paged-KV 或
   request-contiguous allocation，不能隐藏为了适配 kernel 增加的搬运成本。

新路径不应构造原来的完整 Q×K mask；同一个 `PackedLayout` 可把 marker 校验与
cu lengths/gather index 复用到全部层。callback 不得每请求调用 attention，必须让
cu lengths 一次传入真实 varlen kernel。future guard/test 要检查每层 backend call
次数、总 Q 行数、实际选用 backend、K gather 的请求映射与输入 stride/dtype。

## 为什么不直接把 dense mask 换成 Flex score_mod

Flex 的 `score_mod` 在 score 计算后修改值，单靠 -inf 的跨请求 mask 不能证明
跳过了对应 QK 计算。要提供 BlockMask 才有 block-sparse 跳块信息。
此外默认 sparse block 大小为 128；本项目每请求 1–8 个 query，多个请求容易
落入同一个 Q tile。这个 tile 对多个请求的 K 域都有部分可见元素，可能几乎无
完整 block 可跳过。需要先计算实际 nonempty block 密度、tile 边界开销与布局，
不能仅凭理论因果 pair 数就宣称稀疏收益。

当前 `create_block_mask` 未编译路径先生成 dense mask 再压缩；HF flex adapter
初次执行还会触发 `torch.compile`。因此本次只读调查没有调用它们。后续若走 Flex，
应从请求 layout 构造 block metadata，并把 mask 创建/JIT/重编译开销与稳态执行
分开记录，确认 shape/bucket 规则不会抹平 scheduler 的真实成本台阶。

## 门槛与计量

小 tensor 门槛首先覆盖 `q<k`、不同 q/k、GQA、空 query 请求排除、EOS/拒绝 crop
后的 gather、same-ID readd，以及改变其它请求 QKV 的 poison 干预；用 dense per-
request SDPA 和现有 packed dense oracle 双重对照。再把同样 layout 接入 tiny
真实 Qwen 的 hidden/logits/KV 测试。BF16 与 FP32 数值门槛分开；保留此前动态
BF16 gate 的失败，不用新 backend 自动覆盖旧记录。

性能测量需区分逻辑可见 pairs、kernel tile 工作域、真实调用数和实际用时。
报告 gather/mask/layout、attention、完整 Qwen forward、crop/KV 搬运、调度成本，
以及 Q/B、R、context 分布和 backend identity，才能建立 `SPS(B, context/R)`。
native varlen 文档提到 `num_splits=1` 用于减少 batch 组成引起的 reduction 变化，
但这仍是候选控制项，不能未经 ROCm 实测就当成整个模型的 bitwise 等价保证。

## 实现进度

独立 `PackedLayout` / `VarlenPackedTarget` 和显式 CPU-only oracle 已实现；
5 个原 packed tests + 5 个 varlen tests 在 GPU 隐藏的 FP32 CPU 模式通过。
固定小 tensor gate 已准备，默认仅 stdlib dry-run，尚未运行 native GPU/JIT。
见 [实现与固定 GPU 探针协议](varlen-target.md)；真实执行仍需单独协调 GPU 窗口。
