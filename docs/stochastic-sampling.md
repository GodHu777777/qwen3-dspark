# 随机 speculative sampling 的 CPU 数学参考

`dspark_qwen/sampling.py` 是独立、小词表、Python 标准库实现；不调用模型、
GPU、cache、confidence head 或 scheduler。动态 BF16 路径及已有数值失败保持
原状。本阶段验证给定离散概率的算法，不验证模型质量、跨 kernel 数值保真或速度。

## 固定协议与来源

依据 [DSpark v1 §3 与 Appendix A](https://arxiv.org/html/2607.05147v1)、
[DeepSpec 固定 evaluator](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/base_evaluator.py)、
[DeepSpec sampling helper](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/utils/sampling.py)
和 [NeMo 固定 Markov head](https://github.com/NVIDIA-NeMo/Automodel/blob/2d365eda1050dd80ee9bd3bfc651e9ae9bbe8c68/nemo_automodel/components/speculative/dspark/markov_head.py)
及 [_sampling.py](https://github.com/NVIDIA-NeMo/Automodel/blob/2d365eda1050dd80ee9bd3bfc651e9ae9bbe8c68/nemo_automodel/components/speculative/dspark/_sampling.py)。
本模块为独立实现，没有复制上游代码。新读取的两个 helper 保存在 ignored
`output/scope-research/`，并保留原许可：DeepSpec 为 MIT（2026 The DeepSpec
Authors），NeMo 为 Apache-2.0（2026 NVIDIA CORPORATION）。读取文件 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| DeepSpec `sampling.py` | `8f8ff58387e70526452672372628d1b2fbacb778a89f1a46abc9e2b7639d215a` |
| DeepSpec `LICENSE` | `474817ff0d848f0d44cea7f5ac0b488fe74f599f2a656476a94f215e5e4094c7` |
| NeMo `_sampling.py` | `dafe87638e0c6262f1de833cc8807898d89395f42187850d8a61dc6531acd933` |
| 已保留的 `references/NeMo-LICENSE` | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |

设已有 context（含最新 anchor）为 h，已采 draft 前缀为 x[:j]：

- q_j 是**实际采出 x_j 的条件分布**。NeMo vanilla/gated Markov head 在
  block backbone 的 base logits 上加入由前一 token（首位置为 anchor）产生的
  bias；RNN 变体还递推状态。不能用未加 Markov 修正的 base logits，也不能把
  greedy token 当作从原始 softmax q 采出的样本。
- p_j 是 target 在同一 h+x[:j] 后的下一 token 分布。验证输入为
  `[anchor, x_0, ..., x_(n-1)]` 时有 n+1 行：前 n 行验证 draft，最后一行
  p_n 在全部接受后采 bonus。采样温度、词表映射、所有过滤/约束必须先确定；
  sampler 接收最终概率，不自行转换 logits。
- 固定 upstream helper 在 temperature < 1e-5 时取 argmax/one-hot，否则
  softmax(logits/temperature)。实际 `sample_tokens` 未显式转 FP32，
  `logits_to_probs` 却先 `.float()`；低精度输入下两条计算路径未必给出相同 q。
  集成时必须保存实际采样使用的 q，而不能在验证时沿另一条数值路径重建。

对 x_j ~ q_j，接受率为 `alpha_j=min(1,p_j[x_j]/q_j[x_j])`。
遇到首个拒绝即停止，采 `r_j(v)=(p_j(v)-q_j(v))_+/Z_j`，
其中 `Z_j=sum_v (p_j(v)-q_j(v))_+`，丢弃剩余 proposal。
全部接受时独立采 p_n 的 bonus。接受随机数和 categorical 随机数须独立均匀；
调用方提供的 RNG 应满足这个假设，固定 tape 只用于确定性测试。

证明只需逐位置分解：输出 v 的接受质量为 `min(p(v),q(v))`；总拒绝质量
为 `1-sum_v min(p(v),q(v))=Z`；残差贡献为 `(p(v)-q(v))_+`。
两者相加恰为 p(v)，再按实际输出前缀归纳即可恢复 target 序列分布。
`q(v)=0` 的 v 仍可由残差产生，但不能作为已从 q 采出的候选。
p=q 时 Z=0 且拒绝不可能，显式调用零质量 residual 是错误。

## API、EOS、预算与 admission

`sample_proposal(q_callback, max_draft_tokens, rng, admit=...)` 返回不可变
`Proposal(tokens, draft_probs)`，保存每个位置实际采样的归一化 q。
`verify_proposal(proposal, target_rows, rng, max_new_tokens=remaining,
stop_token_ids=...)` 返回 `SampledRound`：本轮新增 token、实际提交的 draft
接受数、首拒绝位置、extra 类型（residual/bonus/无）与停止原因。
结果**不包含原有 anchor**；remaining 也不包含这个 anchor。

EOS 无论来自接受、残差还是 bonus，都包含在输出中并立即结束；未接受的
draft EOS 不终止输出。预算耗尽后不再采 bonus 或消费无用随机数。
预算为 0 时输出为空；proposal 长度为 0 时直接采一枚 target token。
同一步同时达到 EOS 和预算时 stop_reason 记 eos。所有输入，包括未使用的
尾行，均在验证随机数消费前校验。

Admission 必须在当前候选采出前确定，只能依赖已有 prefix、固定容量、
独立的历史状态等。`admit(prefix)` 在本位置 q 采样前调用；接口本身不能
阻止闭包读取未来信息，调用方仍须保证因果性。拒绝 admission 时停止 proposal，
后续由 p 采样。允许根据已采出的前一 token 决定下一位置是否 admission。
函数默认不按 draft EOS 裁掉当前候选，EOS 的输出终止由 verifier 决定。

若先采 X，再仅在 X=0 时 admission，即使 p=q=(1/2,1/2)，普通 verifier
也不能消除选择偏差：admit 时输出 0，否则重采 p，最终 P(0)=3/4。
因此不能把完整当前块的未来 confidence 回溯用于当前 token admission。
本模块没有接入 `scheduler.py`，也没有声称未校准 head 可充当正确调度信号。

## 数值契约与后续集成边界

`probabilities` 要求非空、有限、非负、每项不超过 1，且总和与 1 的绝对误差
不超过 1e-12；仅在这个范围内用总和归一化，然后按 Python float 运算。
这属于本地小词表输入契约，**不兼容把任意 GPU BF16 概率直接送入**；上游
需要定义一致的 FP32/FP64 最终采样 law（并检查精度/和的误差，必要时在采样前
统一提升精度及归一化）。重复归一化不能证明与原模型概率相同。
本模块没有 logits/temperature adapter，也没有 GPU tensor adapter。

固定 DeepSpec evaluator 对选中 q 用 `clamp_min(1e-8)`，两个 helper 对
residual 总质量 <=1e-8 会回退到 target。本参考不复制这些近似：不可能的
q=0 候选报错，非零极小 residual 仍按 residual 采样，不使用接受 epsilon。
类别采样用 `[0,1)` 随机数和严格 `<` 比较；浮点 CDF 尾部舍入缺口回到
最后一个正概率 token，绝不选择零支持 token。

数学恒等式以给定 p/q、理想均匀随机数和精确实数为前提。Python float、有限
PRNG、FP32/BF16 softmax、不同 batch/kernel 的 logits 都有各自数值误差。
此 CPU 验证不意味着这些实现逐 bit 相等，也没有解除当前动态 BF16 输出 gate。
未来 cached stochastic 路径需保存每位置实际 q 与 pre-token confidence logits，
单独检查 target 行/缓存提交回滚，并保留数值诊断；不能直接复用 greedy proposal。

## 精确枚举验证

运行 `python3 -m unittest discover -s tests -p test_sampling.py -v`。
测试用 `Fraction` 枚举全部候选、接受/拒绝、residual 和 bonus 分支，计算
精确概率质量，并以各随机区间内部点驱动实际实现，核对 token、接受数、拒绝
位置、停止原因及随机数消费数。再把多轮输出分布与独立 target 自回归枚举比较。

覆盖 225 对三词表分母 4 的概率分布（包括 q=0、p=q、不相交支持集）、
前缀相关的 p/q、0/1/2 proposal 长度、0/1/3 输出预算、有/无 EOS、基于已有
prefix 的自适应 admission、首拒绝、全接受及 2^-40 residual。
另测归一化、维度/词表错误、非有限输入、RNG 范围与禁止的 hindsight 反例。
Fraction 输出-law 断言无 Monte Carlo 容差；区间内部点测试并非对所有浮点
CDF 边界的形式化证明。固定 seed smoke 只检验 RNG API 可重现性。
