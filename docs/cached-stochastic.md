# Cached stochastic reference：CPU 验证阶段

`tensor_sampling.py`、`cached_sampling.py` 和
`DSparkDraft.propose_stochastic_cached` 提供独立随机路径；旧 greedy 接口及
权重结构不变。动态 `CachedTarget` 为主路径，`CanonicalTarget` 仍是单独数值
控制。本文记录 CPU 实现阶段；后续 [真实 GPU gate](../reports/stochastic-gate-20261009/README.md)
已通过执行/复现检查，但保留了非零 TV 和 argmax 差异，没有解除动态 BF16 数值失败。
数学来源及 residual 推导见 [sampling说明](stochastic-sampling.md)。

## 明确的概率策略

策略标识为 `float64_softmax_normalize_cdf_v1`，temperature 有限且严格大于 0，
无 top-k/top-p 或其他过滤。模型 forward 保持加载时 dtype；target 可保持
BF16，draft FP32 参数可配合 BF16 autocast。logits 转 float64 后除 temperature、
softmax、除行和一次，形成实际采样 law。下溢产生的零概率属于此数值 law。

proposal 每位置保存**实际用于抽样的 q**。verifier 检查 float64、形状、有限性、
非负和行和误差 <=1e-12，不重复归一化 q/p。没有把低精度 row 直接送入 CPU
`sampling.py`。target 首 token、target-only baseline、bonus 和 residual 共用
同一 adapter/CDF sampler。均匀随机数为同设备 float64 `[0,1)`；CDF 采用
`searchsorted(right=True)`，舍入尾部缺口选择最后一个正支持 token。

接受率为 `min(1,p(x)/q(x))`，首拒绝后从 `(p-q)+` 除其总质量采样。
q(x)=0 候选及零质量 residual 报错；无 denominator epsilon、小质量回退或
近似接受阈值。`relu(p-q)` 只是数学上的正部。

概率升精度并没有改变 target forward dtype，不能消除不同 batch/kernel 的
logits 差异；有限 PRNG/CDF 也不是精确实数。其他概率精度方案仍可研究，但须
变更策略标识，重新核对实际 draw 与验证 q 的一致性。

## Proposal 和 cache 规则

固定长度上限在本轮随机数前确定，实际长度为
`min(max_draft_tokens, remaining_budget)`。长度 0 合法，跳过 draft backbone。
非空 proposal 只做一次并行 backbone/base LM-head，随后逐位置 Markov 递推。
前一 token embedding 同时进入 logits bias 和 **pre-token raw confidence**。
confidence 不用于 admission，未被当成已校准 scheduler 信号。

首 target token 从 prompt prefill 最后一行采出。以后 target append
`[old_anchor] + n个draft` 获得 n+1 行 p：行 j 预测 draft[j]，行 n 预测
全接受后的 bonus。提交首拒绝前 prefix 加 residual，或全接受加 bonus。
EOS 无论来自接受/residual/bonus，都保留并停止；未接受的 draft EOS 不停止。
预算耗尽不抽额外 bonus 或无用随机数。

边界处两套 cache 都只包含**已提交序列减最后一个输出 token**。
本轮发出 m 个 token 时，target crop 到 `before+m`；只把 verified context
前 m 行投影加入 draft KV，包含旧 anchor、排除新的最后 token。失败时 reset
target 和本地 draft cache；下一独立请求从新 prefill 开始。

## Observer 和 STS 接口

`cached_speculative_sample` 返回 `(token_ids, round_records)`：

- `verified_cache_end` 为完整验证后的绝对 cache 位置。
  `verified_proposal_length` 是本 block 实际 target-scored 候选数 n。
- `attempted_positions` 只是已抽接受随机数的位置数。首拒绝后已评分 suffix
  的 **prefix event 已知为 0**；STS 分母不能只采用 attempted_positions。
- `accepted_eos_position` 仅对接受 draft EOS 给零基位置；residual/bonus EOS
  为 None。接受 EOS 自身保留，之后位置从校准分母移除。
- `confidence_logits` 是 raw head 输出；`conditional_overlap=sum min(p,q)`
  是 proposal-prefix 诊断，拒绝后的 prefix 未提交。它不能替代 STS 的实际
  accepted-prefix 二值标签。
- `probability_policy` 和 `temperature` 显式记录。实验还须绑定 checkpoint、
  数据、runtime、forward dtype、采样/收集协议及代码哈希。

可选同步 `observer(SamplingObservation)` 在提交后看到真实 p/q、raw confidence、
结果及局部长度/策略。调用者不可修改 tensor；长期保留时自行复制。
默认 records 不积累全词表 GPU tensor。`trace_probabilities=True` 保存每轮
p/q 的 detached CPU 副本；`trace_tokens=True` 记录候选/提交 token。
这些大 trace 仅供私有诊断，不用于计时或直接公开发布。

STS adapter 尚待单独接入。固定完整 proposal 收集和预算截短须披露；未生成/
未评分尾部不能补 0。confidence 选出的样本不能混入 full-proposal 校准集，
greedy 与 stochastic 标签也不能混用。

## 运行和性能边界

```python
import torch
from dspark_qwen.cached_sampling import cached_speculative_sample
from dspark_qwen.tensor_sampling import TensorRandom

rng = TensorRandom(torch.Generator(device=input_ids.device).manual_seed(20261009))
tokens, rounds = cached_speculative_sample(
    target, draft, input_ids, 128, temperature=1.0, rng=rng,
    eos_ids=eos_ids, amp=True, max_draft_tokens=draft.spec.block_size,
)
```

CPU 设 amp=False；真实 device/AMP 取决于模型加载配置。可重现性用同路径/
runtime 的新同 seed generator 检验。不同算法消耗随机数不同，不要求同 seed
的 speculative 逐 token 等于 target-only。

target-only 使用 `cached_target_sample` 和相同 temperature/float64 adapter。
性能 baseline 应构造 `CachedTarget(model, ())`，避免无用的 draft-context hooks。
CPU KV 数值测试共用捕获配置不等于公平性能 baseline。

这是一条可审计 reference：float64 softmax、CDF、scalar 校验/分支和每轮统计
都可能很贵。端到端性能必须计入 draft、概率变换、采样、verify、cache 及实际
启用的 observer/trace，不能只计 verify。canonical physical rows/padding 另计。
float64 不是已经优化好的最终策略，也不是唯一可研究策略。

## 已验证范围

AMD 独立临时目录中以 `HIP_VISIBLE_DEVICES="" CUDA_VISIBLE_DEVICES=""
ROCR_VISIBLE_DEVICES=""` 运行，并断言 `torch.cuda.is_available()` 为 False。
新增 tensor/cached 测试 11 个，加既有 CPU sampler 合计 22 个；完整快照
76 个 CPU unittest 通过，包含旧 greedy/cache/core 和 STS 测试。

`Fraction` oracle 枚举 225 对小词表分布及完整 cached 多轮随机树，覆盖
首 target token、proposal、拒绝/residual、bonus、EOS、0长度和预算；每个
非零分支用随机区间内部点驱动实际代码，精确质量累计与独立 target 自回归
枚举相等。不是 Monte Carlo，也不是全部浮点边界的形式化证明。

真实 tiny Qwen CPU 测试在每个拒绝位置、全接受、EOS/预算之后，将 target KV
和 projected draft KV 的**内容及长度**与完整已提交 prefix 重算比较；另检查
单次 backbone、实际 Markov q/pre-token confidence、seed/request 状态隔离、
observer 失败 reset 和 canonical 控制。后续真实 BF16 GPU gate 见上方独立报告；
不能将此处 CPU 数学测试与真实后端数值保真混为同一个通过结论。
原始日志在 ignored `output/scope-research/*sampling-tests.log` 与
`stochastic-full-cpu-suite.log`。
