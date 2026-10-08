# DSpark 忠实复现范围：算法、全局验证预算与执行系统

核对日期：2026-10-08 UTC。本文区分论文方法、公开训练/评估代码、生产系统描述与
本仓库实测，不将一种证据当成另一种。结论是：**完整 DSpark 系统复现不能止于
parallel drafter、单请求固定 k、confidence 阈值或一张单请求耗时表。其核心系统
机制是多个活跃请求竞争 target 验证 token 容量，并与异步执行及可变长度 kernel
协同。** 本仓库目前尚未实现并验证这条完整路径。

## 1. 精确来源和版本

| 来源 | 固定版本/核对范围 | 用途与边界 |
| --- | --- | --- |
| [DSpark 论文](https://arxiv.org/abs/2607.05147v1) | *DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation*，Cheng et al.；v1 提交于 2026-07-06 14:28:06 UTC | 标识符从官方 DeepSpec README 的 paper/citation 链接取得；本次阅读 §3.1–3.3、§4.3.3、§5.1–5.4、Appendix A |
| [论文 HTML](https://arxiv.org/html/2607.05147v1) | arXiv v1；下面的章节/算法锚点均来自该 HTML | 算法、因果性、校准、异步生产调度的主要依据 |
| [DeepSpec 官方仓库](https://github.com/deepseek-ai/DeepSpec/tree/005e03b81cec38b7da6399833d609ee89a2587f2) | commit `005e03b81cec38b7da6399833d609ee89a2587f2`，commit 日期 2026-07-09 | 本次只读取 README、Qwen3-4B 配置及公开 evaluation 相关小文件，未下载整个仓库 |
| [NeMo DSpark 指南](https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/dspark-speculative-decoding) | 2026-10-08 读取的 Nightly 页面；可读 [Markdown](https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/dspark-speculative-decoding.md) SHA-256：`dcdd28f6246c681480594d208c322c3a375d7381db7accb6388cea28b6898ad4` | 标题为 “Train a DSpark Drafter for Speculative Decoding”，是训练指南，不是已提供完整 serving scheduler 的承诺 |
| [本项目采用的 NeMo 源码](https://github.com/NVIDIA-NeMo/Automodel/tree/2d365eda1050dd80ee9bd3bfc651e9ae9bbe8c68/nemo_automodel/components/speculative/dspark) | commit `2d365eda1050dd80ee9bd3bfc651e9ae9bbe8c68`；本地固定文件/哈希见 `references/nemo-2d365eda/` | 本地 dense Qwen3 draft/loss/mask 的直接来源；不能把后来变化的 Nightly 页面等同于这个源码快照 |

Nightly 现已描述 DeepSeek V4.1 专用结构，与论文 §5 的 V4 preview、以及本项目的
Qwen3-0.6B 都不同。下面不会混用它们的层数、block size 或性能数字。

## 2. 论文真正分配的资源是什么

[§3.2.2](https://arxiv.org/html/2607.05147v1#S3.SS2.SSS2) 及
[Algorithm 1](https://arxiv.org/html/2607.05147v1#alg1) 的输入是 **R 个活跃请求**、
每个请求的 confidence 序列和硬件/引擎容量曲线。决策变量是每个请求的验证
前缀长度 `ell_r ∈ {0,...,gamma}`，不是给某个模型固定挑一个 k。

记条件接受概率为 `c[r,j]`，则前缀存活概率、验证 token batch 和期望产出为：

```text
a[r,j] = product(c[r,i], i=1..j)
B      = sum_r (1 + ell_r)
tau    = sum_r (1 + sum_j=1..ell_r a[r,j])
Theta  = tau * SPS(B)
```

每个请求的 `1` 是本轮基准 target/anchor 工作与产出计数；要把 `tau` 与本仓库
“仅 draft 接受长度”区别开。EOS 与输出预算还会改变实际提交数，评估不能机械
地对每轮都加一个 bonus。

`SPS(B)` 的单位是 **steps per second**，不是 token/s。论文说该容量曲线在
engine initialization 时 profile 一次，并保存为轻量表。在线调度根据当前活跃
请求和 confidence 使用它，而不是每轮重新 benchmark。它体现增加验证 token
对整个 target batch 的成本及机会成本：一个请求的低质量 suffix 可能挤占另一个
请求更有价值的验证位置。

这里的“资源分配”在原文中是 **target 验证 token/batch 容量**。本文所核对章节
没有提出 SM 分区、GPU 核数分配、MPS 配额、按请求分显卡或动态调整张量并行度
的算法。不能为了听起来更“硬件”而给复现添加论文未声称的机制。KV 容量、
并发限制、kernel 吞吐则是调度必须面对的真实系统约束。

Algorithm 1 从所有请求 `ell=0`、`B=R` 开始，按 `a[r,j]` 全局降序逐个增加
候选，更新 `tau*SPS(B)`，**第一次不优于此前值时立即 break**。前缀存活概率
随 j 不增加，因此固定 B 时按边际收益分配有前缀结构；实现仍需处理相等分数，
确保不能先选同请求的 j+1 而漏掉 j。`ell=0` 是必要的合法结果。

原文只在目标沿这条 admission 路径单峰时，声称此 early-stop 搜索达到全局最优。
真实 SPS 的离散台阶会破坏这个条件。不能把“遍历所有 k 取最大值”当成同一个
算法，尤其不能忽略下面的因果性要求。

## 3. Confidence、校准与无损性的条件

[§3.2.1](https://arxiv.org/html/2607.05147v1#S3.SS2.SSS1) 的 head 为
`sigmoid(w · [h_k; W1[x_(k-1)]])`，预测**此前 token 均已接受条件下**的位置 k
接受概率。训练软标签为 `1 - 0.5 * ||p_d - p_t||_1`。条件概率通过累乘形成
prefix survival；不能直接将位置 confidence 相加当成期望连续接受长度。

原文使用 **Sequential Temperature Scaling（STS）**：在 held-out validation
上，从左到右逐位置做一维 temperature grid search，最小化累乘预测的 ECE；
已经校准的前面位置保持不变。温度是在 confidence logits 上校准，而不是调整
target 的采样温度。需要绝对概率校准，是因为 `tau` 使用数值大小，不只需要排序。

本项目当前只有 BCE head 和 teacher-forced overlap/MAE，未实现 STS，也没有
证明实际 rollout prefix 上的校准。特别地，本项目现阶段只做 greedy，而上述
分布 overlap 对应标准随机 speculative rejection sampling 的接受概率；不能
直接把未变换分布的 overlap 解释为 deterministic greedy 的 top-1 命中率。
忠实概率复现还需要明确 temperature、proposal/target 概率变换、接受随机数和
residual correction sampler；greedy 可以保留为先行 correctness 分支。

[Appendix A](https://arxiv.org/html/2607.05147v1#A1) 给出反例：若在当前 draft
完整生成后，利用所有未来 confidence 回溯选择全局最优长度，则下一位置
`c[k+1]` 依赖刚采出的 `x[k]`，会反过来影响是否 admission `x[k]`，造成选择
偏差。原文要求 admission 为 **non-anticipating**；同步版的 early break 是
正确性机制，不是可随意删除的性能剪枝。仅“最后仍调用 target 验证”不足以
证明经过自适应选择后仍恢复 target 分布。

## 4. 生产实现比 Algorithm 1 多了什么

[§5.2](https://arxiv.org/html/2607.05147v1#S5.SS2) 明确处理两个冲突：真实 SPS
曲线离散而不单峰；动态 token 数与 continuous CUDA graph replay、
Zero-Overhead Scheduling（ZOS）的提前 batch-size 要求冲突。

其生产改法是：

1. 用 **two steps prior** 的 confidence 信息估计本轮验证容量上限 K。
2. 用本轮真实、最新的累积 confidence 对当前候选排序，在 K 内做动态 top-K。
   历史信号决定容量，当前信号决定分配顺序；不是拿旧分数直接排列当前 token。
3. 对历史预测做不含 early break 的全局容量搜索，以跨过 jagged SPS cliffs。
   由于容量决定不依赖当前候选的具体采样值，论文将这个时间隔离解释为因果屏障。
4. 把调度与 GPU 执行管线重叠，满足下一步 batch shape 提前已知的 ZOS/graph
   要求；不能用同步 Python 计算后再启动下一轮冒充已经隐藏调度开销。

这里的 K 是**跨请求容量**，不能与每请求 draft 长度 k 混为一谈。请求进入/退出、
EOS、数据稀疏或历史不足时的启动策略，在所读取的公开评估路径中没有生产
实现可供直接核对；本地必须声明约定并测试，不能伪称这些细节已逐字复刻。

[§5.3](https://arxiv.org/html/2607.05147v1#S5.SS3) 还要求在一个 batch 中高效
执行不同请求的可变 query 长度。原文将各请求 token flatten 为独立物理元素，
以 marker tensor 向 sparse attention 传递逻辑序列依赖；DeepSeek-V4 的
index-attention 和 compress kernels 需要适配。简单按最长前缀 padding、
逐请求 Python for-loop，虽然可作 oracle，却没有完成这个执行层复现。

“parallel drafter”本身指 block backbone 一次前向产生所有位置，并不自动表示
draft 与 target 在不同 CUDA stream 并发运行。本文所核对原文也不足以要求
实现某种特定 SM 分割或双模型并行方案；异步 scheduler/ZOS 才是明确描述的机制。

## 5. 公开代码到底提供了什么

- [DeepSpec Qwen3-4B config](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/config/dspark/dspark_qwen3_4b.py)：5 层、block7、rank256、512 anchors、global batch512、10 epochs、BF16、4% warmup。本项目 Qwen3-0.6B 与 32-anchor/累积8 计划是资源适配，不是相同规模训练复现。
- [公开 eval CLI](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/eval.py#L32)：提供 `--confidence-threshold`，默认 0 和 temperature1；没有暴露 Algorithm1 的 SPS/多请求容量参数。
- [draft_ops.py](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/dspark/draft_ops.py#L82)：`_confident_prefix_length` 在第一个低于阈值的位置截断；`build_dspark_proposal` 明确断言 `batch_size=1`。这是静态阈值评估，不是生产全局 scheduler。
- [base_evaluator.py](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/base_evaluator.py#L193)：包含 target/proposal 概率、随机接受和 residual sampling；主生成入口也只支持单序列。
- [confidence_head.py](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/dspark/confidence_head.py#L345)：公开 recorder 对 sigmoid confidence 做 cumprod，与实际 prefix acceptance 比较，统计 ECE/AUROC/Brier，并去掉接受 EOS 后无意义的位置。本次读到的是统计代码，不能据此声称已获得生产 STS 拟合和异步调度实现。
- [论文 §4.3.3](https://arxiv.org/html/2607.05147v1#S4.SS3.SSS3) 明确把 offline threshold sweep 用于单独验证 head，将 hardware-aware scheduler 留在 §5 的真实生产实验。这与公开评估入口的范围一致。
- NeMo 指南提供训练流程、FSDP2、目标特征捕获和三项 loss；其当前 V4.1 段还明确说明 serving-time dynamic verification scheduler 不属于这条训练路径。

上述结论限定于**实际读取的入口和文件**；没有搜索完所有历史分支、内部引擎或
未公开生产代码，因此不声称“全网绝无 scheduler 源码”。公开入口不能提供的
生产实现细节应列为待查/待重建，而不是默认已经具备。

## 6. 本项目范围矩阵

| 机制 | 当前证据 | 忠实复现还需完成的检查 |
| --- | --- | --- |
| Parallel backbone + Markov + confidence | pinned NeMo mask/loss/shift 核对，CPU 真模型测试，单样本对齐轨迹可学到每轮7接受 | 扩数据后的 held-out 质量；与纯 parallel/无 Markov 对照；禁止用单样本拟合代替泛化 |
| 训练配方与数据 | target 重生成、三 split、严格身份/恢复；扩数据生成与训练准备中 | 实际审计后的可用数量、训练曲线、anchor/批量差距及 warmup 差距，不宣称论文规模 |
| KV 增量与 rollback | target/draft cache CPU及逐层 KV 内容审查通过 | 真实 dtype/backend correctness gate；BF16 cached-block 3 prompt 中2失败仍是未解决事实 |
| Stochastic distribution recovery | 独立 CPU 概率参考已实现；11 项测试含 Fraction 精确分支枚举与非预知 admission 反例；模型解码仍仅 greedy | 接入真实 Markov 抽样 q、target 概率和 KV 回滚，核对随机解码分布及调度因果性；区别概率无损理论与不同 kernel 的数值误差 |
| Confidence STS | 尚未实现；现 TF MAE 不是 rollout 校准 | 独立 dev 上拟合并冻结逐位置温度，报告 cumprod ECE、Brier、prefix coverage；test 不参与 |
| R 请求全局 Algorithm1 | 已有独立CPU `scheduler.py` literal planner，输出ell/B/tau/score；7项测试用小R/gamma穷举oracle，保留cliff反例 | 尚未接入解码/engine，没有实际多请求SPS；fixture仅证明算法，不证明性能或因果score来源 |
| 硬件容量 SPS(B) | 已测单请求 eager target 若干块长；不含 draft、并发或服务管线 | 测真实 batched engine 的 SPS/shape 台阶、上下文/并发敏感性；定义计时边界，验证模型预测误差 |
| 两步历史异步容量 K | 尚未实现 | 两步历史状态、因果隔离、当前top-K、启动/新旧请求映射、离散容量 cliffs、调度延迟隐藏 |
| 可变长度批验证执行 | 尚未实现 | 多请求隔离、flatten/marker或等价无padding执行、独立KV crop、graph shape策略和真实成本 |
| 吞吐—交互性 frontier | 尚未测 | 多并发/到达负载下 aggregate tok/s、per-user TPS、TTFT/ITL分位数和SLA达成率；与正确基线比较 |

单请求固定 k 和 cost lookup 仍有价值：它们是基线、成本界和开发步骤。它们不构成
上表后五项的替代品，也不能把“完成一个容易子任务”写成 DSpark 系统复现完成。

新增CPU planner的接口为 `plan_prefixes(confidences, sps, max_batch_tokens=...)`。
`confidences` 是已校准的条件接受概率矩阵，`sps` 必须提供所有可行整数 B 的
steps/s 值，禁止插值跨过cliff；capacity必须能容纳所有R个baseline token，
否则明确报错，不擅自丢请求。输出包含每次尝试是否admit及first-drop停止原因。
它尚不实现STS或两步历史状态。尤其不能对当前候选移除break：测试按论文
Appendix A 的 `[0.8,0.9]` 与 `SPS={1:1,2:0.5,3:0.45}` 明确保留
同步算法返回0长度、而回溯全局最优返回2长度的差异。
独立oracle分别核验 `B <= capacity` 的完整可行集合，以及每次admission的
**精确 B** 的最优期望产出，二者不是同一种测试。Algorithm1第4行显式过滤
`a[r,j]=0` 候选，本实现保留该条件；在反常上升的SPS曲线上，添零收益token
也可能增加吞吐目标，因此这项过滤又是不能泛称“任意SPS曲线全局最优”的原因。

```bash
python -m unittest discover -s tests -p test_scheduler.py -v
```

## 7. AMD 适配与 NVIDIA 验证需求

算法层的 confidence、prefix budget 和非预知决策不绑定 NVIDIA。现有 AMD
Radeon AI PRO R9700 已实际运行 Qwen3-0.6B 训练/缓存测试；不应凭品牌推断无法
做多请求调度。反过来，短序列约 5.18 GB 的 aligned 训练峰值、或单请求 block
cost，也不能保证真实长上下文/并发的 KV 容量和吞吐。应逐步实测目标并发数，
记录 OOM/内存余量/利用率及性能退化位置。

Qwen dense attention 与 V4 sparse index-attention/compress 不是同一 kernel。
AMD 版本应明示为**同调度算法的 dense-Qwen/ROCm 适配**，并验证相同请求预算、
隔离/因果性和可变长度执行代价，而不是声称已经复制 V4 的生产 kernel。

论文 §3.2.2 脚注认为 V4 的上下文影响较小，且 prefill/decode disaggregation
和 DP load balancer 平衡请求数与总上下文，因此用 B 一维建模。这个假设不能
直接移植到本项目：dense Qwen 的长上下文、多请求 KV 读写、batch组织可能显著
改变成本。先测 `SPS(B, context/load bucket)` 的残差；只有证据支持时才简化
成 B 一维表。若扩展状态维度，应标注为硬件适配而不是论文原模型。

若目标是复现 **NVIDIA CUDA graph + ZOS 执行路径**或给出 AMD/NVIDIA 比较，
需要真实 NVIDIA 设备/对应引擎及同模型、dtype、负载的测量。ROCm graph/异步
执行可以作为等价设计候选，但其支持和开销必须实测，不能凭接口名称宣称相同。
原文所核对章节未给出足以确定本实验最低 NVIDIA 型号/显存/卡数的资料；不能
擅自写“必须 H100/8卡”。DeepSpec README 的默认单节点8GPU和 NeMo 指南的2GPU
训练示例是各自配置，不是本调度算法的硬件下限。

生产论文的吞吐数字来自 V4 preview、特定 live traffic/SLA 及未完整公开的
引擎配置；单张 AMD/Qwen0.6B 不可能通过替换百分比就“复现”该实验。可以先完成
可审计的算法和等价执行适配，再将缺少的 NVIDIA/生产引擎复核明确列为未达成。

## 8. 分开算法正确性、数值保真与相同工作量的性能

论文的 lossless 论证以给定的 proposal/target 概率、接受/残差采样及
non-anticipating admission 为基础。本文核对的论证**没有声称不同 batch shape、
kernel、dtype 的浮点执行逐 bit 相同**。本项目此前要求 greedy 输出 token 完全
相同，也不是逐 bit 比较 hidden/logits：它是一项明确而严格的输出数值契约。
已有失败必须继续记作这项契约失败，不能悄悄改成通过；但也不能把它直接等同于
论文数学算法错误，或据此认定所有动态验证都必须放弃。

| 门槛 | 必须证明的内容 | 不能偷换成的结论 |
| --- | --- | --- |
| 算法/状态正确性 | 小词表已知分布的接受/残差采样、prefix admission 因果性、跨请求隔离、KV内容/位置及rollback正确；小R的预算oracle | 一两个实际prompt刚好token相等，就证明完整随机算法或scheduler正确 |
| 数值保真 | 在相同语义prefix下比较顺序/块/不同batch/历史KV路径，保留首次分歧位置、top margins、logit与概率TV/KL差异、完整输出分歧率；不改变argmax或删prompt | CPU逻辑通过、FP32局部通过，便宣称BF16逐token无损；或称近tie一定无害 |
| 性能/资源 | 同模型、精度、SLA、真实请求负载和物理执行规则下，报告实际B、padding、graph buckets、draft/verify/schedule/KV成本及输出质量 | 因人为给baseline额外padding而获得“加速”，或把逻辑ell变小当成真实算力节约 |

更具体的后续数值协议应先冻结在 dev 上：

1. 保留当前 strict cached-sequential token-equality gate 及失败样本作为一个
   单独结果字段。继续做同prefix四路分解，区分历史已接受块产生的KV差异与当前
   verify chunk shape 差异；浮点误差可能级联，不能只比较最终文本相似度。
2. 加测 **target-only 自身**从单请求到真实批量/graph bucket 的数值变异，作为
   serving backend 的数值对照，不能用它自动豁免 speculative 的新增差异。
3. 对随机采样分支，同时检查相同prefix的target/proposal概率、输出分布以及
   调度的非预知属性。论文的抽象单一target分布不自动证明各种有限精度shape
   实现的是完全相同分布；该差距要量化并注明适用的执行契约。
4. 若工程上选择容许有限精度差异，应在新实验协议中预先说明容差来源、指标、
   baseline数值变异和质量门槛，独立报告“算法验证通过/数值不逐token相等”，
   不能追溯修改旧失败或把该模式命名为 lossless。不能仅为改善接受率调epsilon。

固定 query/key 容量的 canonical padding 可以保留为数值 control 或明确命名的
独立执行模式，但不是自动优于动态执行的主方案。若跨整个batch始终执行最大
物理 token 容量，则不同逻辑 `ell` 可能不再改变实际计算/内存工作量，实测的
`SPS(B)` 台阶与机会成本也会被改变甚至抹平。这会削弱乃至消除本文要复现的
资源分配机制；不能以 padded benchmark 的成功代替原scheduler的系统验证。

如果 canonical 使用有限 graph buckets，应同时记录 logical B 与 physical B，
并按实际bucket成本profile；所有baseline也必须披露各自真实执行工作量。
控制实验与原生高效baseline并列，不能只选被最大padding拖慢的target-only来
宣传收益。**保留动态B的BF16主线与数值未决状态**，正确性诊断和工程优化均继续，
而不是通过自设一个过强执行契约把研究目标收缩成恒定padding。

## 9. 可测试的忠实路线与完成条件

1. **先保持正确性边界。** 继续保留 BF16 失败；FP32 gate 只说明该精度配置，
   不混比 BF16。补随机 rejection sampler 后，在小词表已知分布上检验输出分布。
2. **校准真实 prefix 信号。** 在独立 dev rollout 上按论文 STS 顺序拟合，冻结
   温度；用恒定 confidence、未校准 head 作对照，报告位置分母/EOS截断。
3. **实现多请求数学 oracle。** 对小 R/gamma 枚举验证固定 B 的最优 allocation；
   测 ties、c=0/1、ell=0、不同请求质量、预算守恒。同步早停只在单峰曲线上与
   全局最优相等；用 Appendix A 反例验证“当前完整块回溯最优”确会破坏因果性。
4. **测目标 engine 的容量。** 扫 B、context、R，保留真实离散台阶和误差，
   不能平滑掉不利结果。论文 Eq.(1) 同时含 draft 与 verify；无论 SPS 表的具体
   profiling 边界怎样定义，最后都须实测完整 draft/verify/schedule/cache 时间。
5. **实现可变长度多请求执行。** 同批独立请求不泄漏，EOS/拒绝/加入/退出的
   request/KV映射正确，分配后的 batch 真正按不同 ell 执行，不退化为每请求串行。
6. **实现 §5.2 异步机制。** 两步旧信息只决定K，当前信息决定rank；用修改
   当前候选未来 token 的干预测试确认不能改变其自身 admission；验证跨cliff搜索
   和调度与执行重叠，单独报告同步oracle与异步实现的差异。
7. **最终比较全系统。** 在相同正确性/精度/请求负载下比较 target-only、固定
   k、静态confidence阈值、同步Algorithm1和异步scheduler，绘制吞吐—TPS/SLA
   frontier。只有这个层级才回答 hardware-aware 资源分配是否带来系统收益。

论文 [Limitations](https://arxiv.org/html/2607.05147v1#S5.SS4) 还指出 full-block
parallel drafting 的固定前置成本不会因 suffix 不验证而消失。允许禁用 speculation
或探索难度 early exit 是合理后续研究，但其动机和结果应单列，不能暗中替换
上述原方法再声称原论文复现完成。
