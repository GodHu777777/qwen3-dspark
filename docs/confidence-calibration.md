# Sequential Temperature Scaling：CPU 算法与数据契约

本模块 `dspark_qwen/calibration.py` 实现独立、纯标准库 STS 拟合。它没有载入模型、
收集真实 rollout、选择 checkpoint、配置 scheduler 或连接 GPU decoder。
八项 CPU 小样本检查只验证算法与输入边界，不证明当前 confidence head 已校准。
不可变 expanded training 快照仍是 `796fecc`；此新增模块不写入该快照。

## 来源与标签

- [论文 v1 §3.2.1](https://arxiv.org/html/2607.05147v1#S3.SS2.SSS1)：
  在 held-out validation 上左到右搜索每位置温度，以 cumprod ECE 为目标，
  已拟合的更早位置保持不变。
- [DeepSpec recorder](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/dspark/confidence_head.py#L345)：
  `sigmoid(confidence_logits)` 后 float64 cumprod，与 verifier 的
  `accept_prefix_mask` 比较；按 `effective_proposal_length` 去掉接受 EOS 后的尾部。
- [同版本 verifier](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/base_evaluator.py#L240)：
  随机逐 token 接受事件累乘成为二值 prefix 标签。首个拒绝后，后续已验证候选
  的 prefix 标签为 0；这不等于逐 token 独立接受标签，也不是训练时的软 overlap。
- [同版本 evaluator](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/dspark/evaluator.py)：
  已读取的 evaluator 使用 20 个 coarse bins；其 recorder 仅在 confidence threshold
  为 0 时启用。上面链接的仓库文件位置应以 pinned tree 为准；本地已读原件位于
  ignored `output/scope-research/deepspec-evaluator.py`。

`RolloutBlock` 用已实现的接受前缀长度生成位置 j 的标签
`int(j < accepted_prefix_length)`，不从 teacher-forced overlap/MAE 推断标签。
正温度校准的是 confidence logits：`c[j] = sigmoid(z[j] / T[j])`，随后
`prefix[j] = product(c[:j+1])`。不是对已经累乘的概率做 temperature scaling，
也不改变 target/proposal 的采样温度。

## 每条 rollout block 的必需语义

`block_id` 必须全局唯一，`prompt_id` 标识所属 prompt（同 prompt 的多轮可以重复）；
`split` 只接受 `validation`，train/test/未知 split 均拒绝。调用方必须先按
audited development records 核实 ID 的 split；传入字符串本身不是来源证明。

`confidence_logits` 是每个候选采样前的 raw logits，顺序从位置 0 开始。
`proposal_length` 是实际可用的候选数量，不能超过 logits 长度；
`verified_length` 是前缀中有验证证据的位置数量，不能超过 proposal_length。
未验证或因输出/context 预算截断的位置不进入任何统计分母，即使较早位置拒绝
已使后续存活事件必然为零，也不把未观测尾部补成负例。

`accepted_prefix_length <= verified_length`。拒绝后的**已验证**位置仍提供零标签，
不能只保留成功到达该位置的 block，否则会从 prefix 概率变成条件接受率。
`accepted_eos_position` 是接受 EOS 的零基位置，必须落在接受前缀内；EOS 本身为
有效正例，之后全部排除。接受前缀长度可来自截断前或截断后的 verifier 计数。
未被接受的候选 EOS 不设置该字段。长度为 0 的 block 可以出现，但全体均没有
有效位置时拟合明确拒绝。

`sampling_mode` 默认 `stochastic`，所有 block 必须相同。若显式选择 `greedy`，
结果会标注 greedy regime，不能宣称已校准随机接受概率。`collection_policy`
目前只接受 `full_proposal`：用于校准的数据必须来自无 confidence 阈值/动态
admission 选择的 proposal 收集；尾部只可因预先已知的输出/context 预算或 EOS
截断。接口不能从记录反推策略因果性，调用方必须保证并在协议中记载。

`fit_sts` 强制要求 `identity` 字典：

| 字段 | 必须绑定的对象 |
| --- | --- |
| `checkpoint_sha256` | 唯一冻结的 draft 权重文件；不可混合训练 steps |
| `development_records_sha256` | audited train/validation export；收集时仅取 validation |
| `rollout_protocol_sha256` | 完整冻结协议文件，包括 target/tokenizer/runtime、采样变换、seed、panel、预算和收集策略 |
| `probability_policy` | 明确命名实际接受概率所用 target/proposal law 与数值实现 |

例如后续 cached stochastic adapter 若采用 BF16 model forward、正 temperature、
不做 top-k/top-p/min-p 过滤、float64 softmax/归一化，则协议必须写全这些内容及
具体 temperature；不能把原 greedy 日志装入这一 policy。所有 blocks 必须来自
同一个冻结 identity，这仍是调用方必须核对的契约，模块只验证指纹格式并保存。
以上来源文件哈希、rollout 文件哈希与拟合结果应一同归档；不要修改训练 run manifest。

## ECE、分母与本地选择

按位置独立构造分母：仅包括 `effective_length > position` 的 blocks。
`effective_length` 是 verified_length，若有接受 EOS，则为 EOS position + 1。
每个非空 bin 的绝对校准误差按样本数加权，ECE 分母为该位置总样本数；
Brier 是该位置二值 prefix 事件的均方误差。每条 block 一票，不做 prompt 宏平均，
同 prompt 多轮相关性不能当成独立样本置信区间。

参考 recorder 在收到 **cumprod 之后的预测概率** 时，先统一 clamp 到
`[1e-8, 1-1e-8]`，再使用 `floor(p * bins)`（限制到最后一格）。这个 clamp
同时影响 ECE、Brier 和 pred_mean，不是仅用于防 bin 越界；本模块保留该行为。
条件概率与每步 cumprod 本身不 clamp。参考实现使用 Torch sigmoid 后转 float64；
本 CPU oracle 使用 Python float sigmoid/累乘，不能声称逐 bit 相同。

论文与已核对代码没有给出完整 STS 温度网格及平局规则。因此明确采用以下**本地
约定**，而不声称它们是论文原参数：

- 默认 61 个 log2 等距温度 `2**(i/10), i=-30..30`，范围 0.125–8。
  自定义网格必须正、有限并包含 `T=1`，排序去重；输出保存实际全网格与逐位置 ECE。
- 从位置 0 到 block_size-1，选择当前 cumprod ECE 最小的 T，冻结此前位置。
  ECE 精确平局时取 log 空间最接近 1 的温度，再取较小者。不用 epsilon 重排目标。
- 始终报告全 `T=1` baseline 的同一分母指标。无有效样本的位置保存 T=1、
  `fitted=false`、count=0，ECE/Brier 为 null，不能解释为拟合通过。
- num_bins 默认 20，与已读公开 evaluator 相同，可显式改动并在结果中保存；
  本模块不实现 recorder 的 1000-bin AUROC 近似，也不绘制 reliability 图。

当前 API 返回的 calibrated ECE 是**拟合集上的目标值**，不是独立检验集性能。
后续应在预先划定、prompt 不重合的 development 子集上拟合/评估并冻结温度；
最终 test 不参与拟合、网格选择或 policy/checkpoint 选择。拟合温度不能保证
不同 checkpoint、采样法、context/load 分布上的校准有效。

## CPU 检查

```sh
python3 -m unittest discover -s tests -p test_calibration.py -v
```

手算 fixture 有 5 个两 token block，接受前缀长度 `[2,2,2,1,0]`，原 logits
都为 log(4)。第 1 位置 T=1 给出 0.8；第 2 位置
`T=log(4)/log(3)` 给出条件 0.75，从而 prefix=0.6，两个位置 ECE 均为 0。
这同时区分逐位置独立校准（错误目标）与按论文顺序固定前缀的 STS。
其它检查覆盖 EOS/输出预算/未验证尾部、拒绝负例分母、20-bin 同类规则、端点
clamp、极端 logits、网格平局、空位置、dev split/重复身份/采样法/指纹边界。
