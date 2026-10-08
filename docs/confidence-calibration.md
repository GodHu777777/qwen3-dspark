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

## 已绑定的 fit44/eval43 CPU 工作流

`collect_rollout --group fit` 和 `--group eval` 使用同一份原始 development manifest，
分别选择其中完整的 44/43 条 case 和原 seed。默认仍为 `quality`，保留旧 quality
binding 的字段形状；fit/eval 必须把显式 `collection_group` 写入 binding digest。
历史 `a278e5a` quality32 快照、协议与报告不被新实现替换。原先两个 quality TV
probe 的 ID 不属于 fit/eval，因此这两组不会自行挑选替代 TV probe。

以下只展示运行接口。真实 fit/eval GPU 采集需要独立审核窗口和已冻结 checkpoint；
本次 CPU 实现本身不授权或执行采集。独立 decision manifest
`configs/sts-step1280-selection.json` 已按 root 的选择记录冻结 step1280 权重/metadata、
原 panel/protocol 与 quality32 决策依据；fit/eval 必须显式传入它，且它的 bytes 和
digest 进入 collection 与校准 identity。先在同一个不可变新源码快照分别 dry-run：

```sh
python -m dspark_qwen.collect_rollout --group fit --selection configs/sts-step1280-selection.json --checkpoint /private/frozen-checkpoint --manifest /private/panel.json --output /private/fit-dry --dry-run
python -m dspark_qwen.collect_rollout --group eval --selection configs/sts-step1280-selection.json --checkpoint /private/frozen-checkpoint --manifest /private/panel.json --output /private/eval-dry --dry-run
```

获批的两次采集都完成后，纯标准库 CPU CLI 才可以消费各自的 collection 目录：

```sh
python -m dspark_qwen.calibrate_rollout fit --collection /private/fit/collection --output /private/sts-fit.json
python -m dspark_qwen.calibrate_rollout eval --artifact /private/sts-fit.json --fit-collection /private/fit/collection --collection /private/eval/collection --output /private/sts-eval.json
```

CLI 不加载 Torch、模型或 GPU；输出采用排他创建，不覆盖旧 artifact。fit/eval
collection 必须有完整的 `completed` 结果、`execution_checks_passed=true`、由
launcher 实际等待的 `worker-exit.json: returncode=0`、全部 44/43 条 case/seed/output，
以及原始 rounds/blocks。零 block 提示仍必须有其初始输出和完成记录，不能删掉或
补伪标签。所有提示均无有效位置时 fit 明确拒绝；eval 可以保留零样本指标。
launcher/controller 自身 OS 退出、timeout 和 GPU 释放仍由外层运行控制器独立归档；
这个 CPU CLI 不把 worker 的退出冒充为那些进程的退出。

加载时复核源快照文件与 binding、checkpoint 权重/metadata、development records、
完整 manifest、case/seed/split 和 target 指纹；逐轮重算原始 prefix 标签、每位置分母、
提交 token、cache 长度及汇总计数。预算尾部必须符合 `min(block_size, remaining)`，
拒绝后的已验证零标签不可删，接受 EOS 后的尾部不可补。只有显式 fit/eval group
可用于此 CLI；quality 或重新标记/调换的 group 均拒绝。

冻结 artifact 保存 checkpoint/metadata/data/full-manifest/protocol、全部 collector
模块 SHA256、Torch/Transformers/HIP/device、fit binding/文件 SHA256、完整 fit prompt
身份、STS 实现文件 SHA256、温度网格/温度/20 bins 和 fit-only prevalence。它属于
**私有证据**，不能未经审查直接发布。eval 重新验证原 fit 文件未变、artifact digest
未变、实现未变、eval 同 checkpoint/协议/源码/runtime，并明确核对 prompt ID 和
prompt-token hash 与 fit 不相交；eval 不再搜索温度或估计常数。

每位置以同一组 effective prefix 标签比较 unscaled、冻结 STS 和 **只在 fit44
估计的 prefix prevalence 常数**，报告 ECE/Brier/count、block coverage 和 prompt
coverage。constant 不做条件概率累乘，它直接估计该位置的累计接受事件率。
如果 fit 在某位置没有有效标签，常数为 null、`available=false`、预测 count=0，
并单独保留 eval 的 `observed_label_count`；绝不用 eval prevalence 或零常数填补。
STS 对应位置保留 T=1 并标记 `fitted_on_fit=false`。所有方法仍用相同 20-bin/clamp
定义；fit_metrics 始终标记为拟合样本内指标。eval43 只相对 STS fit prompt-held-out，
不是未参与过模型监测的数据；final test 不进入任何一个阶段。

新增测试覆盖 stdlib dry-run/fit/eval、默认 quality 兼容、完整分组与实际 worker exit
要求、原始标签/计数篡改、checkpoint/runtime/source/fit 文件/artifact 变更、分组混入、
零 block 提示、缺失 fit 尾部常数和无替代 TV probe。CPU 检查通过不代表已有真实校准
结果，也不提供原生 BF16 losslessness 或服务性能证据。
