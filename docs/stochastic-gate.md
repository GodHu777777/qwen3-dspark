# 有界真实 stochastic gate（GPU 执行待协调）

`python -m dspark_qwen.eval_stochastic_gate` 固定使用 prior private gate 中
同一个 aligned128 checkpoint、原训练 prompt 和前两个 accepted validation
prompt。不会筛掉先前失败的 case，也不会读取 final test。当前完成了 CPU 测试
及原始文件 dry-run；**尚未运行真实 BF16 GPU gate**。

## Dry-run 与独立执行

在具有原始模型/data/checkpoint 的主机，先用新输出目录：

```sh
python -m dspark_qwen.eval_stochastic_gate \
  --prior-gate /path/to/cached-decode-gate-20261008/result.json \
  --output /path/to/new-stochastic-dry-run --dry-run
```

`--dry-run` 仅使用 Python 标准库，不 import torch/transformers、不查 CUDA、
不加载模型或优化器。它校验并绑定 checkpoint 权重/metadata、target 文件、
records、prior gate、三个 case 与协议，并复制执行 package 源码及全部 `.py`
哈希到输出目录。checkpoint 可用 `--checkpoint` 指向搬迁副本，但其哈希必须
等于原 prior checkpoint。记录 token 原文及私有路径属于 private `run.json`。

正式 GPU 运行在主 agent 协调窗口后执行同一命令，去掉 `--dry-run`，并换一个
**全新**输出目录。程序自动从该输出的 `source/` 启动独立 worker，再次校验
所有绑定才访问 GPU；不在训练快照、生成脚本或服务 checkout 中修改任何源码。
dry-run 输出不能直接覆盖或续跑。输出目录权限为 0700；同账号访问仍是约定，
不是独立安全边界。

## 有界工作与真实数值策略

默认 temperature=1、seed=20261009、max-new-tokens=32；最多两个不同 seed，
预算 1..32，预先固定 probe-rounds=0..2（默认 2），timeout 默认 600 秒且最多
1800 秒。只用动态 CachedTarget，target forward BF16，draft FP32 参数加
BF16 AMP，SDPA；p/q 使用 `float64_softmax_normalize_cdf_v1`，无过滤。
这些参数和源码绑定写入 run.json，实际 torch/transformers/HIP/device 设置写入
runtime.json，不为了过测试切换全模型 FP32。

worker 启动前要求至少 8 GiB 空闲，设置本进程 PyTorch 分配上限 6 GiB。
这是 allocator 限制，不是对驱动、workspace 或其他进程的全系统内存保证。
外部 launcher 到期只杀死自己的 worker，保留已有结果。内存/时间约束不能
替代 GPU 窗口协调；不要与 generation、memory gate 或训练计量同时运行。

每 seed 对三 case 分别跑 reference、立即 repeat，以及跨请求交错 repeat
（最后一轮顺序为 1,0,2，保证每个 case 前面是别的请求）。target-only baseline
使用 `CachedTarget(model,())`，无 draft-context hooks，概率 adapter 和温度
与 speculative 完全相同。检查同路径 token、speculative round/audit 迹复现；
**不比较同 seed 的 target-only 与 speculative token 是否相同**，随机数消耗
本来不同。

每轮检查 p/q、confidence、输出 shape/有限值、target KV 全缓存与 draft 新投影
KV chunk 的形状/有限值、投影提交边界、接受 prefix、残差支持集、EOS/预算和
末 token 不入 cache 的规则。tiny Qwen CPU 的完整 KV 内容对照仍由独立测试
承担；真实 gate 的有限/shape 检查不等于重新证明全部 GPU KV 数值相同。

## Same-prefix 数值探针单独报告

只选择每个 reference case 预先配置的前 K 轮；不按失败程度挑选。
保存实际 block logits、实际 p/q、proposal token 和 raw confidence 后，在
独立 fresh target cache 中 prefill 同一 prompt，再**逐 token**重放已提交输出
（含当前 anchor）及 sampled proposal prefix。逐行比较所有 n+1 个 target 分布，
包括 bonus 行；拒绝后的行明确属于 sampled-proposal 路径而非已提交路径。

记录 TV、最大概率差、最大/平均绝对 logit 差和 argmax 是否变化。比较混合了
历史 block/chunk 与当前 query-shape 数值影响，不能将其解释成已隔离某个 kernel
根因。`native_numerical_fidelity` 单独汇总，不用阈值把非零误差改为“无损通过”。
执行/复现通过也不表示离散模型分布无损得到证明；CPU 数学 oracle、后端数值
保真、模型质量、性能始终是不同的证据。

## 结果、失败与 private traces

- `aggregate.json`：不含 prompt/output token 或 record ID；仅状态、绑定哈希、
  case 序号/split/seed、数量和标量探针统计。公开前仍须检查内容。
- `result.json`、`private-rounds.jsonl`：私有实际 token、每轮实际 selected p/q、
  confidence、接受/拒绝及 cache 信息，分阶段/逐轮保存。
- `private-probe-*.pt`：实际 block/seq logits、float64 p/q 和 prefix token。
  默认 Qwen3-0.6B/7 draft/3 case/2 probe/1 seed 可能约 200 MB；不是轻量日志。
- `stdout.log`：launcher 实时复制 worker 的完整 stdout/stderr，包括原生输出。
  `error.txt`/`launcher-error.txt` 保存异常；`result.json` 和 aggregate 原子替换。
  probe 中途失败保留已完成 row 指标和部分 tensor，worker 超时/退出保留之前
  的私有 JSONL 和结果，不能因失败删掉 case。

这不是性能 benchmark：完整 finite 检查、float64 softmax/CDF、CPU tensor
复制、文件写入和 fresh sequential 探针均有成本。没有速度结论。

CPU 测试：`python -m unittest discover -s tests -p test_stochastic_gate.py -v`。
覆盖禁止 dry-run import torch/启动 worker、输入损坏和 case 漂移拒绝、源码绑定、
fresh 输出、原生 stdout/stderr 保存、超时只终止自有 child、partial evidence、
真实 tiny Qwen 的 9 次运行/6 次复现检查及全部同 prefix probe 行。
