# 原 pilot stochastic gate：执行复现通过，native 数值差异仍存在

2026-10-09，使用 `d6e385d8c41253de7c37ebb47dd7b16169e6b0b5` 独立源码快照，
对原 pilot 的 aligned128 单样本 checkpoint、原训练 prompt 和原两个 validation
prompt 运行一次有界 gate。**这不是 expanded-data step128 checkpoint 的质量结果。**
target forward BF16，draft FP32 参数/BF16 AMP，SDPA；概率策略为
`float64_softmax_normalize_cdf_v1`，temperature=1，无过滤，seed=20261009，
最多输出 32 token，每个 reference case 预先固定前两轮数值探针。

9 次运行和 6 次同路径 repeat/state-isolation 检查全部通过。195 个 speculative
round 的概率/shape/finite、接受 prefix、残差支持集、cache 提交、EOS/预算检查
通过；末 token 保持未进入 cache。没有要求同 seed 的不同算法输出 token 相同。

| 原 case | 输出数 | Speculative 轮数 | 已提交 draft token | 停止原因 | 同 prefix 最大 TV |
| --- | ---: | ---: | ---: | --- | ---: |
| 单条训练 prompt | 32 | 29 | 2 | 预算 | 0.0328185404 |
| Validation 1 | 6 | 5 | 0 | EOS | 0.0416194062 |
| Validation 2 | 32 | 31 | 1 | 预算 | 0.0381776330 |

表中为 reference run；其余两轮使用同 seed 的同路径结果与 audit 迹完全复现。
target-only baseline 使用无 context hooks 的 `CachedTarget(model,())` 和相同
float64 概率 adapter。这里没有速度比较，也不能用这个旧单样本 checkpoint 的
随机接受率推断 expanded 训练质量。

## Native numerical fidelity：保留负面结果

在完全相同的 prompt、已提交输出与 sampled-proposal prefix 上，比较实际 block
路径 p 与 fresh cached sequential target 路径 p：

- 比较 48 行，其中 **47 行 TV 非零**。
- 最大 TV 为 **0.04161940616245697**（约 4.16%）。
- 最大绝对 logit 差为 **0.5**。
- **1 行 argmax 改变**。

这些探针包含 n 个 proposal 行及 bonus 行；拒绝后的行属于 sampled-proposal
路径，不是已提交轨迹。探针混合历史 chunking 与当前 block shape 的影响，未隔离
某个 kernel 为唯一根因。执行/复现通过不能改写成“对 sequential target 分布无损”。
概率升为 float64 也没有消除 BF16 forward 的路径差异。

先前 [dynamic BF16 greedy gate 的 2/3 失败](../cached-decode-gate-20261008/README.md)
继续保留；本次不同采样协议没有推翻它。数学离散分布 oracle、后端数值保真、
模型质量和系统速度是不同的证据。

## 运行与证据边界

运行后 launcher/worker 均已退出进程表，GPU 上只剩原 ASR 服务；VRAM used
回到运行前同一值 8,962,179,072 bytes。ASR 前后 HTTP 200、ready=true、busy=false。
本 gate peak allocated 为 2,260,815,872 bytes。内存数字及 scalar 校验/CPU traces
不构成性能 benchmark。

`result` 状态和最后日志事件均为 completed，没有 worker/launcher error 文件。
此次后台启动没有额外持久化 OS exit-code 文件，因此只能报告这些实际完成证据，
不能把某个 exit code 写成直接观测。未重跑、未改参数或为了通过而丢弃 case。

- [aggregate.json](aggregate.json)：执行/复现与数值差异的独立汇总。
- [numerical-probes.json](numerical-probes.json)：48 行不含 token 的标量探针。
- [audit.json](audit.json)：9 次运行、6 次 repeat 和运行后状态的纯标量审核。
- [protocol.json](protocol.json)、[runtime.json](runtime.json)：冻结协议及真实运行库/设备标签。
- [source-identity.json](source-identity.json)：源码、checkpoint、target/data 与私有证据哈希。

本目录已检查无私有路径、prompt/generated token、实际 p/q 向量或 record ID。
完整 private result、逐轮迹、logits/p/q tensor 和 stdout/stderr 保留在远端原始
run 中，未下载到公开目录。公开的是派生标量与身份标签，不是原始生成内容。

部署清单差异已核对：正式执行来自 git archive，包含 27 个正规 `.py` 模块，
没有 AppleDouble sidecar。此前 CPU dry-run 的 51 项 inventory 包含 26 个正规
模块和 25 个 macOS `._*` sidecar；共同 26 个模块 SHA 完全一致，正式快照另有
`packed_target.py`（本 gate 仍使用动态 CachedTarget）。两次绑定不是同一清单，
没有追改本次已执行 snapshot 或其 identity。
