# BF16 缓存投机解码正确性门槛失败（2026-10-08）

使用 aligned128 单样本 checkpoint，比较增量 target/draft KV 的 speculative
greedy 与同模型的 cached sequential greedy。3 个 prompt 中 2 个不一致，
程序保存完整迹后按设计以 exit code 1 结束。此版本是有明确数值限制的原型，
不能声称 BF16 下逐 token 无损或提速。

| 输入 | Cached speculative = sequential | Sequential = full recompute | 首分歧 |
| --- | --- | --- | --- |
| 单条训练 prompt | 否 | 是 | 输出索引 17，prefix 长度 76 |
| Validation 1 | 是 | 是 | 无，真实 EOS 终止 |
| Validation 2 | 否 | 否 | 输出索引 23，prefix 长度 172 |

训练 prompt 在前两轮各接受 7 个 token 后分歧。在相同 token 前缀下，sequential
的前两名 logit 为 23.5/23.375，block verification 将这两个候选的次序对调，
数值仍为 23.5/23.375。Validation 2 的 sequential 前两名量化为 18.75/18.75，
block 为 18.75/18.625，排名也变化。公开报告省略 token ID，原始同前缀迹仍保留。

这不是只看最终文本推断数值误差：报告记录了首次分歧前相同的 prefix 长度、
sequential 与实际 block 验证所得的 top-k logits。它仍不能排除所有实现问题；
应继续用全 FP32 target 与 draft 对同一 cached sequential 基线做独立诊断。
不得将 FP32 结果与 BF16 基线混比或用宽松文本相似度把失败改成通过。

CPU 的实际 tiny Qwen 测试共 29 项通过，其中缓存投机新增 5 项覆盖每个拒绝
位置、全接受 bonus、EOS、输出预算，以及 rollback 后继续 target 预测；缓存
draft backbone 也与完整前缀计算比较。其证据支持缓存逻辑，不推翻上述真实
BF16 失败。

`summary.json` 为不含生成 token 的结果；`source-identity.json` 绑定执行源码、
checkpoint 和运行库。完整结果、源码快照位于本地忽略目录
`output/cached-decode-gate-20261008/`，快照所有模块与执行哈希一致。测试日志
保留在 `output/diagnostic-aligned-20261008/cached-decode-cpu-tests.log`。

后续 `eval_cached_decode --target-dtype float32` 会同时使用 FP32 draft 计算，
是独立精度实验；本目录未记录该尚未执行模式的结果。
