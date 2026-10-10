# 两个固定初始状态的全词表分布对照

按[预先审查的协议](../../docs/matched-prefix-diagnostic.md)，只读恢复两份原预选 round 0 probe × 三个 checkpoint，共六文件 **196,953,054 bytes**。这次是已保存概率的 CPU 重析，没有新模型 forward、训练、生成或 GPU 实验。

每个状态固定 `p_ref = step 128 的 sequential target 首行`，比较 `O_ref = Σ min(p_ref, q_step)`。对归一化概率，这等于该状态下一次 q 抽样的期望首位接受概率；它不是整段 rollout 的实测接受率。

## 固定 target 分布的结果

| 原始 ordinal | Step 128 | Step 512 | Step 1280 | 512→1280（百分点） |
|---|---:|---:|---:|---:|
| 0 | 0.043384% | 0.343341% | 0.176278% | -0.167064 |
| 1 | 4.716073% | 97.353413% | 98.110884% | +0.757471 |

两个状态高度不同；512→1280 一降一升，不用两状态平均掩盖分歧。Ordinal 0 在 step 1280 的低接受概率是这个状态上的完整分布不匹配，不能仅归于那一次抽中了低 alpha 的 token。

## 数值对照与解释边界

每个状态的三份 sequential target 概率逐字节相同。因此共同参照下的 checkpoint 差异确实来自保存的 draft q，不能归于本次参照 target 的漂移。所有已保存 sequential 参照下的变化符号一致；原 block 配对的变化方向也一致。完整 O_block、O_seq、O_ref、质量残差和 TV 敏感性界保留在 `summary.json`，不修改或重新归一化原概率。

这是两处预选初始状态的描述性对照，不能外推全部 32 个 prompt 或后续生成状态；也不能区分训练目标、训练量、数据分布、容量或曝光偏差的原因。它不证明 dense/cached draft 等价。S47 的真实 rollout 统计仍保留，两者的状态分布不同。

## 验证与来源

六 payload 共 **138 个概率行**通过 FP64、形状、有限值、范围及归一化检查；质量对照仅用六个首行。与原始 proposal、初始输出前缀及 selected p/q 的精确 gather 一致，原首行数值控制复核通过。输入与分析源码前后哈希不变。

12 项聚焦测试通过，唯一正式分析及 worker/supervisor 均退出 0。运行时间只记录 CPU 数据处理，不是推理性能。分析器编写期间代理连接中断一次，原代理从已保存脚本继续；没有测试或正式分析失败，没有重复取数或实验。来源、资源记录及证据哈希见 `audit.json` 和 `source-identity.json`。原始 tensor、token、ID、seed、选中 token 的概率仅留在 ignored 本地证据中。

恢复时哈希证明现有文件与传输字节一致，不能替代不存在的历史预提交完整 probe 哈希。模型权重及共享生成文件没有重开；原来源/绑定/完成记录的核验承接 S47。

尚无新的端到端收益、confidence admission 或 hardware-aware 调度证据。保留[同 backend R2 控制](../paired-r2-benchmark-20261009-17da0b4/README.md)及其独立 vLLM 强基线，保留已失败的 sampler 优化结果，不从本次重叠数值推演吞吐。


独立实现先以 `math.fsum` 重算六个首行，180 项检查通过；其数值在正式分析完成前没有发给分析器作者。随后仅用已保存 JSON 做 337 项对照，最大差约 6.66e−16，低于预定 1e−12 容差。详见 `independent-comparison.json`。

下一步接受的方向是先设计原 32 个 quality prompt 的共同初始前缀对照，比较 step 512/1280，以判断两状态的差异在多大范围内存在。复用已保存初始 token，不生成新轨迹；覆盖、共同 target 分布、数值控制和资源上限需在任何新模型执行前明确。本轮没有实施或运行这个扩展，也不改变 checkpoint、训练或校准。
