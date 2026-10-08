# 固定样本学习诊断（2026-10-08）

这次实验验证了模型能优化训练目标，但没有达到真实 rollout 平均每轮接受 3 个
draft token 的目标。它不能回答泛化或加速问题。

同一条训练记录、4 个固定 anchor、28 个监督位置，训练 128 步：

| 指标 | 初始化 | 128 步 |
| --- | ---: | ---: |
| Teacher-forced 分布重叠度 | 0.079% | 82.775% |
| Teacher top-1 一致率 | 0% | 92.857% |
| Label top-1 准确率 | 0% | 96.429% |
| 同一 prompt 的真实 greedy 平均接受长度 | 0 | 0.632 |

五个检查点的 32-token 输出均与 target greedy 逐 token 相同；最终接受 12 个
draft token，共 19 轮验证。target 冻结检查通过，PyTorch 峰值已分配显存约
4.61 GB。26.85 秒是缓存特征后这次诊断训练、评估及保存的总耗时，不是推理性能。

## 关键限制与下一步

训练回答是预先随机采样的 target completion，与 greedy rollout 只有前 7 个
token 的完整前缀相同。训练 anchor 为 `[58, 65, 72, 79]`；最终 rollout
anchor 只在位置 79 与它们重合，但此时完整前缀已经不同。**最终 rollout 没有
任何一个 anchor 同时具有训练过的位置和完整前缀。** 因此较低接受率不能单独
归因于 Markov 的训练/推理差异，也不能假设扩数据必然解决。

下一诊断应使用同一条 target greedy 轨迹，并覆盖该轨迹中每一个可能 rollout
anchor，隔离上下文覆盖不足的影响。`diagnose_learning` 新增了
`--trajectory target-greedy --anchor-coverage contiguous`；本目录记录的是
原始 saved/spaced 实验，不是新增模式的结果。

## 正确性审查与证据

mask、label 的下一 token 移位、teacher hidden 对齐、Markov 的训练
teacher-forcing 和推理逐步依赖与固定 NeMo 源码一致。新增 CPU 测试核对了
NeMo loss 数值、全序列训练与截断前缀推理的 backbone 一致性，以及串行 Markov
proposal。连同缓存基线测试，共 15 项通过。

第一次测试有 1 项因部署缺少固定 NeMo `loss.py` 参考文件失败；补齐该文件后
全部通过。没有为此修改 Python 环境。

另修复 checkpoint 完整性缺口：新增 optimizer/RNG resume 文件的 SHA-256
校验与 step 一致性检查，且校验失败发生在写入模型参数之前。旧 checkpoint
仍可只加载权重，不再允许未经 checksum 验证的 optimizer resume。

`summary.json` 保存不含生成 token 的汇总；`source-identity.json` 保存此次执行
的每个模块 SHA-256、target/data 指纹与运行库版本。它们描述的是执行时源码，
不能用之后提交的 Git SHA 代替。原始 token 轨迹及测试日志保留在本地忽略目录
`output/diagnostic-20261008/`，权重和优化器文件仅保留在实验主机。
