# 对齐轨迹学习诊断（2026-10-08）

单样本机制诊断通过：第 64、96、128 步均连续 4 轮接受全部 7 个 draft token，
32-token 输出与 full-recompute target greedy 逐 token 相同。这支持模型具有
拟合该轨迹的能力，不支持泛化或提速结论。

为消除前次实验的覆盖混淆，本次先从同一 prompt 生成 target greedy 的 32-token
轨迹，再训练连续 anchor `[59..89]`，涵盖所有可能 rollout 前缀。每步 196 个
有效监督位置，128 步，学习率 0.0006，结构与前次相同。

| 指标 | 0 步 | 32 步 | 64 步 | 128 步 |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forced 分布重叠度 | 0.063% | 69.656% | 85.279% | 86.138% |
| Teacher top-1 一致率 | 0% | 79.592% | 100% | 100% |
| 首位置 teacher top-1 | 0% | 100% | 100% | 100% |
| 每轮验证接受长度 | 0 | 2.556 | 7.000 | 7.000 |

最终每一个 block 位置的 teacher top-1 都是 100%。`summary.json` 包含每个检查点
全部位置的指标，并区分“验证接受”与“受输出预算约束后实际提交”的数量。

最终输出构成为首个 target token 1 个、draft token 28 个、bonus token 3 个。
最后一轮接受的 7 个 token 全部提交，其 bonus 因预算耗尽不提交。**本次未遇到
真实 EOS，终止原因是 32-token 上限**；不能将此记录描述为实际 EOS 训练验证。
已有 CPU 测试另外覆盖 EOS 和输出预算分支。

target 冻结检查通过；PyTorch 峰值已分配显存约 5.18 GB。缓存特征后的训练、
五次评估及保存耗时 21.50 秒，不是推理性能指标。使用 BF16 target/AMP 和
full-recompute 参考核对；本次没有出现 token 分歧，不代表其他 prompt 或缓存
执行路径也必然逐 token 一致。

精确执行模块哈希、target/data 指纹和运行库在 `source-identity.json`。
原始证据及源码快照保留在本地忽略目录 `output/diagnostic-aligned-20261008/`。
执行后的快照中 `bench_cached_target.py` 被另一实验更新，故该文件与执行身份
中的哈希不同；它不被本诊断导入。其余执行模块快照均与记录哈希一致。
