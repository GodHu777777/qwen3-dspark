# 扩数据训练计划：先通过正确性门槛，再评估泛化

状态（2026-10-09）：扩数据审计、CPU preflight、两周期 GPU 显存门槛及首段
step32 已完成，实际为 932 train / 119 dev；证据见[实验日志 T04/T05](lab-notebook.md#expanded-resource-gate)。
下文保留分段训练与评估协议；各后续阶段是否执行需查看对应结果，不能由计划推断。
示例配置为 `configs/train-expanded.example.json`，路径相对运行工作目录；执行前
复制为忽略的机器配置，替换模型、数据、manifest 和新输出目录，然后冻结配置。

## 已知结果与要回答的问题

旧 sampled/spaced 单样本诊断有上下文覆盖混淆。改用同一 target-greedy 轨迹和
全部连续 anchor 后，64 步起每轮接受 7 个 draft token，证明该结构能拟合已见
轨迹。它不证明泛化。进一步的 BF16 cached-block 解码在 3 个 prompt 中 2 个
与 cached sequential target 不一致，已公开保留失败；尚不能声称该执行路径
逐 token 无损。因此本轮须分别回答训练能否泛化、选定精度下解码是否正确，
性能测试只能在后者通过后开展。

## 数据与实验身份

- 原始选择规模为 1,024 train / 128 validation / 128 final test。实际可用条数
  以生成后自然 EOS、长度和模板审计结果为准，不能将选择条数当作训练条数。
- 训练只读取 `generated/records.jsonl`，其中只有 train/validation；final test
  的记录在独立目录。禁止把 test 的完成内容、准确率或接受率用于选 checkpoint、
  rank、anchor 数、精度或调度策略。结构性哈希/格式审计不等于解锁效果评估。
- 使用已排除 pilot prompt 的新数据。保留源分片哈希、selection、生成 manifest、
  输出哈希和审计结果。共享同一用户 ID 的文件权限不构成防止人为读取 test 的
  安全隔离，test lock 是实验约定。
- 从头训练作为首个基线；不加载 pilot 或单样本 overfit 权重。记录 Git commit、
  未提交 diff、每个模块 SHA-256、目标/数据指纹、运行库、完整配置及有效样本数。
  训练执行期间不修改部署包，因为所有模块都属于严格 resume 的源码身份。
  建议在本次训练专用、不可变的源码快照目录运行所有分段；主开发部署可继续
  改进评估/缓存代码，但不覆盖训练快照。否则即便改动的是未导入模块，也会让
  当前实现的全包 hash 校验拒绝 resume。

## 首个配置与内存门槛

五层 draft、block size 7、Markov rank 256、32 anchors、micro-batch 1、累积 8，
共 1,280 个 optimizer step，学习率固定为 6e-4，梯度裁剪 1.0。保持三项 loss
和 decay gamma 4。沿用当前训练入口，不为本轮引入新调度器或改写训练循环。

32 anchors 每 micro-step 最多 224 个监督位置，累积 8 时最多 1,792 个
block-position 权重项；重叠 block 会重复监督同一 token，这不是唯一 token 数。
NeMo 参考配置使用 512 anchors，本计划只有其 1/16。累积 8 与参考单机设置一致，
并不达到论文 effective batch 512，也没有复刻其 warmup/学习率衰减。

aligned31 诊断峰值约 5.18 GB，但它的序列很短，不能据此担保 4,096-token
训练内存。先以审计后最长 train 序列作独立内存检查：冻结 target capture、
32-anchor forward/backward、AdamW 第一次 update（必须创建优化器状态），
再完成第二个 accumulation/update 周期，让 Adam moments 常驻时的 forward/backward
峰值也实际出现。分别记录首次分配与常驻状态周期的峰值，不能假设前者覆盖后者。
检查 loss/grad 有限、全部组件梯度、target 冻结及峰值显存。也记录当时其它服务
占用；不停止无关进程。内存检查使用独立输出，不产生正式实验 checkpoint，
正式训练重新 seed、重新初始化。

最长序列也未必最占显存：其 completion 可能很短，实际 anchor 很少，而稍短
序列可能有完整32 anchors。memory gate 选取真实训练记录在 `(sequence length,
actual anchors)` 两个维度的非支配集合；最长记录若也有全数据最大 actual
anchors，通常只需一个 probe，否则每个非支配形状独立初始化并测两周期。
这是实际数据的经验资源检查，不是对所有 kernel workspace/allocator 行为的
形式化 worst-case 保证。

若内存有足够余量，64 anchors 可另做独立内存检查，但首个正式基线仍为 32。
64-anchor 比较必须是新配置、新 run、从头初始化，不能在 32-anchor resume
中直接改配置。64 相对 NeMo512 仍为 1/8。

## Anchor 覆盖与训练顺序

`select_anchors` 从所有满足 `token[a+1]` 属于 assistant/EOS 的位置中均匀、
无放回抽至多 32 个 anchor。它包括最后 prompt token 对第一个回答 token 的
预测，也包括倒数第二 token 对 EOS 的预测。长度不足 32 时使用全部可用位置。

每个 block 预测 a+1 到 a+7，越过序列末尾的位置通过 `valid` mask 排除。
teacher 使用对应的 `h[label_position-1]`，不把 clamp 的尾部占位当真实标签。
因为靠近 EOS 的 block 较短、较早 block 有指数衰减，目标是 block-position
加权目标，不是对所有唯一 completion token 等权的 corpus loss。

训练每次重新抽 anchor，RNG 随 checkpoint 保存；不会像固定4-anchor诊断那样
永久漏掉大多数位置。数据行顺序目前按选择时的随机顺序固定循环，每个 epoch
不再重新 shuffle。实际数据遍历次数为 `steps * 8 / accepted_train_count`；
若恰有 1,024 条，128/512/1,024/1,280 步对应约 1/4/8/10 次遍历。拒绝数据
会使相同步数的遍历次数更高，报告应按实际数量计算。

## 分段训练与严格恢复

先冻结同一配置的 `max_steps=1280`，使用已有 `--stop-after` 分段：

```bash
python -m dspark_qwen.train --config configs/train-expanded.local.json --stop-after 128
python -m dspark_qwen.train --config configs/train-expanded.local.json --resume --stop-after 512
python -m dspark_qwen.train --config configs/train-expanded.local.json --resume --stop-after 1024
python -m dspark_qwen.train --config configs/train-expanded.local.json --resume
```

输出目录必须开始不存在。每段结束才保存 checkpoint，所以中途进程故障会丢失
本段尚未保存的更新；首次训练可先在 32 步结束以更早核验，再原样恢复到 128。
不要修改 `max_steps`、LR、anchor 数、源码、数据、模型或输出目录来“继续”旧 run。
所有段同一配置身份；resume 校验权重与 optimizer/RNG 文件哈希及 step 一致性。

现有入口支持严格同身份 resume，**不支持跨数据/配置 warm-start**。如后续确需
warm-start，应增加独立参数，只加载权重、重建 optimizer/RNG、记录父 checkpoint
哈希和新实验身份，不能伪装为 resume。本轮不需要这个功能。

## Dev 指标与 checkpoint 选择

每段前后，现有训练入口会对全部可用 validation 做固定 seed 的 anchor 评估，
anchor RNG 与训练独立。保留 CE、L1、teacher-forced overlap、confidence BCE/MAE，
注明当前统计是每样本 macro 平均。不要仅因总 loss 或 confidence MAE 下降选模型；
接近零的接受标签会让恒低 confidence 看起来不错。

预先固定 validation 的前 32 条（按冻结 records 的顺序，若不足则全部），在
128/512/1,024/1,280 步上做相同的最多 128-token greedy rollout。主要质量指标：
每轮**实际提交**的 draft token 平均数、prompt-level 分布和区间，以及首位置
top-1；报告 block 位置 1..7 的覆盖/正确率。统计 EOS 和长度终止，区分验证接受数
与输出预算下实际提交数。选择后再用全部 dev 复核，不触碰 final test。
Rollout 中第 k 个位置的接受统计只在前 k-1 个位置存活的轮次上计算，并报告
对应分母；拒绝点后基于错误 proposal 前缀的 logits 不属于正确目标轨迹。

当前可用工具为 `eval_cached_decode`，支持独立输出目录和 `--target-dtype float32`。
下一步先在短 panel 上验证 FP32 target + FP32 draft 与 FP32 cached sequential
基线的一致性；通过后才能将该明确精度的 rollout 接受作为候选选择依据。FP32
不是 BF16 无损性的修复证据，也不能与 BF16 target-only 性能混比。如果 FP32
门槛仍失败，保存同 prefix 的分解诊断并排查逻辑，暂停基于它的模型选择。

每个 checkpoint 的评估必须使用新输出目录，保留执行源码身份与 checkpoint
权重哈希。旧 `eval_decode` 默认把 `greedy-evaluation.json` 写到 checkpoint
父目录，会覆盖同 run 的先前评估；本计划不依赖该共享文件名保存多检查点证据。

## 独立工具与停止条件

已实现独立 `memory_gate`，复用 `forward_loss`，对真实非支配训练形状分别执行
两次完整 gradient accumulation/update 周期：第一次创建 Adam moments，第二次
覆盖 moments 常驻时的 forward/backward。分别重置并记录周期峰值，同时保存两者
最大值。仅第一次 optimizer update 的峰值不能保证覆盖稳态；只做一个 micro-step 会漏掉
已经分配梯度后的下一次 forward 峰值，因此不能替代这个检查。工具保存实际
free/allocated/reserved/peak 内存、梯度与 target 冻结核验；失败时也保留结果，
不保存 checkpoint，不影响后续正式从头初始化。

```bash
python -m dspark_qwen.memory_gate \
  --config configs/train-expanded.local.json --output runs/memory-gate-32anchors
```

`eval_dev_tf` 只读 checkpoint 的 validation 数据，固定 records 顺序前 32 条及
anchor seed，保存 panel 身份、每样本 macro loss、逐位置分母/teacher top-1/
label top-1/分布重叠度/confidence MAE。任何 test 行（包括被拒绝的 test 行）
出现在输入 records 都会拒绝；它不会调用或解锁 final test。

```bash
python -m dspark_qwen.eval_dev_tf \
  --checkpoint runs/train-expanded-32anchors-seed20261008/step-000128 \
  --output runs/dev-tf-step128 --panel-size 32 --anchor-seed 20262007
```

该报告明确记录 rollout gate 为 `not_run`，不能把 TF 评估成功当成 cached
解码正确或实际接受率通过。真实 rollout 仍需独立数值 gate。

待补充的 Dev aggregate：从已保存的 token/round 私有迹计算 committed acceptance、
   EOS/长度边界和按 prompt bootstrap 区间；公开报告不包含生成 token。

NaN/Inf、target 获得梯度或被修改、数据身份不一致、缓存正确性门槛失败都是
停止相应阶段并保留证据的条件。Dev 指标停滞则按预定分段停止，记录负面结果；
不能以继续增加 step 或修改数据/精度来掩盖失败。训练质量与正确性通过后再测
固定 k=1..7 成本及 confidence/cost-aware 调度，允许策略选择关闭 speculation。
