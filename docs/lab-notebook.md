# 实验日志

最近记录核对：**2026-10-09 00:03（UTC+8）**。本轮直接读取 reproduction scope、canonical/scheduler 源码、39/44/46 项 CPU 日志、真实 canonical gate 聚合/暂停证据及源码哈希；其他条目也分别注明直接读取的聚合证据、实时检查或协调者转达。并非所有后续进度都是转达。本文持续追加；旧结论若被修正，保留原结论并说明修正依据。历史实验与实时进程状态分开记录。

早期研究问题：冻结 Qwen3-0.6B target 后，并行 DSpark 草稿能否比带 KV cache 的 target-only greedy 更快地产出完全相同的 token？训练可运行、loss 下降、回退输出一致，各自只回答这个问题的一部分。早期阶段门槛见[实验计划](experiment-plan.md)，下面历史实验的协议与失败口径不回改。

当前完整目标见[忠实复现范围](dspark-reproduction-scope.md)：还需随机分布恢复与非预知 admission、confidence 校准、跨活跃请求的全局验证 budget、可变长度多请求执行及两步历史异步容量机制，并在真实负载/相同物理工作量下评估系统收益。Greedy correctness 与 canonical 数值 control 是分层检查，不能缩小或替代这个完整目标。

公开链接只指向聚合报告。生成的 prompt、回答、精确 token 轨迹、机器路径、权重、优化器状态及凭据不进入日志。`output/` 是 **未发布的本地证据**。已有实验早于首个项目提交，实际身份由当时 manifest 中的源码、配置、数据、模型哈希绑定；后来的 Git 基线不被描述成启动这些历史实验的 commit。

复盘索引：

- 数据：[pilot 聚合](../reports/pilot-20261008/README.md)、[扩充准备与 smoke](../reports/data-expansion-20261009/README.md)。
- 训练与对齐：[采样轨迹诊断](../reports/diagnostic-20261008/README.md)、[greedy 对齐诊断](../reports/diagnostic-aligned-20261008/README.md)。
- 缓存数值：[dynamic BF16 失败](../reports/cached-decode-gate-20261008/README.md)、[canonical 真实 drafter gate](../reports/canonical-real-draft-gate-20261008/README.md)。
- 调度：[Algorithm 1 与生产异步机制范围](dspark-reproduction-scope.md)。
- 资源门槛：[M01 的两周期/Pareto 复盘](#memory-gate)。

## 2026-10-08 — P01：pilot 数据重生成，已完成

**目标与问题。** 先取得可审计的 non-thinking target 回答，保存精确生成 token，并在生成前隔离训练/验证 prompt。真实 Qwen3 权重上的准备、生成、拒绝与审计流程能否完整执行？

**方法。** 数据源为 `mlabonne/open-perfectblend`，固定 revision `af60f3c18201652a83a93f46fcfee1b646ba3df7`。从第一个 Parquet 分片抽取 64 个规范化后唯一的首轮用户 prompt；保留初始 system，丢弃来源回答和后续轮次；生成前划分 56 train / 8 validation。冻结 Qwen3-0.6B，seed 20261008，batch size 8，关闭 thinking，temperature 0.7、top-p 0.8、top-k 20、min-p 0；回答上限 1024 token，prompt 上限 2048，完整序列上限 4096。环境：Python 3.12.14、PyTorch 2.12.0+rocm7.2、Transformers 5.17.0、HIP 7.2.53211；AMD 上 BF16 + SDPA。

**验证与结果。** 已直接读取本地 selection、generation manifest、summary 和 audit。64 条输入全部有记录；56 条自然 EOS 完成并接受，最终 49 train / 7 validation；8 条达到长度上限，进入隔离记录。审计通过 prompt 身份、split 互斥、无来源 assistant 轮次、导出内容与 records 一致、精确 token 长度、EOS/模板前缀对齐、序列预算及输出 SHA-256。生成 23,430 token，生成循环记录 284.45 秒，PyTorch 峰值已分配显存 3,327,124,480 bytes（约 3.10 GiB）。

**结论边界与问题。** 这是单分片、小样本执行检查，不代表全量领域分布。长度拒绝会引入回答长度偏置。未按事实正确性过滤，因为学习目标是模仿 target。最终 validation 只有 7 条。生成时间是本次造数据记录，不是推理引擎 benchmark。

**证据与身份。** 本地未发布：`output/pilot-20261008/input/selection.json`、`generated/manifest.json`、`generated/summary.json`、`audit.json` 和 generation log。公开[初始聚合报告](../reports/pilot-20261008/README.md)记录样本量与训练背景。源分片 SHA-256 为 `06d223e493c9e3822c7862ae0705c1a0e4705fd21385276917fc8348e7ce7b0f`；prompt 文件 SHA-256 为 `29e039d4c697cb732dcd6ef360a2c5514600472d6499b9ad1abc651cdab2d408`。manifest 保存模型/tokenizer 文件指纹与生成脚本哈希；本日志不复制实际样本。

**下一步。** 先做微型学习诊断，再决定扩充生成；扩充必须排除全部 pilot prompt，分离 dev 与最终 test，继续保留来源与恢复身份。

## 2026-10-08 — T01：8 step 训练与恢复，已完成

**目标与假设。** 核对真实 forward/backward、冻结 target 和优化器恢复。短跑可以证明训练链路可执行，是否学出有用草稿仍待验证。

**方法。** 49 条 pilot 训练记录；NeMo 参考配置保留零基 decoder block 输出 [1, 7, 14, 21, 26]、5 层 draft、block size 7、Markov rank 256、每步 4 个 anchor、micro-batch 1、梯度累积 2；AdamW 学习率 0.0006、梯度裁剪 1.0；FP32 trainable + BF16 AMP。可训练参数 161,692,161。v2 在 optimizer step 4 保存并结束进程，再恢复到 step 8，共 16 个 microstep。7 条 validation 均使用固定 seed 的 anchor，指标按样本做 macro 平均。

**验证与结果。** 8 项 CPU unittest 通过，覆盖固定 NeMo mask 对照、未来/跨块信息隔离、label 与梯度、冻结 target 特征、checkpoint/优化器恢复和 greedy 边界分支。单元测试核对恢复后的下一步参数更新一致；真机实验另外核对真实停止/重启。两段真机报告均确认 target 冻结。峰值 PyTorch 已分配显存 4,612,584,960 bytes（约 4.30 GiB）。

| 验证时点 | Macro loss | CE | 概率 L1 | Teacher-forced overlap | Confidence MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 训练前 | 3.421146 | 12.251003 | 1.996324 | 0.001838 | 0.324706 |
| Step 4 | 2.746994 | 8.962182 | 1.969907 | 0.015046 | 0.017389 |
| Step 8 | 2.714265 | 8.684708 | 1.974229 | 0.012885 | 0.013045 |

**发现与结论边界。** Step 4→8 的 loss 继续下降，overlap 却从 1.5046% 降到 1.2885%；因此不能只凭总 loss 选择更有用的草稿。confidence 学习的是 detach 的分布 overlap，不是二值 greedy 接受事件。同样的权重与 macro 平均下，恒零 confidence 的 MAE 等于平均 overlap：step 8 为 0.012885，优于学习 head 的 0.013045。此比较由指标定义和现有聚合数值推导，并非另一次 rollout 校准实验。Teacher-forced overlap 不等于实际连续接受长度或加速比。16 个 microstep 尚未遍历全部 49 条数据，不支持收敛结论。

**证据。** 公开聚合：[step 4 报告](../reports/pilot-20261008/result-step-000004.json)、[step 8 报告](../reports/pilot-20261008/result-step-000008.json)、[逐步指标](../reports/pilot-20261008/metrics.jsonl)、[CPU 测试日志](../reports/pilot-20261008/core-tests-20261008.log)。本地未发布的 run manifest 和 checkpoint 身份在 `output/train-pilot-20261008-v2/`。测试日志开头有两条 `(null): No such file or directory`，随后全部 8 项用例通过；原因尚无证据，不删除也不解释成已定位故障。

**下一步。** 在很小的 target-greedy 训练集合上测固定 anchor 的逐位置 top-1 与实际连续接受数，再决定是否投入扩充数据；针对实际部署的接受事件比较 confidence 和常数基线。

## 2026-10-08 — G01：真实 greedy verifier 对照，已完成

**目标与问题。** 真实权重下，accept/reject/correct 参考循环能否保持 target-only greedy 的 token 语义，尤其是草稿很差时？

**方法与结果。** 2 条真实留出 prompt 共输出 30 token，全部与 target-only greedy 一致，**接受草稿 token 数为 0**。验证采用完整前缀重算。直接解码的前一轮 8-step checkpoint 与最终 v2 checkpoint 的 trainable 权重逐字节相同；解码源码、target 指纹也一致。共同权重 SHA-256 为 `5df9b6e052aff2a5cee2542e7b3356f75a071b560d56195bed1a729818ae4c04`。

**结论边界。** 支持这两条短轨迹上的回退正确性，不支持正接受率、留出质量、KV cache 正确性或加速。v2 等价检查通过哈希把最终权重连接到直接验证过的 checkpoint，不是第二次独立真实解码实验。

**证据。** [公开聚合报告](../reports/pilot-20261008/README.md)。本地未发布：`output/train-pilot-20261008-v2/greedy-evaluation.json` 和 `greedy-equivalence.json`；prompt 与 token 轨迹不进入公开记录。

**下一步。** 保留该参考循环为正确性 oracle；增量 cache 完成后，先测拒绝裁剪、全接受 bonus、EOS 与长度边界，再与带 cache 的 target-only 比较成本。

## 2026-10-08 22:41–22:46（UTC+8）— R01：Git 与 CI 基线，已核实

**目标。** 形成可公开复现的源码/聚合证据里程碑，同时不发布生成样本、机器路径、权重、优化器状态或凭据。

**核实结果。** 首 commit 为 `410f070f914ec7a5fd564d9a863b7b765036f2ea`，时间 22:41:28，标题 `feat: establish Qwen3 DSpark training and greedy verification baseline`。本地 main 跟踪 origin/main；22:43 实时 `git ls-remote` 核实[GitHub 仓库](https://github.com/GodHu777777/qwen3-dspark)的远端 `refs/heads/main` 同 SHA，确认该基线已推送，不代表后续未提交工作已发布。

22:46 再通过 GitHub public API 核对[CPU correctness 工作流](https://github.com/GodHu777777/qwen3-dspark/actions/runs/37794557536)：run 37794557536 对应上述完整 commit，status `completed`、conclusion `success`；执行时间 22:41:41–22:42:39。该 CI 只支持 CPU correctness 范围，不提供 GPU 质量或性能证据。

`git ls-files output` 无条目；ignore 已覆盖 output/data/runs/checkpoints、权重/优化器文件及机器特定配置。`reports/pilot-20261008/` 聚合证据已跟踪。协调者报告 GitHub CLI 未登录，但 SSH 认证成功，因此通过 Git SSH 推送，无需额外用户登录；本记录者独立核实了远端 SHA，未执行认证变更、commit 或 push。

**下一步。** 日志与文档入口形成可审阅版本后，由协调者独立提交；新实验完成后追加实际 source commit 和 evidence，再核实新的远端 SHA。

## 2026-10-08 22:46 起（UTC+8）— D02/C01：在途工作，尚无新 GPU 结果

**Astra core 第一性原理审查与小集合学习诊断：进行中。** 协调者转达 agent 报告：mask、shift、Markov 已与 NeMo 对照一致，暂未找到移位或泄漏 bug。审查另外发现原 checkpoint 对 `resume.pt` 未存哈希，存在 optimizer/RNG 与权重错配的风险；已增加 SHA 与 step 校验，修复尚待形成提交和完整实验记录。

CPU 首轮 14 项通过、1 项失败，报告原因是远端缺少 NeMo `loss.py` 参考文件；补部署后重跑，最新 agent 报告 15 项全部通过，包含 direction 的 5 项 cache 测试。记录者尚未直接核对这次新日志，故不将转达状态混写为独立验证。

后续 GPU 128-step 微型学习诊断已启动，执行 session 标识 75160，运行标签 `diagnostic-overfit-20261008-128`。计划在 32/64/96 step 记录 `evaluation.jsonl` / `metrics.jsonl`，128 step 结束后保存 checkpoint；agent 报告 `run.json` 绑定全部 `.py` SHA、配置、runtime、target 与 data 指纹。**启动不等于完成，尚无新诊断结果交给记录者。** 实验计划中的固定训练 anchor 首位置 top-1 ≥90%、训练 prompt 平均连续接受 ≥3 草稿 token 是诊断目标，不能写成已达结果。若不通过，先调查对齐与优化，再扩规模。

**随后收到完成报告，证据待归档核对。** 协调者转达 core：128 step 已完成，记录耗时 26.85 秒，GPU 已释放；固定 28 个受监督 token 的 overlap 由 0.0788% 升到 82.775%，teacher top-1 从 0 升到 92.857%，label top-1 从 0 升到 96.429%。同一训练 prompt 的 32-token 真实 greedy 对照，accepted draft tokens / round 从 0 升到 0.632（12/19）；steps 0/32/64/96/128 的完整输出均与 target 一致，target 冻结，峰值已分配显存约 4.61 GB。以上是 agent 报告而非记录者已读取聚合文件；远端证据标签同上，下一轮归档后补 commit 和公开聚合链接。

**实验设计混淆与修正。** core 进一步明确：本次实际训练用 pilot temperature 0.7 的采样回答（137 token）和固定 anchors `[58, 65, 72, 79]`，并非原计划的 target-greedy 训练轨迹。采样回答与 greedy 回答只有前 7 个 token 共同前缀。真实 rollout 先由 target 发出首 token，从 anchor 59 开始，再使用后续自适应位置。此前转达成“训练/rollout anchor 不重合”过于绝对：最新报告指出只有 79 这个数值重合，但该位置的完整 prefix 已分叉。故本次没有同 anchor 且同完整 prefix 的训练实例，不能用 0.632 与原 ≥3 门槛的比较判断架构学习成功或失败，也不能单归因 exposure bias。

teacher top-1、采样轨迹 label top-1、分布 overlap、实际 rollout 接受数口径不同，不能互相替代；现有报告也未明确 92.857% 是首位置还是所有固定位置平均，不能声称计划中的首位置门槛已过。该报告支持微型固定任务可优化、真实 rollout 有非零接受，不支持留出泛化或加速。下一诊断拟用同 prompt 的 greedy 32-token 轨迹，并覆盖 anchors 59–89，使训练与评估的轨迹/anchor 对齐；这是新计划，尚非完成结果。

**Astra direction 方向审查与 cached target-only 基线：性能待结果。** 协调者转达 agent 报告：Transformers 5.17 的 `DynamicCache.crop(0)` 是 no-op，不能作为清空 cache。`cached_target.py` 对外采用绝对保留 prefix 长度，内部转换为负数移除语义。5 项 CPU cache 测试覆盖裁剪至 0 / 全部回滚、chunk logits/features 对照、EOS/长度上限及非法输入；如上一段所述，agent 报告测试通过，记录者尚未直接核对新日志。GPU 性能未测。完整前缀重算参考不能代替 cached 基线；性能主张仍需该基线和 break-even 成本核算。

**扩充数据：暂停，尚未写脚本/配置或选择新样本。** 已读准备、生成、审计脚本及固定 pilot 配置，CPU 扩充选择尚未执行。拟先准备 1024 train / 128 dev（使用现有 validation 接口）/ 128 final test，排除全部 pilot 身份。最终 test 不参与 checkpoint 或策略选择；至少 100 条独立且完成的 test 回答为目标。长度停止可能降低有效条数，因此需以实际审计为准；必要时另行记录补充样本批次。当前脚本还没有 pilot 排除和 final-test 导出/审计路径。

按 pilot 的 284.45 秒/64 prompt 做线性估算，1280 prompt 约需 95 分钟、约 468,600 生成 token；这是未验证的资源估算，长度、拒绝率、padding 与其他 GPU 工作都会改变它。未启动扩充生成。实时 SSH 观察到 AMD GPU use 3%、总已用 VRAM 约 8.96 GB；这只是当时快照，不证明 GPU 无占用。未停止无关进程，未改环境或模型。

**记录机制。** 用户要求专职实验记录；新建 subagent 因 thread limit 被拒绝，协调者复用当前记录者。协调者报告 30 分钟 heartbeat `dspark-astra` 已包含实验日志更新要求；本日志不把定时器安排当作实验结果。后续每轮记录目标、问题/假设、方法、验证、结论边界、下一步、commit 与聚合证据。没有证据的失败尝试不补写虚构详情。

**下一步。** 优先交付本版记录机制；后续收到诊断或缓存测量就追加。学习诊断支持扩充后，再实现规范化 prompt 排除、三路 split、不可变选择来源；保存精确 prompt/output IDs、源 revision、模型/配置/脚本身份与整批原子恢复；跑 CPU 选择与泄漏检查，给出生成命令，并协调 GPU 窗口。

## 2026-10-08 23:06 起（UTC+8）— D03：对齐轨迹学习诊断，已完成

**问题与修正。** 前次采样回答/稀疏 anchor 与 greedy rollout 的完整前缀覆盖不同，不能把它作为严格同任务过拟合结论。本次先生成同 prompt 的 target-greedy 32-token 轨迹，使用连续 anchors 59–89，覆盖该轨迹所有可能 rollout prefix。每步 196 个有效监督位置，训练 128 step，学习率 0.0006，draft 结构保持不变。

**验证与结果。** 已直接读取[聚合报告](../reports/diagnostic-aligned-20261008/README.md)、[summary](../reports/diagnostic-aligned-20261008/summary.json)及[执行身份](../reports/diagnostic-aligned-20261008/source-identity.json)。Step 64/96/128 的逐位置 teacher top-1 均为 100%，首位置也为 100%；每次真实 rollout 的 4 轮接受长度均为 `[7, 7, 7, 7]`，完整 32 token 与参考 target greedy 一致。最终输出组成是首个 target token 1、实际提交 draft token 28、bonus 3；最后一轮 bonus 因预算耗尽未提交。最终 overlap 86.138%，target 冻结通过，峰值已分配显存 5,179,751,936 bytes，缓存特征后的训练、评估和保存共 21.495 秒。

**结论边界。** 同任务学习机制门槛通过，证明模型可以拟合这一条训练前缀；不证明留出泛化、EOS 训练验证或提速。本次真实结束原因为 32-token 上限，没有遇到 EOS。执行后 `bench_cached_target.py` 被另一个实验修改，与记录的执行哈希不同，但本诊断不导入它；其余执行模块快照核对一致，因此 `source_snapshot_all_modules_verified=false` 如实保留。不能把快照缺口描述成所有文件逐位相同。

**版本与下一步。** 聚合证据随 `3e4514d8be504050bb2acc03dab68e9c87c0a958` 提交；23:08 左右的 live `git ls-remote` 核实远端 main 同 SHA。该 commit 是归档版本，实际执行身份仍以 source-identity 为准。原采样实验已另有[公开报告](../reports/diagnostic-20261008/README.md)，可核对前述方法混淆。下一步扩大训练样本并测真正的 held-out 质量，同时建立可比较的 cache 语义与成本基线。

## 2026-10-08 — C02：缓存 benchmark gate 失败与数值调查

**失败记录（协调者转达 direction）。** 初次正式成本测量在第 2 条 prompt 的第 15 个生成 token 发现 cached 与 full-recompute greedy argmax 不同，程序退出，没有正式成本报告。v2 保留第 1 条 prompt 的部分测量（5 trials / 40 blocks），随后释放 GPU；不能把 partial 数据说成成功性能结果。最初怀疑数值 tie，当时尚未证实。

**后续调查（agent 来源，记录者尚未读取完整聚合）。** 在相同 prefix 上，BF16 第 15-token 的 cached/prefix 预测均正确，top-2 logits 同为 19.125；全模型 FP32 的同 24 步没有 token 分歧，最大 logit 差约 1.16e-4；仅把 head 改 FP32 仍会翻序。这为数值敏感性提供证据，但不能扩大解释成所有原分歧均已定位，亦不能悄悄跳过失败 prompt。

**方法更新与当前边界。** 后续 benchmark 改为对照相同 dtype/backend/mask 且无额外 logits processors 的 HF.generate；BF16 full/cache 差异继续单列 numerical fidelity。协调者随后报告 v4 成功并释放 GPU；尚待公开成本聚合和执行身份归档，本日志不提前填写吞吐或加速倍数。成本测量也不能证明 speculative verifier 正确。core 的 cached loop 据报告已有 29 项 CPU 测试通过，下一步使用短 GPU 窗口做真实 correctness，再由扩数据使用 GPU。

## 2026-10-08 23:03–23:11（UTC+8）— P02：扩充数据 CPU 准备，已完成，GPU 待窗口

**目标与方法。** 准备 1024 train / 128 validation（dev）/ 128 final test，排除全部 pilot 身份，保持来源、精确 token 和恢复身份。重新读取 AMD 源分片，SHA-256 与 pilot 的 `06d223e…` 完全相同。按 seed 20261009 选择 1280 个规范化首轮 user prompt，排除来源包含全部 64 条 pilot 输入（包括长度拒绝），不读 source assistant。保留精确 prompt token IDs；生成时还要逐项比对这些 IDs。

**验证与结果。** 实际选择计数为 1024/128/128，pilot overlap 为 0；prompt 长度 16–1845 token，跳过 1 个长 prompt、3 个重复 prompt。prompt 文件 SHA-256 `777e883a373cfb22cd4aaf47c5ed9441e410cc4e5e29380ed60b2254942292d3`。原参数 `prepare.py --resume` 成功核对并复用同一选择；另用配置副本只追加空白换行，恢复被 identity mismatch 拒绝，原选择保持不变。新增 8 项 CPU 数据测试本地/AMD 均通过，覆盖规范化排除、三路导出、泄漏与重哈希后的语义篡改、同长度 token 替换、EOS/重复导出、恢复损坏以及小型三路 smoke 选择。旧 pilot 通过新版 audit，仍为 49 train / 7 validation / 8 rejects。公开[准备报告](../reports/data-expansion-20261009/README.md)包含 counts/长度分布/哈希、执行身份及测试日志，没有 prompt、回答或 token 数组。

**隔离与审查。** 训练路径 `generated/records.jsonl` 完全不含 test；test records/export/rejects 独立放在 `final-test/`。final-test 目录 700、文件 600、lock 声明不允许用于 checkpoint/policy 选择。core 已独立审查，未发现阻止生成的泄漏或恢复问题，并明确：这些权限是同用户工作约定，不是访问隔离，完整 batches 仍包含 test。test 只做格式/完整性检查，不参与质量或超参选择。总有效 test 回答达到 100 仍需生成后审计，128 个输入不保证 128 个完成回答。

**下一步与版本冻结。** 数据脚本/portable 配置和 CPU 报告现已 ready，待协调者提交。正式生成要求传入完整 `--source-commit` SHA，并绑定 config 字节、selection、模型、runtime、生成脚本与共享数据代码；已完成 batch 另有 checksum，部分写入或中断按整批重做。待 core 真实 GPU correctness 窗口释放，先用 `prepare_smoke.py` 机械取每个 split 前 2 条，在独立新目录运行真实生成、三路审计和重复启动恢复，不以 test 内容作选择；再使用冻结已提交源码生成全部 1280。扩充生成尚未启动；不可一边生成一边修改 pipeline。95 分钟仅是由 pilot 线性外推的资源估计。

## 2026-10-08 23:14–23:16（UTC+8）— P03：真实三路 smoke / 恢复通过，正式生成运行中

**版本与方法。** 协调者提交并推送数据 pipeline `d5e1538960afded86d487eab71908ce2c7531a7e` 后，记录者比对 5 个 scripts 的 Git blob、本地与远端 SHA-256 全部一致，按冻结源码部署。`prepare_smoke.py` 机械取每个 split 前 2 条，在全新 smoke 目录生成；不查看 test 内容、不按回答表现选样。运行 manifest 绑定该 source commit 及配置/选择/源码/模型/runtime 身份。core 短 GPU gate 结束释放后才启动；其 cached-block BF16 gate 失败属于独立正确性调查，不能当作通过，但不改变冻结 Transformers target 的数据生成路径。

**验证与结果。** 六条均 EOS 接受，2 train / 2 dev / 2 test、0 rejects，共生成 2,074 token。初次生成退出 0，audit 退出 0；同命令恢复退出 0，所有导出 SHA-256 与首次完全相同，再 audit 退出 0。[公开准备报告](../reports/data-expansion-20261009/README.md)已追加首次生成汇总、审计、恢复等价和生成执行身份。恢复会重写 summary 的本 invocation 耗时和显存，因此首次生成指标取保留的原 generation log，不将恢复进程的 0 秒写成造数据耗时。

**运行状态与下一步。** 通过全部 smoke 阶段后，bounded runner 自动启动完整 1280 输入生成；runner PID 3402974，generation PID 3405911，运行标签 `expand-20261009`。各阶段留 PID/log/exit-code；任一阶段错误就停止后续，不修改运行身份。当前是“运行中”，不是已完成数据；完整结束后还要自动 audit。持续核对完成 batch、实际接受/拒绝数量和 final-test ≥100 完成回答目标，仅按聚合数据报告。正式运行不混入 smoke completion，不改 pipeline 源码，不停止无关进程。

## 2026-10-08 23:30（UTC+8）— C03：缓存原型的逻辑证据与 BF16 真实门槛失败

**目标。** 分开核对 cache bookkeeping、真实逐 token 语义和性能，避免 CPU 通过或 target-only 成本被误读为 speculative BF16 无损。

**CPU/独立审计。** 已读取[独立 cache-content 审计日志](../reports/cached-target-20261008/cpu-cache-content-audit.log)与[报告](../reports/cached-target-20261008/README.md)：42 cases / 429 proposal rounds，在 block size 1/3/7、prompt length 1/4/9、每个拒绝位置/全接受情况下，输出 token 和 cache length 全部一致。逐层 projected KV 与 fresh committed-prefix 计算相比，最大绝对差 1.9446e-6，draft backbone 最大差 7.1526e-7。最初 tensor closeness 的 1e-6 断言遇到一处 1.1325e-6 误差；修正为直接报告数值误差，仍保留精确 token equality 门槛，没有用 epsilon 改 token 选择。

当前 README.ai.md 与协调者报告 CPU 共 30 项通过。记录者直接核对的本地 `cached-decode-cpu-tests.log` 是较早的 29 项通过日志，公开[BF16 gate 报告](../reports/cached-decode-gate-20261008/README.md)也保留当次 29 的口径；不把它改写成同一次执行 30 项。新增用例后的具体日志仍待归档。

**真实 BF16 gate，失败。** 已直接读取[summary](../reports/cached-decode-gate-20261008/summary.json)和[执行身份](../reports/cached-decode-gate-20261008/source-identity.json)：使用单样本 aligned128 checkpoint，3 个 prompt 中 2 个 cached speculative 与 cached sequential target 不一致，按门槛 exit 1，完整迹被保留。

| 输入范围 | Speculative = cached sequential | Cached sequential = full recompute | 首分歧（零基输出索引 / prefix 长度） |
| --- | --- | --- | --- |
| 单条训练 prompt | 否 | 是 | 17 / 76 |
| Pilot validation 1 | 是 | 是 | 无；真实 EOS 终止 |
| Pilot validation 2 | 否 | 否 | 23 / 172 |

训练 prompt 前两轮各接受 7 token，随后同完整 prefix 的 sequential top-2 为 23.5/23.375，block verification 对调候选排名，数值仍为 23.5/23.375。Validation 2 sequential top-2 量化为 18.75/18.75，block 为 18.75/18.625，排名变化。公开记录保留位置和 logits 数值，不复制 token ID 或生成内容。

**结论边界与逐 shape 数值疑问。** 独立 CPU 审计支持 rollback/cache 内容管理，没有推翻真实 BF16 失败；目前不能声称 cached speculative 逐 token 无损或提速。已观察到相同 prefix 下 one-token、multi-token block、full recompute 的接近候选排名不同，但不能因此排除所有实现问题。下一步需要固定同 prefix、同精度/后端/mask，分离 query length、block shape、prefix length 和完整 cache state 的影响；全 FP32 target+draft 是独立精度实验，不能与 BF16 baseline 混比。计划中的稳定 BF16 路径也不能靠跳过 prompt、放宽文本相似度或修改 argmax epsilon 达成。

**Target-only 成本证据范围。** [已归档 baseline](../reports/cached-target-20261008/README.md)使用 7 条 pilot validation、每条 2 warmup / 5 repeats、batch 1、BF16/SDPA、最多 64 token。同 cached pure-greedy HF policy 在 117 个受检查 token 上一致；聚合 end-to-end 29.9215 tokens/s、remaining-token decode 30.0387 tokens/s，prefill+first logit 中位 36.29 ms，1 个新输入的 verification 中位 32.90 ms，2–8 个约 36.54–38.01 ms。这些 eager Transformers target-only 数据没有 draft、特征捕获或 controller 成本，不能推导 speculative 提速。完整前缀 BF16 差异仍单列 numerical fidelity，没有删失败 prompt。

**版本。** `4b3c80c6f4598f2525c2501bfb9ad57a3a419ae3`（23:24:51）归档缓存原型与明确失败门槛；`f8a35f3fdeca5683e03f92fa2555ed2f2f997811`（23:25:30）归档三路 smoke/生成启动日志。23:30 live `git ls-remote` 核实远端 main 为 f8a35f3，且本地 ancestry 核实其中包含 4b3c80c；因此两者已推送。归档 commit 与实际执行源码哈希继续分开。

## 2026-10-08 23:30（UTC+8）— P04：生成进度与短诊断暂停计划

**一次实际核对。** 已读取 stages 与 generation log：四个 smoke 阶段均 exit 0，正式生成尚无完成/退出记录。最近完成 232/1280 输入；PID 3405911 为 `Rl`、已运行 877 秒，仍绑定 source commit d5e1538。未重启、未修改脚本，最终 test 保持锁定。该快照不表示完整数据已经完成。

**尚未执行的计划。** 协调者提出可能给予 Astra 一个 ≤5 分钟关键 BF16 诊断窗口：只对本次 generator SIGSTOP，runner 保持等待；确认已有 GPU 工作排空后诊断，使用 timeout/finally 必须 SIGCONT 原 PID。此时仍待协调者授权，记录者未发送任何信号。无关 ASR 等进程不停止。若执行，T 状态应记录为预期暂停，不判作失败，不重启或另起生成进程；之后补暂停开始/恢复时间、原 PID 是否恢复和 batch/hash 完整性。暂停会影响 wall time，需要与实际 generation batch time 区分。诊断计划本身不算新数值证据。

## 2026-10-08 23:44（UTC+8）— C04：短暂停后 canonical BF16 探针，已报告完成

**执行与恢复（direction 报告，由协调者转达）。** 先前待授权的短诊断窗口已执行并结束：通过 pidfd 只暂停本次 generator，暂停 28.893 秒；GPU probe 19.05 秒、峰值已分配显存约 1.404 GB。诊断结束 SIGCONT 原进程，PID 3405911 回到 `Rl` 且 generation log 继续增长，未重启生成、未改生成源码，未停止无关 ASR。这次实际暂停应计入 wall time；不能把暂停记录成失败或把等待时间混称模型计算成本。

**结果口径与边界（公开报告待归档）。** 三个 case 的 canonical sequential 与 oracle blocks 输出一致；两个原失败 prefix 的各条受检查路径 logits 差为 0。但 canonical 与 stock target 只有 2/3 一致，说明更换执行约定后的一致性不能被说成复现原 stock 语义。探针未包含 draft，也未测速度，不是 cached speculative 全流程正确性或性能通过。逐 shape 排名差异仍需以冻结的执行约定和 same-prefix 数据解释；不能用这次 canonical 检查覆盖原 BF16 gate 的失败。完整公开 probe 聚合/源码身份待 direction 归档，当前数值明确为转达报告。

<a id="memory-gate"></a>

## 2026-10-08 23:44（UTC+8）— M01：训练资源门槛设计复盘，CPU 通过，真实 GPU 未执行

**问题与 root review 修正。** 原“最长序列 + 一次 optimizer update”资源检查有两个陷阱：首次 AdamW 更新才分配 moments，下一轮完整 gradient accumulation 在 moments 常驻时可能出现更高峰值；最长序列如果 completion 短、实际可选 anchors 少，未必比稍短但 anchors 更多的序列更耗内存。因此不能只测一个最大长度记录或只看到首个 Adam 分配成功就启动长训。

**已实现方法。** 记录者读取 `memory_gate.py` 与 `experiment_inputs.py`：从真实、已审计且拒绝 final-test 的 train records 中，以 `(sequence_tokens, actual_anchors)` 选不被两维同时支配的 Pareto 形状；actual anchors 为 requested anchors 与 completion token 数的最小值。每个形状建立新的 draft/AdamW，执行两个完整 accumulation/update 周期：第一轮核首分配，第二轮核 optimizer states 常驻下的完整训练。报告各轮 state bytes、allocated/reserved/peak、梯度通路、投影确实更新、冻结 target；不保存可继续训练的 checkpoint、不作为后续 warm start。

**验证。** 已直接读取本地未发布 `output/cached-decode-gate-20261008/cpu-training-tools-tests.log`：**34 项测试全部通过，2.954 秒**，包含四项新 training-tools 测试（固定 dev panel 的 EOS 尾部/位置/重复性、Adam states 与两轮训练、较短序列但更多 anchors 的 frontier、拒绝 test/split 身份泄漏）。这是新日志的实际计数，不从旧 29 项日志推断。日志保留 `(null)` 以及 HIP/CUDA visibility `-1` 的启动提示；全部用例随后通过，不虚构提示原因。当前尚未执行真实 GPU resource gate，也没有 expanded-data 内存数字。

**结论边界与下一步。** 此工具只是选定数据/配置/硬件上的经验资源门槛，不是形式化最坏内存保证；kernel workspace 与 allocator 行为未必随两维单调。扩数据完成并冻结训练配置后，应对真实 Pareto 形状运行 GPU 两周期测量，保留失败/OOM 原始报告与运行身份，再判断可用余量；短序列诊断显存不能替代它。新工具与日志将在本轮归档提交，当前没有臆造新 commit 或推送结果。

## 2026-10-08 23:56 起（UTC+8）— C05：动态 BF16 主线与 canonical 数值 control 分开

**问题。** C03 的真实 BF16 cached-block 失败与 C04 的固定形状 target-only 探针，验证的是不同执行契约。若仅为得到 token equality 把主线改成恒定最大 padding，就可能消除原本要研究的验证容量机会成本；数值对照通过不能代替动态 DSpark 系统验证。本条只补执行契约与 CPU adapter 的新增证据，不重复旧探针结果。

**假设。** 共享 prefill、固定 query/key 物理形状及 LM-head 投影形状，可能减小不同 chunk/history 路径的数值变异；这是可测的 control 假设，不是 stock BF16 逐 token 等价或动态执行无损的保证。论文数学 lossless 条件与不同有限精度 shape 的输出数值契约应分别检验；保留旧失败，不把近 tie 自动解释为无害，也不把它直接解释为论文算法错误。

**方法。** 已读 [reproduction scope](dspark-reproduction-scope.md) 的数值/性能分层以及 `canonical_target.py`、`eval_canonical_decode.py`、`test_canonical_target.py`。adapter 共享完整 prompt prefill 和全行 LM-head；后续 append 固定 query width 8、StaticCache 固定 key capacity，使用显式 causal/valid mask，并先投影全部 padded hidden rows，再选择真实位置。只有真实输入行进入逻辑 cache/返回 draft context。`predict(features)` 让普通 dynamic target 继续按原请求行数投影，与 optional canonical 分支分开。HF 5.17 StaticLayer 没有 crop；实验 adapter 原地调整 cumulative length 并屏蔽 stale backing slots，不能推广成通用 cache API。

**观察与证据。** 已直接读取[canonical CPU 日志](../reports/static-shape-candidate-20261008/canonical-cpu-tests.log)：39 项全过，4.688 秒，其中 5 项新 canonical tests 覆盖 padded hidden/head shape、真实 context 行数、dummy/stale suffix 非干扰、指针不变的 rollback、全部 crop 边界/容量 headroom、tiny Qwen 实际 draft 与每个拒绝位置、EOS/输出预算/reset。四个 adapter 源码及该日志的 SHA 均与[执行身份](../reports/static-shape-candidate-20261008/source-identity.json)匹配。C04 的 target-only 探针现有[公开报告](../reports/static-shape-candidate-20261008/README.md)可查，但它使用 oracle proposals，仍不等于真实 trained drafter GPU gate。scope scheduler 加入后的本地未发布 `output/scope-research/cpu-scheduler-suite.log`另记录 44 项全过、4.157 秒；39 与 44 是不同测试轮次，不合并成同一执行计数。

**局限。** canonical 当前是 CPU-ready 的 optional 数值 control。实际 trained drafter BF16 GPU 尚无完成结果，未得到速度或 stock 语义通过结论。恒定 key/query 容量有实际成本；逻辑 `ell`/B 变小可能并未减少 physical work，也会改变 SPS(B) 台阶。必须分别记录 logical B、physical B/graph bucket 及真实 padding，不能只与被同样最大 padding 拖慢的 target-only 比较来制造加速。动态 BF16 主线与数值未决状态继续保留；任何预注册容差新协议均需独立命名，不能回改旧 token-equality 失败或称为 lossless。

**下一决策。** 协调者已授权 direction 第二次有界 GPU 窗口执行真实 drafter canonical gate，**目前只记录授权/计划，结果未到，不能写通过**。应复用原 gate 的全部例子和失败位置，分别报告 canonical sequential/speculative、stock target、同 prefix 的 shape/history 分解，后续再测长输入、容量边界和同工作量成本。记录者不启动 GPU、不暂停任何进程。

## 2026-10-08 23:56 起（UTC+8）— S01：Algorithm 1 与生产 stale capacity 的范围复盘

**问题。** 单请求固定 k、confidence 阈值或 target block 耗时表，不足以复现论文的全局资源分配。同步 Algorithm 1 和生产的两步历史容量策略也不是同一算法，不能用一个全局回溯搜索替代并说成更忠实。

**假设。** 按校准的条件接受概率形成 prefix survival，用硬件容量曲线分配 R 个活跃请求共享验证预算，才可能体现某请求低质量 suffix 挤占另一请求的机会成本。其数值收益依赖真实 SPS 台阶、prefix 因果性与实际执行工作量；CPU 数学最优分配并不自动说明 serving 获益。

**方法与来源。** 本轮读取[忠实复现范围](dspark-reproduction-scope.md)，该文区分论文 v1 §3.2.2/Algorithm 1、Appendix A、§5.2 生产描述，固定 DeepSpec 公开入口与 pinned NeMo 训练来源。同步 Algorithm 1 从所有 `ell=0`、B=R 开始，按累乘得到的 prefix survival 全局排序，以 `tau*SPS(B)` 判断 admission，首次不改善就 break；SPS 单位是 steps/s，B 是跨请求 target 验证 token 数，tau 含每请求一个 baseline 产出项。相等分数仍需 prefix 顺序，ell=0 必须合法。其全局最优主张只适用于目标沿 admission 路径单峰的条件，不能扩展到任意 jagged SPS。

生产 §5.2 采用 **two steps prior** 的 confidence 估计本轮跨请求容量 K，再用本轮最新累积 confidence 在 K 内做动态 top-K；历史信号决定容量、当前信号决定顺序，不是用旧分数排序当前候选。历史容量搜索不 early break，以跨越离散 SPS cliffs；容量对当前采样值的时间隔离也是因果屏障。还需要把 scheduling 与 GPU 执行重叠以满足 ZOS/graph 的提前 shape 要求，并高效执行不同请求的可变 query 长度。同步 Python 算完后才启动 kernel，或每请求串行 oracle，都不能被写成该生产路径已实现。

**观察。** 已读 `scheduler.py` 和五项 `test_scheduler.py`：当前只是同步 Algorithm 1 的 CPU pure planner，保存首次不改善 early-stop；独立枚举小 R/gamma 的分配，在固定预算/平滑单峰曲线上核最优性，并保留 jagged curve 反例，证明 hindsight 全局搜索会与 early-stop 不同。输入 SPS 是合成 fixture，不是硬件测量。上述 44 项 suite 日志通过包含这些 tests；planner 明确把 pre-token、non-anticipating confidence 的来源责任留给 caller，不能自行验证 scores 的因果性。没有实现 STS、随机接受/残差 correction sampler、两步异步状态、真实可变长度 batch engine 或 serving 集成。

**局限。** pinned NeMo 支持训练结构/loss/mask，并不等于完整 scheduler；本轮实际核对的 DeepSpec 公开单序列 threshold eval 也不提供已复刻生产实现的证据。当前 greedy overlap/MAE 不能替代论文随机接受事件的条件概率校准。论文中的资源是全局 target verification token/batch 容量，不能无依据改写成 SM 分区或每请求 GPU 核配额。AMD dense-Qwen 的 context/load 成本模型要实际核对，不能直接继承 V4 一维 B 假设或生产吞吐数字。

**下一决策。** 继续分层：小词表随机 sampler/因果性与小 R allocation oracle；dev rollout STS 拟合后冻结；真实 engine profile SPS(B, context/load) 并保留台阶；多请求 KV/预算/可变长度隔离；两步历史 K 和当前 top-K 的干预与异步测试；最终按同模型、dtype、物理工作、到达负载及 SLA 比较 target-only/固定 k/静态阈值/同步/异步系统。canonical 仅作为并列数值 control，不替代这条动态主线，也不把尚未实现项写成完成。

## 2026-10-09 00:00（UTC+8）— S02：exact-B oracle 与正 survival 候选范围补证

**问题。** “容量不超过 B”的穷举最优，不能直接代替“恰好 B”的边际 admission 核对；过滤零 survival 也不能只凭工程直觉推定忠实。

**假设与方法。** core 报告逐行核对论文 Algorithm 1 第 4 行，候选集合明确为 `E={a>0}`。记录者已读取当前 planner 的对应过滤和新增两项 test：逐 admission 使用真正 exact-B 独立穷举比较 expected progress；另构造 zero-survival 且 SPS 上升的合成反例，保留原文正候选行为，即使无约束全局搜索可因增加无收益 token 而提升目标值。

**观察。** 直接读取新版本地未发布 `output/scope-research/cpu-scheduler-suite.log`：46 项全部通过，5.065 秒；这是新增测试后的轮次，不覆盖前述 44 项历史。合成反例中 literal planner 保留 ell=0/目标值 1，而 unrestricted oracle 可得 100，明确没有宣称任意容量曲线上的全局最优。当前 planner/scope/test 归档 commit 为 `69e86be5e8f60cfedf7fd208c78aac495760fc27`（本地 commit 时间 2026-10-08 23:59:43）；协调者确认 Git push exit 0、origin/main 从 4f2bd24 更新至 69e86be。最新 CI 尚未核实，不写 CI 通过。

**局限与下一决策。** exact-B 与正 survival tests 支持 CPU 论文 planner 的范围，不证明 scores 的采样因果性、异步生产容量搜索或真实硬件收益；SPS fixture 仍是假设曲线。保留反例及早停/正候选限制，后续使用真实离散容量曲线和非预知干预测试，不把忠实性修正为任意全局搜索。

## 2026-10-09 00:01（UTC+8）— C06：真实 drafter canonical gate 完成，独立执行契约通过

**问题。** C04 的 oracle target-only 探针不包含真实 drafter，C05 当时只到 CPU-ready。此次需要把实际 trained draft proposal、target 中间特征和 rollback 接入 canonical 执行契约，并保留 stock 对照与原 dynamic BF16 失败。

**假设与方法。** 使用 aligned step-128 checkpoint，复用原 gate 的全部 3 个例子，BF16 target、FP32 draft 参数/BF16 AMP、SDPA、query width 8、key capacity 256；canonical sequential/speculative 共享 full-prompt prefill、全部 padded 行的 LM-head 投影，再选 valid rows。与 stock dynamic cached greedy 同时比较，不删除原失败例子、不改 argmax epsilon。开始前 39 项 CPU suite 已通过；这次不是后来包含 scheduler 的 46 项轮次。

**观察与直接证据。** 已读取[真实 drafter 报告](../reports/canonical-real-draft-gate-20261008/README.md)、[summary](../reports/canonical-real-draft-gate-20261008/summary.json)、[pause evidence](../reports/canonical-real-draft-gate-20261008/pause-evidence.json)及[source identity](../reports/canonical-real-draft-gate-20261008/source-identity.json)。GPU gate exit 0：3/3 canonical sequential 与实际 cached drafter 输出相等，共 70 token；stock 对照仍只 2/3 相等。训练轨迹 32 token/4 rounds，每轮接受 7、实际提交 draft 28；两条 dev 分别 6 token/5 rounds、32 token/31 rounds，**草稿接受均为 0**。最后一条 canonical/stock 首分歧为零基输出索引 10。四个 API 的当前源码 SHA 与执行身份一致，原始 token 迹仍未发布。

授权暂停仅作用于身份核对后的原 generator：23:57:32.621845–23:58:08.277109（UTC+8），35.655255 秒；报告记录暂停状态 T、恢复状态 R，日志大小 6670→6748 bytes，23:58:38 确认继续增长。pidfd/timeout/finally 控制器没有重启 generator 或停止 ASR。GPU probe 用时 22.0796 秒、峰值已分配显存 2,221,295,104 bytes；这些是诊断资源数据，不是速度比较或总 VRAM。

**局限。** 此次通过的是 optional canonical 契约中这三个例子的真实 draft 集成，不是 stock BF16 无损、动态预算系统通过或普遍数值等价。原 2/3 dynamic gate 失败继续成立。两条 dev 接受 0，不能声称 held-out draft 有收益；固定 padding 改变物理工作，且未比较 speculative/canonical/stock 性能，不能从 gate elapsed time 计算加速。

**下一决策。** 保留 canonical 对照与动态 BF16 数值未决主线并列；后续扩大独立 dev 前缀、做同 prefix 的 history/chunk/batch 分解和真实物理工作成本，再按 S01/S02 的分布恢复、全局预算与多请求执行路线推进。报告代码/聚合将由协调者归档提交；本条不臆造新的 commit/push 或 CI 结果。记录者本轮只更新日志，未启动 GPU、发送暂停信号或改生成脚本。

**补充口径修正与版本。** 最新源码/报告归档的本地 commit 为 `796fecc0d9142c1a733ad6df47b105fab4ee9f05`（2026-10-09 00:03:13）；此时未 push，不写已发布。已读最新 README.ai.md 与早期 static probe 的执行源码/summary：该 target-only probe 内还调用了 same-prefix dynamic 四路 helper，但这不是原 native drafter evaluator 的重跑，也不是 native gate 通过。C04 中“各路径 logits 差为 0”应限定于 canonical 路径，不能扩展到该动态 helper；公开 summary 仍保留动态 current-chunk/history/full 路径的非零差异。新的 native 四路字段重跑和全 FP32 cached-block control 仍 pending。索引、数值契约和真实 gate 的归档到此交接；其他工作按下一轮新证据追加。
