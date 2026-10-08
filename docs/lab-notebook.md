# 实验日志

最近记录核对：**2026-10-09 01:19（UTC+8）**。专职 experiment_journal 已直接读取本地 CPU preflight、显存 gate、step32 报告与冻结训练源码，形成 T04/T05；因当前 agent 唤醒受线程额度限制，本轮由 sol_data 临时代为续记 stochastic gate 的 CPU/dry-run、expanded step128/512及真实GPU gate聚合，保留专职记录角色。远端 checkpoint 哈希核对结果引用已有留证报告。此前 STS、随机缓存、packed target 和正式数据审计的检查来源保留在各自条目中。本文持续追加；旧结论若被修正，保留原结论并说明修正依据。历史实验与实时进程状态分开记录。

早期研究问题：冻结 Qwen3-0.6B target 后，并行 DSpark 草稿能否比带 KV cache 的 target-only greedy 更快地产出完全相同的 token？训练可运行、loss 下降、回退输出一致，各自只回答这个问题的一部分。早期阶段门槛见[实验计划](experiment-plan.md)，下面历史实验的协议与失败口径不回改。

当前完整目标见[忠实复现范围](dspark-reproduction-scope.md)：还需随机分布恢复与非预知 admission、confidence 校准、跨活跃请求的全局验证 budget、可变长度多请求执行及两步历史异步容量机制，并在真实负载/相同物理工作量下评估系统收益。Greedy correctness 与 canonical 数值 control 是分层检查，不能缩小或替代这个完整目标。

公开链接只指向聚合报告。生成的 prompt、回答、精确 token 轨迹、机器路径、权重、优化器状态及凭据不进入日志。`output/` 是 **未发布的本地证据**。已有实验早于首个项目提交，实际身份由当时 manifest 中的源码、配置、数据、模型哈希绑定；后来的 Git 基线不被描述成启动这些历史实验的 commit。

复盘索引：

- 数据质量/长度：[P06 正式审计与拒绝](#data-final-audit)、[T04 模板与实际输入长度](#expanded-resource-gate)。
- Hidden-state/训练对齐：[D03 同轨迹学习诊断](#aligned-learning)、[T05 首段训练与 TF 边界](#expanded-step32)。
- 数值差异：[C02 dynamic BF16 调查](#dynamic-numerics)、[C06 canonical 真实 drafter control](#canonical-drafter-gate)。
- KV 正确性：[C03 缓存内容/回退](#kv-correctness)、[R04 随机路径提交](#stochastic-cache)、[S03 多请求隔离](#packed-isolation)。
- 资源与调度：[M01 两周期/Pareto 设计](#memory-gate)、[T04 实测显存](#expanded-resource-gate)、[S01 异步机制范围](#scheduler-scope)。

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

<a id="aligned-learning"></a>
## 2026-10-08 23:06 起（UTC+8）— D03：对齐轨迹学习诊断，已完成

**问题与修正。** 前次采样回答/稀疏 anchor 与 greedy rollout 的完整前缀覆盖不同，不能把它作为严格同任务过拟合结论。本次先生成同 prompt 的 target-greedy 32-token 轨迹，使用连续 anchors 59–89，覆盖该轨迹所有可能 rollout prefix。每步 196 个有效监督位置，训练 128 step，学习率 0.0006，draft 结构保持不变。

**验证与结果。** 已直接读取[聚合报告](../reports/diagnostic-aligned-20261008/README.md)、[summary](../reports/diagnostic-aligned-20261008/summary.json)及[执行身份](../reports/diagnostic-aligned-20261008/source-identity.json)。Step 64/96/128 的逐位置 teacher top-1 均为 100%，首位置也为 100%；每次真实 rollout 的 4 轮接受长度均为 `[7, 7, 7, 7]`，完整 32 token 与参考 target greedy 一致。最终输出组成是首个 target token 1、实际提交 draft token 28、bonus 3；最后一轮 bonus 因预算耗尽未提交。最终 overlap 86.138%，target 冻结通过，峰值已分配显存 5,179,751,936 bytes，缓存特征后的训练、评估和保存共 21.495 秒。

**结论边界。** 同任务学习机制门槛通过，证明模型可以拟合这一条训练前缀；不证明留出泛化、EOS 训练验证或提速。本次真实结束原因为 32-token 上限，没有遇到 EOS。执行后 `bench_cached_target.py` 被另一个实验修改，与记录的执行哈希不同，但本诊断不导入它；其余执行模块快照核对一致，因此 `source_snapshot_all_modules_verified=false` 如实保留。不能把快照缺口描述成所有文件逐位相同。

**版本与下一步。** 聚合证据随 `3e4514d8be504050bb2acc03dab68e9c87c0a958` 提交；23:08 左右的 live `git ls-remote` 核实远端 main 同 SHA。该 commit 是归档版本，实际执行身份仍以 source-identity 为准。原采样实验已另有[公开报告](../reports/diagnostic-20261008/README.md)，可核对前述方法混淆。下一步扩大训练样本并测真正的 held-out 质量，同时建立可比较的 cache 语义与成本基线。

<a id="dynamic-numerics"></a>
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

<a id="kv-correctness"></a>
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

<a id="scheduler-scope"></a>
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

<a id="canonical-drafter-gate"></a>
## 2026-10-09 00:01（UTC+8）— C06：真实 drafter canonical gate 完成，独立执行契约通过

**问题。** C04 的 oracle target-only 探针不包含真实 drafter，C05 当时只到 CPU-ready。此次需要把实际 trained draft proposal、target 中间特征和 rollback 接入 canonical 执行契约，并保留 stock 对照与原 dynamic BF16 失败。

**假设与方法。** 使用 aligned step-128 checkpoint，复用原 gate 的全部 3 个例子，BF16 target、FP32 draft 参数/BF16 AMP、SDPA、query width 8、key capacity 256；canonical sequential/speculative 共享 full-prompt prefill、全部 padded 行的 LM-head 投影，再选 valid rows。与 stock dynamic cached greedy 同时比较，不删除原失败例子、不改 argmax epsilon。开始前 39 项 CPU suite 已通过；这次不是后来包含 scheduler 的 46 项轮次。

**观察与直接证据。** 已读取[真实 drafter 报告](../reports/canonical-real-draft-gate-20261008/README.md)、[summary](../reports/canonical-real-draft-gate-20261008/summary.json)、[pause evidence](../reports/canonical-real-draft-gate-20261008/pause-evidence.json)及[source identity](../reports/canonical-real-draft-gate-20261008/source-identity.json)。GPU gate exit 0：3/3 canonical sequential 与实际 cached drafter 输出相等，共 70 token；stock 对照仍只 2/3 相等。训练轨迹 32 token/4 rounds，每轮接受 7、实际提交 draft 28；两条 dev 分别 6 token/5 rounds、32 token/31 rounds，**草稿接受均为 0**。最后一条 canonical/stock 首分歧为零基输出索引 10。四个 API 的当前源码 SHA 与执行身份一致，原始 token 迹仍未发布。

授权暂停仅作用于身份核对后的原 generator：23:57:32.621845–23:58:08.277109（UTC+8），35.655255 秒；报告记录暂停状态 T、恢复状态 R，日志大小 6670→6748 bytes，23:58:38 确认继续增长。pidfd/timeout/finally 控制器没有重启 generator 或停止 ASR。GPU probe 用时 22.0796 秒、峰值已分配显存 2,221,295,104 bytes；这些是诊断资源数据，不是速度比较或总 VRAM。

**局限。** 此次通过的是 optional canonical 契约中这三个例子的真实 draft 集成，不是 stock BF16 无损、动态预算系统通过或普遍数值等价。原 2/3 dynamic gate 失败继续成立。两条 dev 接受 0，不能声称 held-out draft 有收益；固定 padding 改变物理工作，且未比较 speculative/canonical/stock 性能，不能从 gate elapsed time 计算加速。

**下一决策。** 保留 canonical 对照与动态 BF16 数值未决主线并列；后续扩大独立 dev 前缀、做同 prefix 的 history/chunk/batch 分解和真实物理工作成本，再按 S01/S02 的分布恢复、全局预算与多请求执行路线推进。报告代码/聚合将由协调者归档提交；本条不臆造新的 commit/push 或 CI 结果。记录者本轮只更新日志，未启动 GPU、发送暂停信号或改生成脚本。

**补充口径修正与版本。** 最新源码/报告归档的本地 commit 为 `796fecc0d9142c1a733ad6df47b105fab4ee9f05`（2026-10-09 00:03:13）；此时未 push，不写已发布。已读最新 README.ai.md 与早期 static probe 的执行源码/summary：该 target-only probe 内还调用了 same-prefix dynamic 四路 helper，但这不是原 native drafter evaluator 的重跑，也不是 native gate 通过。C04 中“各路径 logits 差为 0”应限定于 canonical 路径，不能扩展到该动态 helper；公开 summary 仍保留动态 current-chunk/history/full 路径的非零差异。新的 native 四路字段重跑和全 FP32 cached-block control 仍 pending。索引、数值契约和真实 gate 的归档到此交接；其他工作按下一轮新证据追加。

## 2026-10-09 00:07（UTC+8）— P05：正式生成一次聚合检查，仍在运行

**问题与方法。** 在不干预 generator、不读 test prompt/回答的前提下，核对是否可进入完整审计。记录者只读 generation 的 batch 进度 JSON、batch/checksum 文件名、runner stages/退出标识和进程状态；按冻结的 8 条 batch 与先 128 dev、再 128 test、后 train 顺序累计 stdout 接受计数。

**直接观察。** 最近完成 816/1280 输入，102 个 batch payload 与 102 个 checksum；stdout 聚合累计接受 745、拒绝 71、生成 token 301,027。按 batch 顺序得到已完成部分 train 507 / dev 119 / test 119。这些是运行日志的聚合，不是完整 exports 的审计结论，未读取最终 test 内容。generator PID 3405911 为 `Rl`，runner PID 3402974 为 `S`，分别 elapsed 3122/3183 秒；仍是同进程，未重启。四个 smoke 阶段 exit 0；正式 full-generation/full-audit 尚无结束记录，runner exit code 未出现，summary 与三路完整 exports 均未出现。

**版本核实。** 本轮 live `git ls-remote` 核实 origin/main 为 `60d33c3ba1b5aca32e4dc5c9349daa272dd5198a`；本地相邻历史包含 `796fecc0d9142c1a733ad6df47b105fab4ee9f05`，后续文档/数值 control 版本已推送。数据 generation 仍绑定冻结的 d5e1538，不因后续文档/核心提交更换身份。本轮没有新的性能结论。

**局限与下一决策。** 仅确认目前正常推进，未宣布生成完成、test 最终合格或完整审计通过，不高频轮询、不操作 GPU/信号。已有 runner 会在完整生成成功后自动 audit。后续只有出现终态与完整 exports 才检查 full-generation/full-audit/runner exit code、重新核 source/config/selection/output 哈希、读取 summary/audit 的最终 accepted split/reject 数及 EOS/预算分布，归档仅含 aggregate 的报告；保持 test 不参与选参。core 的 sampler CPU 工具与 direction 的 immutable 训练 snapshot 当时仍在准备，未在本条当作已完成实验。

## 2026-10-09 00:11（UTC+8）— R02：随机 sampler 的 CPU 输出-law 参考

**问题与假设。** Greedy cache 相等检查不验证随机 speculative 的 target 分布恢复；必须用真正采出 proposal 的条件 q、同输出 prefix 的 target p 和非预知 admission。对于给定 p/q，接受质量 `min(p,q)` 加残差质量 `(p-q)+` 应恢复 p，再按实际输出前缀逐位置归纳。

**方法。** 已读[随机采样协议](stochastic-sampling.md)、独立 `sampling.py` 及 Fraction 测试。Proposal 保存每位置实际归一化 q；q 应包括 Markov 条件修正、实际温度/过滤和采样计算 dtype，不能用 backbone base logits 或 greedy token 配原 softmax 冒充。n 个 proposal 对齐 n+1 个 target 行，首拒绝采规范化 `(p-q)+`，全接受采最后行 bonus；EOS/输出预算决定实际提交与随机数消费。Admission 在当前候选采样前调用，但闭包可读什么仍是调用方因果性责任。

**直接观察。** 本地未发布 `output/scope-research/sampling-unit-tests.log` 是 **11 项全过、0.025 秒**；协调者另一次审查重跑报告 11 项全过、0.022 秒，两个计时分开。Fraction oracle 精确枚举全部候选、接受/拒绝、residual/bonus 分支，组合为完整多轮输出 law，并用区间内部点驱动实际实现核 token、接受数、拒绝位置、终止原因和随机数消费。覆盖 225 对三词表有理分布（含零支持、p=q、互不相交支持）、prefix 相关 p/q、adaptive admission、EOS/预算和 2^-40 residual；概率-law 断言不是 Monte Carlo 近似。Hindsight 反例在 p=q=(1/2,1/2) 下先看 X 再只允许 X=0，最终 P(0)=3/4，说明“之后仍 target verify”不能修复选择偏差。

**上游差异与局限。** 固定 DeepSpec evaluator 对选中 q 使用 clamp_min(1e-8)，读取的 helper 对 residual 总质量 ≤1e-8 fallback 到 target；本 CPU reference 保留非零极小 residual，q=0 的已采候选报错，不引入接受 epsilon。upstream 实际 sample_tokens 未显式转 FP32，而 logits_to_probs 先 float；不能沿另一 dtype 路径重建验证 q。当前 input sum 的 1e-12 契约是小词表 Python-float 参考，不是可直接接 GPU BF16 tensor 的接口；Fraction 区间测试也不是所有浮点 CDF 边界的形式化证明。没有 GPU model/cache/scheduler 集成、随机解码结果或速度结论，动态 BF16 失败没有被解除。

**下一决策与版本。** 后续先冻结 logits/temperature/filter/dtype/RNG 契约，保存真实每位置 q 和 pre-token confidence，对 target 行/cache rollback 做独立集成验证，再处理 STS/非预知 scheduler。三个文件归档在 `943f435593ed483bdd2372aa8f855835a991cde1`（00:09:51）；本轮协调者报告该提交待 push，未写已发布或 CI 通过。

**随后推送核实（协调者）。** Git push 已 exit 0，origin/main 从 `60d33c3` 更新至 `6ea2739`，包含上述 `943f435` 及区分 CPU 概率参考和真实模型集成的文档修正。最新 CI 尚未核实。

## 2026-10-09 00:11（UTC+8）— T02：immutable 训练源码部署与未完成数据 guard

**问题与方法。** 长训练 strict resume 哈希包含全部 package 模块，开发中的 sampler/数值工具不能混入既有训练身份。direction 为训练导出独立的已提交 `796fecc0d9142c1a733ad6df47b105fab4ee9f05` 快照；记录者直接核本地 ignored `output/expanded-training-preflight-20261009/` 的 deployment、archive、COMMANDS 和 preflight 代码，**112 个文件全部与 deployment SHA 和该 commit blobs 一致**，archive SHA 为 `52c14bc7d698f53aff46b794d15c3181d14f7ea388651b0bc4dd0351aac32fb0`。只读文件模式是工作约定，不是安全边界；运行时/模型/生成数据/config/GPU co-tenants 在 source snapshot 外，仍需核对。

**观察与证据等级。** direction 报告远端已部署只读快照，实际 CPU preflight 在 `full-generation` 尚未 complete/successful 处按预期拒绝，没有读取 records、import runtime 或构造模型/GPU。记录者已从代码独立确认：snapshot 完整性及 package membership 后，先检查 full-generation/full-audit/runner exit 0 和先后阶段，再读 audit/config、import runtime 与 development records。当前本地包没有该拒绝执行的独立 log/report，因此实际执行拒绝来自 direction 转达，不虚构一个不存在的成功 preflight JSON。最新 840/1280 也是 direction 报告，**不是记录者新的 live 查询**；本轮未再次 polldata。

**局限与下一决策。** 预期拒绝证明的是门槛拒绝行为，不算 CPU eligibility 通过、GPU resource gate 或训练通过。待完整 generation/audit/runner 成功后，用同快照核真实 eligible train/dev 与 hashes，再按实际 rows 计算各段 traversals；计划 max_steps 1280、accumulation 8、anchors 32，从新 draft 开始，手工分段 32/128/512/1024/1280，不自动启动。真实 Pareto 两周期 GPU memory gate 需另协调并审结果，最终 test 不参与选参；本轮记录者只改日志，未执行这些未来命令。


## 2026-10-09 00:35（UTC+8）— R03：STS 的 CPU 算法与 prefix 标签契约

**问题与假设。** 训练中的软 overlap 和 greedy MAE 不能证明随机 rollout 的 confidence 已校准。应先对每位置条件 confidence logit 做 `sigmoid(z[j]/T[j])`，再 cumprod，以真实存活到该位置的二值 prefix event 拟合，而不是独立条件接受标签或已累乘概率的温度变换。

**方法。** 已直接读取 `calibration.py`、[STS 协议](confidence-calibration.md)和原始日志。每条 block 用实际 accepted-prefix length 标记 `int(j < accepted_prefix_length)`；首次拒绝后，已被 target 评分的 suffix 保留零 prefix 标签，未验证/因预算截断的尾部不进入分母，绝不补零。接受 EOS 自身仍是正例，其后全部排除。只接收 validation，拒绝 train/final test；要求同一冻结 checkpoint、development records、rollout protocol SHA 及明确 probability policy，调用方仍须核真实来源与采样因果性。收集采用无 confidence threshold/admission 选择的 `full_proposal`，预算/EOS 截短需披露，greedy/stochastic 不混合。

温度搜索是明确披露的本地约定：默认 61 个 log2 等距点、范围 0.125–8、包含 T=1；从左至右最小化该位置 cumprod ECE，冻结早先温度；精确平局先取 log 空间最接近 1，再取较小值。默认 20 bins，cumprod 后统一 clamp 至 `[1e-8,1-1e-8]`，同时影响 ECE、Brier、pred_mean；无观测位置保存 T=1、fitted=false 和空指标。按 block 而非 prompt 宏平均，多轮相关性不被当成独立置信区间。

**直接观察。** 本地未发布 `output/scope-research/calibration-unit-tests.log` 为 **8 项通过、0.002 秒**；手算 fixture、顺序目标、EOS/拒绝分母、未验证尾部、端点 clamp、极端 logits、平局/空位置及身份边界均有检查。没有真实 rollout 收集、真实温度拟合或 decoder/scheduler 接入。API 的 calibrated ECE 是拟合群体目标值，不能写成独立评估结果或当前 head 已校准。

**版本与下一决策。** 本地 Git 直接确认算法 commit `f54da037174598ce31c092d642bccc4ef1656733` 与文档 commit `a7abb256cefa6e8bb3a0046306c86968ee198673`；协调者报告 push exit 0、从 cbc65b1 更新至 a7abb25，记录者本轮未做远端 refs/CI 查询。后续先冻结实际概率/收集协议，在预划定且 prompt 不重合的 development 子集上分别拟合与评估，再冻结温度；最终 test 不参与选参。immutable expanded training 快照仍是 796fecc，此新模块没有进入该快照。

## 2026-10-09 00:35（UTC+8）— T03：未完成 generation 的新留证拒绝，补充 T02

**问题与方法。** T02 当时的拒绝执行只有 direction 转达、本地独立 log 尚未存在。随后 direction 做了一次新的留证执行；本轮直接读取本地未发布 `output/expanded-training-preflight-20261009/preflight-incomplete-refusal.json` 与 `.log`。这条新证据不回改 T02 的历史来源等级，也不冒称是当时那次执行。

**观察。** JSON 保存精确 argv，exit_code=1、expected_refusal_observed=true；traceback 在 preflight 第 40 行 completion marker guard 抛出 `Not complete/successful: full-generation`。按已读代码顺序，这发生在 import runtime、读 development records 或构造模型/GPU 之前。它确认完成门槛拒绝行为，不是成功 CPU eligibility/preflight、GPU memory gate 或训练。此前 archive SHA 与全部 112 个文件/commit blobs 的独立核对仍适用；只读权限继续仅是同用户工作约定。

**状态来源与下一决策。** 协调者于 00:26 转达 1112/1280，generator 为 Rl、runner 为 S、退出标识未出现；这是转达的运行快照，记录者本轮没有再次 poll，也不由此宣称完整数据审计通过。只有终态 full-generation/full-audit/runner exit 0 与完整 exports 到齐后，才核 audited train/dev 的真实 eligibility/hashes，进入另行协调的 GPU 资源门槛；最终 test 仍不读取用于调参。

<a id="stochastic-cache"></a>
## 2026-10-09 00:35（UTC+8）— R04：真实 Markov proposal 与 cached stochastic 的首轮 CPU 集成

**问题与方法。** R02 小词表 sampler 仍需与真实 tensor proposal、Markov q、target 行和两套 KV 提交语义接通。已读[缓存随机路径协议](cached-stochastic.md)及 `tensor_sampling.py`、`cached_sampling.py`：采用独立策略 `float64_softmax_normalize_cdf_v1`，temperature 正且有限、无过滤；model forward dtype 不变，实际 logits 升为 float64、softmax 后一次归一化，保存真正用于抽样的 q。verifier 检查已定义 p/q，不再次归一化，不加 denominator epsilon 或小 residual fallback。

固定 proposal 长度在本轮随机数前承诺，非空 proposal 一次 backbone 后逐位置 Markov，保留当前 token 抽样前 raw confidence。两套 cache 在轮次边界包含 committed prefix、排除最新输出 anchor；首拒绝 crop、全接受 bonus、EOS/预算和异常 reset 依该边界处理。`verified_proposal_length` 表示 target-scored 候选数，`attempted_positions` 仅是已抽接受随机数的位置；STS 不能只用 attempted 作为分母。raw confidence/conditional overlap 仍不是已校准信号或真实 prefix 标签。可选 observer/全词表 trace 有同步、传输与存储成本，私有诊断不能作计时路径或公开样本。

**直接观察。** 首轮 `output/scope-research/tensor-cached-sampling-tests.log` 是 **22 项通过、1.942 秒**（新增 tensor/cached 11 项加既有 sampler 11 项）；另一次 `tensor-cached-sampling-final-tests.log` 为 **22 项通过、1.640 秒**，两个轮次分别保留。`stochastic-full-cpu-suite.log` 为 **76 项通过、5.768 秒**，包含既有 greedy/cache/core、scheduler 和 STS。完整 cached 随机树的 Fraction law oracle 与独立 target 自回归枚举质量相等；tiny Qwen CPU 检查每层 target/projected draft KV 的内容和长度、一次 backbone、实际 Markov q/pre-token confidence、各拒绝位置、EOS/预算、状态隔离和失败 reset。

**身份与局限。** 文档声明独立临时 CPU 目录及隐藏 GPU 的环境，并断言 cuda unavailable；记录者直接核了上述本地原始测试日志，未执行任何 GPU gate。日志不能独立补齐精确执行命令与完整源码/runtime 哈希绑定。当前本地归档 commit 为 `4255f23992cab62c6915eb5428edcaa91abff1c7`（00:28:46）。协调者另核 `stochastic-cpu-manifest.json` 中最终七个交付文件的 SHA 全部与提交前源码相符；这项核对不补成完整运行环境身份。数学 oracle 不是所有浮点边界的形式化证明；float64 概率运算不消除 BF16 不同 kernel 的 logits 差异，原动态 BF16 失败继续成立。

**下一决策。** 保留 target-only 同 probability adapter 的 baseline；同 seed 跨不同算法不要求输出 token 相同，只要求同路径/runtime 重跑语义。真实 GPU stochastic gate、实际 validation rollout、STS 接入及性能都尚未完成；采样、probability transform、verify、KV 与 observer 的物理代价必须完整计入后续比较。

<a id="packed-isolation"></a>
## 2026-10-09 00:35（UTC+8）— S03：单次 packed Qwen 前向的多请求 CPU 状态参考

**问题与假设。** 全局 B 分配需要把不同请求的可变新片段放入一次 target forward，且各请求的 position、可见 KV、crop/退出独立。无 query padding 不自动代表 attention 高效，必须把 dense score 域及显式 mask 工作分开披露。

**方法。** 已读[packed target 协议](packed-target.md)：请求片段拼成 `[1,sum(query_lengths)]`，一次 Qwen backbone 和一次全部新行 LM-head；request marker 与本地 position 形成可见条件 `same_request and key_position <= query_position`。共享 DynamicCache 的物理 KV 可交错；inactive resident 请求无 query 但保留 keys。每请求 crop 逐层 gather，remove 后同外部 ID re-add 使用新 marker/本地位置 0；输入错误保留状态，前向中途失败清空全部请求。仅支持 pinned HF 5.17 dense Qwen/full attention、SDPA、default RoPE。EOS/随机验证/draft KV/预算仍是调用方职责。

**直接观察与身份。** 本地未发布 `output/packed-target-cpu-20261009/tests-final.log` 为 **5 项通过、0.858 秒**；`execution.json` 保存命令、exit 0、CPU FP32、Torch 2.12.0+rocm7.2、Transformers 5.17.0。执行时间为 00:26:47（UTC+8），通过独立临时文件追加 package 搜索路径载入，未改 796fecc 只读快照或 live generation。执行绑定 `packed_target.py` SHA `fece900ab97d4611b093b1ed95263a6b589cf8f99bd704e167884e33c95797ca`、test SHA `cf1321994aa6ec060240a4408cd66238b13cb14bea4dd2e32dedc7c2796f40eb`，本轮另核当前两个本地文件 SHA 与该执行身份一致。日志的两条 `(null): No such file or directory` 原样保留，后续五项均通过；尚无定位该提示原因的证据。

混合 query 长度、次序变化与 inactive keys 均通过 forward hook 确认一次真实调用；每请求 hidden/context/logits 和逐层 KV 内容对照独立 cached/fresh oracle。还覆盖每个 crop 边界、全部拒绝/partial、crop 到 0 后继续、退出/同 ID 重新加入，以及跨请求 KV/new-token 和同请求 future-token poison 隔离、exception reset。CPU FP32 比较容差为 atol=2e-6/rtol=1e-5，不能说成逐 bit 相同。

**物理工作边界与下一决策。** Q 是实际新行之和、无 query padding；K 是包含 inactive keys 的全部 resident KV，当前 dense score 域仍为 Q×K，并分配 `[1,1,Q,K]` boolean mask。allowed/cross-request/future 三类计数相加为 Q×K，不是实测 FLOPs 或高效 varlen kernel 的证据；crop gather 也不是 paged KV/graph stable 管理。此次未读真实数据/test，未用真实 target 权重或 GPU，未测接受率、SPS/吞吐/SLA。后续先用相同状态 oracle 审核实际可用的 ROCm varlen/block-sparse 路径，再测 mask/tile/KV gather 和 context/load 物理代价，不能仅凭 API 存在或 CPU 通过声称生产异步机制完成。

**随后版本核实（协调者）。** packed 三文件归档为 `acef238`；Git push exit 0，origin/main 从 `a7abb25` 更新至 `6f8b244`，包含 `4255f23`、`acef238` 与入口/范围文档更新。以上是源码推送证据，最新 CI 尚未核实。


<a id="data-final-audit"></a>
## 2026-10-09 00:43（UTC+8）— P06：正式 1280 输入生成终态与独立结构复核

**问题与方法。** P05/T02/T03 的进度或 guard 拒绝都不是最终审计。协调者新授权记录者等待约 45 秒后仅查一次 live 终态；本轮直接确认原 generator 3405911 和 runner 3402974 已不存在，full-generation/full-audit/runner exit code 均为 0，六个 stages 全部成功。此前一次 SSH 命令在远端 Python 启动前因 heredoc 换行转义错误失败，未产生 live 查询或更改；随后的正确命令才是这次终态检查，没有重启/信号/GPU 工作。

**独立复核。** 先将远端 prepare/generate/audit/data_pipeline 四个实际脚本 SHA 与冻结 `d5e1538960afded86d487eab71908ce2c7531a7e` 的 Git blobs 比较，全部一致；再调用其 stdlib `audit_run`，不覆盖 runner 原 audit、不 import model runtime。复核结果与 runner audit 相同：规范化 prompt/排除身份、来源、split/export 顺序、精确 token/EOS/预算、三路隔离及所有 output SHA 通过。另核全部 160 个 batch 的 payload/checksum、1280 条记录与 exports 内容一致；没有读 test 做质量判断或调参。

**观察。** 输入 1280，接受 1170：train 932 / validation 119 / final test 119；拒绝 110：train 92（截断 69、模板 token 不匹配 23）、dev 9（5、4）、test 9（6、3）。全体 finish reasons 为 EOS 1200、length 80，故 EOS 完成不等于接受，30 条 EOS 回答仍因模板不匹配拒绝；全部接受记录终止 EOS 151645。不修补或重生成来掩盖拒绝，模板不匹配机制尚未定位。

精确 accepted output token 为 train 297385 / dev 38692 / test 40748。accepted training-template token 为 425199 / 54608 / 57122；summary 的 training_tokens 536929 包含三路，不能称 train-only token。train/dev/test 最大模板长度分别 2225/2000/2153。全生成 token 469958 含拒绝；invocation 4853.66 秒、峰值 allocated 3863313920 bytes 只记录造数据资源，包含暂停与运行开销，不是 serving 性能。每 split 的长度 min/nearest-rank p50/p90/p95/p99/max/sum 与来源聚合另存公开报告。

**证据与身份。** 新增[正式 summary](../reports/data-expansion-20261009/full-summary.json)、[最终 audit](../reports/data-expansion-20261009/full-audit.json)、[completion](../reports/data-expansion-20261009/full-completion.json)、[正式 source identity](../reports/data-expansion-20261009/full-source-identity.json)及[长度/batch aggregate](../reports/data-expansion-20261009/full-aggregate.json)；保留原准备/smoke/启动阶段报告。正式 config SHA 为 `042c644900c56c136971d11d94e1a4814b82ccef55b8b6495e7214a937a569d6`，selection SHA 为 `b934967810270f932f915b8eecc74e5d6bce01d8bddea97284aa6fcac9822b40`；model 私有路径在公开 projection 中替换为标签，config hash 绑定原始 bytes。正式 development records SHA 为 `69daac1d39af961c75fcb0795c12b22d7a9a18d747cd05f0673202060f6914e2`，最终 test 独立 exports 哈希齐备。各类完整原件/脚本 collector stdout 保留 ignored，不公开样本。

**下一决策与边界。** 已告知协调者可进入同 immutable 796fecc 快照的 eligibility/resource preflight 协调，但本轮尚未执行 expanded training/GPU memory gate、训练或 STS rollout；新的终态不回改此前真实拒绝。训练数据仅 932 train，dev 119；final test 不参与 checkpoint/policy 选择，权限仍非安全边界。单分片/截断/模板拒绝限制数据代表性。后续实际训练形状/eligible rows/anchor 数由 preflight 再核，真实显存与质量逐门槛推进。记录者没有 commit/push 或调度其他进程。


<a id="expanded-resource-gate"></a>
## 2026-10-09 — T04：完整数据 CPU eligibility、长度口径与真实显存 gate

**问题与假设。** T02/T03 的 completion guard 拒绝是正确的早期门槛行为，完整生成后需重新确认数据身份、实际 eligible rows 与训练形状，再判断累积更新和 Adam 状态常驻时是否能运行。生成 summary 的 training-template 长度不能直接代替训练输入长度。

**方法与解决办法。** 本轮直接读未发布 `output/expanded-training-preflight-20261009/cpu-preflight-execution.json`、`cpu-preflight.json`、`length-convention.json`，以及 `output/expanded-training-memory-gate-20261009/{completion,result,aggregate-summary}.json`。CPU preflight 于 00:39:22 执行 exit 0，source 冻结为 `796fecc0d9142c1a733ad6df47b105fab4ee9f05`，archive SHA 延续 T02；配置 SHA `ea8091ca8d124434696ccd15f88249839aaf1d71bdf540c3db9142bcbdda4762`。本轮独立核本地 config bytes、preflight SHA、gate/run identity 与冻结 package SHA 一致。没有执行 GPU 或重新读取生成记录/test 内容。

**长度观察。** eligible train/dev 为 932/119，train 实际 sequence tokens 总计 **424267**，completion 含 EOS 总计 297385；生成 audit 的 train 模板总计 425199，比实际输入多 932，恰为每条一 token。直接长度结构证据确认最长 accepted train 行实际输入 **2224**（prompt 1588 + completion 含 EOS 636），rendered training field/重分词为 **2225**；generated IDs 是 rendered prefix，EOS 后模板还含 token 198（换行）。这是模板末尾换行与训练实际 IDs 的口径差异，未据此改数据或宣称模板拒绝机制已定位。

**显存与正确性观察。** 00:43:45–00:44:08 memory gate exit 0、未 timeout，runner wall 22.936719 秒。932 行产生一个非支配真实形状 `(sequence=2224, actual anchors=32)`；累积 8，执行两次完整更新周期，分别覆盖首次 Adam 分配及状态常驻。projection/backbone/Markov embedding/Markov projection/confidence gradient checks 均通过，首次更新与第二周期状态常驻通过，target frozen=true。optimizer tensor state 1293537536 bytes，峰值 allocated **5636591616 bytes**，reserved **5827985408 bytes**；runner 启动前设备 free 25246564352 bytes。该 gate 不保存正式 checkpoint，探针自身更新不作为训练质量结果。

**局限与下一决策。** Pareto gate 是实测资源门槛，kernel workspace/allocator 不必随长度或 anchor 数单调，不是全形状最坏显存保证，也不保证未来 co-tenant 内存。wall/显存是资源事实，不能推出 serving SPS、speedup 或训练质量。现有 preflight 与 gate 允许进入另行授权的 step32；此前拒绝条目仍保留原来源和结果。训练/source/config 外的 runtime、模型、数据及 GPU 共用状态仍须在未来执行前核对。

<a id="expanded-step32"></a>
## 2026-10-09 — T05：expanded training 首段 step32 与 checkpoint 身份核对

**问题与假设。** 显存 gate 通过后，需确认全 eligible 数据配置下真正的 32 optimizer updates、冻结 target 和可恢复 checkpoint 能完成，并观察 dev teacher-forced 学习信号；这不能替代独立 rollout 质量门槛。

**方法与尝试。** 本轮直接读未发布 `output/expanded-training-step32-20261009/{completion,run,result-step-000032,checkpoint-metadata,step32-verification}.json`、`metrics.jsonl`、runner exit，以及冻结源码 `output/expanded-training-preflight-20261009/source/dspark_qwen/train.py`。runner 从 immutable 796fecc snapshot 启动 fresh `--stop-after 32`，没有自动 resume。配置累积 8、LR 0.0006、32 anchors、block 7、BF16 AMP/FP32 trainables；共有 **161692161** trainables、932 train/119 dev。256 microsteps / 932 = **0.2746781116 遍**，不是完整一遍训练；按固定行顺序循环，anchor 重采样。

**正确性与身份观察。** 00:46:57–00:47:59 runner exit 0、未 timeout，result 存在；本轮独立核 optimizer 日志恰为 1–32、全部数值 finite、本地冻结 package SHA、config bytes、run/preflight 的 source/config/data/target/runtime 字段，以及 metadata/run identity 一致。`step32-verification.json` 记录 latest/metadata/resume-state step 都为 32、weights 与 optimizer/RNG resume hashes 匹配、source/config 未变；大型 checkpoint 原件未在本地，本轮 checksum 匹配依据是该留证报告，没有冒称重新 hash 远端权重。target frozen=true，源码检查对应无 target gradients 且 parameter version 未变；这是已实现冻结检查的范围。

**模型质量观察。** 同 119 dev、同 anchor RNG 的逐样本宏平均 teacher-forced loss **3.385942285 → 2.582734973**，CE 11.99816623 → 7.180634186，L1 1.998285316 → 1.957873026；字段 teacher_forced_accept 的软 overlap **0.00085734024 → 0.02106348236**，confidence BCE 0.387668913 → 0.102585854、MAE 0.318861928 → 0.020433038。supervised tokens 的样本均值前后均为 213.8739496。可观察到初期 teacher-forced 学习信号，不能称 2.106% rollout acceptance、confidence 已校准、held-out rollout 已通过或 test 质量提升。

**计时与显存口径。** runner wall **61.78925495 秒**，report elapsed **36.97336006 秒**，allocated 峰值 **5571854848 bytes**。源码 `began` 在 before validation 之后，而 report 在 after validation 与 checkpoint 保存之后计算，所以 36.973 秒包含训练更新、after validation 和 checkpoint 保存；纯更新循环末条 elapsed 为 **20.99824730 秒**，也没有独立逐阶段同步计时。不能将 report elapsed 写成纯 training loop、从 wall 推断训练吞吐或 serving 性能。

**局限、解决办法与下一决策。** 本段无运行失败；用源码纠正 elapsed 名称，用身份/step/hash 留证排除混用源码或配置，用分开记录的 TF 指标避免误读为 rollout。只完成首段执行及初期学习检查；更长训练、恢复实际加载、独立 development rollout、STS 拟合/验证和真实性能仍需各自证据，后续只在新的明确授权下推进。final test 未打开、未做质量分析，未修改训练快照/实现或 commit/push。


<a id="stochastic-gate-cpu"></a>
## 2026-10-09 01:02（UTC+8）— R05：bounded stochastic gate 的 CPU 门槛与真实文件 dry-run

**问题与假设。** R04 的随机缓存 CPU 路径要进入真实 BF16 model gate，需先冻结原 case/checkpoint 身份、执行边界和失败留证，避免用新的训练权重或删掉原失败 case 偷换问题。可重现执行、same-prefix 后端数值差异、分布数学证明、模型质量和性能必须分别报告。

**方法。** 直接读取[有界 stochastic gate 协议](stochastic-gate.md)、`eval_stochastic_gate.py`、final CPU tests/dry-run log、final manifest 及其 private run/aggregate 的身份字段。三例固定为原 pilot aligned128 checkpoint 的训练 prompt 和前两条 accepted validation prompt，沿用 prior gate；**这不是新 expanded step128 checkpoint**，不读 final test。checkpoint 权重/metadata、target 文件、development records、prior gate、三个 case 与 dry-run 的 51 个 `*.py` inventory 条目哈希一起绑定；其中 26 个是普通源码，25 个是匹配 glob 的 macOS `._*` sidecar，并非 51 个可执行模块。dry-run 只用标准库，复制源码快照，禁止 import torch/transformers、启动 worker 或接触 GPU；正式执行须另一个 fresh 输出，worker 先重核 binding，不能在训练快照中修改代码。

协议绑定 `float64_softmax_normalize_cdf_v1`、temperature=1、seed=20261009、预算 32、固定 block capped by budget，target BF16/draft FP32+BF16 AMP/SDPA、无过滤；reference/立即 repeat/跨请求 1,0,2 repeat 分开。预先指定 reference 的前两轮，重放同 prompt、committed prefix 和 proposal prefix，逐行比较全部 n+1 行 p/logits（含 bonus），拒绝后的行标为 proposal 路径。Same-prefix probe 混合 history/chunk/current shape 数值影响，不能据此定位单一 kernel 根因；不把非零 TV/logit 差按任意阈值写成无损通过。默认超时 600 秒、上限 1800，只结束自有 worker；6 GiB allocator 上限和启动前 8 GiB free 门槛不保证驱动/workspace/co-tenant 全系统内存。

**直接观察。** `output/scope-research/stochastic-gate-final-cpu-tests.log` 为 **7 项通过、6.308 秒**：标准库 dry-run 禁止 runtime/worker、原输入损坏与 case 漂移拒绝、fresh 输出/source binding、native streams、超时只结束自有 child、partial evidence、tiny Qwen 的 9 次执行/6 次 repeat 比较与全部 same-prefix probe 行。负向 fixture 的 failed/timeout 事件是预期测试，不能写成真实 GPU gate 失败；两条 `(null)` 提示仍保留。真实文件 dry-run log 为 dry_run_complete、cases=3、gpu_touched=false、verified_no_torch_import=True；aggregate completed_runs=0、reproducibility_checks=0、execution_checks_passed=null，native fidelity 为 not_measured/0 rows。它证明原文件身份与流程准备，不证明任何真实 rollout 或数值保真通过。

**版本与独立核对。** 本地确认 commit `d6e385d8c41253de7c37ebb47dd7b16169e6b0b5`（00:57:28）；final manifest 四个交付文件 SHA 都与该 commit blobs 一致，当前 gate 源码/test/协议也一致。当前 README.ai.md 因后续入口更新已不同于当时哈希，未把现有 README 说成执行原件。重新计算 unsigned run JSON 的 binding SHA 为 `b47856093fbc3b4b71d4e55cd3277fa4dd14ccc7a827c64aa099623754e1d132`，与 manifest/log/aggregate 一致。原 aligned 权重 SHA 为 `8c634a2e0b9b3b6a46d573836c1e4ab54c810b060affa65a13ea80822cf585ca`，pilot development records SHA 为 `9bd09a5958c45485f4f7b7fdbb4bd42387d28b6a2bbaa173e20f3f62f15ec247`。记录者核的是本地留证和 hash binding，不虚构本轮重新读取远端大型权重，亦未查询 push/CI。

**局限与下一决策。** 协调者报告 core 正在执行真实 GPU gate，另报告 expanded step32→128 自然 exit 0、其结果还在归档；这里仅记录协调状态，不把 GPU intent 或未到的训练 result 写成通过。真实 gate 有结果后，分开记录 execution/repeat/cache 边界与 native fidelity，同时保留失败 case、private partial traces 和原 dynamic BF16 问题。GPU finite/shape 检查也不替代 tiny CPU 全 KV 内容 oracle。没有 STS 拟合、scheduler/异步 serving 或速度结论；同 seed 跨 speculative/target-only token 相等不被要求。记录者只核证据、改 notebook，未操作 GPU/源码/进程、commit/push 或 final test 质量。


<a id="expanded-step128"></a>
## 2026-10-09 01:04（UTC+8）— T06：expanded strict resume 32→128 与 confidence 指标边界

**问题与方法。** T05 只完成 fresh step32；继续到 128 必须实际加载 weights/optimizer/RNG，而不是从头训练或改变 max_steps/config。记录者直接读取本地 `output/expanded-training-step128-20261009/` 的 completion/result/run/started/checkpoint metadata/latest/metrics/train log/step128-verification，以及[expanded 聚合报告](../reports/expanded-training-20261009/README.md)和[summary](../reports/expanded-training-20261009/summary.json)。train log 明确 resumed_step=32，冻结源码 loader 在恢复前核 weights/resume-state SHA、metadata identity 和 step；原配置 max_steps=1280 不变，执行仅 `--resume --stop-after 128`，timeout 1800，不自动续下一段。

**直接观察与身份。** 00:57:35.979–00:59:23.251（UTC+8）exit 0、timed_out=false；run.json 与 step32 原 run 完全一致，metadata identity 匹配。metrics 恰为 1–128 且全部 scalar finite，最初 32 行与 T05 的解析记录完全相同；33–128 新增 96 updates/768 microsteps，累计 1024 microsteps / 932 = **1.0987124464 遍**，本段增加 0.8240343348 遍。恢复初始 dev 指标与 step32 末次逐字段完全相等，排除换 panel/评估 seed 的该项漂移；仍不是未恢复连续 128-step 的逐 bit 更新等价实验。

留证 verification 记录 latest/metadata/resume-state step 均 128、62 个 optimizer state entries 的 step 全为 128、source files/config bytes 未变、checkpoint/run/preflight identity 一致及 hashes 匹配。weights SHA 为 `5e6c2cbaf9eaca081ba3c598748952bc735e7d8b54ca5ecf77a47a6bf1c561e0`，resume SHA 为 `ea68520d884458e9187cf3a730e53b37a2e48922a7708ca326157994e15b8137`；大文件远端 hash 重算依据该留证报告。记录者另外独立核公开 summary 的 result/completion/verification SHA 与本地三份原始证据一致，不冒称本轮重 hash 远端权重。

**模型指标与失败信号。** 同 119 dev/固定 anchors 宏平均，step32→128 loss **2.582734973→2.525808697**、CE **7.180634186→6.329201546**、L1 **1.957873026→1.812801690**；软 teacher-forced overlap **0.021063482→0.093599154**（2.1063%→9.3599%）。与此同时 confidence BCE **0.102585854→0.261367040**、MAE **0.020433038→0.098902311** 上升。teacher overlap 目标分布也随 draft 改变，因此跨 checkpoint 的误差绝对值上升本身不能证明 head 预测能力退化；应优先比较同 checkpoint 的基线及独立 rollout 校准。按冻结 loss 的同一 weights/宏平均定义，constant-zero confidence 的 MAE 等于 overlap，即 step128 为 0.093599154，仍好于 head 的 0.098902311；这是现有指标与定义推导，非新 rollout 校准实验。不能因 loss/overlap 改善选定 checkpoint，不能将 9.3599% 写成真实接受率或 confidence 已校准。

**资源与计时。** target frozen=true，trainable parameters 161692161，932 train/119 dev，BF16 AMP/FP32 trainables；peak allocated **5633036288 bytes**。controller wall **107.27052365 秒**；result elapsed **76.42861543 秒**仍含末次 validation/checkpoint 保存；末条 optimizer metric elapsed **60.14590007 秒**是本段更新循环及 Python/logging 口径，不是端到端 serving 性能或独立同步阶段计时。controller 的退出后设备内存含此时其他工作，未将其当训练独占实测峰值或实时空闲。

**局限与下一决策。** 此段证明已授权 strict resume 和初期 TF 学习推进；step128 的置信头尚未优于同 checkpoint 恒零 MAE 基线。独立 development rollout、STS 拟合/验证、checkpoint/policy 选择及后续512/1024/1280阶段未由本条执行或宣布完成；真实随机 gate 使用的是 R05 旧 pilot aligned128，不能用其结果代表 expanded step128 质量。保留未校准信号与原 dynamic BF16 数值问题。记录者只读取小型聚合/身份证据并追加日志，未读 final test 质量、运行 GPU、修改源码或 commit/push。

**T06 随后归档核实。** 公开 expanded 五文件报告的本地 commit 为 `4a85847d584b01e2696797c01c7744a04022c07c`（01:04:12），记录者直接核 Git 条目；push/CI 未在本轮查询。协调者随后授权 direction 下一段 128→512，这仅是计划，不在 T06 写成已执行。


**R05/T06 随后 CI 证据（01:06）。** 协调者成功请求公开 GitHub API，记录者直接读取 ignored `output/scope-research/github-actions-20261009-0106.json` 及 workflow。`d6e385d` 的 [run 37812874291](https://github.com/GodHu777777/qwen3-dspark/actions/runs/37812874291) 与 `d9289d3e1c871eef8184498ca51e0df86633cd13` 的 [run 37813260774](https://github.com/GodHu777777/qwen3-dspark/actions/runs/37813260774) 为 completed/success；`4a85847` 的 [run 37813847597](https://github.com/GodHu777777/qwen3-dspark/actions/runs/37813847597) 在该快照仍为 in_progress/conclusion=null，不能写通过。该 workflow 在 Ubuntu/Python 3.11 安装 Torch CPU 和 package[data]，执行 unittest discover；只支持这些 commit 的 CPU CI 状态，不支持 GPU gate、AMD 后端数值或后续 head 的 CI 结论。本条修正的是此前未查询的 CI 证据等级，不抹除原时间点的未知状态。


<a id="stochastic-native-gate"></a>
## 2026-10-09 01:09（UTC+8）— R06：真实 BF16 stochastic 执行可复现，native 数值差异仍存在

**问题与方法。** R05 只完成 CPU/真实文件 dry-run；本轮记录者直接读取新公开[aggregate](../reports/stochastic-gate-20261009/aggregate.json)、[运行/复现 audit](../reports/stochastic-gate-20261009/audit.json)、[protocol](../reports/stochastic-gate-20261009/protocol.json)、[runtime](../reports/stochastic-gate-20261009/runtime.json)、[源码身份](../reports/stochastic-gate-20261009/source-identity.json)及[逐行数值探针](../reports/stochastic-gate-20261009/numerical-probes.json)。它们记录 core 在新 source snapshot 的真实 AMD 执行，仍用旧 pilot aligned128 权重/原三 case，不是 expanded128，也未读 final test。target BF16、draft FP32+BF16 AMP、SDPA、float64 law、temperature1、seed20261009、固定32输出预算与前两轮 probe，均未为过 gate 切换 FP32/canonical。

**执行与复现观察。** aggregate 为 completed、execution_checks_passed=true：三 case 各 reference/立即 repeat/跨请求 repeat 共 **9 runs**，六项同路径 target-only/speculative token/round/audit 复现检查全部通过。finite/shape/commit audit 全部通过，确认实际 p/q/confidence、target KV/新 projected draft KV 的有限值、形状与提交边界；不是完整 GPU KV 数值 oracle。Reference 训练 case 输出32、29 rounds、committed draft2；dev case1输出6/EOS、5 rounds、draft0；dev case2输出32/预算、31 rounds、draft1。同算法/runtime 的可复现不要求 speculative 与 target-only 在同 seed 下 token 相同。两条 dev 的0/1接受也不是代表性 rollout 质量结论。

**数值保真观察。** 预设三例前两轮全部 n+1 行，共 **48 probe rows**；47行 TV非零、最大TV **0.04161940616245697**，最大绝对logit差 **0.5**，1行 argmax 改变（case2/round1/position0，TV0.030732573863166624）。记录者独立从逐行 JSON 重算上述计数与 maxima，均与 aggregate 一致；45行有非零logit差，另2行虽logit差0仍有浮点量级非零TV，未把所有非零TV解释成同等大小的模型误差。Probe 重放同语义 prefix，包含拒绝后 proposal 路径与bonus；混合 history/chunk/shape影响，不是单一 kernel 因果定位。CPU residual 数学恢复的是实际 verifier p，不能因执行通过就认为 block p 与 sequential target p 在真实 BF16 下相同；**native模型分布无损没有被证明，原 dynamic BF16 问题未解除**。

**独立身份核对与 dry-run inventory 修正。** 生产绑定为 `81022db4947f68b25c7375350c86284e049e260adde3c69e6f430350f5b4d49c`，commit仍 `d6e385d8c41253de7c37ebb47dd7b16169e6b0b5`；其27个正常package源码SHA逐个与该commit blobs一致，archive SHA `f438f9d257fcf67974603c2a52bd0b0be404aab39af9854e7410667da2526a07` 与本地留存归档相同。它不是 R05 的 b47856 dry-run binding/inventory：dry-run26个普通源码加25个macOS sidecar，共51条，生产去sidecar并包含packed_target.py，共27个普通源码；共有26个普通源码SHA一致。R05此处修正计数含义，不将dry-run称为生产inventory。生产报告仍绑定旧aligned权重8c634a和pilot records9bd09a，私有result/round/probe/stdout只公开哈希。

**终态与证据限制。** audit postcheck 记 worker completed event/result、launcher和worker均已不存在、无launcher/worker error文件；协调者另外live核GPU释放。最初background launcher没有持久化独立OS exit-code文件，**不写“实测exit0”**。ASR HTTP200、ready=true/busy=false，postcheck仅原ASR进程；设备 used before/after均8962179072 bytes，gate peak allocated2260815872 bytes。这些是core/协调者的留证及观察，记录者本轮未重新查询远端进程/endpoint，亦不用于速度结论。runtime为Torch2.12.0+rocm7.2、Transformers5.17.0、HIP7.2.53211；deterministic_algorithms=false，有限复现结果不提升为跨runtime/device保证。

**局限与下一决策。** 此gate完成有界执行/随机状态隔离与后端差异测量，保留执行通过和数值差异两条结论；全词表CPU复制、finite检查、文件写入与fresh sequential探针有显著成本，无性能比较或加速主张。没有用本结果选择expanded checkpoint/采样policy，没有STS拟合或同步/异步scheduler证明。后续用独立development rollout/STS评价训练质量，并按同prefix更细拆分动态BF16数值来源；真实packed/全局预算/物理成本仍待各自门槛。记录者仅改notebook，无GPU/源码/进程/commit/push动作。


<a id="expanded-step512"></a>
## 2026-10-09 01:19（UTC+8）— T07：expanded strict resume 128→512 与同 checkpoint confidence 基线

**问题与方法。** T06 完成 step128 后，只在新授权下恢复到512，不改变训练源码/config/max_steps，需检验旧日志继承、optimizer/RNG恢复和 dev panel 连续性。记录者直接读取[本段公开报告](../reports/expanded-training-step512-20261009/README.md)、[summary](../reports/expanded-training-step512-20261009/summary.json)、[checkpoint核验](../reports/expanded-training-step512-20261009/checkpoint-verification.json)、[source identity](../reports/expanded-training-step512-20261009/source-identity.json)及公开 steps129–512 metrics，另读本地未发布 `output/expanded-training-step512-20261009/` 中 completion/result/run/metrics/latest/metadata/verification 和退出标识。原796fecc快照、配置SHA `ea8091ca8d124434696ccd15f88249839aaf1d71bdf540c3db9142bcbdda4762`、max_steps1280继续绑定，执行仅 strict resume `--stop-after 512`，没有自动推进1024。

**直接观察与身份。** 01:05:05.252–01:09:54.362（UTC+8）process exit0、timed_out=false；本段新增384 updates/3072 microsteps，累计4096/932=**4.3948497854遍**，本段增加3.2961373391遍。直接核完整metrics恰为1–512、全部scalar finite，最初128条解析记录与原step128完全相同；公开metrics恰为129–512并与本地新增部分逐项相同。run identity与step128完全一致，metadata identity匹配；恢复before dev逐字段等于128final，仍用119dev/固定anchor seed、样本内weights加权再宏平均，不是独立rollout panel。

本地verification与公开checkpoint-verification完全一致，记录latest/metadata/resume-state step均512，62个optimizer state entries的step均512，source112/config bytes未变、run/preflight一致、weights/resume hashes通过。权重SHA `667d2dd6e8ad7d11e2115e936af1b6cf7e44357d68e3412f5c9ef433f26690ae`，resume SHA `a8b433cbcee71e73ca32696f1755e18bc41b62618934bff92ba91dacf1a59c73`；记录者独立核metadata中的这两值与verification相同，并核summary的四个raw evidence SHA与本地result/completion/verification/full metrics一致。大文件远端重hash、CPU加载resume-state的事实依据direction留证，没有冒称记录者重新读取远端checkpoint。

**TF结果与confidence比较。** Step128→512，dev loss **2.525808697→2.191068616**、CE **6.329201546→4.807051978**、L1 **1.812801690→1.425409559**；teacher-forced软overlap **0.093599154→0.287295218**（9.3599%→28.7295%）。confidence BCE **0.261367040→0.427494834**、MAE **0.098902311→0.184831666**绝对值上升，但overlap目标随draft变化，不能跨checkpoint单凭这些值宣布预测能力退化。**同step512**的constant-zero MAE按同weights/宏平均定义为 **0.287295218**，learned MAE **0.184831666**较低；只支持优于这个零基线。constant-mean/median基线和真实rollout校准尚未测，TF MAE改善/相对基线优势不能写成confidence已校准或scheduler有效。28.7295%也不是实测连续接受率、答案质量或速度。

**计时与资源。** target frozen=true、161692161 trainables、932train/119dev；本段peak allocated **5632856064 bytes**。optimizer-loop末条elapsed **240.04818938秒**包含Python/logging；result timer **256.27566592秒**从initial dev之后计，含末dev与checkpoint保存；process wall **289.10937890秒**另含startup、identity、模型/strict-resume加载与两次dev。三个口径分别保存，不从它们推serving吞吐或speedup；没有重新测全形状资源保证。

**局限与下一决策。** 本段授权执行/严格恢复与TF学习信号已完成，早期32/128报告保持独立；没有选择checkpoint/policy、做final-test质量、拟合STS或执行独立dev随机rollout。旧pilot R06 GPU gate不能代表expanded512质量，原dynamicBF16问题与高效多请求/异步资源机制仍各需证据。后续1024/1280需另行授权并核身份/状态；当前native varlen最终模块/probe尚未交付，不写提前完成。记录者只核证据与notebook，不开GPU、不改实现/进程/冻结快照，不commit/push。
