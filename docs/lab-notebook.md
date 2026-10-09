# 实验日志

最近记录核对：**2026-10-09（S35 finite physical-B family的CPU基础与复审）**。experiment_journal 此前已恢复并接回唯一编辑权；本轮再次唤醒因 agent thread limit 失败，root 将 S32及后续轮次的唯一临时编辑权交给 sol_data。历史交接与各轮来源保留，root 负责最终审核与提交。记录者不操作 GPU/进程、不改实现、不读 final test 或 private 样本；远端大型 checkpoint/tensor 的核验事实引用已有留证并标明来源。本文持续追加：修正旧判断时保留原结论及修正依据，历史证据与实时状态分开。

早期研究问题：冻结 Qwen3-0.6B target 后，并行 DSpark 草稿能否比带 KV cache 的 target-only greedy 更快地产出完全相同的 token？训练可运行、loss 下降、回退输出一致，各自只回答这个问题的一部分。早期阶段门槛见[实验计划](experiment-plan.md)，下面历史实验的协议与失败口径不回改。

当前完整目标见[忠实复现范围](dspark-reproduction-scope.md)：还需随机分布恢复与非预知 admission、confidence 校准、跨活跃请求的全局验证 budget、可变长度多请求执行及两步历史异步容量机制，并在真实负载/相同物理工作量下评估系统收益。Greedy correctness 与 canonical 数值 control 是分层检查，不能缩小或替代这个完整目标。

公开链接只指向聚合报告。生成的 prompt、回答、精确 token 轨迹、机器路径、权重、优化器状态及凭据不进入日志。`output/` 是 **未发布的本地证据**。已有实验早于首个项目提交，实际身份由当时 manifest 中的源码、配置、数据、模型哈希绑定；后来的 Git 基线不被描述成启动这些历史实验的 commit。

复盘索引：

- 数据质量/长度：[P06 正式审计与拒绝](#data-final-audit)、[T04 模板与实际输入长度](#expanded-resource-gate)。
- Hidden-state/训练对齐：[D03 同轨迹学习诊断](#aligned-learning)、[T05 首段训练与 TF 边界](#expanded-step32)。
- Confidence/校准：[R12 冻结STS与raw/constant对照](#sts-fit-eval-result)。
- 数值差异：[C02 dynamic BF16 调查](#dynamic-numerics)、[C06 canonical 真实 drafter control](#canonical-drafter-gate)、[R08 quality128 TV](#expanded-quality128)、[S05 varlen 诊断准备](#native-varlen-diagnostic-cpu)。
- KV 正确性：[C03 缓存内容/回退](#kv-correctness)、[R04 随机路径提交](#stochastic-cache)、[S03 多请求隔离](#packed-isolation)。
- 资源与调度：[M01 两周期/Pareto 设计](#memory-gate)、[T04 实测显存](#expanded-resource-gate)、[S01 异步机制范围](#scheduler-scope)。
- 正式benchmark与计时边界：[S20 vLLM完整六case](#vllm-formal-benchmark)、[S21 native full64局部成本](#native-full64-profile)、[S22 matched native E2E准备/执行状态](#native-e2e-cpu-prep)、[S24 完整native E2E与vLLM比较](#native-e2e-result)、[S29 同backend target-only控制的CPU验证](#packed-target-only-cpu)、[S32 配对profile的两项P1修复与Linux CPU验证](#paired-profile-cpu-repair)、[S33 trace限额失败与部分诊断](#paired-profile-trace-limit)、[S34 signature成本与缓存安全否决](#signature-cpu-safety)、[S35 finite actual-Q family的CPU基础与兼容失败修复](#query-family-cpu-foundation)。
- 持久KV与设备身份：[S23 committed/scratch事务、CPU device alias复现修复及native/graph缺口](#persistent-target-kv-cpu)、[S25 native capacity tail与gather+attention真实capture/replay](#native-capacity-graph-result)、[S26 全部HF Qwen层的persistent事务CPU集成](#persistent-full-qwen-cpu)、[S27 随机session接入、bucket失败与feature生命周期](#persistent-session-cpu)、[S28 full-target graph的CPU准备与三项阻塞审查](#persistent-full-qwen-graph-cpu)、[S30 完整target真实graph保真](#full-target-graph-result)、[S31 配对R1完整请求负收益](#paired-r1-result)。

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


<a id="expanded-rollout-panels"></a>
## 2026-10-09 01:32（UTC+8）— R07：quality32 collector 与预冻结 development 分组，尚无 GPU rollout

**问题与方法。** 训练TF指标不能直接选出有用的随机drafter，需对兼容expanded checkpoints复用固定development panel/种子，同时将之后STS的fit与eval隔开。直接读取[rollout协议](rollout-collection.md)、`rollout_protocol.py`及本地未发布 `output/rollout-collector-cpu-20261009/` 的verification、final tests、dry128/512 log/aggregate、source SHA与private panel的身份/分组字段，不读final test。实际CPU部署为baseline `8a5d24c50f91ec55d8706b677279942e2c9027f4` 的git archive加三个交付module/test；后续归档commit `a278e5a5d7d74a1f77f8dcd475700959ee158cb0`（01:28:50）。四个module/test/doc执行SHA与该commit blobs独立核对一致。

**冻结选择与身份。** 必须恰为119个accepted validation：immutable records顺序前32为quality，其余87按 `SHA256(salt + NUL + record_id)` 排序，前44为fit、后43为eval；salt固定为dspark-expanded-dev-sts-v1。记录完整row/prompt-token/规范化首user身份唯一性、数据/generation/model哈希及每prompt seed，种子不依赖checkpoint。记录者直接核manifest的32/44/43计数、119 IDs互斥，并重算unsigned canonical JSON SHA为 **`fd8efdd09f58c20a24e4c248532bb1460282b6eadb2e73d51754e9acaeeaa2ec`**。改变rows/split/来源/静默replacement会拒绝；checkpoint weights/metadata另绑定，不能混checkpoint标签。全部119dev已用于TF监测，eval43只是相对STS fit的prompt留出，不是未经模型选择影响的数据。

**协议与分母。** quality执行协议为nativeBF16/SDPA、draftFP32+BF16 AMP、temperature1、无filter、actual float64 q、max128输出、完整block只因预算截短。预选quality rows0/1的round0做same-prefix TV，EOS导致未到时不换例子/轮次。分别保存proposed、target-verified、attempted-uniform与effective prefix-label分母；拒绝后的已验证suffix为零prefix事件，未生成/未评分尾部缺席；accepted EOS自身计入，其后排除，residual/bonus EOS不冒充accepted draft EOS。首token EOS或budget1产生零blocks，不填造零标签。CPU有限/形状/cache/实际q与标签证据不替代真实GPU分布保真或性能。

**直接观察。** final测试 **9项通过、5.517秒**，HIP/CUDA/ROCR隐藏、OMP2；覆盖exact partition/leakage、checkpoint-independent seeds/identity、stdlib dry-run/tamper、真实tinyQwen实际q/repeat/boundedTV、rejection/EOS分母、budget1/firstEOS、timeout与partial probe evidence。真实expanded128/512文件binding dry-run均dry_run_complete、32 prompts、gpu_touched=false，共用上述panel；两次aggregate的completed prompts/blocks/positions均0、execution_checks_passed=null、TV rows0，不能写成32真实rollouts通过。128 binding为052f737b…，512为4a76b220…，不同checkpoint绑定不被说成相同run身份。原始日志中的两条 `(null)` 提示保持记录。

**局限与下一决策。** 交付的是quality32 collector；尚未GPU quality采集或STS拟合。fit44/eval43预留给quality评估后选定并冻结的同一checkpoint/同protocol；必须用fit44估计每位置prefix prevalence常数、在eval43比较unscaled head/frozen STS/fit-prevalence的ECE/Brier，不能用constant-zero替代该校准基线或把fitting ECE当留出性能。选择checkpoint仍待真实quality结果，final test保持锁定。新collector源码在独立执行snapshot，不更改796fecc训练快照；记录者未执行GPU/源码修改/commit/push。

<a id="native-varlen-failure"></a>
## 2026-10-09 01:32（UTC+8）— S04：varlen CPU adapter通过，唯一真实native首组数值gate失败

**问题与方法。** S03 denseQ×K参考没有实现高效native varlen；新adapter将请求末段query和active KV映射为THD/int32累计长度，在HF attention callback中保留Qwen QKV/QKnorm/RoPE/cache语义，每层一次native varlen调用，无dense fallback。直接读[varlen说明](varlen-target.md)、[不可变失败报告](../reports/native-varlen-probe-20261009/README.md)、[summary](../reports/native-varlen-probe-20261009/summary.json)、[预声明protocol](../reports/native-varlen-probe-20261009/protocol.json)，及CPU/controller/raw failure/completion/worker-result/stderr和pre/postflight小型证据。实现/probecommit `d45442722aecb0f555df35b1f66feb8c7b44c01f`（01:22:59），失败聚合commit `6fa63e5d5d95b0f45a945615c2bcf4a0f6812416`（01:28:50）。

**CPU前提。** CPU FP32显式注入dense oracle kernel，5既有packed+5新增varlen共 **10 tests通过、1.253秒**；另 **3 controller tests通过、0.012秒**，覆盖stdlib dry-run禁backend imports、timeout只结束自有group、interrupt回收。前者核terminal-suffix layout/bottom-right可见性、active gather排除inactiveKV、tinyQwen hidden/context/logits/逐层KV内容、mixed Q/crop/exit/readd/cropzero、poison与kernel异常清状态，仍不是native GPU成功。记录者独立核CPU execution中4个sourceSHA与d454427 blobs相同；gate固定atol0.02/rtol0.02及RMS≤0.005，不因失败调阈值。

**真实尝试与失败。** 早期preflight把现有桌面render/card句柄误判为compute冲突，在native worker创建前拒绝；随后经授权修正guard，只拒绝新增compute owners并保留桌面inventory。原guard失败保留，这不算第二次native尝试；桌面和ASR未停止。真正native仅 **1次**，gfx1201/Torch2.12.0+rocm7.2/HIP7.2.53211，BF16/Hq16/Hkv8/D128/window(-1,0)/GQA。首组query lengths[1,8]、activeK[17,29]、inactive13；queryshape[9,16,128]、gatheredKshape[46,8,128]。native输出shape/dtype/device/finite通过，single `aten::_flash_attention_forward` assertion及两个FP32 MATH oracles彼此一致的检查在数值失败前通过；这些dispatch事实由源码执行顺序/traceback支持，原profiler event map未落盘。AOTriton是ROCm偏好getter结果，不是独立device-kernel trace。

与per-request FP32 MATH oracle比较时，**17841/18432=96.7936%**元素超fixed tolerance，max绝对差 **3.7750649452209473**、max相对差33742.24609375；raw traceback/failure JSON与公开summary一致。process **exit1、no timeout**，controller wall11.59582493秒。尚未到RMS assertion、native-versus-packed独立比较、causal ramp sentinel、active/inactive poison、crop-exit-readd和cropzero两组；RMS与peak allocation为null，不能填造测量。误差幅度本身不定位causal alignment、kernel或layout根因，也不宣称算子普遍不可用。

**身份与终态。** 本地source archive SHA `9114c062aab7586bb906ca1df27ca21d57892e164c96da7279def9eac857a8c5`与summary一致；九个raw evidence SHA均有对应本地文件且一致（guard-rejection SHA对应最初controller log，不是preflight JSON）。completion和failure确定已终止失败；worker-result保留初始running/empty cases是尚未append便抛异常的partial文件，不能当正在运行。postflight留证worker PID已不存在、compute句柄只剩原ASR、ASRready=true/busy=false、原PID保留；before/after VRAMused均8947318784bytes。post-integrity记录157个archive文件不变，796fecc训练源码/config未改；本轮记录者未再live查设备/ASR。

**局限与下一决策。** 不重跑、不fallback、不松阈值、不安装环境，不从11.6秒推性能；此native路径未通过correctness，不能用于后续质量或速度主张。保留CPU语义通过与native数值失败两层，不把缺失ramp/poison/crop证据说成通过。下一次诊断须新协议/另协调GPU，在原同输入双oracle上拆分数值与dispatch原因，最终仍需full-Qwen/KV/物理gather-attention-crop门槛；没有checkpoint、final-test、校准或async调度收益结论。记录者仅追加日志，无GPU/实现/进程/commit/push操作。


<a id="expanded-quality128"></a>
## 2026-10-09 — R08：expanded step128 quality32 真实随机 rollout 完成

**问题与假设。** R07 的 CPU collector/dry-run 与训练 TF overlap 不足以判断实际连续 draft 接受；需对预冻结 quality32 真正运行同协议，分别记录执行正确性、模型质量和 native BF16 数值保真。

**方法与身份核查。** 本轮直接读取[八份公开聚合报告入口](../reports/expanded-quality128-20261009/README.md)及 aggregate/audit/per-prompt/numerical-probes/protocol/runtime/source-identity，核八份文件 SHA 全部等于本地未发布 `output/expanded-quality128-20261009-a278e5a/public-verification.json`。30个package SHA 独立与 `a278e5a5d7d74a1f77f8dcd475700959ee158cb0` Git blobs 比较全部一致，worker/package inventory 一致；协调者另行独立核查也一致。公开报告归档commit为 `6e5c777`（协调者报告，本轮尚待push），执行source仍为a278e5a，归档提交不冒充执行身份；远端实际 deployment 比较及私有字段扫描依据 verification 留证，不冒称记录者重新查远端。archive SHA `669b48cd2a11693422f34b798607dbae307bafd6de0af2d2b461fdf3afc636bf`、panel manifest `fd8efdd09f58c20a24e4c248532bb1460282b6eadb2e73d51754e9acaeeaa2ec`、binding `df2366c25b0ef9c2e72136ef1dd32b1cb4c092ba0412ab58bbb137f1004e7a24` 固定；实际是 expanded step128 权重，不是旧pilot aligned128。沿用 native BF16/SDPA、draft FP32+BF16 AMP、temperature1/无filter、实际float64 q、block7仅预算截短、max128及checkpoint无关prompt seeds。

**执行正确性观察。** 已完成32 prompts，worker/launcher/controller均有exit0留证；全部observed finite/shape/cache/probability/EOS/budget checks通过，独立标签/计数及cache delta复核通过。actual q 为float64，最大q/p归一化误差分别6.66134e-16/5.55112e-16。runtime pre/post记录01:31:35–01:36:46，ASR HTTP成功/ready/not busy、原compute occupancy保留、设备used VRAM前后均8962183168 bytes，peak allocated2267426816 bytes；这是已有留证，本轮未做live查询。没有运行失败，执行通过与数值保真问题分别保留。

**模型质量与分母。** 32例共输出 **3874 tokens**（5例EOS），**3268 rounds**、**581 accepted draft tokens**，平均 **0.17778458 draft/round**。本轮从per-prompt重算输出、draft和各位置分母，与aggregate一致。proposed/target-verified/effective prefix labels各 **22415**，attempted acceptance uniforms **3842**；首拒绝后的已验证suffix是零prefix事件，不能以attempted分母替代effective。accepted-prefix长度0/1/2/3的round数为2749/461/54/4，4–7均0；位置1/2/3存活事件为519/58/4，分母3268/3247/3223，位置4–7均0。7个全接受block都是预算截短，没有全接受7-token block。全部5个EOS来自residual、accepted draft EOS与bonus EOS均0。每例接受draft为1–31；每轮提交均值3842/3268=1.17564259，含预算缺失bonus，不能机械写为1+draft/round。验证query rows为25683，含old anchor+n proposals，排除prefill。

**数值观察与未解决问题。** 预冻结quality cases0/1 round0的16个n+1行，本轮重算 **16行TV全非零、max TV=0.045764141753646695、argmax改变0行**，与aggregate一致。同semantic prefix的native block和fresh sequential概率存在差异；argmax相同不等于随机law相同，CPU residual恢复实际verifier p不证明其等于sequential target p。此次没有解决dynamic BF16数值问题，也不是varlen失败的复测。

**局限与下一决策。** 这是native协议下的有界quality32观察，不是speed benchmark、答案事实质量或native distribution无损证明；rounds不当作独立统计重复，未给置信区间。fit44/eval43未采集/拟合；全部development已用于TF监测，eval43仅对STS fit留出。该窗口未启动step512 collection，不能写已有128/512 rollout比较或已选checkpoint。后续以同冻结panel/protocol比较另checkpoint，再冻结选择并分别采fit/eval做STS及fit-prevalence基线。final test未读，未公开私有样本/机器路径；本轮仅改notebook。

<a id="native-varlen-diagnostic-cpu"></a>
## 2026-10-09 — S05：六调用 native varlen 对齐诊断的 CPU 准备，GPU 未执行

**问题与假设。** S04 原生首组数值gate失败需定位原因，不能改阈值或fallback掩盖。直接读[独立诊断计划](native-varlen-diagnostic-plan.md)、本地未发布 `output/native-varlen-diagnostic-cpu-20261009/{manifest,dry-run}.json` 与 tests-final.log。静态源码链从Torch wrapper window(-1,0)、adapter window转换到AOTriton literal right=0，预测public cached suffix可能使用upper-left而非lower-right边界；这是待检验假设，revision对应源码不证明实际binary输出或失败根因。

**方法与身份。** 新诊断独立于生产adapter与原失败gate，冻结protocol SHA `c370831c863364f0da90a658e7bb2935ec937293016a95ba0f5393967bff8de8`。本轮独立核manifest中script/test/plan三个SHA均与当前文件一致，tests_log SHA匹配tests-final.log；较早tests.log不匹配该最终日志SHA，保留两份而不混称同一次测试。manifest保存installed/static upstream source SHA与revision，记录者未重新获取/运行这些backend源文件。dry-run status为 `dry_run_no_backend_import_no_gpu`，manifest gpu_executed=false。

**六调用固定矩阵。** 复现原seed/FP32 K→V→Q抽样顺序，再转BF16；Q长度[1,8]、active K[17,29]、inactive13、Hq16/Hkv8/D128。三variant各跑random与zero-Q/position-ramp一次：public GQA、public复制KV到16heads/noGQA、private ATen causal且windows=None。每个输出同时对照FP32 MATH upper-left/lower-right，ramp另记full-attention fingerprint；private variant是独立mapping control，不是失败后fallback。保持atol0.02/rtol0.02/RMS≤0.005与120秒timeout；先保存identity/dispatch，再保存每调用tensor/errors，数值不匹配继续预声明矩阵，backend异常/nonfinite/timeout则失败退出。FP64 error reduction与固定1e-30相对误差floor只用于诊断标量，不改变FP32 oracle。

**CPU观察与失败保留。** final日志 **7 tests通过、5.326秒**，GPU可见变量隐藏；覆盖stdlib默认guard、timeout/interrupt只回收自有child、原seed/draw顺序、GQA与复制KV同reference、ramp区分边界、大finite error可JSON保存、数值失败仍保留raw tensors/metrics。日志两条 `(null): No such file or directory` 未定位，原样保留；七项最终均OK。这些是CPU oracle/controller准备证据，没有六次native GPU输出、profiling或alignment结论。

**局限与下一决策。** 原S04 failed gate仍失败；不将源码推测写成已确认upper-left，也不把private control写成修复方案。只有另行GPU授权、独立source/archive身份与新证据目录齐备后才执行矩阵；public对upper-left而private对lower-right仅支持该runtime/shape的mapping解释，GQA差异或两oracle皆不符需继续定位。诊断完成也不自动提升为full-Qwen/KV正确性或性能成功。无模型/checkpoint/final-test输入，本轮未启动GPU/训练/部署、未改实现或commit/push。

**归档与独立复核补记。** 诊断源码、测试与协议已归档为 `cd317f7`。Astra core 独立复核报告无阻断项；隐藏 GPU 重跑 7/7 CPU tests 通过，额外 CPU 替身 worker 检查六调用顺序/参数、数值 mismatch 后继续矩阵、原始证据先保存及 production_gate_passed 保持 false。证据保存在未发布 `output/native-varlen-independent-review-20261009/`。本补记引用独立审查交付，未把 CPU 替身称为原生 GPU 验证；本轮仍未启动新 GPU 诊断。


<a id="native-varlen-diagnostic-gpu"></a>
## 2026-10-09 01:55（UTC+8）— S06：唯一六调用 GPU 诊断支持本 runtime/shape 的 explicit-window mapping 解释

**问题与预声明假设。** S04 的大数值差异本身没有定位根因，S05 静态源码链只是待检验的window mapping假设：public wrapper的literal right-window0可能对cached terminal suffix使用top-left，private无window的causal special case可能使用需要的bottom-right。应在同输入同时对照两种FP32 MATH oracle，以GQA/复制KV和解析ramp分辨解释，而不是修改原gate阈值或失败后换backend算通过。

**方法与固定矩阵。** 直接读取[七份公开诊断报告](../reports/native-varlen-diagnostic-20261009/README.md)及aggregate/audit/input-identity/protocol/runtime/source-identity，另读本地未发布 `output/native-varlen-diagnostic-20261009-cd317f7/public-verification.json`、archive和worker/launcher/controller终态。唯一授权matrix含三variant×两inputs，恰六次native调用：publicGQA、public复制每个KVhead两份/noGQA、privateATen `is_causal=true/window_left=None/window_right=None`，各跑原random和zero-Q/local-position ramp一次；private始终是预声明mapping control，不是fallback。保持seed20261009、原CPU FP32 K→V→Q抽样再转BF16、Q[1,8]/K[17,29]/inactive13、Hq16/Hkv8/D128、scale1/√128/dropout0，以及atol0.02/rtol0.02/RMS≤0.005、timeout120。数值不匹配保留并继续既定matrix，backend异常/nonfinite/timeout另记失败。没有补跑或松阈值。

**实测观察。** 六例输出均finite且shape/dtype/device通过，逐例已保存operator map，各恰一个 `aten::_flash_attention_forward`。四个public结果均仅匹配top-left，两个private control仅匹配bottom-right；diagnostic_completed/三层exit0不改 `production_gate_passed=false`。原random的public GQA及重复KV相同：对top-left maxabs **0.0076074600**、RMS **0.0009926664**，对bottom-right maxabs **3.7750650644**、RMS **0.6740736880**、17841/18432超阈值。private random对bottom-right maxabs **0.0042855740**、RMS **0.0006106310**，0元素超阈值，对top-left仍不匹配。

解析ramp `V=local_position/32`：public输出与BF16 top-left均值逐bit相同（首请求0，第二请求0..7/64）；private与BF16 bottom-right均值逐bit相同（首请求0.25，第二请求21..28/64）。对FP32 oracle的tiny非零误差来自reference averaging口径，不把oracle误差写成解析指纹不一致。random和ramp两种输入的publicGQA/重复KV输出hash均相同，audit另逐bit核相同；该矩阵不支持GQA grouping是这一次失败原因。结合revision-matched源码链，支持**当前pinned ROCm runtime、原首组shape**中explicit right-window0映射top-left，而private无window causal special case映射bottom-right的解释；不泛化到其他形状、runtime、架构或全Qwen。

**身份与独立核验。** execution source为 `cd317f79a572dc976ddf660f3bcb2d4a4724635d`，archive SHA **`3537bf3185e1b5f5ae07f477bb4f14c13db6d97c648c1633900c8e7caad58c40`**、script SHA `ac47f2ba1e7b1a940f3a0e87081b6063bf9d199f51fa9f62ac654a467254962f`、protocol SHA延续S05的c370831c…。记录者独立核七个public文件SHA与public-verification一致、31个worker源码SHA逐个与该commit blobs一致、archive与本地保留件相同，重核6例分类/flash计数及两个public raw-output哈希相等。GPU-hidden CPU审计的 **49 tensors hash/shape/dtype通过、原input按seed/draw顺序重构exact**、analytic ramp逐bit核与before/aftersource相同等事实依据公开audit和verification留证；完整tensors仅远端私有，本轮不冒称记录者再加载全部raw tensor。raw output/dispatch先保存后classification由源码审查/独立CPU流程测试支持，与S04缺失event map的失败轮次分开。

**终态与隔离。** worker completion exit0/no timeout、launcher return0、controller return0均有本地留证；runtime pre/post为01:48:44–01:48:54（UTC+8），三diagnostic进程已退出的事实据既有postcheck，非本轮记录者live轮询。peakallocated **81909248bytes**，pre/postVRAMused均8962183168bytes，compute句柄只保留原ASR、ASRHTTP正常/ready/notbusy，桌面render/card活动未改。Torch2.12.0+rocm7.2、HIP7.2.53211/gfx1201，AOTriton偏好与nativeATen dispatch已记录，但确切devicekernel未独立追踪。9.55196299秒含profiler/oracles/persistence，不是benchmark。

**结论边界与下一决策。** 原S04production gate继续失败；本次完成诊断不证明整个Qwen/KV修复、全部cached/poison/crop形状通过、模型分布无损、质量或性能。生产adapter/backend默认没有由这次diag改变。若后续明确pin private backend，仍必须先以原三组tensor gate/原阈值全部验证，再做wholeQwen/KV及实际物理工作/成本检查，CPU实现授权也不等于这些GPU门槛已完成。协调者报告step512quality正在执行，结果未到，本条不写完成或checkpoint选择。公开报告归档为 `e32fbb1`（协调者已核 push 成功），不得冒充execution cd317f7；scope/README入口更新也不等于deployment或新实验。记录者只续notebook，未改root scope/core源码、开GPU、操作进程或commit/push；final test未读。


<a id="expanded-quality512"></a>
## 2026-10-09 02:02（UTC+8）— R09：expanded step512 quality32 完成，同panel配对接受量提升

**问题与方法。** R08只观察expanded128，需要在不变panel/source/protocol下比较中间checkpoint512，不能靠改变temperature、budget、seed或删失败prompt制造接受收益。直接读取[八份quality512报告](../reports/expanded-quality512-20261009/README.md)及aggregate/audit/per-prompt/numerical-probes/protocol/runtime/source-identity、本地未发布 `output/expanded-quality512-20261009-a278e5a/public-verification.json`，并逐字段对照R08。八份public SHA与verification一致；30个executed package SHA与a278e5a Git blobs/128worker逐个相同，panel文件SHA b5119c7f…、manifest fd8efdd09f58c20a24e4c248532bb1460282b6eadb2e73d51754e9acaeeaa2ec、data/target identity与32case ordinal/seed均相同，protocol文件bytes完全一致。此前CPU512dry-run的inventory较早；正式执行前在128的同一archive上fresh dry-run通过的事实据报告/verification，未将旧dryrun binding冒充此次执行。

仍为nativeBF16/SDPA、draftFP32+BF16 AMP、temperature1、无filter、actualfloat64 q、fullblock7只因预算截短、max128。checkpoint weights SHA为 `667d2dd6e8ad7d11e2115e936af1b6cf7e44357d68e3412f5c9ef433f26690ae`，metadata SHA `30d8813eaa5a4db7a50ddbf164f17b8c045e79ae38434e378aa8e703bc699ddb`（独立核本地T07metadata同SHA），本次binding `41d39d07cd6c8a55efab5ea2ddd75e39f55e7a570e3de264ae7c278379db4b56`；不同checkpoint完整run绑定仍不同。execution source仍 `a278e5a5d7d74a1f77f8dcd475700959ee158cb0`，公开归档本地commit为 `329bb049a14a142d58383a4e4ebda4df8331382f`（02:00:29，协调者随后工具核实push成功），不将归档commit当execution身份。

**执行观察。** 32/32完成，worker/launcher/controller各有exit0、no timeout；2620observed blocks的finite/shape/actualprobability/cache/EOS/budget检查通过，actualq为float64，最大q/p normalization误差6.66134e-16/5.55112e-16。运行资源留证pre/post01:51:11–01:55:32（UTC+8），三collector PID均已退出、只有原ASR持compute句柄，ASRHTTP正常/ready/notbusy，VRAMused前后8962183168bytes，peakallocated2267403776bytes；本轮依据已导出的runtime/postcheck，不重新live轮询。

**配对模型行为。** Step512共 **3941 outputs/2620rounds/1299accepted draft**，draft/round **0.495801527**，对照128为581/3268=**0.177784578**。记录者独立从32个per-prompt重算所有总数、histogram和prefix分母，并按每prompt等权重重算macro：128 **0.176533098**、512 **0.528943552**；31例ratio上升、1例下降、0例持平，paired delta中位数 **0.321620822**。这支持固定native协议/panel下接受行为的更广改善，保留下降例子，非答案事实质量或置信校准结果。相同prompt/seed不保证跨checkpoint相同已生成prefix；3例EOS outcome改变，输出3874→3941、EOS5→4，不能用round/work数量比当speedup，rounds也不当独立重复给CI。

**Prefix与四分母。** prefix length0..7的round histogram为 **1814/515/173/74/24/8/4/8**，加和2620，按长度加权1299；8个全7接受block来自2515个7-token proposal，长prefix仍少。各位置prefix event/count为 **806/2620、291/2602、118/2580、44/2563、20/2547、12/2529、8/2514**。proposed与target-verified位置各 **17960**，effective labels **17955**；一次accepted draft EOS位于第2候选，其自身与第1位置仍保留，之后5个已验证尾部排除。四EOS为3residual+1accepted，bonusEOS0；未观测预算尾部不补零。

attempted uniforms **3901=1299accepted+2602rejected rounds**，不能替代17955prefix-label分母。initial32draw之后实际committed为 **3909=3941−32=1299+2602+8bonus**，与3901attempted不同。17个全proposal接受block中9个达到budget不抽bonus，另acceptedEOS round也不抽bonus，故不机械以rounds+accepted算commit。实际commit/round为 **1.491984733**，target verification query rows为 **20580=17960+2620**、含每轮oldanchor但不含prefill；这些是执行计数，不是同物理负载的时延比较。

**数值保真。** 预选cases0/1 round0共16rows，记录者逐行重算 **16行TV全非零、maxTV0.05288759555199574、argmaxchanges0**，与aggregate相同。128maxTV0.0457641418对应另一组sampledprefix，不能把max差写成checkpoint对target数值稳定性的受控因果作用。argmax不变不等于随机law不变；execution通过仍不证明nativeBF16 block p等于freshsequential p。未解除旧数值问题，也不是新pinned varlen backend的gate。

**局限与下一决策。** Step512仍是max1280训练计划的中间checkpoint；只支持复核下一段有界训练，不将它写成最终选定checkpoint，不改变panel/seed/policy。协调者已授权strict512→1024，但本条实际启动PID/结果尚未交付，授权不作完成证据；1024/1280在本条未记录启动，后续训练另起T08。fit44/eval43采集与STS/fit-prevalence常数留出比较仍待后续冻结选择。全部development已用于TF监测，eval43只是相对STSfit留出；final test未读。性能、native分布无损、outputquality及全局异步调度收益未测。记录者仅改notebook，不运行GPU、改实现/进程或commit/push。


<a id="pinned-varlen-cpu"></a>
## 2026-10-09 02:10（UTC+8）— S07：显式 pinned private backend 的 CPU 准备与 inference-mode guard 修复

**问题与方法。** S06诊断支持原shape/window mapping解释，但不能靠private control将原S04 gate改成成功。新backend必须显式opt-in、保留public默认/旧失败，并先核精确runtime/input契约和原完整三组tensor gate的控制流。直接读[pinned backend说明](pinned-rocm-varlen.md)、`rocm_varlen.py`对应guard/test、未发布 `output/pinned-rocm-varlen-cpu-20261009/` 的manifest、final tests、pinned-protocol，以及direction独立review目录的pinned/new-public guarded dry-run输出。源码归档 `5281a6af05ddd6b3e80cd3aeded892c61d2c91a5`（02:07:47）；记录者独立核manifest五个sourceSHA逐个与该commit blobs相同、final tests log SHA一致，并重算pinned protocol JSON SHA。

**显式契约。** 默认仍public `native_varlen`；`rocm_aten_no_window_pinned_v1`只经显式选择进入。pin Torch2.12.0+rocm7.2、torch git7661cd9c…、HIP7.2.53211、gfx1201、AOTriton preference、未设置CK preference环境及精确privateATen schema，不匹配即拒绝；pin不是独立devicekernel binary认证。只接收selectedGPU contiguous BF16 THD/Hq16/Hkv8/D128、int32累计长度、非空terminal suffix、scale1/√128、无autograd，调用privateflash dropout0/is_causal=true/windows=None及其余optional controls=None。异常直接传播，不mutate backend preference、不重试或fallback；CPU注入与private显式选择互斥。

**发现的合法推理误拒与修复。** 协调者review发现原guard直接读tensor `_version`，而`torch.inference_mode`创建的张量没有version counter，会使合法推理在native调用前意外抛错。修复后normal/no_grad累计tensor仅在同layout/同version时复用校验，inference tensor读version的RuntimeError转为“无version”，每call重新核累计values/metadata，不把它默认为immutable cache。只留最近layout、不无限增长。直接读新增normal/no_grad/inference-mode测试，三模式合法两次调用后mutation被拒；requires_grad guard另测拒绝。这里修复的是输入校验与运行契约，不证明native attention数值或原模型问题已经修复。

**CPU与独立dry-run观察。** final **25tests通过、0.530秒**，HIP/CUDA/ROCR隐藏、OMP2：10新pinned tests+5varlen+3controller+7diagnostic；覆盖所有runtime pin字段、exactATen参数、tensor/scale/cu/autograd边界、version/inference mutation、public默认与选择互斥、operator异常无fallback、原完整gate **11-call CPU替身**和首失败raw output/oracle/metric保留。两条 `(null)` 提示保留，最终全部OK。direction独立review报告stdlib guard禁backend imports的dry-run通过、无阻断；本地pinned/public输出status均 `dry_run_no_torch_import_no_gpu`，记录者直接读取，不冒称是真privateGPU execution。

**协议身份与证据顺序。** 原public protocol SHA **08c501349bb59d30a888f1a5669a6c5bcb0bdfbf5f9e894c409fb2e702d49ff9**未变；新pinned SHA **f42729a4e857b1e1f0023888ba1f5c0e86602b36e764243d52007f96866cb98a**显式绑定backend/runtime/windowNone/privateentry及旧protocol hash。原三组shapes/draw顺序、FP32双oracle、atol0.02/rtol0.02/RMS≤0.005、ramp/active/inactivepoison/crop检查都保留；11次native调用只在真实原assertions全部通过时发生。新增atomicJSON与每call先保存QKV/cu/raw output、oracle/scalar再assert，是失败留证修复，不松数值标准，不重写旧archives。CPU替身11-call通过不能代替真实full tensor gate。

**局限与下一决策。** 协调者报告1024训练当时仍live，完整tensorGPU仅获等待释放后的条件授权；manifest GPU_executed=false，S07不写已跑。原S04失败和S06的production_gate_passed=false保持原结论，显式新backend没有生产晋升或wholeQwen/KV通过。后续先按新绑定/原阈值执行完整tensor gate，再冻结实际pretrainedQwen/KV的结构与数值报告协议；拟议wholemodel synthetic schedule不是已交付/已运行结果，观察后不得临时发明BF16pass阈值。没有quality/STS/性能或final-test结论；记录者只notebook，无GPU/实现/进程/commit/push动作，实际tensor结果另记新条。


<a id="expanded-step1024"></a>
## 2026-10-09 02:14（UTC+8）— T08：expanded strict resume 512→1024 与动态标签基线

**问题与方法。** T07完成512、R09观察同panel接受行为改善后，下一段仍须保持冻结训练身份，实际恢复weights/optimizer/RNG，不能把TF信号当rollout1024结论。直接读取[五份本段报告](../reports/expanded-training-step1024-20261009/README.md)、[summary](../reports/expanded-training-step1024-20261009/summary.json)、[checkpoint核验](../reports/expanded-training-step1024-20261009/checkpoint-verification.json)、[source identity](../reports/expanded-training-step1024-20261009/source-identity.json)和公开steps513–1024 metrics，另核未发布 `output/expanded-training-step1024-20261009/` 的public-verification/result/completion/run/metadata/latest/full metrics及runner OS退出标识。执行仅获授权strict resume512→1024、`--stop-after 1024`；原source `796fecc0d9142c1a733ad6df47b105fab4ee9f05`、config SHA `ea8091ca8d124434696ccd15f88249839aaf1d71bdf540c3db9142bcbdda4762`、max_steps1280保持不变。

**直接观察与继承核对。** 02:03:15.985–02:09:24.680（UTC+8）training child exit0、runner OS exit0、timed_out=false；最外层shell wrapper没有由该verifier独立wait，不添第三个exit0。本段新增512 optimizer updates/4096 microsteps，累计8192/932=**8.7896995708遍**，本段增加4.3948497854遍。记录者从本地full metrics重核恰为1–1024、所有scalar finite，最初512行保留换行逐字节等于T07原文件，SHA `1d305e9d85b0ad33b62aa350649584ee60c0f1978edbc30501472e8f4e14c2bb`；公开512条新增metrics与raw513–1024解析记录逐项相同。run identity逐字段等于step512、metadata identity匹配，initial dev逐字段等于step512 final，仍非未中断连续训练的逐bit对照。

**Checkpoint与执行身份。** 公开checkpoint-verification和本地step1024-verification bytes/SHA一致，latest/metadata/resume-state均1024，报告记录62个Adam state entries的step全1024、source112/config bytes未变、preflight/run身份一致、weights/resume重hash通过。weights SHA `5b9984de9817531ca29544890dc2988a1c0aa3aff6866a828ec8a81a2641a765`、resume SHA `5d083a262a252baa19b0164e9737d172948093d755781d4cfd39f610260f3fd1`，与本地metadata逐项一致。记录者独立重算五个public SHA和summary所列五个raw SHA；source identity与512报告相同，21个训练package源码SHA逐个等于796fecc Git blobs。source112是远端完整树核验计数，不等于21个package模块；大型weights/resume-state的重hash/加载及完整source/config bytes核验依据direction留证，未声称记录者本轮加载远端大文件。公开归档本地commit `5a96d7d14eb0199166b7c498ded68899a4300a0d`（02:13:28）独立核实；执行身份仍796fecc，本条未查询push/CI。

**TF结果与confidence边界。** 同119dev、固定anchors、样本内weights加权再宏平均，512→1024 loss **2.191068616→2.064107772**、CE **4.807051978→4.370612417**、L1 **1.425409559→1.293027493**；teacher-forced软overlap **0.287295218→0.353486253**（28.7295%→35.3486%），supervised tokens宏平均213.87394958不变。confidence BCE **0.427494834→0.463321805**、MAE **0.184831666→0.208602766**绝对值上升，但overlap目标随draft改变，不能据此跨checkpoint称预测退化。**同step1024**的constant-zero MAE按同weights/宏平均定义为 **0.353486253**，learned MAE **0.208602766**较低；只支持优于零基线，constant-mean/median与真实rollout校准未测。35.3486%不是实测连续接受率、答案质量或confidence已校准，也不能替代独立quality32。

**计时与资源。** target frozen=true、161692161 trainables、932train/119dev、exact train sequence tokens424267、BF16 AMP/FP32 trainables；peakallocated **5632856064 bytes**。optimizer-loop末条elapsed **323.60860635秒**含Python/logging；result **338.22575630秒**从initial dev后开始，含末dev/save；child wall **368.69495988秒**另含startup、identity、模型/strict-resume加载、两次dev及保存。三种计时口径分别保留，无serving吞吐、独占全系统峰值或speedup结论。

**局限与下一决策。** 已完成的仅是本段严格恢复和TF监测；step1024仍为1280计划中间checkpoint，没有rollout1024、checkpoint/policy最终选择、STSfit/eval或scheduler收益。协调者随后授权1024→1280，但本条实际PID/结果未交付，不能把授权写成启动或完成。pinned private backend完整tensor gate据协调者报告已通过、公开报告仍在导出；本条不填初步数值、不替代S04原失败或wholeQwen/KV门槛，完整交付后另起S08。final test未读；记录者仅追加notebook，无GPU、实现/进程、冻结快照或commit/push动作。


<a id="pinned-varlen-tensor-gate"></a>
## 2026-10-09 02:19（UTC+8）— S08：显式 pinned private backend 完整小 tensor gate 通过

**问题与方法。** S07的CPU替身不是nativeGPU证据；本轮需按同三组输入与原阈值完成显式private backend全部11call，同时保持S04 public失败结论。直接读取[六份完整报告](../reports/pinned-varlen-tensor-gate-20261009/README.md)、[aggregate](../reports/pinned-varlen-tensor-gate-20261009/aggregate.json)、[CPU audit](../reports/pinned-varlen-tensor-gate-20261009/audit.json)、[protocol](../reports/pinned-varlen-tensor-gate-20261009/protocol.json)、[runtime](../reports/pinned-varlen-tensor-gate-20261009/runtime.json)、[source identity](../reports/pinned-varlen-tensor-gate-20261009/source-identity.json)，另核未发布 `output/pinned-varlen-tensor-gate-20261009-5281a6a/` 的public-verification、source archive、verification log及三层退出标识。这是唯一获授权的小tensor运行，没有重试、fallback或观察后松阈值。

**保持原协议。** 显式选择 `rocm_aten_no_window_pinned_v1`，runtime/schema pin精确匹配，is_causal=true、window左右均None。原seed20261009、CPU K→V→Q抽样转BF16、Hq16/Hkv8/D128、FP32 MATH per-request及packed bottom-right双oracle、atol0.02/rtol0.02/RMS≤0.005和exact isolation均不变。三组Q/K/inactive lengths为 **[1,8]/[17,29]/13**、**[3,2,1]/[11,5,1]/0**、**[2]/[2]/6**，包含cached tail、crop/exit/re-add和cropzero/append。原public协议SHA仍 **08c501349bb59d30a888f1a5669a6c5bcb0bdfbf5f9e894c409fb2e702d49ff9**，新pinned canonical JSON SHA **f42729a4e857b1e1f0023888ba1f5c0e86602b36e764243d52007f96866cb98a**；记录者重算后者，不混同pretty-printed公开protocol文件bytes SHA。

**真实观察。** **3cases/11nativecalls全部通过**，11call逐项记录恰一次 `aten::_flash_attention_forward`，finite/shape/dtype/device通过，raw先保存。三case对per-request FP32 oracle的maxabs/RMS依次为 **0.0042855740/0.0006106310**、**0.0071058273/0.0008891412**、**0.0077950954/0.0010230795**；packed oracle各也满足同原阈值。三个zero-Q位置ramp均与bottom-right解析均值exact（maxabs/RMS均0）；三项other-request poison和两项适用inactiveKV poison输出均逐bit不变。记录者独立核aggregate的case/call序号、11flash计数、3ramp与3+2isolation计数，不把scalar maxima alone当全元素allclose的独立重演。

**身份与审计证据等级。** 执行source **5281a6af05ddd6b3e80cd3aeded892c61d2c91a5**，archive SHA **233d3f493ce084b18761beccfe34c56f0002a56cd098b9c0c31d1873cbaec38e**，script SHA **0da7f5b52671257a0d33c08b0e55829132c0a3fa5c5753b0aabd7779f802915e**；worker/launcher identity一致、before/after未变。记录者独立重算六public文件SHA与public-verification一致、32个executed script/package SHA逐个等于该commit Git blobs、本地archive SHA相同。GPU-hidden CPU独立审计记录11份raw native tensor-file hash和3个重算FP32oracle通过、CPU/savedGPU MATH oracle一致，并重核ramp/poison exact；verification log保留一条 `(null)` 提示和最终independent_cpu_verification_passed。完整QKV/cu/output/oracle留在远端私有，本轮记录者没有再加载它们。协调者另报告已独立核六report/32Git/counters并直读remote completion及原worker result；其远端检查与本轮本地核验分开。公开归档commit **d35254fbf0ae95a4feef24fe7214e5385a5c424d**（02:17:38）已直接核Git，不冒充execution source或本轮push/CI证据。

**终态、隔离与计时。** Worker completion exit0、launcher return0、controller return0、timed_out=false均有本地标识并逐字段等于runtime。运行pre/post为02:11:32–02:11:44（UTC+8），peakallocated **82391552 bytes**、VRAMused前后 **8962183168 bytes**；既有ASR是唯一KFD compute持有者且HTTP正常/ready/notbusy，desktop render/card使用者记录并保留。全部gate进程退出依据既有postcheck/交付事实，本轮不live操作。Torch2.12.0+rocm7.2、torchgit7661cd9c…、HIP7.2.53211/gfx1201、AOTriton preference与private schema精确符合pin，仍非独立devicekernel binary认证。**10.75362557秒**含profiler/oracles/persistence，不是benchmark。无安装、targetweights或冻结训练快照改动。

**已知失败留证局限。** 实际5281a6a继续使用FP32 error reduction；极大的finite BF16差可能在square/mean等reduction中overflow，继而不能用allow_nan=false序列化scalar报告。raw native tensors在此之前已经保存，但不能因此声称该极端失败会留下完整scalar/oracle终态。本次未触发该局限；没有事后悄悄修改执行代码或将重写后的实现当此次原件。后续健壮化需另留source/检查证据，不改变本次成功范围。

**边界与下一决策。** 仅显式private backend完整小tensor gate通过，**原public S04 gate仍failed**、public默认不因本结果晋升；不证明whole pretrainedQwen/KV、native分布无损、quality、confidence/STS或speedup。WholeQwen尚无GPU执行；实际每层sameQKV固定0.02/0.02/0.005与精确cache/position/isolation的CPU准备仍在另行实现，end-to-end BF16 hidden/KV/logit/TV须分别报告。协调者已收到direction实际1024→1280启动PID（shell3919877/runner3919883/train3920175），训练拥有当前GPU窗口；这是启动证据，不是1280完成，本轮无并发GPU实验。final test未读；记录者只notebook，无GPU/实现/进程/commit/push动作。


<a id="expanded-step1280"></a>
## 2026-10-09 02:24（UTC+8）— T09：最终 strict resume 1024→1280，原训练计划完成

**问题与方法。** T08完成1024后，最后256步仍须严格恢复原训练身份与optimizer/RNG，完成预定max_steps1280不代表完整复现目标完成。直接读取[五份最终训练报告](../reports/expanded-training-step1280-20261009/README.md)、[summary](../reports/expanded-training-step1280-20261009/summary.json)、[checkpoint核验](../reports/expanded-training-step1280-20261009/checkpoint-verification.json)、[source identity](../reports/expanded-training-step1280-20261009/source-identity.json)及公开steps1025–1280 metrics；另核未发布 `output/expanded-training-step1280-20261009/` 的public-verification、root-public-verification、result/completion/run/metadata/full metrics、runnerOSexit及postrelease。本段只执行授权strict resume `--stop-after 1280`，source796fecc/config ea8091ca…/development records69daac1d…与max_steps1280未改，没有自动启动quality/STS。

**完成与继承观察。** 02:16:24.709–02:19:53.916（UTC+8）training child exit0、runner OS exit0、timed_out=false；最外层shell仍无独立wait证据，不添加第三退出码。本段256 updates/2048 microsteps，累计10240/932=**10.9871244635遍**，本段增加2.1974248927遍；**原1280-step训练计划完成**。记录者从full metrics独立核恰为1–1280、所有scalar finite，前1024行保留换行逐字节等于T08，SHA **b1aa6471ff73580e6afb394330b135134396cd73ab48b85975ff0135e1b2373e**；公开256条与raw新增段逐项相同。run与1024逐字段相同、metadata identity匹配，initial dev恰等于step1024 final，119dev/固定anchor seed和样本内加权再宏平均保持；不声称中断恢复与无中断训练逐bit等价。

**Checkpoint与hash核验。** Latest/metadata/resume-state均1280；核验报告记录62个Adam states全部step1280、source112/config/development records bytes未变、preflight/run identity匹配、planned_training_schedule_complete=true。权重SHA **d6f21ab3af5187ed181efdbde54118a9ada21b458d9957515a17d6b3693d0de4**、resume SHA **947809cc3a8ce0f4b49e6dc8a5e05dc4260a6ec00a56e507cd69a48d940d2280**与本地metadata一致；完整大文件重hash/optimizer加载及source112核验事实依据direction留证，记录者未再加载远端checkpoint。原public-verification没有列五public SHA；协调者额外root-public-verification绑定五份报告，记录者按该文件逐份重算通过，并重算summary的五raw SHA、核公开checkpoint-verification等于本地step1280-verification。source identity与1024报告完全相同。归档commit **2c3a6686e4c384f93668f1127fc315af01b0e51c**（02:22:46）直接核Git；execution仍 **796fecc0d9142c1a733ad6df47b105fab4ee9f05**，本条未查询push/CI。

**TF结果与动态目标基线。** 1024→1280 dev loss **2.064107772→2.029411892**、CE **4.370612417→4.252141251**、L1 **1.293027493→1.253140956**；teacher-forced软overlap **0.353486253→0.373429528**（35.3486%→37.3430%），supervised tokens宏平均213.87394958不变。confidence BCE **0.463321805→0.476370916**、MAE **0.208602766→0.214375955**绝对值上升，目标随draft改变，不能据此跨checkpoint宣布预测退化或校准变化。**同step1280** constant-zero MAE按同weights/宏平均定义为 **0.373429528**，learned **0.214375955**较低，只支持胜过零基线；constant-mean/median及真实rollout校准仍未测。37.3430%不是实际接受率、答案质量或confidence已校准，不能用最后训练步直接选最终checkpoint。

**计时、资源与释放。** target frozen=true、161692161 trainables、932train/119dev、train input tokens424267、BF16 AMP/FP32 trainables；peakallocated **5632856064 bytes**。optimizer-loop **160.13023932秒**含Python/logging；result **176.85806779秒**在initial dev之后开始并含末dev/save；child wall **209.20728205秒**另含startup、身份核验、模型/strict-resume加载与两次dev，三口径不推serving速度。02:20:51.044的postrelease记录shell3919877/runner3919883/train3920175已不存在（记录者另核process snapshot无三PID），KFD只保留ASR1208354且ready/notbusy，VRAMused **8962183168 bytes**、free **25246560256 bytes**；这是该时间点留证，非本轮live状态，不把退出即刻尚未完全释放的memory_after_exit误当独占训练峰值。

**局限与下一决策。** 训练schedule完成，完整goal仍未完成：没有真实quality1280结果、最终checkpoint/policy选择、STSfit/eval、final-test质量、native分布无损或speedup/全局异步系统收益。协调者已给direction quality1280条件授权，其正在准备；本条尚未收到实际PID，不写已启动或完成。Core的wholeQwen脚本仍在CPU测试，没有wholeQwenGPU结果；S08小tensor通过与S04public失败均保持。已用全部dev做TF监测，后续eval43只相对STSfit留出；final test未读。记录者仅改notebook，无GPU/实现/进程/冻结训练源码/commit/push动作。


<a id="expanded-quality1280"></a>
## 2026-10-09 02:35（UTC+8）— R10：step1280 quality32 完成，多数prompt接受量改善但仍有下降例

**问题与假设。** T09完成1280-step训练计划，不自动选定checkpoint。R10检验：在R08/R09相同source/panel/protocol/seed下，最终训练步是否提高实际随机accepted draft/round？预先保留下降例和native数值差异，不把接受量提升等同答案质量、confidence校准或系统速度。专职experiment_journal创建/恢复两次受系统thread limit拒绝，本轮复用现存Sol暂代持续记录，沿用本日志。

**方法与身份。** 直接读取[八份完整报告](../reports/expanded-quality1280-20261009/README.md)及未发布 `output/expanded-quality1280-20261009-a278e5a/` 的public-verification、comparison/preflight、部署前后identity、denominator scalar审计和退出标识，不读private样本或final test。记录者独立核八public SHA、30worker模块对Git blobs，与128/512的protocol bytes、source/archive/panel/data/target和32case/seed均相同；metadata SHA与T09本地原件一致。执行source仍 **a278e5a**、checkpoint weights **d6f21ab3…**，完整hash见[source identity](../reports/expanded-quality1280-20261009/source-identity.json)。Fresh stdlib dry-run/import blocker通过，只容许checkpoint及其绑定字段变化；161archive files核验依据preflight，不混同30模块。保持nativeBF16/SDPA、draftFP32+BF16 AMP、actualfloat64 q、temperature1/无filter、fullblock7只按budget截短/max128。本轮无实现失败/修复、无删case或松policy；原数值保真问题继续保留。

**结果与配对复核。** 32/32完成，**3856outputs/2270rounds/1573accepted draft**；2270blocks的finite/shape/probability/cache/EOS/budget检查通过。Draft/round **0.692951542**，对512为 **0.495801527**，对128为 **0.177784578**。记录者从[32per-prompt](../reports/expanded-quality1280-20261009/per-prompt.json)重算总数、histogram、macro和paired delta：512→1280 macro **0.528943552→0.749223719**，**28改善/4下降/0持平**，median delta **0.200878206**；下降ordinal0/11/14/24保留。对128为32/32改善。与512有3例EOS outcome改变，3941→3856outputs、4→5EOS；同prompt/seed不保证generatedprefix/horizon相同，相关round不当独立重复给CI，round减少350不作speedup。

**EOS与分母。** Prefix length0..7 histogram **1427/448/211/100/43/24/8/9**，和2270、长度加权1573；9全7接受来自2174个7-token proposal。位置event/label为 **843/2270、395/2257、184/2235、84/2221、41/2205、17/2188、9/2170**。Proposed/verified均 **15563**，effective **15546**；4accepted EOS排除 **5+2+5+5=17** 个已验证tail labels，另1residual EOS、0bonus EOS。记录者按proposal length histogram/EOS scalar独立重算分母、按prefix histogram suffix重算events，均一致；raw-round/cache细查依据[audit](../reports/expanded-quality1280-20261009/audit.json)。Attempted **3815=1573+2242rejected**；32initialdraw后commit **3824=3856−32=1573+2242residual+9bonus**。24全proposal接受中15达budget无bonus、9有bonus；acceptedEOS也不抽bonus。Commit/round **1.684581498**、query rows **17833=15563+2270**含oldanchor而不含prefill，只是工作计数，不能互代label/attempt/commit分母或推速度。

**数值与终态。** 预选case0/1 round0的[16rows](../reports/expanded-quality1280-20261009/numerical-probes.json)独立重算为 **TV全非零/max0.0927704844638265/argmaxchanges0**；前两轮max来自不同sampledprefix，不能解释成checkpoint对target稳定性的受控因果效果。Argmax不变仍不等于随机law相同，native sequential-target-law无损未证明。Worker/launcher/controller原OS退出证据均0/no timeout，shell未独立wait，不造第四exit0。[Runtime](../reports/expanded-quality1280-20261009/runtime.json) pre/post02:23:56–02:27:49、postrelease02:27:54（UTC+8）记录四自有PID已退出、KFDonlyASR/ready/notbusy、VRAMfree25246560256bytes；peakallocated **2266962944 bytes**。本轮核已有snapshot，不live操作；同步检查、diagnostic copies、float64及JSON输出使collector成为correctness reference，wall/counts不作servingbenchmark。

**证据与下一决策。** Root另独立核八SHA、protocol bytes、32case totals/histogram/position/EOS、seeds和28+4/median通过。公开归档 **2f40e7cb8b5ed52c7d7c30c744cc6ab06bc54c17**（02:33:08）记录者直接核Git，root随后核push成功；execution仍a278e5a，CI本条未查询。Root最终审核/选checkpoint尚未冻结。STS未执行，fit44/eval43 collector/CLI只获CPU准备授权，eval43仅相对STSfit留出，全部dev已用于TF监测，final test未读。WholeQwenCPU准备刚交付，待独立后续里程碑，GPU未授权；S08显式小tensor通过和S04原public失败不改。没有答案质量、校准、无损或speedup主张；本轮仅改日志，无GPU/实现/进程/提交动作。


<a id="whole-qwen-cpu-gate"></a>
## 2026-10-09 02:40（UTC+8）— S09：whole pretrainedQwen/KV gate 的 CPU 准备与空prefix修复

**问题与方法。** S08只验证小tensor，下一门槛需实际28层Qwen3-0.6B的KV内容、实际sameQKV attention和端到端数值报告各自独立。直接读取[冻结协议说明](whole-qwen-varlen-gate.md)、probe/test及未发布 `output/whole-qwen-varlen-cpu-20261009/` 的manifest、三次tests log和real-target-dry-run。CPU测试用tinyBF16模型检验控制流；正式CLI只接受真实模型/ tokenizer指纹与架构，不提供tiny替换入口，不生成文本或读取final test。记录者独立核三交付文件SHA与 **7d1fcdbd4f3415b1b63bb803965d441545319487**（02:36:50）Git blobs一致；root已核push。CPU实际source是3f2b95d archive加probe/test overlay，32script/package身份、archive/三log/dryrun SHA及protocol hash均独立核通过。并行collector改动未纳入此前CPU执行，manifest保留差异；正式launch须另绑fresh reviewed source，不把当前整树冒充CPU原件。

**失败、修复与测试。** 首轮7tests/7.092秒有2失败：新请求首次append前，KV layers尚未初始化，旧快照为空；原prefix-preservation断言误报“KV layer count changed”。修复仅将 `prefix==0 and not expected` 判为空保留prefix，其余非空layer/count/content检查继续；并新增实际model position_ids篡改应在attention前拒绝的测试。第二轮8/8、6.724秒；最终 **8/8通过、7.588秒**，原失败log和两条 `(null)` 提示保留。覆盖stdlib/tiny拒绝、weights篡改拒绝、自有group timeout、tiny正常lifecycle/poison、固定阈值mismatch留证、独立gather损坏、真实position注入和第二层operator异常。CPU替身的12normal/24total layercalls不是正式112/224。Final log/hash与manifest相符，GPU设备隐藏、cuda_available=false；没有privateGPU执行。

**固定机制与三个状态。** 正式协议4normal阶段prime/cached-tail/crop-exit-readd/cropzero，另2poison control pairs，共8native model forwards/**224layercalls**，其中4×28=**112normal层**强制用该层实际BF16 QKV对独立FP32 MATH bottom-right oracle、pooled及逐request都满足原0.02/0.02/RMS0.005。FP64 reduction避免S08所记极大finite误差的FP32overflow留证局限；不事后回改S08执行代码。结构状态精确核input/RoPE positions、marker/local spans、独立gather、实际KV旧prefix/inactive/crop/remove/readd，poison pairs要求A的QKV/hidden/logits/各层KV逐bit相同，非shape-only。Normal层finite mismatch保留tensor并完成既定矩阵，最终layergate仍failed；nonfinite/operator/结构异常停并留partial。端到端67normal query rows只量化densepacked/independentSDPA的hidden/selectedfeatures/KV/logit/TV，不发明pass阈值。`structural_status`、`layer_attention_numerical_status`、`numerical_comparison_status`分开，completed不代表等价，system_pass_claimed始终false。

**真实文件dry-run与授权边界。** 标准库dry-run状态为dry_run_no_backend_import_no_gpu，真实模型/ tokenizer指纹与T09 target一致、generation manifest绑定一致，protocol SHA独立重算 **d83dfd50…**；不把指纹读取写成pretrained forward。预定300秒、8GiB free门槛、6GiB processallocator cap、三独立实际模型以及外部process/KFD/desktop/ASR pre/post均保留。CPU manifest当时gpu_authorized=false；root随后授权唯一一次wholeQwen GPU，最新交付只到core preflight handle54055、未收到实际启动证据，S09不写执行或通过。后续真实结果另起条；原public失败/小tensorpass/quality/速度各不互代。

**独立的checkpoint决策。** Root与core完成quality1280独立review，已决定冻结 **step1280作为后续STS研究checkpoint**；这是development证据下的研究选择，不是最终产品checkpoint或最优泛化结论。R10当时“未冻结”保留其历史状态，本条记录后续决定。Direction将落盘selection manifest，身份/协议链接待完整交付后补记；STS仍未执行。记录者本轮只核证据和日志，无GPU/进程/实现/提交动作，final test及private样本未读。


<a id="whole-qwen-real-gate"></a>
## 2026-10-09 02:50（UTC+8）— S10：wholeQwen结构通过、固定layer RMS失败，CPU舍入诊断不改判定

**问题与方法。** S09准备之后，唯一获授权的实际Qwen3-0.6B矩阵检验了真实KV/attention；必须保留三个独立状态，不能把执行完成或poison隔离通过写成整体通过。直接读取[九份完整报告](../reports/whole-qwen-varlen-gate-20261009/README.md)和未发布 `output/whole-qwen-varlen-gpu-20261009-7d1fcdb/` 的public-verification、archive、completion/exit及三次CPU审计log。记录者独立核九SHA、archive内 **220文件逐个对7d1fcdb Git blobs**、224layer rows/112normal/252request计数、68结构检查/2poison和67query rows各endpoint；执行source仍 **7d1fcdb**、protocol延续S09，完整身份见[source identity](../reports/whole-qwen-varlen-gate-20261009/source-identity.json)。Root另独立复核同项且直读remote completion/原结果；归档 **08220d5**已直接核Git，root已核push，不与execution身份混同。

**固定门槛失败。** 8native forwards各28flash，共224callbacks，三个结果为 **structural passed / layer_attention failed / numerical_comparison completed**、system_pass_claimed=false。112normal层中3失败、252request中5失败，全部在zero-based decoder **layer26**：prime/B RMS **0.0051563464**；cached-tail/A **0.0051409812**、B **0.0056739500**；crop-exit-readd/A **0.0051131269**、B **0.0058588637**。所有elementwise原0.02/0.02检查满足，但原RMS≤0.005不满足；prime/crop pooled可通过而逐request抓到失败，cached-tail pooled0.005617229也失败。Cropzero全部层通过。三组失败QKV/output/FP32oracle先保存后继续既定finite矩阵，没有松阈值、换backend/source或GPU重跑。

**结构与端点结论。** [68项内容检查](../reports/whole-qwen-varlen-gate-20261009/structural.json)全通过：actual tokens/RoPE/marker独立gather、旧KVprefix/inactive/crop/remove/readd及两个poison pairs的A QKV/hidden/logits/全部KV exact。端点67rows各对densepacked/independentSDPA完成：TV非零 **48/47**，max **0.0229268524/0.0532455264**，argmax changes **1/3**；隐藏态/features/各层KV/logit详见[end-to-end](../reports/whole-qwen-varlen-gate-20261009/end-to-end.json)。这些synthetic cache/attention fixture差异未获新pass阈值，不是quality、losslessness或speed测量。Worker/launcher/controller原OS均 **1**、no timeout，表示完整执行但mandatory gate失败，shell未独立wait。既有postrelease记录四PID退出/KFDonlyASR/ready/notbusy，peakallocated3808959488bytes；45.77555289秒含加载/oracle/留证，不作benchmark。

**CPU诊断与失败审计保留。** 后续仅CPU读取保存tensor，按同savedGPU FP32oracle取nearestBF16，失败5request中 **2/5** 的表示下限RMS已超0.005；其余3下限低于阈值而native超阈值，不能把全部失败归为纯舍入。Native也不等于nearest-rounded oracle；CPU FP32重构与savedGPUoracle最多差0.0000534058，另行报告而不替换原gate。独立审计核9tensor artifacts、1398endpoint scalar、2poison并重算134probability rows。前两次因审计额外probability相等断言过严而失败，原log保留；最终只量化CPU/GPU float64 TV统计差最大 **2.0468124e-8**，未发明新的等价阈值或回写GPU指标。记录者重算公开floor分类/TV差上限，完整private tensor重核依据[CPU audit](../reports/whole-qwen-varlen-gate-20261009/independent-cpu-audit.json)，未声称本轮加载远端raw。

**边界与下一决策。** 原public gate继续失败，pinned小tensor通过不提升为wholeQwen通过，固定RMS原失败不改。下一步需先解释正常层剩余数值误差及门槛适用范围，任何新实验/规则须预声明并另留证；本条不授权自动GPU重跑。独立SDPA STS工作流不因该失败改协议。记录者仅追加日志，无GPU/实现/进程/提交或final-test/private样本动作。


<a id="sts-cpu-workflow"></a>
## 2026-10-09 02:50（UTC+8）— R11：冻结step1280的STS CPU工作流，fit44刚启动尚无结果

**问题与方法。** R10接受量改善与T09 TF零基线优势均不证明真实confidence校准。后续必须先冻结checkpoint，以fit44拟合温度与常数，再在独立eval43比较，不能让eval参与选择。直接读[selection manifest](../configs/sts-step1280-selection.json)、[校准工作流](confidence-calibration.md)、[采集协议](rollout-collection.md)及未发布 `output/sts-cpu-preflight-20261009-v2/` 的handoff、tests、fit/eval dry-run与archive。Source归档 **034064bf8fea7f67c9039c2dd103b65ca12e803a**（02:44:11）已直接核Git，root已核push；记录者核10handoff文件SHA对其Git blobs和archive SHA、selection file SHA及canonical digest通过，详细身份保留在manifest/报告。

**选择、接口与隔离。** S09决策现已落盘：step1280 weights d6f21ab3…/metadata8bb0c511…绑定原panel/protocol、quality1280依据，selection digest **3810939b…**。它只冻结STS研究身份，不表示最优泛化/最终产品或自行授权GPU。Collector显式group与required selection进入binding，历史quality a278报告保持原件。CPU `calibrate_rollout` 核完整44/43、实际workerOS0、rawround/effective/EOS/零block保留及checkpoint/data/source/runtime；fit artifact冻结实现身份、温度、**61grid/20bins**与fit-only prefix prevalence常数，eval复核artifact/fit未变及prompt/token hash不相交，不搜索温度或估计eval常数。Raw、STS、fit-only常数按同ECE/Brier定义比较；全dev已TF监测，eval43仅相对STSfit留出。

**CPU观察与审计口径修正。** GPU隐藏的远程最终 **29tests/6.243秒、exit0**（8算法+10collector+11workflow），root另报告独立local11tests通过。真实step1280 dry-fit44/dry-eval43绑定通过，记录者核44/43数量、group/selection一致，aggregate均0blocks/0completed、execution_checks_passed=null；这不是实采或STS拟合。Root最初把public protocol完整对象直接digest对selection canonical协议hash而assert失败，随后逐字段核fit-run.manifest.protocol一致；public多出的6项描述字段说明了差异，这是审计对象口径错误，未作实现修复或改协议。

**下一决策与执行状态。** Direction获 **fit44采集→CPUfit artifact冻结→eval43采集/冻结artifact评价** 的有序授权，新增任何GPU阶段仍须核窗口。实际fit启动shell4011829/controller4011831/launcher4012129；随后root通过远端ps独立确认worker4012492正在运行且父进程为该launcher，direction亦确认同PID。只记录启动和当时live状态，不写fit完成或pass。尚无真实STS温度、ECE/Brier比较、eval采集、性能或调度收益结果；潜在serving热点只是未测假说，float64采样在tensor device，不误写成CPU逐词表复制。Final test未读；本轮记录者只日志，不开GPU、改实现/进程或提交。


<a id="packed-draft-cpu"></a>
## 2026-10-09 03:00（UTC+8）— S11：真正flattened packed draft的CPU实现、AMP与原子提交修复

**问题与方法。** 多请求draft必须共享一次flattened backbone，不能用逐request forward伪装batch，也不能用target causal kernel改变draft整块双向语义。直接读取[packed draft说明](packed-draft.md)、module/test及未发布 `output/packed-draft-cpu-20261009/` 的manifest/四tests日志。记录者独立核三文件SHA对 **ac64e7d6416371f2306a83c2380fbd9c10970554**（02:57:46）Git blobs、四log及base08220d5 archive SHA；root已核commit/push。CPU实际执行为base archive加三overlay，不重写冻结训练/wholeQwen快照。完整hash及失败原件留在manifest/log，未读final test/private样本。

**机制与物理工作。** `append_committed`先验证所有chunks，一次contextFC/norm/RoPE、每层一次全newcontext KVprojection。Backbone把active anchors一次embedding/RoPE，每层一次全batch Q/blockKV/attention/o_proj/MLP；request loops只做metadata/gather。每request看到自身committed context及**全部own draft block**，显式新noncausal/no-window privatebackend，无fallback；target的causal gate不能验证它。Persistent cache仅存已提交target features，不含新emitanchor或拒绝tail，blockKV临时不提交，backbone失败也不改cache。Admission截短proposal不减少整块计算。记录fullquery/gather/pairdomain和concat/gather/castbytes，inactive context仍进入当前physical concat成本；这些是工作域计数，不是实测流量/速度。Markov heads、stochastic draws、verify loop、globalallocator/async尚未接入。

**失败与两项修复。** 最早8FP32tests/0.177秒通过；新增AMP后，9tests/0.355秒和原子修复后的10tests/0.159秒均有1项AMP error，原log保留。原断言错假定K/V同dtype，而未改的cached backbone在FP32 RMSNorm下可为 **KFP32/VBF16**；privateATen又不能假定SDPA autocast自动处理输入。最终保留各cache component dtype与FP32权重/RoPE，在attention边界按active autocast显式cast QKV，cast工作另计、finalhidden可仍FP32，未靠downcast训练参数规避错误。Root同时发现投影完成后先赋cache、再分配metadata会在late allocation失败时半提交；修成全部candidate cache/metadata/work构造且验证后一次commit，crop也先验证candidate。注入late metadata/crop validation失败核旧引用、内容、marker均不变，不限于早期projection异常。

**CPU结果与边界。** 最终 **10/10通过、0.331秒**，GPU三设备隐藏/cuda_available=false，两条 `(null)` 提示保留。涵盖独立single-request cached/fullbackbone的实际KV/hidden、不同position/order、零context/cropzero/exitreadd、active/inactivepoison、fullblock双向可见、module callcounts、原子失败和AMP精度；native spy只核literal noncausal/no-window参数。AMP与原cachedSDPA的fixture-only0.02/0.02比较不是正式GPU阈值或分布等价证明。新noncausalGPU未验证，无quality/speedup结论；wholeQwen原layer26失败保持独立。Fit/eval尚等完整报告，本条不提前写结果。

**另行只读baseline盘点。** 已完成的私有 `output/target-baseline-inventory-20261009/` inventory/notes确认现有vLLM0.30.0+rocm723 metadata、offlineLLM与bench latency/throughput/serve源码入口和gfx1201/R9700识别；入口存在不等于GPU实际load/kernel/graph成功，也不写“无支持”。隐藏GPU的CLIhelp在25秒退出124，停止该路径、未重试或修环境；精确自有help/timeout进程检查为空，本地父进程已返回。参数/ASR共存预算/关闭instrumentation的baseline提案有file:line，比例及KVbytes不冒称硬全系统cap，性能与采样law仍未测。该盘点未启动server/GPU、调用podman、安装或读凭据。S11本轮仅日志，无实现/进程/GPU/提交动作。


<a id="sts-fit-eval-result"></a>
## 2026-10-09 10:40（UTC+8）— R12：冻结STS fit44/eval43完成，改善不一致，保留raw与constant对照

**问题与假设。** R11冻结研究checkpoint与校准流程，需检验逐位置STS在prompt-held-out eval上是否优于raw confidence，并对照只用fit prevalence的常数。假设是fit44上最小化cumprod ECE的温度可能改善eval校准；不能预先保证Brier、尾部稳定性或scheduler收益。直接读取[九份最终报告](../reports/sts-step1280-20261009/README.md)和未发布 `output/sts1280-20261009-034064b/` 的handoff/public-verification/最终verification，不读raw样本或final test。

**方法与冻结顺序。** 执行source **034064b**、已选step1280、原panel/cases/seeds、nativeBF16/SDPA/temp1/nofilter/fullproposal7/预算128不变。先采fit44，在CPU以原61点grid/20bins、条件logit scaling后cumprod、从左到右冻结先前位置最小化prefix ECE；常数为fit有效标签的累计prefix prevalence，不是条件率乘积。Artifact在 **02:53:36（UTC+8）**冻结后才启动eval43，eval不搜索温度、不拟合常数、不选checkpoint。最终温度约 **0.9330/0.8123/0.8123/0.6598/0.6598/5.6569/1.5157**；完整网格/身份见[fit metrics](../reports/sts-step1280-20261009/fit-metrics.json)和[source identity](../reports/sts-step1280-20261009/source-identity.json)。

**完成、分母与证据。** Fit **44prompts/3157blocks/21637effective labels**，eval **43/3280/22437**；proposed/verified分别21657/22465，acceptedEOS后tail排除20/28，拒绝后已验证tail继续记零、不补未proposal budget尾部。两组zero-block prompts均0，完成/coverage定义仍保留此类prompt。记录者核13handoff SHA对归档Git blobs、九public SHA、executed package SHA对034064b，重算位置count/event总和、各方法bin ECE、加权指标、fit-only constant prevalence及其Brier公式，均与报告一致。对raw logits的独立sigmoid/cumprod/ECE/Brier、223archive文件与prompt disjoint/frozen ordering全核事实依据[verification](../reports/sts-step1280-20261009/verification.json)及公开[核验脚本](../reports/sts-step1280-20261009/verify_sts.py)，本轮未重读完整raw。两组worker/launcher/controller OS均0、CPUfit/eval/verifier均0/no timeout；shell不添独立退出码。既有postrelease确认自有PID退出/KFDonlyASR/ready/notbusy，fit/eval peak2381026304/2474884608bytes；非本轮live探测或性能实验。公开归档 **02c584490553a66b5a1446d02c98bfcc9761a845**（10:34:18）已直接核Git、root已核push，execution仍034064b。

**结果与判断修正。** Eval上STS对raw ECE仅 **3/4/6/7改善**，**1/2/5变差**；Brier仅 **7改善**，**1–6变差**。Fit-only constant ECE在1–5优于STS，但raw/STS Brier在全部7位置均优于constant，不能仅因constant低ECE断言head无用。按同22437有效标签的加权结果为：

| 方法 | 按label count加权的位置ECE均值 | Label-weighted Brier |
| --- | ---: | ---: |
| raw/unscaled | 0.021306579 | **0.04629602443** |
| frozen STS | 0.021254273 | **0.04674456667** |
| fit-only constant | 0.011116911 | **0.06261451037** |

这里ECE是位置ECE的count加权均值，未把不同位置合并为pooled-bin ECE；小幅均值下降不抵消各位置失败或Brier变差。最终采集/拟合没有执行失败、重试或实现修复；被证据修正的是“STS应普遍改善”的预期，保留不利结果，不据eval调整grid/温度/checkpoint/sampler/admission。尾部eval位置6/7仅 **19/9正例**（3159/3135labels），位置3–7只有42/43prompt有观测，round相关性与稀疏事件限制外推，不能用尾部低ECE泛称校准可靠。完整position/bin/coverage见[eval metrics](../reports/sts-step1280-20261009/eval-metrics.json)。

**下一决策与记录责任。** 保留冻结STS作为论文方法分支，同时保留raw/fit-only constant对照；未来scheduler还须验证因果集成与实测成本，不因本轮低ECE/Brier宣布系统收益。Eval43仅相对STSfit prompt-held-out，全部119dev已用于TF监测；final test未使用，既有native数值差异/wholeQwen固定失败不解除，无答案质量/无损/性能结论。专职独立experiment_journal创建仍遭系统thread limit拒绝，继续由现存sol_data承担6.1 Sol专职记录；本轮同步修正README.ai.md的当前角色描述和顶部STS索引，旧历史条目保留。记录者只日志及入口角色说明，无GPU/实现/进程/提交动作。


<a id="packed-sampling-cpu"></a>
## 2026-10-09 10:50（UTC+8）— S12：packed proposal/verify/commit 的同步 CPU 集成通过

**问题与假设。** S11只有flat draft backbone，仍需连接真正批量Markov heads、实际q采样、一次packed target验证与一次draft提交，并证明跨request没有KV污染。直接读[集成说明](packed-sampling.md)、源码/tests及未发布 `output/packed-sampling-cpu-20261009/` 的manifest、overlay-verification和测试日志。假设是复用已有stochastic verifier，在批量执行中仍保持每个request的独立采样law与“已提交prefix减最新anchor”缓存边界；这不预设native GPU数值等价或加速。

**批量路径与工作口径。** Admission一次packed prefill，只选每request最后hidden row，调用一次 **R-row target LM head**，不对整段prompt做vocab投影；全部已提交prompt features一次投影进draft KV。Proposal一次flat backbone/base LM head，每位置对仍需该位置的request批量调用Markov embedding/projection与confidence head，request循环只做RNG draw。Confidence在当前token抽样前计算，实际q保留原float64 law。Verify一次packed append `[旧anchor,选择的proposal prefix]` 和一次logits投影，按request执行既有接受/residual/bonus/EOS逻辑；裁剪target后全部committed features一次投影进draft。拒绝tail与最新输出不进入persistent KV。默认shadow即使后续allocation为0仍付全block工作，只有抽样前冻结长度的fixed-budget允许zero-skip；fullshadow/basehead/serialheads/targetrows/commit/copies分别计数，不当作耗时或速度。

**身份、私有来源与拒绝机制。** Issued handle绑定request/incarnation/session mutation epoch/cache length/nonce/limit/mode及精确对象身份；移除后同名重加取得新incarnation/markers，stale/foreign/reused或非法prefix在target forward前拒绝。Admission/commit/remove推进的session epoch不是planner的连续t−2 round clock。原tokens/q/confidence由session私有保留，改公开observation copy不能替换verification q或policy score。Typed validator收到原始confidence的immutable CPU tuples、roster/remaining budget及七字段身份，拒绝也发生在target前；external nonanticipating只是caller声明，`planner_causality_proven=False`仍保留。执行/draw/copy/commit异常清空两套cache、使session失效并拒绝后续操作，不承诺RNG replay或事务回滚。

**CPU证据与边界。** 最终 **26/26通过、1.869秒**（11新sampling+10draft+5target），三设备隐藏/cuda_available=false；此前8/10/25项阶段也通过，原log及两条 `(null)` 提示保留。实际tiny Qwen多request rollout对照独立single-request cached reference，每轮从完整committed prefix重建并比较各层target/draft KV；另核实际q/pre-token confidence、inactive exact isolation、R-row/head callcounts、public-copy篡改、生命周期、typed拒绝、BF16 AMP mixed KV/FP32 trainable与失败后关闭。受控每个拒绝位置/fullaccept/EOS分支仍运行真实models/cache，仅替换acceptance决策，验证集成commit边界，不冒称新的独立接受law证明。GPU入口测试是metadata-only，要求target causal与draft noncausal两侧显式pinned backend并拒绝public default；没有执行GPU。

**身份核对与下一决策。** 记录者独立核manifest SHA **366bc955…**、三交付文件对 **ba3fa56575697bda5bd6c4a3b24bf3053a7aad36**（10:46:58）Git blobs、六log SHA及overlay身份表；最终log **ddf53928…**与26项计数相符，root已核push。CPU执行为manifest所记 **08220d5 archive+明确overlay**，还含packed_draft/async_capacity/async_round；不把三新交付或当前整树冒充完整tested snapshot。Trained noncausal draft与integrated native GPU仍未验证，wholeQwen固定layer26 RMS失败不变；没有overlap/graph/真实SPS或性能结果。同步driver smoke不提升为实际异步，正式planner/driver交付另记S13；本条未开启新GPU，记录者只改日志，无实现/进程/提交或final-test/private样本动作。


<a id="two-step-capacity-cpu"></a>
## 2026-10-09 10:52（UTC+8）— S13：t−2容量与同步driver CPU通过，修正输出预算末位收益

**问题与假设。** S12已打通packed sampling，但current proposal对own future的干预可能污染容量/接纳，冷启动若永远target-only且不产shadow history也会死锁。新[两步容量说明](async-capacity.md)把历史absolute K搜索与当前prefix接纳分开：连续planner round严格取 **t−2** sealed frame，对全部可行离散SPS大小搜索expected progress×SPS，包含cliff/zero-score扩展、exact tie取较小K；当前只在draw前冻结的K内按cumprod score接纳positive prefix。原scheduler的同步first-non-improvement规则未修改。这是CPU参考策略；SPS曲线与physical bucket需完整供给，不能插值或当作实测硬件。

**review发现的失败假设与修复。** Root发现旧预算收益会把remaining=1时接受最后一位算为额外output；在不考虑EOS停机的预算oracle中，每轮进度是 **E[min(1+A,remaining)] = 1 + sum(P(A≥j))**，求和只到 **j≤min(length,remaining−1)**。Direction据此把historical候选与current admission均截到remaining−1，remaining=1只保留一个baseline token；fullshadow仍包含全部gamma位置。独立oracle逐一枚举reject paths/fullaccept，覆盖remaining1/2/3及所有prefix长度，与公式和容量/接纳一致。这个修复纠正收益与预算边界，未改sampler law或裁掉shadow成本；root另报告本地18项review检查通过，工具输出无独立文件，本日志不将其冒充额外GPU或另份归档实验。

**实际调用顺序与校准来源。** `CapacityRoundDriver`强制 **freeze→full shadow propose→bind→private validate/verify/commit→finish**，拒绝已有outstanding draws，partial failure后拒绝复用；planner round与session mutation epoch分开。Calibration是只读temperature tuple/source-artifact SHA身份，draw前固定，默认raw transform亦有独立身份；本轮未声称实际接入R12的trained STS artifact。Bind核精确handles、七字段identity/cache/budget/epoch，verify从session私有原confidence重算score digest与allocation，公开copy篡改不可替换来源。前两round本地cold-start只接纳baseline，但每轮照付 **R×gamma全shadow**并封存history，第三round才能取frame0；这只是显式reference startup选择，不宣称论文exact startup。Current roster/budget在draw前clamp旧absolute K，departed history仍参加旧搜索，新incarnation不继承旧score，profile/context不兼容则拒绝。Reserved K、actual logical B与physical bucket分别报告。

**证据、结果与独立集成。** 直接读未发布 `output/async-capacity-cpu-20261009/{handoff.json,runtime.json,real-tests.log}`，正式 **20/20、0.234秒、OS0**（11planner+2真实tiny Qwen driver+7原scheduler），cuda_available=false。三轮用实际target/draft重建全部committed KV，事件严格freeze/propose交替；前两轮B=3/alloc0仍各付9shadow positions与9private confidence host values，第三轮history0/B=12，同样付9shadow。测试另覆盖jagged exhaustive allocation、own-token later-score干预、tie/zero underfill、churn/budget clamp、growth/empty roster、stale/profile/context与partial failure。记录者独立核五交付文件对 **6b1c2bca15d2f6802c2dd6404a04c87f6ceb2ca2**（10:49:52）Git blobs、runtime六源码SHA，handoff SHA **4c9d2dd4…**；root已核push。

Root随后在该immutable source archive运行完整unittest suite，未发布 `output/integration-cpu-6b1c2bc/{verification.json,tests.log}` 记录 **176/176、10.878秒、SSH实际OS0、120秒bound**，三GPU环境变量为空。记录者独立核log SHA **e624d933…**、176test entries、最终Ran/OK及source identity；archive SHA依据root verification，未在本轮重复解包。日志有用例JSON/stdout穿插，不能仅按同一行“... ok”计数；所有原输出保留。GitHub Actions API本轮HTTP403 rate limit使CI状态未核，不写CI失败，也不把CPU suite替代CI状态。

**边界与下一决策。** Driver全round计入calibration/host copies/hash/private验证/model work与显式device sync，各stage时间都属同步开销；fixture用synthetic SPS/test-only dense kernels，`hardware_overlap_proven=False`。Progress公式/预算oracle固定无EOS，真实随机EOS停机尚未建模，也不能用当前sampled EOS事后改变自身admission；root随后在说明文档补记该局限，原五文件handoff仍绑定6b1c2bc blobs。数值干预与provenance只验证这份参考集成，不是模型losslessness因果定理、实际异步buffer/overlap、graph或speedup。wholeQwen固定RMS失败与trained noncausal/integrated GPU未验证状态均保留。后续需分别解决native gate、测量真实capacity及同物理成本baseline，不能据本条自动开GPU。本轮记录者只追加日志，无实现/进程/GPU/提交或final-test/private样本动作。


<a id="vllm-baseline-cpu-prep"></a>
## 2026-10-09 10:58（UTC+8）— S14：强target-only vLLM smoke 的 CPU 准备，真实engine尚未启动留证

**问题与方法。** S11只读盘点发现现有vLLM入口，尚不知default graph/init与Qwen3-0.6B能否实际运行；不能只与逐token Python reference比较就宣布系统加速。直接读[冻结smoke协议](vllm-offline-smoke.md)、五交付源码/tests及未发布 `output/vllm-offline-smoke-cpu-20261009/` 的handoff/tested-source/preparation、三tests log、两preflight及tokenizer logs。准备一个新公开synthetic prompt，真实non-thinking tokenizer得 **35tokens/seed20261009**，request SHA **8b676c69…**、八model/tokenizer files绑定；无dataset/final-test输入。原始tokens不写入本日志。

**配置与执行边界。** 预定现有vLLM0.30.0+rocm723、BF16/T1/top_p1/top_k0/min_p0/no penalties、TP1、max_model_len4096/max_num_seqs1、128fixed output tokens/ignoreEOS、generation_config=vllm、memory ratio **0.18**。保留default graph配置/enforce_eager=False，无eager fallback、retry、改backend/安装环境；该开关不证明capture/replay，0.18是engine预算比例，不是全系统/全process硬显存cap。Worker **600秒deadline**包含import/init/compile/capture/generation；blocking health/KFD最多额外6秒、TERM grace10秒/KILL观察2秒及bounded post-release另记controller wall time，不偷算进worker限时。执行前后需fresh ASR/KFD/VRAM/source/model核验，启动free≥10GiB、运行free≥2GiB；单独核worker与controller实际OS exit、所有自有PIDs退出及ASR ready/notbusy，不以worker-written completed替代。

**失败与review修正。** 初次tokenizer准备由direction报告 **exit1**，原log留有失败traceback：Transformers5默认 `apply_chat_template(tokenize=True)` 返回BatchEncoding，JSON序列化报TypeError；明确 `return_dict=False` 后第二轮 **exit0**，原 `tokenizer.log` / `tokenizer-v2.log` 及 `(null)` 提示保留。这与P01/README已记的同一返回类型陷阱一致，应复用显式接口约定，不能假定tokenize=True总返回list。Review还发现仅ppid/group认ownership会把recycled parent PID后的无关child误纳；改为live parent startticks核验，并用Linux subreaper接管orphan/跨group descendant，仅对仍匹配已观察身份的自有PID发TERM/KILL。Mock PID-reuse反例与真实子进程timeout/orphan tests均覆盖。另拒绝config-only模型指纹：完整weights/tokenizer/config及indexed shards必须存在且在frozen binding，不以config hash代替实际权重身份。

**CPU证据与身份。** 最终 **7/7、1.348秒、tests OS0**，前5项/0.943秒与冻结7项/1.315秒也通过；覆盖exact graph-preserving config/非法tokens、缺weights/index shards拒绝、PID reuse、真实zero/nonzero worker OS exit、escaped-session timeout清理、成功parent后orphan adoption和ASR guard失败。真实35-token输入worker/controller stdlib preflight均 **OS0**、prepared-only，不import vLLM/Torch或执行GPU。记录者独立核五交付对 **e0f5ce8dd511ecce8b2f77e1f4c870a607bb1833**（10:55:53）Git blobs、四tested-source SHA、request原文件SHA与preparation35/8计数；handoff SHA **2824e84c…**、final test log SHA **4b41d0f9…**。远端大型模型文件实际哈希依据preparation/已有绑定，未在本轮重新读取weights。Root已核commit/push。

**当前状态与下一决策。** Handoff当时no GPU authorized；root随后授权direction唯一一次按committed archive与installed-source身份冻结的真实smoke，但本条时尚未收到启动PID，不写已运行、完成或通过。仅CPU准备通过；真实init/kernel/graph、sampling fidelity、速度均未验证，fixedignoreEOS工作也不是EOS-stopping quality评估。此前隐藏GPU的vLLM help25秒timeout仍只表示该入口尝试未完成，不写engine unsupported。实际结果收到后另起条；无benchmark/speedup、无自动重试授权，wholeQwen原固定RMS失败不解除。记录者只日志，无GPU/进程/实现/提交动作。


<a id="trained-packed-gate-cpu"></a>
## 2026-10-09 11:06（UTC+8）— S15：trained packed decoder gate CPU准备，纠正AMP下的FP32 oracle

**问题与方法。** S12/S13的tiny CPU集成不能验证实际trained draft的native noncausal attention。新[冻结协议](packed-decoder-gate.md)绑定真实Qwen3-0.6B与已选step1280（weights d6f21ab3…/metadata8bb0c511…），固定synthetic三轮pre-draw allocation、remove/readd、active/inactive poison与完整KV；target causal/draft noncausal两侧显式pinned backend，无tiny正式入口/fallback。正常draft15层同actualQKV对FP32 MATH，原elementwise0.02/0.02、RMS≤0.005不变；结构、draft层数值与endpoint completed三状态独立，system_pass_claimed=false。300秒/8GiB free/6GiB processallocation边界保持。

**失败假设与修复。** 首轮 **4/4、6.236秒**通过，但root review发现 `.float()` 输入加MATH SDPA仍被外层BF16 autocast影响，不能声称FP32 oracle。Core显式包 `torch.autocast(...,enabled=False)`、assert oracle.dtype==FP32并测试persisted tensor dtype，另加helper/protocol threshold drift拒绝。原first log保留为开发历史，不作为最终oracle证明。修复后 **5/5、5.948秒**，GPU三设备隐藏；finite故意bias用例仍完成矩阵并保留failed状态、未松原阈值。真实target/1280 stdlib dry-run **exit0**，只核身份不import backend/GPU。

**证据与状态。** 直接读未发布 `output/packed-decoder-gate-cpu-20261009/` manifest、两test logs及real-binding-dryrun。独立核manifest SHA **93198e55…**、三文件对 **8378ea7af51fdb84e369a4b306259e88d8dd22b7**（11:02:57）Git blobs、四evidence SHA，按实际helper序列化重算protocol **a311e144…**、核dry-run checkpoint digests与冻结协议一致；CPU执行为6b1c2bc archive加明确overlay，不称当前整树tested。Root已授权core唯一一次真实GPU，但本条未收到启动PID，不写实际运行或通过；原wholeQwen固定RMS失败不解除，无quality/async/性能或系统通过结论。记录者仅日志，未读private trace/final test。


<a id="vllm-first-version-failure"></a>
## 2026-10-09 11:06（UTC+8）— S16：vLLM第一真实attempt在版本guard失败，engine尚未初始化

**问题、结果与原因。** S14之后唯一授权的 **e0f5ce8** attempt实际执行，import后 **13.2115秒**抛runner版本guard异常，engine_initialization_started=false、generated_tokens=0；worker/controller实际OS均 **1**、无timeout/retry/fallback，controller15.9002秒不是benchmark。Runner把distribution metadata **0.30.0+rocm723**与module.__version__ **0.30.0**当作同一字段；已在launch前绑定的 `_version.py` 经后续stdlib AST核定module值，未二次import vLLM。Fatal是版本契约错误，optional torch-c-dlpack warning不是本次fatal原因；不能据此断言BF16/GPU/attention/graph失败。

**证据与释放。** 直接读[六份公开失败报告](../reports/vllm-offline-smoke-20261009-e0f5ce8/README.md)并核对 **631892ce5db5a847df2d1005dcd956001aa73abe**（11:04:39）Git blobs；归档commit已核，root当时push进行中。执行source仍e0f5ce8，完整身份见[source identity](../reports/vllm-offline-smoke-20261009-e0f5ce8/source-identity.json)。257source/3147installed/3425binding files及八model/tokenizer文件前后不变；四自有shell/controller/worker/descendant PID均不存在、无cleanup signals、KFDonlyASR ready/notbusy、VRAM前后used8967499776/free25241243648bytes。完整raw日志/远端proc与版本再次核验依据root直接独立检查及[verification](../reports/vllm-offline-smoke-20261009-e0f5ce8/verification.json)，记录者未读private样本或重做远端进程动作。`verification.passed`只指失败边界/释放通过，不是engine通过；不添独立shell退出码。

**修正与下一决策。** 原失败保留，direction仅CPU修双字段guard，分别钉distribution/module并报告actual/expected，不改installed packages。尚无第二次执行授权，不能沿用首次窗口重跑；真实init、graph capture/replay、完整request、fidelity与强性能baseline仍未验证。原35-token公开synthetic/128fixedignoreEOS/defaultgraph/.18配置身份保留，graph开关与显存比例仍不是replay或全系统硬cap证据；wholeQwen固定RMS失败独立保留。本轮只追加日志，无GPU/实现/进程/提交动作。


<a id="trained-packed-gate-real"></a>
## 2026-10-09 11:15（UTC+8）— S17：trained noncausal draft与packed集成固定GPU gate通过，target失败仍独立

**问题与方法。** S15修正oracle后，唯一获授权的 **8378ea7** GPU矩阵实际检验已选step1280与真实Qwen3-0.6B，BF16 target/FP32 draft trainable与RoPE/BF16 AMP、原pinned causal/noncausal backend及protocol **a311e144…**不变。直接读[公开聚合报告](../reports/packed-decoder-gate-20261009/README.md)与未发布 `output/packed-decoder-gpu-20261009-8378ea7/` 的source archive、completion/exits及release身份字段；未读raw proposal/token样本。记录者独立核六public SHA、archive SHA **401822fc…**及 **260文件逐一对8378ea7 Git blobs**、38executed-source SHA；报告经root复核后归档至95038bf并核push；root另核原JSON的结构/层/TV/接受数及执行archive与Git一致。

**通过的范围。** 固定状态为 **structural passed / draft_layer_numerical passed / endpoint completed**、system_pass_claimed=false。15normal pooled层与35request-layer全部满足原elementwise0.02/0.02和RMS≤0.005，max pooled/request RMS **0.0035258595/0.0038259146**；15saved oracle均FP32，S15首CPU错误已在GPU前修复。75结构内容检查和2exact draft poison pairs全部通过，覆盖committed projection/KV、oldprefix/inactive、remove/readd incarnation/markers及A actualQKV/output/logits/cache exact。5target forwards/**140events**、7draft forwards/**35events**（3normal+4poison/control），73normal target hidden query rows。记录者独立重算15/35计数/maxRMS、75passed/2exact及22endpoint概率行；完整actual tensor复核依据[CPU audit](../reports/packed-decoder-gate-20261009/cpu-audit.json)，本轮未重载远端raw。

**端点差异与成本边界。** Independent SDPA按native实际input chunks/committed lengths replay，不另采轨迹。22verification概率行 **18非零TV、max0.0293895273、argmax changes0**，只是endpoint差异量化；resident target K/V各420comparisons中304不相同、projected draft K/V各75中55不相同，这些含重复retained-prefix观察，不是独立token或pool RMS。Native exact旧cache保留与cross-backend KV不相同可同时成立，完整幅度见[numerical](../reports/packed-decoder-gate-20261009/numerical.json)。两shadow轮各21draft/basehead rows，allocation0仍全付；fixed-zero只运行B的7-row backbone/basehead而抽2positions、仍复制inactive context，逻辑copy/gather量不冒称实际traffic。三轮14selected proposal tokens接受0，只是synthetic矩阵，不是quality估计。Hidden-GPU独立CPU audit核25artifacts/1022endpoint scalars/35request gate/22TV/**78law rows**；CPU/GPU TV统计maxdelta **7.0494474e−9**、重构FP32oracle maxabs1.2397766e−5仅诊断，不换GPU判定或阈值，记录者重算公开78计数/TV差上限。

**退出、supervisor局限与下一决策。** Worker/launcher/controller/SSH均 **OS0**，无timeout/retry/fallback；peak3492600832bytes、instrumented35.456秒非benchmark。已保存release确认四ownedPIDgone、ASR同startticks/ready/notbusy、KFDonlyASR、VRAM恢复used8967499776/free25241243648bytes，非本轮live探测。Root指出outer controller裸PID复用弱点时实验已运行，未改source/重跑；postlaunch补存startticks、release独核且无termination signals，只证明本次释放，不能回溯宣称supervisor reuse-safe。后续须用已测试的identity-aware supervisor。原wholeQwen target固定layer26 RMS gate **failed unchanged**，本次draft/integration通过不提升为whole-system/target-law losslessness；无speedup/async overlap/graph或质量结论。Root给下一方向是先在CPU设计真实R/context/B成本profile与graph友好执行，未授权新GPU或继续同类门禁。本轮记录者只追加日志，无GPU/实现/进程/提交或final-test动作。


<a id="vllm-functional-smoke"></a>
## 2026-10-09 11:26（UTC+8）— S18：修正双版本guard后vLLM functional smoke完成，保留实际Triton路径与replay边界

**问题、假设与修复。** S16失败发生在engine初始化前，只暴露runner误把distribution与module版本当同一字段。修正 **03b80a2b573d42cb9c90a573e1ecdd1bd12f1d17**（11:07:38）分别钉distribution **0.30.0+rocm723**、module **0.30.0**并留actual/expected/matches；新假设是修正该契约后，原BF16/defaultgraph配置可完成一个实际request，不预设fidelity或速度。另行获授权的这一次运行保留原request SHA **8b676c69…**、seed20261009、T1/no filtering、generation_config=vllm、ignoreEOS/fixed128、memory ratio0.18与enforce_eager=False；没有手动eager重跑/backend override、安装或patch环境。S16原失败条目和报告不改判定。

**实际执行与backend。** 直接读[六份功能报告](../reports/vllm-offline-smoke-20261009-03b80a2/README.md)及未发布 `output/vllm-offline-smoke-20261009-03b80a2/` 的非样本stage/exit/health/integrity/release留证，并只检索worker log的capture/backend行。双版本分别matches，engine初始化完成，**35input→128output tokens**、finish_reason=length，worker/controller实际 **OS0**、无timeout/自动retry。Engine报告 **FULL_AND_PIECEWISE / capture sizes[1,2]**，PIECEWISE/FULL capture日志完成；默认 **ROCM_ATTN** 的custom paged kernel不适用，内部转用 **Triton implementation**。这是engine实际默认dispatch，必须记录，不能误写“无任何fallback”或手动切backend。默认init有compile/JIT cache行为，但无package安装/环境补丁。Capture支持graph初始化，**replay未独立instrument**。

**身份核对与释放证据。** 记录者独立核六public SHA对report-handoff及 **3a16f4593202b3b286b8312514d380633d2df70c**（11:20:14）Git blobs、source archive SHA **6a596f80…**及 **266文件逐个对03b80a2 Git blobs**、worker/controller SHA与七份非样本private evidence SHA。Root已核code/report push。Binding共 **3434files**（266source/3147installed vLLM/八model-tokenizer等），执行前后未变依据完整integrity留证，记录者另核binding count/hash与postpassed；未在本轮重读大型weights或generated tokens。独立重核stage双版本/128输出、13owned PID全部消失、worker/controller0及 **25health samples全ready/notbusy**；ASR身份保持/KFDonlyASR、VRAM恢复used8967499776/free25241243648bytes。Supervisor未发cleanup signals，engine自身正常teardown对EngineCore发SIGTERM，两者分开；不添独立shell退出码或将历史release写成本轮live探测。

**另行集成CPU验证。** Root在immutable **3a16f45** archive运行完整suite，未发布 `output/integration-cpu-3a16f45/{verification.json,tests.log}` 记录三GPU环境变量为空、120秒bound、**189/189、12.861秒、SSH实际OS0**。记录者独立核log SHA **49154ee5…**、189test entries/最终Ran/OK及source identity；archive SHA **d51b4bc5…**依据root verification，未本轮重复解包。两Astra当前未提交benchmark/profiler WIP不在该snapshot内，这不是CI状态、GPU或性能证据。

**结论边界与下一实验。** 现在成立的是该配置的strong-engine **functional baseline**：能init并完成单个固定request。Controller127.931秒属单次diagnostic lifecycle，不算吞吐/latency基准或DSpark加速。五秒guard采样全局VRAM最高 **15065853952bytes**、最低free19142889472bytes，记录者重算采样max；这不是exact allocator peak、孤立engine显存或0.18全系统硬cap。没有参考float64 sampling law fidelity、多请求capacity/arrival-load、graph replay或性能优势证明，原wholeQwen target固定RMS失败仍独立。下一步先在CPU冻结共同synthetic **R/context长度/固定输出** workload manifest、warmups/repeats与latency边界，设计单engine复用及真实R/context/B成本profile；需另审查和GPU窗口，当前没有新GPU授权。本轮只改日志，无GPU/进程/源代码/实验/提交或final-test/private样本动作。


<a id="formal-performance-cpu-prep"></a>
## 2026-10-09 11:38（UTC+8）— S19：full64 native成本与共享54-batch baseline的CPU测量准备

**问题、单位与共享域。** S17/S18建立bounded正确性/功能证据，下一步要在冻结工作量下测成本，不能拿单次smoke或target-only tok/s替代planner曲线。[Native测量设计](packed-performance-plan.md)规定目标函数expected progress×SPS所需SPS为 **1/T_round（rounds/s）**，不是B/T或output tok/s；[vLLM协议](vllm-offline-benchmark.md)的output tok/s是独立end-to-end baseline。共享manifest SHA **3f2faab1…**冻结六组 **R1/2/4×supplied C64/256、固定128ignoreEOS**、exact IDs/seeds/T1/nofilter/no额外template。Native local section单独是active/residentR2、**committed C=(128,128)**、gamma7，从r2-c256的前128IDs/原seed构造snapshot并另emitanchor，不冒称整个C256 request。正式覆盖 **全64个ell∈{0,…,7}²、B2…16**；26balanced/leftconcentrated只是展示子集，不减少域或为未测layout作保证。

**Native计时与capability。** 每cell2warmup/5primary/1diagnostic，repeat-major遍历全部64，按17×repeat旋转、odd repeat反向；保留outlier/partial/unmeasuredcells。Snapshot精确恢复target/draft KV、RNG、markers/request状态，restore/content核验成本在样本外单列；pending/failed session拒restore，epoch/nonce即使重现，旧handle仍因对象身份拒绝。Primary同步完整fullshadow round，含实际proposal/private confidence hostcopy、q/p/合法prefix验证、target/head、commit/crop；不运行真实capacity search/calibration/history driver，明确 **capacity_round_driver_profile_eligible=false**。Lookup只接受已完整测量的exact domain/cell，拒R/context/layout跨域、缺cell/partial、插值与CapacityRoundDriver冒用；局部1/T不能隐藏planner排除成本。Diagnostic events/host spans可能nested，不能相加当独立kernel成本或替primary。复用identity-aware supervisor，300秒外层/290秒cooperative留partial，无缩域/retry；本轮尚无GPU成本数据。

**共享依赖漂移与CPU证据。** Pre-handoff捕获direction-owned helper在WIP中由615bd318…变为68dd26f3…，`dependency-drift.json`保留原失配与旧test/dryrun，不把先前pass当最终freeze。重新封存后native **8/8、5.676秒**（`tests-frozen-dependencies.log`）、supervisor **8/8、1.345秒**，两suite及真实Qwen/1280 stdlib finalbind均OS0、三GPU设备隐藏；tiny realKV/RNG restore、旧capability拒绝、partial/deadline/lookup边界均覆盖。直接读未发布 `output/packed-profile-cpu-20261009/` manifest/漂移/各log/finalbinding，独立核三交付及三dependency对 **700bfa60746f1315baaeec5ead28cc5ee553ae52**（11:34:33）Git blobs、九evidence SHA；重算native protocol **1014eb9e…**并核final source/dependency/workload绑定一致。此前6/8项不同阶段通过日志与 `(null)` 提示保留，final文件名不替代冻结顺序。

**单engine54batch与API边界。** vLLM预定只load一次engine，依次 **12warmup+30primary+12diagnostic=54**，primary六case按repeat旋转，fresh request/prefixcacheoff；完整54ordered (phase,repeat,case) identities由worker/controller独立验证/重算，缺diagnostic或reorder不能completed。Primary只计完整LLM.generate batch entry→return，包括submission/prefill/decode/IPC；params准备及output验证/hash在外，stats关闭故per-request latency/TTFT为null，不能用batchwall/R代替。五primary以总output/总wall算pooled tok/s，排除warmup/diag，不能平均batchrates。Diagnostic add_request/step DELTA按各request自己的host提交→first nonempty chunk/final计时，包含queue/observer/IPC/采样，不等于pureprefill，不能替primary。实际installed API源码留证核签名、DELTA prompt-ID echo与statsNone；missing echo拒绝，但source/fake engine tests不证明真实新multi-request/diagnostic可执行。

**Baseline准备身份与当前状态。** 直接读未发布 `output/vllm-benchmark-cpu-20261009/` handoff、installed-api-evidence、final tests与real-input preflights。Linux **15/15、1.307秒、OS0**（7measurement contract+8guard），worker/controller preflight均OS0；独立核六交付对700bfa6、四API source SHA、共享六case/full64域和54identity唯一性/phase计数，finalpreflight与native一致。Native manifest SHA **f1ea3ea2…**、baseline handoff SHA **fa477e60…**，详细完整身份保留在handoffs；十文件含README的700bfa6 root已核commit/push。Root刚给direction唯一正式vLLM窗口，但本条无启动状态，不写已运行/性能完成；native profile尚未执行。下一步须真实maxseq4/defaultgraph/diagnostic路径与完整54sample/释放验证后才报告baseline测量，再另核native窗口，不能由准备推出真实SPS/overlap/graph收益。原wholeQwen固定RMS失败及sampling fidelity边界保留。本轮记录者只日志，无GPU/源代码/实验/进程/提交或final-test/private样本动作。


<a id="vllm-formal-benchmark"></a>
## 2026-10-09 11:52（UTC+8）— S20：共享六shape正式vLLM baseline完成，单engine54批全部留证

**问题、假设与方法。** S19冻结协议后，检验强target-only engine在共享synthetic **R1/2/4×C64/256、每request128输出**下的实际吞吐，而不是从S18单request推速度。唯一正式执行source **700bfa6**、共享manifest **3f2faab1…**、BF16/T1/nofilter/ignoreEOS、memory0.18/defaultgraph/maxseq4不变，单engine复用、prefixcacheoff、每批fresh requests，按固定旋转顺序保留 **12warmup+30primary+12diagnostic=54**。直接读[九份报告](../reports/vllm-offline-benchmark-20261009-700bfa6/README.md)与公开scalar-samples，未读raw generated tokens或final test。所有54ordered identities完成，共 **16128output tokens**，其中primary **30批/8960tokens**；本轮无失败、超时、自动retry或手动eager/backend切换，所有五repeats与outliers保留。

**主结果与计时。** Pooled output tok/s按每cell总primary输出/总primary batchwall重算，排除warmup/diag，六结果如下（各五samples）：

| 活跃R | Supplied C64：output tok/s | Supplied C256：output tok/s |
| ---: | ---: | ---: |
| 1 | 131.264 | 127.340 |
| 2 | 252.165 | 246.567 |
| 4 | 515.998 | 475.260 |

计时是完整同步LLM.generate entry→return，含submission/prefill/decode/completion；params准备/output核验与hash在样本外。Primary **per-request latency/TTFT均null**，batchwall不是requestlatency，不能除R。每cell另两批diagnostic用各request自己的host submission→first nonempty chunk/final clocks，含queue/IPC/observer与第一token工作，**不等于pureprefill**、不替主结果；同批request相关、五repeat不支持高percentile serving tail。完整times/median/min/max和diagnostic统计见[primary](../reports/vllm-offline-benchmark-20261009-700bfa6/primary-metrics.json)与[diagnostic](../reports/vllm-offline-benchmark-20261009-700bfa6/diagnostic-metrics.json)。Engine init98.778713秒、imports/version20.506031秒在sample外；compiler/JIT/allocator自然warmcache与OMP2 warning保留，未按结果调参。

**实际backend、退出与释放。** 两版本分别核distribution0.30.0+rocm723/module0.30.0；默认ROCM_ATTN记录内部 **Triton paged-attention fallback**。Engine完成 **FULL_AND_PIECEWISE、capture sizes[1,2,4,8]** capture日志，**replay未独立instrument**。Worker/controller实际 **OS0**，supervisor cleanup为空；vLLM自身正常EngineCore SIGTERM与guard清理分开，不推断每个descendant都单独OS0，也不添shell独立exit。七ownedPID独立absent，ASR同身份/readyidle、KFDonlyASR、VRAM恢复used8967499776/free25241243648bytes；**35health samples**均ready/notbusy。采样全局VRAMmax **15068602368bytes**含常驻ASR，不是exact allocator peak/连续max或0.18硬cap；这是saved release/采样记录，不是本轮liveGPU探测。

**独立证据与CPU集成。** 记录者核九public handoff SHA对 **7bf281bae9dcf71293b2ea015c27517dcd650a22**（11:50:35）Git blobs，root已核commit/push；执行仍700bfa6。独立核archive SHA **71213ffb…**及 **288文件逐一对700bfa6 Git blobs**、八criticalsourceSHA、binding **3456files** count/hash及八份非样本stage/exit/health/release/integrity evidence SHA；完整bound前后检查依据verification，不重读大型weights。公开samples SHA **69c1ece5…**与raw scalar留证一致依据root独立检查，记录者另重构54顺序/phase、所有request128、16128/8960分母、六rates/medians、primary null和diagnostic clock bounds及35samplemax/七gone。独立stdlib verifier与root审计见[verification](../reports/vllm-offline-benchmark-20261009-700bfa6/verification.json)及[root review](../reports/vllm-offline-benchmark-20261009-700bfa6/root-verification.json)。另immutable700bfa6完整CPU suite **204/204、13.793秒、SSHOS0、120秒bound/三GPU隐藏**，证据 `output/integration-cpu-700bfa6/{verification.json,tests.log}`；记录者核log **ece14761…**/204entries/最终Ran/OK与source identity，不含native end-to-end WIP，不是新的GPU或CI验证。

**结论与下一决策。** 当前成立的是这组六synthetic shape、固定批次/固定输出的vLLM吞吐基线；没有自然输入质量/性能分布、float64-law fidelity、arrival-load/HTTP frontier、verification SPS/B曲线或DSpark speedup结论。Native R2/committedC128/full64 allocation是独立成本实验，不能与suppliedC256端到端request混同；core当前独立profile窗口的结果尚等完整交付，本条不提前写完成。待native全轮分解与matched trajectory再选优化，不以单kernel或graph capture替代系统收益。S16原失败/wholeQwen固定RMS失败仍保留；本轮记录者只日志，无GPU/remoteCPU/源代码/实验/进程/提交动作。


<a id="native-full64-profile"></a>
## 2026-10-09 12:03（UTC+8）— S21：full64 native local成本全部完成，same-B差异与嵌套计时不作因果/加速结论

**问题、假设与方法。** S19冻结local域后，唯一获授权的 **700bfa6** GPU执行测量active/resident **R2、committed C=(128,128)、gamma7、fullshadow eager**的全64ell，protocol **1014eb9e…**、actualtrained1280/target/runtime/FP64q/p不变。要检验同B不同allocation的成本与粗阶段热点，不能预设B-only transfer或拿local rate直接对vLLM tok/s。每repeat全64按17旋转/odd反向，snapshot/restore/exact内容检查在timer外并单列；实际round内付proposal/private confidence/能力验证/q/p/target/commitcrop，但不付真实capacity search/calibration/history，**CapacityRoundDriver用途仍拒绝**。直接读[完整报告](../reports/packed-profile-20261009-700bfa6/README.md)和公开scalar样本，不读raw token/probability traces。

**完成与主计时。** **64/64cells complete、128warmup+320primary+64diagnostic=512records**，每exact cell五primary/outliers全保留。320primary整体median **94.9779755ms**、range **78.388–153.418ms**，restore median **9.574ms**在样本外，model/setup/evidence亦不在timer。每cell localrate为 **1/median T_round**，不是B/time或output tok/s，更不是含真实scheduler的容量曲线。Same-B最大事后cell-medianratio **1.158 at B9**：ell(0,7) **91.215ms**，ell(3,4) **105.638ms**；按repeat成对5/5同向、差 **0.646–24.954ms**，但ranges **86.513–96.968 /90.597–118.258ms**重叠。两cell是observed extreme medians事后选择，accepted/committed分别 **0/2 vs1/3**；noise/drift与实际commit工作不同仍可能贡献，不能纯归因layout、不能事后发明transfer阈值。预声明balanced/left/right集中配对全部保留在[CPU audit](../reports/packed-profile-20261009-700bfa6/cpu-audit.json)。

**诊断区域与下一调查。** 独立diagnostic hostmedian：fullshadow **33.129ms**（含backbone **11.074**），private verify/commit **61.121ms**（含target append **43.461**、draft committed projection **4.388**、crop每requestcall **1.651**）；diagnostic roundmedian95.092ms、两topspan外remainder1.069ms。它们nested，不可相加为独立组件；GPUeventinterval含enqueue/dispatch gaps/同步影响，不是isolatedkernel时间，crosspass subtraction也不证明精确host/device拆分。当前证据把target append和proposal中backbone外工作列为下一调查区域，尚未区分其中attention/MLP/metadata/host同步/copy谁主导。Persistent KV/layout buffers、deterministic target subgraph是待比较候选，不宣布graph或优化收益。

**身份、退出与边界。** 记录者核六public SHA对 **53d1202626b89237d685c525deb8c37ae7a8e16f**（12:01:34）Git blobs，root已核commit/push；source仍700bfa6。独立核source archive **71213ffb…**与S20逐Git核的288文件archive同SHA、41executedsourceSHA、公开完整samples SHA **ca8c40af…**；从scalar重构512顺序/phase/64全域、各cell median/1T、全primary/restore统计、B9pair/commit及所有diagnostic region counts/host-event medians。原512work逐项独立核与rawscalar逐字节一致依据core本地stdlib audit和root `output/packed-profile-gpu-20261009-700bfa6/root-scalar-verification.json`，不冒称本轮重新执行GPU。Worker/controller/SSH **OS0**、no timeout/retry/fallback、三ownedPID保存独核gone；ASR同身份ready/notbusy/KFDonlyASR，VRAM恢复used8967499776/free25241243648bytes。Peakallocated/reserved **2268122112/2348810240bytes**；supervisor107.010秒含setup/restore/evidence，不是serving吞吐。Root另同步更新README的过期STS未拟合/wholegate pending状态，历史失败留存。只能查exact frozen R/C/input/cell，无growing context/churn/其他R外推；不与S20 end-to-end tok/s直接求speedup。WholeQwen target固定RMS失败未解除，下一matched E2E尚未执行/最终CPU修复交付另记；本条不提前写解决factory或memory review问题。记录者仅日志，无远端/GPU/实现/进程/实验/提交或final-test动作。


<a id="native-e2e-cpu-prep"></a>
## 2026-10-09 12:11（UTC+8）— S22：native六case端到端CPU准备，修复重复adapter与测试/内存口径缺口

**问题与方法。** S20已有强vLLM吞吐、S21只有local成本，仍缺matched native完整trajectory。[Native协议](native-offline-benchmark.md)复用共享manifest **3f2faab1…** 的 **R1/2/4×suppliedC64/256×output128** 与单model/adapter的54batch（12warmup/30primary/12diagnostic），eager fullshadow、draw前固定 **ell=max(0,min(7,remaining−1))**，不接capacity scheduler/history/graph。Admission首token计入128；remaining1虽ell0仍付完整shadow与anchor验证。Finished request inactive但KV resident到batchend，后续copy/gather如实收费。FP64实际q/p/check/RNG/reject/commit语义不改，仅在完成verify/scalar记录后释放观测q/p引用，不跨round保留fullvocab tensors。固定 **1800秒worker/1790秒cooperative、8GiB free/6GiB processallocator cap**与identity-aware supervisor，不能按进度缩panel或自动retry。

**失败、修复与S21待办闭合。** 首轮 **6tests/5.324秒、2errors**：在已经注册native的model config上重新构造VarlenPackedTarget，触发explicit-mask SDPA guard。改为production `make_session_factory`只构造/注册一个target adapter，每batch在timer内reset target cache、创建新的draft/session cache；同target/draft model复用。后续6/6与7/7中间日志保留；root/direction review又指出tests曾手写duplicate factory，不能验证生产路径，以及memory peak未说明继承上一batch targetKV。最终tests直接调用生产helper、只注入backend constructors；明记 **pre_reset allocated/reserved baseline + whole_operation_peak**，从reset前到结束的peak包含上一batch残留targetKV、model与预备输入，不当作currentcell独立steady-state峰值。S21留下的两个review缺口在CPU准备中修正，不倒改其当时状态。

**计时、诊断与CPU证据。** Primary从fresh native cache/request构建前到admission/fullshadow所有round/私有验证/target/commitcrop/finalsync，token tensor/RNG准备成本单列在外，serialization/hash/重复finaloutput验证在后；request latency/TTFT仍null。Native diagnostic以共同batch提交为start、观测admission返回/各request结束round，含setup/sampling/projection/observer，区别于vLLM各request add_request时间边界，不把两者当相同服务timestamp或pureprefill。Worker/controller仍要求54完整有序identity，缺diag不能completed；deadline只保留此前完整samples和全panel，未完成batch不伪造计时。最终 **8/8、5.474秒、OS0、三GPU隐藏**，真实target/selected1280/workload/source stdlib finalbind **OS0**。实际tiny BF16连续batch验证单adapter/新cache、首tokenbudget、ell0fullshadow与diagnostic；新增unequal A/B→A fixture核finished B的 **15KV rows**仍resident/收费，下一实际batch rounddeadline核此前sample保留。

**身份与当前边界。** 直接读未发布 `output/native-end-to-end-cpu-20261009/` manifest、各失败/修复logs及finaldryrun；独立核manifest SHA **ace4ec8f…**、三交付当前SHA、八evidence SHA，按真实compact序列化重算protocol **b6a7b1dd…**、核54identity/phase分母及script/selectedweights/sharedmanifest绑定一致。CPU执行为immutable700bfa6 archive加明确overlay，source.tar仍 **71213ffb…**；三新文件当时交root review、未添不存在的commit/push，不把含其他WIP的current tree当tested snapshot。**初次交付时没有native E2E GPU授权或执行**、无速度/whole-system pass/graph/overlap/fidelity结论；原target固定RMS失败仍限制速度比较的law解释。下一步root审查冻结source与独立窗口后才能测matched trajectory，本条不自行开GPU或将local94.978ms外推成已测E2E时间。记录者只日志，未提交、未改实现/实验/进程或读取private trace/final test。


**12:20后续核对：提交、完整suite与执行状态。** 上述初次交付只证明新增路径的CPU准备；要启动完整trajectory，还需冻结整套源码并核完整suite。Root随后确认代码已提交/推送 **0c36b03416311c0ca529d10ff0a10663ebd407fd**（Git时间12:09:58；这是代码提交时间，不把它当作记录者知悉/窗口授权时间）。记录者直接核 `output/native-e2e-gpu-20261009-0c36b03/expected-source.json` 与 `source.tar`：archive **ad8ee63b…** 的307文件逐项SHA与该commit Git blobs一致，原三交付文件亦与初次manifest一致，protocol/workload绑定未变。完整CPU日志 **06f99105…** 明记 **212tests/13.648秒、OK**；OS0及三GPU隐藏执行条件由core/root确认，本轮没有重新跑suite。原开头两行 `(null): No such file or directory` 原样保留，不把它们删掉或误作测试失败。随后root授权首个all54/1800秒独立窗口并进行preflight；最新root交接已报告controller/worker启动，因此当前是 **native E2E执行中，尚无完成、54batch完整性、速度或释放证据**。PID/startticks属于其执行留证，不据此判断成功。本条等待真实结束后另补结果，旧target RMS失败与law边界继续保留。


<a id="persistent-target-kv-cpu"></a>
## 2026-10-09 12:20（UTC+8）— S23：persistent target KV独立存储候选，CPU事务/真实tiny Qwen验证与设备别名修复

**问题与假设。** S21把target append列为较大成本区域，但其43.461ms host median含模型计算、动态layout、cache拼接与gather，不能直接当作可消除的开销；crop每call1.651ms同样不是独立kernel成本。当前先把已提交KV与验证中的KV分开，让缓存地址与所有权可被验证，再判断能否形成稳定的target子图。[候选设计](persistent-target-kv.md)引用S21同一raw scalar结果 **ca8c40af…**；这一步没有提前承诺graph或速度收益，也未修改现有decoder/oracle/sampling/benchmark。

**存储与提交方法。** Resident KV固定为 `[L,slot,Ccapacity,Hkv,D]`，verification scratch另有 `[L,max_query_tokens,Hkv,D]` 分配；`load_prefix`导入模型实际含位置/RoPE的KV。Store签发的Slot绑定request/slot/incarnation及对象身份，拒绝旧、复制、伪造或跨store handle；`begin`冻结活跃顺序/context和metadata，单事务pending时禁止churn。每layer新KV先copy到scratch，再gather committed+scratch供attention；只有全部layer完成stage且整组决策通过检查，才把验证前缀copy回resident。提交数包含旧anchor，通常为accepted+1，拒绝尾部不进入resident；abort不改已提交内容。Metadata准备失败不发布事务，retry重写metadata且旧capability仍失效；resident bootstrap/commit发生copy或device错误则poison pool，不声称多layer原子回滚。Inactive request的KV继续resident并计入费用，审计接口返回detached副本；单owner约束也不等于防止任意Python直接改公开tensor。

**稳定地址为何仍可能更贵。** Bucket固定ordered Q向量和context ceilings，实际C、positions、slot及cuK可在上限内增长；physical Q严格等于logical Q，不用maxQ padding抹平allocation差异。Work分别报告真实K、capacity/padding、inactive K、causal pairs及resident/scratch/workspace bytes。然而每bucket有六个capacity-shaped K/V staging tensor，对每个capacity row同时gather resident与scratch再选择，可能比旧exact-size路径增加流量。注册有限且显式；若枚举大R的全部ordered ell向量，workspace会组合爆炸，生产graph cache还需选择/淘汰及总预算。稳定指针只是可测前提，既未证明更快，也未解决graph-cache容量。

**真实review失败与修复。** 首次本地七测试 **1.190秒、OK** 后，root review发现device身份保留了未解析的 `torch.device('cuda')`/`cpu:0`，可能与实际分配device不等。Direction实际CPU复现 `cpu:0` 配置分配到 `cpu`，合法prefix因device比较报ValueError；随后用首次分配的 `keys.device` 规范化后续分配与检查。新增实际indexed-CPU bootstrap/stage/attention/commit回归后为 **8tests/1.281秒、OS0**。七测试log与before-fix失败留存，最终证据读取handoff指定的 `tests-device-fixed-eight.log`，不把旧 `tests.log` 当八测试结果。CUDA别名采用同机制但未执行，不能写作GPU验证。

**CPU证据与身份。** 八测试核增长C时resident/scratch/staging pointers稳定、ordered allocations、active subset/inactive residents、scratch/partial或zero commit/abort隔离、capability拒绝、非法决策及注入metadata准备失败/retry；独立variable-prefix CPU SDPA对照验证staging语义。实际随机初始化tiny Qwen的所有layer KV导入并部分提交，再与独立完整prefix forward比较，不是预训练target或native dispatch验证。使用现存本地 **Torch2.11.0/Transformers5.4.0、CPU、HIP/CUDA/ROCR隐藏**，与native固定 **Torch2.12.0+rocm7.2/Transformers5.17.0** 不同；没有安装包或远端运行。Root审查后提交/推送 **2010503571d551ec887eb411e53ad54e53f45eb2**（12:17:56）。记录者独立核最终handoff/current文件/该commit三者SHA一致：module **b7eea1b3…**、tests **77f5768b…**、doc **00ec7cb1…**；final log **64292238…**、alias失败 **ac775b9f…**，均在未发布 `output/persistent-target-kv-cpu-20261009/`。旧七测试与最初handoff属于修复前阶段，未用于证明最终版本。

**缺失机制与下一决策。** 当前private ROCm wrapper要求K tensor长度等于actual cuK末值，并把GPU layout值读回Python，不能直接消费capacity tail；CPU切片attention也证明不了native支持。下一项单独授权的bounded capability check应在同一小bucket的两个增长context复用指针，先比较exact-length eager与capacity-tail native，poison未用尾部确认不读，再考虑fixed maxK下capture/replay及变化positions/cuK后的输出/commit核对。Exact-Q bucket也不能从t−2全局K预选：当前各request ell仍依赖当前confidence；未来R/physical B/maxQ家族需要动态cumulative值的native支持，多余physical queries须隔离并明确收费。单事务不支持CPU/GPU overlap；未来双bank还需producer/consumer所有权、stream events以及shared resident commit/scratch hazard顺序。当前 **没有native/capture/replay/graph/overlap、decoder integration或speed证据**，也不解除旧whole-Qwen RMS失败或cross-backend law限制；记录者只核日志与公开/scalar/source证据，不执行下一GPU实验。


**12:30后续：固定AMD环境的CPU对照完成。** S23首次本地验证使用较旧依赖，不能据此假定固定native环境也能通过。Root随后将immutable **2010503** Git archive的最小package/init、module、test流式送入AMD临时目录，在 **Torch2.12.0+rocm7.2/Transformers5.17.0** 上仅跑同一CPU suite；CUDA/HIP/ROCR全隐藏，日志明记 `torch.cuda.is_available=False`。实际 **8tests/3.664秒、SSH OS0**，记录者直接核本地 `output/persistent-target-kv-cpu-2010503-amd/{tests.log,ssh.os-exit,verification.json}`，log SHA **7ebb9f47…** 与verification一致，三执行源码SHA与2010503 Git blobs一致。本次操作由root执行，记录者没有远端动作；原 `(null): No such file or directory` stderr原样保存，未猜测原因。这个对照消除了该CPU suite在固定依赖版本上的未测状态，仍只有CPU存储/attention/tiny-model语义证据，没有GPU device alias、native capacity-tail、capture/replay或overlap证据。


<a id="native-e2e-result"></a>
## 2026-10-09 12:30（UTC+8）— S24：matched native E2E完整完成并释放，六case均显著慢于vLLM

**问题与方法。** S21固定cache的一轮成本不能回答完整生成是否获益：实际还要admission、不断增长的KV、草稿采样、概率检查、接受/回退与提交。S24执行S22冻结的 **0c36b03** 单model/adapter eager full-shadow协议，保持共享R1/2/4×suppliedC64/256×output128、相同输入/输出预算和FP64实际q/p法则；draw前固定最大合法prefix，不使用capacity scheduler/history、persistent KV、graph或overlap。它测的是当前完整实现能否胜过S20的vLLM engine；执行路径与sampler不同，工作负载匹配并不构成输出law等价。结果已完整留证，不再沿用S22启动阶段的pending判断。

**完成与结果。** 54/54批按原顺序完成（12warmup、30primary、12diagnostic），全部 **16128输出token**，primary **8960**；每cell五primary完整保留。[正式报告](../reports/native-offline-benchmark-20261009-0c36b03/README.md)与[primary证据](../reports/native-offline-benchmark-20261009-0c36b03/primary-metrics.json)按sum(outputs)/sum(batch wall)计算，不平均每batch tok/s，也不使用supervisor总时间。下表时间倍率为同等输出数的pooled总batch wall之比。

| R / supplied C | Native tok/s | vLLM tok/s | Native / vLLM吞吐 | Native / vLLM耗时 |
| --- | ---: | ---: | ---: | ---: |
| 1 / 64 | 16.0924 | 131.2641 | 12.26% | 8.157× |
| 1 / 256 | 13.9283 | 127.3399 | 10.94% | 9.143× |
| 2 / 64 | 23.6857 | 252.1654 | 9.39% | 10.646× |
| 2 / 256 | 24.0317 | 246.5667 | 9.75% | 10.260× |
| 4 / 64 | 36.4076 | 515.9980 | 7.06% | 14.173× |
| 4 / 256 | 33.8484 | 475.2599 | 7.12% | 14.041× |

当前native只达到vLLM吞吐的 **7.06%–12.26%**，没有加速；R增加时native吞吐上升，但相对engine差距扩大，不能把多请求打包可执行等同于engine效率已达标。Native各cell primary batch median为 **7.960/9.122/10.745/10.693/13.962/15.092秒**，不是S21的94.978ms local round外推结果。比较值与限制见[matched comparison](../reports/native-offline-benchmark-20261009-0c36b03/matched-comparison.json)。

**接受与成本能支持什么。** 这次确实接受了草稿，问题不是完全零接受：每cell每primary batch的accepted为 **36/21/54/44/126/85**，五repeat计数一致。但accepted/selected只有 **2.94%–5.77%**，每活跃request-round平均只提交 **1.198–1.396token**；128输出含admission首token，剩余生成仍需 **91/106/108/106/112/119轮**。与此同时，每活跃请求每轮付七位置fullshadow；selected target query（旧anchor+选中草稿）为 **715/820/1551/1626/2969/3291行**，约是admission后最终提交行的 **5.63–6.48倍**。因此少量接受减少部分顺序步骤，却仍伴随大量未提交query、草稿/概率/验证/crop及动态cache工作；全样本还记录 **99个含零prefix的round、621个存在inactive resident的round**，没有因结束或ell0隐去费用。[Work统计](../reports/native-offline-benchmark-20261009-0c36b03/work-metrics.json)是该固定synthetic轨迹的描述，不能外推为真实prompt接受率、校准收益或纯算法成本。比127个target-only顺序步骤少 **6.3%–28.3%** 的batch append调用只是反事实调用数对照，不证明相同每call成本或吞吐获益。

**为何完整链路与engine不同，尚不能归因什么。** Native执行真实draft+target与FP64采样/校验、动态metadata和cache提交；vLLM是自己的target-only执行/采样路径，已有graph capture配置与专门engine。完整结果显示减少调用数没有转化为吞吐收益，但没有独立消融去分出接受率、host同步、数据移动、模型kernel或engine优化各占多少。同backend native target-only控制尚未运行，因而还不能将native执行本身与新增draft/verification成本分开归因。S21的append/fullshadow nested host spans只指出调查区域，不能直接扣减本次wall time；vLLM replay仍未独立追踪，更不能把全部差距归因于graph。Primary TTFT/request latency仍null；另12diagnostic以共同batch提交为start，TTFT medians约 **54–61ms**，含整个batched admission/projection/observer，不是pureprefill，也不同于vLLM各request add_request边界，不能直接比作相同服务延迟。

**身份、结束与证据边界。** 执行源码仍0c36b03、archive **ad8ee63b…**（307文件已在S22逐Git核），固定 **Torch2.12.0+rocm7.2/Transformers5.17.0、gfx1201、pinned native ROCm ATen**，单target与trained step1280 draft；raw samples字节SHA **4fac289b…**。记录者重核公开scalar **fb0ea6d8…** 与source identity一致，从54行重构phase/输出/round分母及每cell五wall times、native pooled吞吐，再从S20五wall times独立重算vLLM吞吐与全部比值；另核每cell九次输出hash向量一致，仅是same-path重复证据；每轮allocation、active/retained roster、commit/KV计费的原始核对依据core independent audit及root `root-scalar-verification.json`，不冒称本轮重跑GPU或重新逐轮审计。公开报告去掉raw round列表，保留scalar聚合与原始SHA，记录者未发布或读取生成token/private样本。Worker/controller/outer **OS0**、no timeout/cleanup/retry/fallback；core与root分别检查owned worker/controller/shell消失，保留ASR同start身份、ready/notbusy，KFD仅ASR，VRAM回到used **8967499776** / free **25241243648bytes**。这些是保存的结束检查，不宣称此刻实时状态。[Runtime audit](../reports/native-offline-benchmark-20261009-0c36b03/runtime-audit.json)记录124health samples、global sampled peak **12178845696bytes**（含ASR），whole-operation allocated/reserved peak **2459580928/2728394752bytes**；后者从reset前至结束包含继承targetKV/预备输入，不能作独立cell稳态峰值。Supervisor **639.694秒**含setup/evidence，不能作生成吞吐分母。

**下一决策。** 当前证据足以否定这六个固定batch上“已实现加速”的判断，并支持把低接受与完整实现成本同时作为改进问题。下一步应先核S23的native capacity-tail固定buffer可行性，再考虑target子图与端到端消融；persistent存储尚未接入本次decoder，不能认领收益。Graph capture/replay、CPU/GPU overlap、真实capacity planner、arrival负载frontier仍缺证据。旧whole-pretrained-Qwen layer26固定RMS gate仍失败，same-input endpoint TV及cross-backend law非等价限制不因本次完成而消失；该速度比较不是分布忠实性pass、quality改进或完整系统复现。记录者只编辑本日志，未操作GPU/进程、改实现、提交或读final test。

S24公开报告与项目状态入口已由root提交并推送 **2934342114b99943d78351265d4b8566b8270479**。随后独立verifier补充核对两引擎的workload、八个model/tokenizer指纹及vLLM原始sample SHA，重跑通过；不改变任何测量值。


<a id="native-capacity-graph-result"></a>
## 2026-10-09 12:39（UTC+8）— S25：native capacity-tail与gather+attention真实capture/replay通过，仍非完整target graph

**问题与因果进展。** S23只验证固定resident/scratch/storage语义，S24又显示完整native链路明显慢于vLLM；稳定地址并不自动意味着native算子能接受capacity tail，更不意味着能capture。当前按[固定probe协议](native-capacity-graph-probe.md)先测尾部不会被attention使用，再测固定地址的gather+attention能否在context增长时真实replay。[正式报告](../reports/native-capacity-graph-20261009-f5d03d4/README.md)记录三步现已分别有证据：CPU持久存储、同QKV native tail隔离、这个局部subgraph真实capture/replay；它们仍没有连接成完整target执行或吞吐优化。

**准备、review与冻结。** 最初五CPU contract tests通过后，root review发现隔离检查只覆盖committed K，漏了V；即使输出数值正确，也可能已污染resident values。Direction加入K/V双before-clone与双torch.equal检查，以及eager和replay的V-only污染反例；本地最终 **6tests/1.738秒、OS0**，旧五测试与中间六测试日志保留，不将review前覆盖不足写成已经完整隔离。CPU用了本地 **Torch2.11.0/Transformers5.4.0** 的真实tiny Qwen和明确标注的CPU replay emulator，不是真实GPU graph。源码随后经root审查提交/推送 **f5d03d4ca82482a65963429e94e4068896f5a77a**（12:32:47）；core在其immutable archive上跑完整 **226tests/13.806秒、OS0、GPU隐藏** 后，才进入唯一原定 **300秒/290秒cooperative** 窗口，无retry、缩domain或阈值回调。

**真实输入与同QKV参照。** 正式worker绑定实际pretrained Qwen3-0.6B与共享r2-c256 synthetic输入，BF16首层的normalized、已RoPE Q/K及projected V由模型实际forward提取，随后恢复原attention setting。QKV生成与RoPE在capture外；未来context改变通过外部刷新实际QKV体现，不能把更新positions tensor说成graph已执行RoPE。单bucket固定 **R2、ordered Q=(1,4)、physical Q5、ceilings(144,144)、Kcapacity293、maxQ4/maxK148**；context依次 **(128,128)→(129,131)→(130,135)**，cuK末为 **261/265/270**，尾容量 **32/28/23** 行。Reference独立拼接当前已提交KV与新实际scratch bytes，避免把另一条完整prefix浮点重算的KV当作同QKV oracle。人工commit为 **(1,3)/(1,4)/(0,0)**，只是事务fixture，不是实际采样/接受事件。

**先eager tail，后capture。** 三state各五variant：exact actual maxK、exact fixed maxK148、capacity zero、capacity finite（K尾100/V尾−100）、capacity NaN。前三按执行前冻结的elementwise **atol0.02/rtol0.02、RMS≤0.005** 做pooled/逐request比较；后两必须输出finite且与zero-tail **bit-identical**，NaN gate没有被豁免。全部 **15 eager** 通过后才新建pool、两次side-stream warmup并一次真实capture。Body只有fixed-size index_select双source gathers、where选择、invalid-tail zeroing和private native attention；metadata/host copy、QKV/RoPE、模型其余计算、采样与commit均在外。随后三个增长context都更新真实QKV及cumulative metadata并 **真实replay**：与各state exact eager native数值比较通过，同时与对应zero-tail capacity eager bit-identical；Runtime逐次检查18个input buffer地址稳定，capture output地址逐次直接记录并保持稳定。18个主比较×pooled+两request的 **54行** 在本次实测均finite、maxabs/RMS0且bit-equal；这是比原阈值更好的观察值，不改协议或外推到其他shape。

**事务隔离的证据层级。** Eager与replay在人工commit前均用runtime双K/V snapshot比较断言resident未变，随后commit又与独立old-prefix+选定scratch prefix比对，标量记录全部隔离通过。K/V before clones未dump为完整resident快照，所以不能声称离线复算了这些未保存的全resident bytes；这部分是source-bound runtime assertions、V-only CPU反例与保存scalar证据。当前production wrapper的exact-K guard没有修改，直接调用private native schema限于该probe；成功证明这个固定bucket下cuK末小于Kcapacity可用，不证明所有capacity形状都可放行。

**身份、退出与核对。** 正式环境 **Torch2.12.0+rocm7.2/Transformers5.17.0、gfx1201、固定ROCm ATen/AOTriton schema**。记录者核本地 `output/native-capacity-graph-gpu-20261009-f5d03d4/`：archive **7d2714bc…** 的324文件逐SHA与f5d03d4 Git blobs一致；三准备文件与final handoff一致、本地six-test log **ea1c9f3b…**，full suite log **49a11c11…** 明记226/13.806/OK。Result **6ecbe61f…** 记录real_qkv/eager_tail/capture_replay三stage passed、完整固定18观察顺序；记录者独立重构比较分母、cuK、初始input pointer字典与每次实际output pointer一致，与root scalar+exact-Git核对一致；保存的input字典是初始快照，后续input地址稳定依据执行中的比较flag，不能把重复字典当作独立重测当前地址。Worker/controller/outer **OS0**、no timeout/cleanup/retry；保存独立release显示worker/controller/shell消失、ASR同start identity ready/notbusy、KFD only ASR、VRAM回used **8967499776** / free **25241243648bytes**，source/model/workload post-integrity仍通过。Supervisor **19.940秒**包含模型提取、warmup和disk evidence，不能作速度指标；四health samples的global peak **10742800384bytes**含ASR，allocator peak未记录，6GiB配置cap不冒充实测峰值。记录者没有操作远端/GPU/进程。

**独立raw tensor复算与公开证据。** Core随后在CPU独立读取42个已保存tensor artifacts，从初始prefix与人工commit重构增长context的K/V，并逐项核actual QKV、cumulative lengths、positions与zero/finite/NaN尾；共108个exact input tensor对照、21组输出比较（含三个replay对capacity的额外组）/63个pooled或request比较，maxabs/RMS0，output storage bytes也相同。记录者核该audit的scalar结果、42文件字节SHA及[公开verification](../reports/native-capacity-graph-20261009-f5d03d4/verification.json)与原audit逐JSON一致，[公开result](../reports/native-capacity-graph-20261009-f5d03d4/result.json)与原result一致；没有自行加载private QKV tensor或冒称执行了core的离线tensor复算。Core audit **dd97233e…** 明列未保存full resident snapshots及input pointer初始快照的局限，前述runtime隔离不能越级为离线full-pool证明。公开只保存scalar/哈希/审计代码，42个raw tensor保留在未发布output证据中。

**结论边界与下一决策。** 这解除S23在该固定first-layer shape上的native tail与局部capture能力未知，但 **不是whole-layer/whole-model graph、sampling-law pass或加速结果**。Ordered Q=(1,4)已知且不变；当前ell依赖当前confidence，所以不能从t−2 global K单独预选该graph，也未实现zero-overhead scheduling。未来还需确定full-target deterministic边界、有限bucket cache/总workspace预算、sampling/commit与resident所有权，再测double-bank+events及CPU/GPU overlap；这些不由pointer稳定自动成立。S24的无加速结果没有被本probe改写，旧whole-pretrained-Qwen固定layer26 RMS失败及cross-backend law/endpoint TV限制仍保留。下一步是有界集成和独立验证，不把这次disk-heavy能力probe转换成SPS或serving吞吐。

S25报告与项目入口已由root提交/推送 **51d1c3d**。Root另在本地CPU重跑公开tensor verifier，实际OS0，输出与core原audit逐字节一致；运行证据保存在同一私有目录的 `root-rerun-public-tensor-verifier.json`。


<a id="persistent-full-qwen-cpu"></a>
## 2026-10-09 12:51（UTC+8）— S26：完整HF Qwen层接入persistent事务，CPU语义与固定5.17接口验证通过

**问题与方法。** S25只capture首层gather+attention；要形成完整target路径，还需各layer将新KV写入scratch、继续原模型计算并保留正确的draft上下文特征。S26的[CPU集成候选](persistent-qwen-target.md)直接复用HF Qwen原embedding、全部decoder layers的QKV/QK norm/RoPE/output projection/MLP、final model norm与LM head，新增Cache.update与request-local causal callback把存储边界接到真实层执行，避免另写一份模型算术。它仍是eager、opt-in、明确CPU-only；constructor要求CPU model和显式CPU test kernel，GPU被拒绝，S25局部native成功不被当作全target native授权或证据。

**缓存、位置与特征为何分开。** 每layer真实已RoPE K/V由Cache.update复制到该layer scratch，gather committed prefix+scratch后返回同一HF attention层，模型继续执行原残差/MLP与后续层。Explicit per-request position IDs来自事务各自C，不能使用flattened resident总长度作为各request RoPE位置；mask mapping由callback显式处理，generic mask推断被拒绝。Cache明确 `is_compileable=False`，空layers列表不成为可编译/可capture声明。Selected-layer hooks在取得model所有权时注册一次，按layer-ID顺序收raw block outputs；最后decoder block的raw输出若被选中，也不能与final-normalized `features.last`混同。独立tiny Qwen测试选择了最后block并核两者不同，logits始终来自原LM head对final norm输出。

**显式事务与失败边界。** `prefill`只接受新admission的空request，执行完整forward后提交全部prompt；`verify`执行全部层但仅写scratch，返回严格对象身份绑定的features capability；`commit`整组验证输入前缀计数后才写resident，计数包含旧anchor，不是accepted draft数。`abort`或ordinary layer-forward失败丢弃scratch事务，保留所有resident K/V和lengths以便retry；pending时第二forward、admission/removal/reset被拒绝。Resident copy/device失败仍可poison pool，不承诺跨layer原子rollback或与draft cache的联合事务。Inactive resident保留并收费；request复用slot时incarnation更新，reset释放requests而保留bucket allocations，close移除hooks并恢复原backend。Work里的scratch-only write policy是契约字段，实际pre/post双K/V检查来自CPU测试，不把字符串当runtime全pool审计。

**本地CPU证据与冻结提交。** 实际随机初始化 **四layer tiny Qwen** 与独立逐request HF forward比较全部四层KV、selected raw block features、final norm及full/last-row logits，检查不等context的prefill、两次增长context partial/full commit、pending resident双KV不变、非法/复制features、abort及layer1注入failure后的retry。Changing inactive request内容时，active features/logits **bit-identical**；另核inactive KV保留、slot reuse/reset stable pointers与精确bucket/input拒绝。初始 **4tests/0.052秒** 后的最终 **4tests/0.066秒、OS0** 使用现存本地 **Torch2.11.0/Transformers5.4.0、CPU、三GPU visibility隐藏**，没有预训练target/fullnative/graph或性能验证。Root审查后提交/推送 **f46e63f554f7144738e72accc37eb6836103c469**（12:46:42）。记录者核 `output/persistent-qwen-cpu-20261009/` final log **b7092b75…** 与handoff一致，三文件current/handoff/Git blob SHA一致：module **015a21ed…**、test **5d0dcc79…**、doc **27062511…**；旧initial log原样留存。

**固定版本的后续验证。** 初次本地通过尚不能证明固定5.17接口兼容，core随后在immutable f46e63f AMD archive上用 **Torch2.12.0+rocm7.2/Transformers5.17.0** 跑完整suite，实际 **230tests/13.647秒、SSH OS0**；CUDA/HIP/ROCR变量均空，`cuda_available=False/device_count0`。记录者直接核本地 `output/persistent-qwen-pinned-cpu-20261009-f46e63f/` 的final summary、versions、source verification、full suite log与OS code：summary六evidence SHA全一致，分别明记test process与实际SSH OS0；archive **073edbb4…** 的334文件逐SHA/Git blobs一致，full log **3c25570f…** 明记230/13.647/OK；原 `(null)` stderr保留，不猜原因。固定5.17 API留证显示model接收explicit mask dict/per-request positions，attention在原RoPE后调用Cache.update再走registered callback，decoder block raw输出后另执行final model norm；本次suite实跑通过，未观察到需要修复的版本兼容失败。它解决此CPU candidate在固定依赖上的未测状态，不能外推为完整预训练模型或GPU路径已通过。

**尚未接入与下一决策。** 当前不是旧append/crop drop-in：session的admission需明确改走prefill，verification改走verify，并在所有决策确定后一次target commit，而不能沿用逐request crop。Direction已开始该接入，但本条没有完成或集成测试证据；实际q、FP64 law、RNG、EOS/output budget及committed context排除最新anchor的约束继续原样保留。当前 `predict(last_only=True)`仍先投影全部Q行；未来admission必须维持先选R末行再LM head的优化，不能新增full-prompt vocab projection费用。Full-model graph也未实现：Python Cache.update的staged-set/hook bookkeeping只在capture时执行，replay不会重新发布新事务；还需外部staging/feature所有权、persistent outputs及stream/events证明完成后才能commit。Exact ordered-Q不能仅由t−2 K确定，bucket预算和双bank/overlap仍未解决。旧target RMS失败、cross-backend law限制和S24无加速结果均保留；四CPU测试与230suite不解除这些边界。

**工作协调续记（不作实验结果）。** Root核现有 `dspark-astra` 30分钟ACTIVE heartbeat，并把prompt里已不存在的experiment_journal改为现存sol_data，要求继续保留vLLM强baseline及同backend control的区分；周期与通知规则未变。此处只记录root的协调状态，不新增实验、GPU授权或性能结论；记录者没有修改automation、远端环境或实现。


<a id="persistent-session-cpu"></a>
## 2026-10-09 13:10（UTC+8）— S27：persistent target接入随机session，完成CPU缓存/采样/feature生命周期验证

**遇到的问题。** S26已有完整CPU target，但旧session在验证时先append、再逐request crop；新的target要求先把所有层KV放在scratch，等所有请求的接受/回退决策确定后一次commit。若直接伪装成旧接口，会混淆哪些token已经提交，也可能在draft投影仍读取selected features时过早释放输出。本轮要解决的是把这两种生命周期明确接起来，同时保留原实际q/p、RNG和输出预算规则，不借接口重构改变采样行为。

**为何这样接入。** [Session设计](persistent-sampling.md)新增显式可选target strategy：默认仍走原append/crop；选择persistent时，admission走prefill，verification走verify，按原request顺序计算完所有决策后，只调用一次target commit，再把同一选定raw feature前缀投影进draft KV，最后release features。Admission仍先选每request的final-normalized末行，再做单次 **R行LM head**，测试直接观测该输入shape，避免把全部prompt行投影到vocabulary。通常提交验证输入中的旧anchor+accepted前缀；遇EOS或budget提前停止时，按实际新输出数提交相应输入前缀，让target/draft都保持 **prompt+output[:-1]**，最新输出仍是下一次的excluded anchor。Finished/inactive request的KV继续保留；remove/readmit更新incarnation并拒绝旧proposal。

**实际失败与修正。** Direction交接及handoff记录，早期working-tree八测试首次暴露：output-budget case会需要 **ordered Q=(3,)**，fixture没有声明该verification bucket。修复是补上明确bucket，不改budget或接受算法、也不隐式补任意shape；随后八测试通过，再加入admission failure测试成为最终九项。此段只依据提供的交接/manifest描述，不补造未提供的初次失败时间、时长或exit。这个问题也说明有限bucket不是装饰：调用方必须声明admission、active subset、改变allocation与收尾budget所需形状。选择只在已声明的exact ordered Q与context ceilings中进行，以最小兼容Kcapacity优先、声明顺序打破平手，不枚举全部shape或暗加maxQ padding。

**拒绝和故障如何处理。** Admission在加slot和消费RNG前检查bucket；固定budget step在draft forward/draw前检查已知shape。Full-shadow的proposal与真实draw已经发生，随后bucket拒绝只能保证不再做target forward/verification draw，不能倒称shadow RNG没消费。正常bucket拒绝保留可供合法allocation继续使用的request/cache/proposal；ell0仍付fullshadow，只验证anchor一行，私有实际q不受观测副本修改影响。真正执行错误则永久invalidate session，按pending scratch abort→feature release→target reset尝试清理，并总是清空draft cache；清理失败保存cleanup_error，所有public操作仍拒绝复用。没有承诺撤回已消费RNG、回滚device部分写入或target/draft联合原子提交；测试分别覆盖commit前故障、commit后draft故障、poisoned commit/abort清理，以及admission的slot/commit/draft失败。

**九项CPU测试能证明什么。** Tests执行完整tiny Qwen target与真实draft投影，并用独立逐request HF cache重构检查两套KV。Same-seed原默认、显式append/crop与persistent路径在fixture上的输出/RNG trace相同，实际q/p按声明数值容差对照；原默认与显式旧strategy的target p另作bit-exact检查，不把persistent的容差比较说成全backend bit-exact。合法受控uniform根据模型实际p/q覆盖每个reject index、partial/full接受及EOS/residual/bonus/budget；概率、proposal或decision未被替换。测试还在一次commit入口核所有请求的决策draw已完成，检查zero ell、inactive resident、移除重入、有限bucket拒绝不做后续forward/draw，及admission **R末行**投影。Feature lease测试使用真实model features加 **fake CPU lease**，观测verification正常顺序为verify→head→commit→draft→release，admission则为verify→commit→head→draft→release，verification failure为verify→abort→release→reset；这证明调用顺序，**没有执行GPU stream/event，也不是实际graph输出异步生命周期证明**。

**冻结证据与提交身份。** 测试没有直接使用并发修改的checkout，而是 **f46e63f immutable archive+恰四owned overlays**：packed sampling、target strategy、新tests和design doc；core正在开发的full-model graph文件未混入。记录者直接核 `output/persistent-sampling-cpu-20261009/{handoff.json,source-manifest.json,cpu-tests.log}` 及其source副本：实际337文件无extra/missing，每个SHA吻合，非overlay内容逐项等于base Git blobs；四overlay与当前文件、随后 **933ed8846f97ef97e6615f44e00e62fb6895e0ec** Git blobs一致。Manifest **37bfc630…**、log **400b8863…** 与handoff吻合，`source_changed_after_run=[]`。完整本地 **239tests/4.155秒、235通过/4跳过、实际OS0**，九个新session tests全通过；四skip明确为Mac无法执行的Linux `/proc` lifecycle tests，不能写成239全部通过。Runtime为现存 **Python3.14.3/Torch2.11.0/Transformers5.4.0、CPU、CUDA/HIP/ROCR隐藏**，没有安装包或远端/GPU运行；suite时间不作模型性能数字。Root已审查并提交（13:08:02），随后确认push实际OS0。测试身份仍是上述base+overlay，不把随后整个commit冒称本次Mac实际执行snapshot。

**仍未解决与下一决策。** Direction正在安排immutable933ed88固定AMD依赖的隐藏GPU CPU suite，本条尚无其完成结果，届时另续记；S26旧版本兼容结果不能自动代替这四overlay的验证。Core full-model graph仍在实现，没有新的whole-target native/capture/replay成功证据。后续需将真实device完成事件、输出lease直到最后draft consumer的寿命、bucket总预算与sampling/commit所有权一起验证；bucket选择、commit/release和draft投影都在session call内，未来round计时必须收费，注册/分配只可单列setup。Exact ordered Q仍不能仅从t−2 K预选，真实capacity scheduler、ZOS与CPU/GPU overlap未完成。S24 native仅有vLLM **7.06%–12.26%吞吐**的强baseline比较继续保留，同backend native target-only控制尚未测；旧target固定RMS失败及cross-backend law限制未解除。本轮没有速度、质量或完整系统pass结论。

用户再次强调便于复盘的问题、方法与过程，要求新建 **6.1 Sol**记录子代理；我们因既有系统agent数量限制无法新建，继续复用现存 **sol_data**专职记录。这是受限后的执行安排，不是把复用既有代理表述为用户的原要求；本轮沿用此日志及索引，不另建重复记录。记录者只做本地source/scalar/hash核验和本日志编辑，未stage/commit、改实现、操作远端/GPU/进程或读取private样本/final test；root负责审核提交。


**13:13后续：immutable933ed88固定AMD环境CPU suite完成。** S27初次本地测试留下两个范围缺口：新session路径尚未在固定5.17依赖上运行，且Mac跳过了四项Linux进程生命周期测试。Direction在root授权的CPU窗口中仅使用immutable **933ed88** archive，跑 **Python3.12.14/Torch2.12.0+rocm7.2/Transformers5.17.0** 完整suite：**239tests/16.039秒、全部OK、无skip，test process与实际SSH均OS0**，九项新session实模型测试也在此版本通过。CUDA/HIP/ROCR visibility均空，`cuda_available=False/device_count0`，没有GPU执行或环境变更。记录者核 `output/persistent-sampling-pinned-cpu-20261009-933ed88/summary.json` 与原log/versions/两个OS code，八项summary/remote/local evidence SHA全部一致；archive **b1b7d1e4…** 的337文件逐SHA与933ed88 Git blobs一致，保存的前后source verification均passed/337files/differences为空。Full log **7569ea75…**、runtime身份 **4a89f67d…**；runtime import前置两行 `(null): No such file or directory` 原样保留，不赋予未验证原因。这个续记消除了上述固定CPU版本与Linux skip缺口，未扩大到full-target native/graph、真实GPU lease/stream、overlap或性能证明；S27原Mac239/235pass4skip历史仍保留。本轮只续记session CPU结果，不将新的graph实现交接混入本节。


<a id="persistent-full-qwen-graph-cpu"></a>
## 2026-10-09 13:25（UTC+8）— S28：full-target graph完成CPU准备，但跨pool事件、取消失效与显存计费三项阻塞审查

**要解决的问题。** S27已经把scratch验证、一次target commit、draft消费与feature release接进session，但真实graph replay不会重新执行Python的layer-ready集合或hook bookkeeping。若仅把完整HF forward capture起来，新的事务仍可能拿不到可信的完成状态；若commit后立刻复用固定输出，又可能覆盖draft尚在读取的features。本轮[full-target候选](persistent-qwen-graph.md)因此同时准备固定计算体、外部写入receipt、完成事件及输出lease。它是显式opt-in的native eager/graph实现与CPU emulator验证，**不是新的完整target GPU成功结果**；独立审查随后发现三个blocking问题，原CPU通过不足以放行。

**固定计算体与外部边界。** 候选调用原 `model.model`，计划capture embedding、全部原decoder层的QKV/QK norm/RoPE、scratch KV copy、capacity gather、native attention、output projection/残差/MLP及final norm。Selected hooks把raw block outputs复制到固定context slices，final-normalized hidden单独存储，最后block raw与final norm仍不混用。LM head、实际q/p及采样、confidence/acceptance、host metadata和resident commit留在外面；admission仍只投影R个末行。Capture时的seen-layer/feature集合只检查固定程序覆盖，不被当作真实replay的新执行证明；CPUReplayEmulator每次重新执行Python tensor body，因此CPU结果不能证明GPU capture允许这些操作或replay不会再运行Python。Native constructor计划限定原BF16/Hq16/Hkv8/D128、固定5.17和ROCm runtime/schema，旧exact-K wrapper未被放宽；这些守卫也不等于GPU执行已经验证。

**事务和lease怎样连接。** 准备阶段先由pool.begin更新固定metadata/positions/token IDs，并给当前transaction签发精确ticket与external-write receipt；submit执行固定程序并记录backend event，finish等待完成后才发布全部layer scratch readiness及新的features。`wait=False`遇未完成event时保持pending且不发布ready层；capture/event不确定失败poison store/executor，不暗中fallback到eager。Head与原决策随后运行，一次commit后，固定graph outputs仍由lease保护，直到draft最后消费后显式release；pending lease也阻止reset、slot churn与下一replay。Prepare/finish/head/commit/release及外部draft消费者要求同owner stream，用同stream排队保证后续覆盖在读取之后；当前CPU stream身份patch只核调用边界，未证明真实GPU stream顺序或跨stream消费安全。Model/config/selected layers、参数版本/地址和固定buffer签名在复用前检查。这些是设计和现有覆盖范围，其中event绑定的完整性已被下述反例推翻。

**最初失败与修正保留。** Core保存的 `initial-focused-failures.md`明确是tool-captured记录，**不是重建的完整raw log**：首次combined persistent selection实际21tests/0.510秒、exit1，含一assertion failure和一error。一个旧断言期待 `explicit CPU`，新拒绝消息却以 `Explicit CPU`开始，修复只统一大小写并保留拒绝；另一个问题是ordinary eager prefill直接走verify_eager，绕开S27 fake-lease fixture patch的public verify，commit断言发生 `IndexError: list index out of range`。随后ordinary prefill恢复public verify，注册graph bucket的prefill仍显式eager，没有删弱lease断言。Initial graph-specific七测试0.077秒已通过，并不抵消combined失败；修复后combined28/0.525秒通过，保存focused log又记28/0.553秒，之后加入第八项model/output-buffer drift测试。

**审查前CPU证据及源身份。** 原 `output/persistent-qwen-graph-cpu-20261009/handoff.json`状态是 `local_cpu_prepared_no_gpu_execution`，base **933ed8846f97ef97e6615f44e00e62fb6895e0ec**。实际final log为 **247tests/4.179秒、243通过/4跳过、OS0**；四skip仍是Mac无法执行的Linux `/proc` lifecycle测试，八项新graph-specific tests全ok。真实随机四layer tiny Qwen eager与emulated body对照all-layer KV、selected raw/final norm/logits，覆盖不等context增长、partial/full commit、abort、inactive residents、地址与model签名漂移、completion未ready/failure、capture失败、有限registry拒绝与commit后的lease。新tests没有覆盖共享backend下两个pool的generation碰撞，CPU budget拒绝也没有测GPU私有pool。记录者于修复前逐项核七owned文件与handoff SHA一致：target **39180db7…**、store **47664716…**、graph **7b5b9aa4…**、tests **88fcf57d…**、三doc **420006ae…/7ccb6741…/5aba829e…**；四evidence SHA均一致，final log **48e86702…**、focused **c1db6248…**、failure record **baf4c23e…**，早先full log247/4.168秒 **25fe0b05…**也保留。该轮使用现存本地 **Torch2.11.0/Transformers5.4.0、CPU**，没有提供immutable全tree archive或source-manifest；只能绑定这七文件及保存日志，不能冒称933ed88完整commit是本次实际测试snapshot，后续修改也不能借用旧hash。

**阻塞一：完成event来自backend，仍可能来自另一pool的另一程序。** Root转达direction的独立CPU反例已复现：两个pool共享同一backend，各自第一个receipt的generation都是1。旧checker核event是该backend持有的精确对象且generation吻合，却未把event绑定到本次receipt/固定program/pool；将A已完成的generation1 event交给B，可以让B发布实际未执行的scratch为全层ready。记录者只读源码核该机制：backend `_owns`只查自己event字典与generation，pool.complete_external_write调用checker后直接设置完整staged集合；普通foreign-pool receipt拒绝测试没有覆盖这个同backend、合法本pool receipt配外来event的组合。由此“backend event ownership+本地单调generation足以证明当前写入完成”的假设被反例否定。该问题已退回core，需把完成证据绑定到实际提交程序及当前receipt，并补跨pool/跨program碰撞回归；Direction随后交接了实际script/log及冻结source，记录者已直接核其标量输出与SHA，未亲自重跑。原handoff关于exact-event绑定的陈述保留为审查前声称，不记为审查通过。

**阻塞二：allocated差值不是graph保留显存预算证明。** 只读审查发现旧TorchGraphBackend用capture前后 `torch.cuda.memory_allocated` 差值与reserve_bytes比较。临时激活在capture结束后可能没有活跃tensor引用，却仍由graph私有allocator pool保留供replay复用；这些reserved bytes不会完整出现在allocated净增长中。因而“固定input/output+workspace+声明reserve”的registry算账虽然有CPU拒绝测试，其GPU reserve执行检查仍可能漏计，不能称已建立真实graph cache总显存预算。本项是allocator语义与源码的审查结论，**没有新的GPU显存测量或反例运行**；core需改用能够覆盖private pool保留分配的计费及拒绝机制，再分别留证retained量、setup/transient峰值与进程cap，不能把三者混为同一指标。Direction核本地Torch2.11 API文档发现pool-scoped `memory_snapshot(mempool_id=graph.pool(), include_traces=False)`可枚举私有pool segments及total_size，作为修复调查方向；固定2.12 API支持仍须单独检查，不能把本地存在的接口当作native环境已验证。

**阻塞三：取消路径的完成检查异常没有fail closed。** Direction的第二个实际CPU反例先提交，再让completion checker抛 `injected completion checker device failure`。`cancel_graph`传播错误后pool.failed仍false、ticket仍为同一对象；恢复checker后可以再次cancel，并成功做下一verify，复用了原storage。这里应区分正常query返回False（保持pending、以后可确认完成）与检查本身抛错（完成/写入状态不确定，按已声明契约必须poison）。Finish已有异常invalidate处理，而cancel→pool.abort路径缺失同等保护；core需修复并覆盖submitted receipt在abort/cancel中检查异常后的永久拒绝复用，不能把普通not-ready也误判为永久故障。

**独立审查留证与覆盖缺口。** Direction在 `output/persistent-qwen-graph-review-20261009/`保留review、source manifest、focused反例script/log和实际OS code；反例跑在immutable933ed88加原七overlay的独立source副本，**没有重跑完整suite或执行remote/GPU**。记录者核全部五evidence SHA一致，focused log **220e7eba…**、review-source **53677ccb…**、review **b8ff875b…**、script **93afc180…**、OS code实际0；逐项核source的340个声明文件，其中933ed88原337文件加三新overlay，非overlay逐Git blob一致，七overlay逐旧handoff一致，无difference及非bytecode extra。Log实际显示两pool generation1、foreign_event_published=true、B的never-written KV为0且length1，以及cancel exception后pool_failed=false/verification_reused=true；这是两项已运行CPU反例，memory accounting仍仅代码/API审查。这个后来冻结的review snapshot不被追溯冒称原247suite的运行snapshot。审查另指出composition覆盖缺口：graph测试commit后用context.square().sum()消费输出，S27 session用eager target加fake lease；二者尚未共同执行真实session+CPU emulator+draft投影。现有test支持各自边界，不能拼接成已验证完整graph-session的断言。

**当前决定与未解除的限制。** Root已把三项blocking退回core优先修复；本节截止仍是原CPU准备通过、独立审查未通过、新修复与重新验证待交接，full-target GPU probe仅在准备，**尚未执行**。后续先保留旧handoff与反例，再核新source/hash、三项回归与完整suite；只有完成独立复审并冻结bounded协议后，新的实际pretrained full-target native eager/capture/replay才能回答RoPE、全部layer、KV及特征是否一致。当前没有性能、sampling-law、whole-model correctness或overlap结论。Exact ordered Q仍依赖当前confidence/allocation，不能单靠t−2全局K预选；有限R/physical-B/maxQ家族、ZOS、双bank/CPU-GPU overlap尚未完成。S24仅有vLLM **7.06%–12.26%吞吐**的无加速结果、未运行的同backend native target-only控制、旧whole-pretrained-Qwen layer26固定RMS失败与cross-backend law/endpoint TV限制全部保留。记录者仅编辑本日志及核本地source/scalar/hash，没有改实现、stage/commit/push、远端/GPU/进程操作或private样本/final test读取。


**13:32修复续记：保留原反例，三个缺口分别关闭。** Core的新backend event绑定精确receipt对象，receipt再绑定writer/transaction/incarnations/generation；execution和backend提交前均检查captured program的writer，不能靠同generation混用另一pool事件。取消路径在checker返回False时保留健康pending，checker抛错则poison pool、invalidate executor，恢复checker也不能复用。显存检查改为按graph.pool()取scoped memory_snapshot并累加segment total_size，把inactive retained blocks计入reserve；不可用/异常schema、foreign device或shared pool拒绝，不fallback到allocated差值，global reserved前后/差值另列。CPU synthetic snapshot只验证这些host算账/拒绝规则。另加真实 **PackedSpeculativeSession+DSparkDraft+CPUReplayEmulator两轮**组合，核actual draft projection在commit后、lease仍持有时执行，并对照eager的outputs/probabilities/projected KV，补上原分开测试的composition缺口；仍不是GPU stream完成证明。

**修复版CPU来源与接口证据。** 这次完整suite使用immutable **21ce2bb+恰七owned overlays**，完整343文件冻结副本，实际 **260tests/4.376秒、256通过/4项Linux skip、OS0**；13项graph聚焦测试 **0.156秒、OS0**。创建snapshot的第一次命令因cwd尚不存在，在进程创建前被拒绝；更正后仅跑这一次完整suite，不抹掉原失败或把启动错误写成测试失败。记录者直接核 `output/persistent-qwen-graph-cpu-20261009/repair/` 的handoff、source-manifest、base.tar、repair-only.diff、notes和log：五主evidence及四additional SHA均吻合，343源码逐manifest/非overlay Git blobs吻合、archive逐Git一致、七current owned SHA吻合；full log **0f640a09…**、manifest **2176fe62…**、archive **55acd9f0…**，graph module已变为 **0bc68d8e…**，旧 **7b5b9aa4…**及旧247suite仍独立保留。Runtime仍本地 **Torch2.11.0/Transformers5.4.0、CPU、三GPU visibility空**。Core和root各做一次固定 **Torch2.12.0+rocm7.2** 的只读API检查，分别留在 `pinned-memory-api-source.txt`（SHA **c151ffdb…**）及 `output/persistent-qwen-graph-pinned-api-20261009/api.json`；root证据CUDA不可用/device0、SSH OS0，均确认 `memory_snapshot(mempool_id=None, include_traces=True)`与CUDAGraph.pool接口存在。它们没有调用真实graph pool测segments，也没有GPU capture/native执行，不能称真实reserve测量已通过。

**独立复审随后完成，只限CPU/代码范围。** Direction使用同21ce2bb+core冻结七overlay，再加一项target-only composition test overlay，独立核旧repair diff并改接口重跑原反例； `output/persistent-qwen-graph-review-20261009/repair/handoff.json`明确三原blocker resolved。实际反例检查现在拒绝foreign event/copied receipt/foreign program，B length0/staged0；synthetic schema在allocated_live0/global_delta0时仍计 **4096bytes**，拒绝reserve2048、接受4096，并拒绝五项坏schema；正常not-ready保留pending，checker异常永久poison，恢复后verify/reset拒绝。独立script OS0、log **c380b762…**；记录者核九evidence SHA与两个实际OS0、343文件逐manifest/Git/八overlay一致，七core文件没有被direction改动。Direction没有再跑完整suite或remote/GPU；新增第九target-only测试 **9tests/0.214秒、OS0**另在S29记。原“审查未通过”是修复前状态；此续记只解除三个已指出的CPU/代码阻塞，**未解除actual pretrained full-target native capture/replay、真实private-pool实测、旧RMS/分布/速度/overlap限制**。下一项仍需冻结有界真实设备数值/KV协议；本条截止probe GPU尚未执行。


<a id="packed-target-only-cpu"></a>
## 2026-10-09 13:32（UTC+8）— S29：同backend batched target-only控制CPU通过，尚无paired E2E速度结果

**问题与方法。** S24与vLLM的强baseline比较证明当前native没有加速，却不能分离native target执行成本与draft/verification新增成本。Direction新增[独立target-only session](packed-target-only.md)，复用显式target strategy与原FP64 categorical law/每request RNG；admission只做R末行head，每decode活跃request只验证一anchor、消费一次真实draw，再一次batch commit/release，无draft/shadow/confidence/acceptance工作。缓存仍为prompt+output[:-1]，inactive KV保留；缺bucket在forward/RNG前拒绝，执行错误永久invalidate。未来paired E2E还需匹配backend、dtype、输入/output预算/EOS、selected-feature配置与setup收费：若target-only仍选raw layers，它仍付特征copy，不能隐藏这一成本或用不同配置归因收益。

**原八测试、冻结身份与提交。** 实际四layer tiny Qwen对照独立cached_target_sample及HF emitted-prefix oracle，检查actual p、完整RNG trace、outputs与全部layer KV，覆盖不等prompt/budget、active subset、EOS、lifecycle/inactive、failure及bucket拒绝。原 **8tests/0.183秒、OS0**来自immutable **844887f+恰三overlay** 的340文件副本，不含core graph WIP；本地Torch2.11.0/Transformers5.4.0、CPU、GPU隐藏。记录者核 `output/packed-target-only-cpu-20261009/handoff.json` 五evidence SHA、340source逐manifest/非overlay Git一致，三overlay与随后 **21ce2bb2833c126555106fae47d88f7614a4eec1** Git blobs一致；log **ee390b8e…**、manifest **e7542d80…**。Root已于13:25:59提交并确认push；当前test后来被direction追加第九项，不能借原test SHA **1ab32163…**认领新文件。

**追加真实graph-emulator composition与边界。** S28修复冻结后，direction的独立第九测试把真实PackedTargetOnlySession接CPUReplayEmulator，跑两轮；outputs、FP64 p与RNG trace相对eager exact，全部committed KV按实际emitted prefixes独立重建，request-local positions **[4,2]→[5,3]**，核inactive resident不变、固定buffer地址稳定及commit后的lease直到release。独立修复副本中的新增test SHA **e38ad05e…**、九测试log **4a0ab25a…**，实际 **9tests/0.214秒、OS0**；这份snapshot与core260suite的七overlay副本不同，未把第九项追溯写成260suite已跑。它为后续同backend paired E2E补上控制实现与CPU组成证据，**没有remote/GPU/benchmark或速度数字，不替代vLLM强baseline**。原whole-Qwen layer26 RMS失败、cross-backend law/endpoint TV限制、S24无加速、真实graph/private-pool/overlap未测均保留；后续仍需实际完整请求/轮次计时与工作量收费，不能把0.183/0.214秒测试时间当吞吐。记录者仅编辑日志及本地证据核验，未操作GPU/进程、改实现或stage/commit/push。


**13:35统一续记：修复与第九composition已提交，固定AMD依赖完整CPU suite通过。** Root审核后将S28七文件修复及S29第九组成测试一并提交/推送 **c2d11a03e6a28dfccc5c2dc47da935d1410ef067**（13:31:44，push实际OS0）；原S29八测试仍对应21ce2bb。Direction随后仅用immutable c2d11a0在 **Python3.12.14/Torch2.12.0+rocm7.2/Transformers5.17.0** 跑完整 **261tests/17.615秒、全部OK、无skip，test与真实SSH均OS0**，三GPU visibility空、cuda_available=false/device_count0，没有环境更改或GPU执行。记录者直接核 `output/persistent-qwen-graph-pinned-cpu-20261009-c2d11a0/summary.json`：八remote/local evidence SHA一致，archive **10cd9f7c…** 的343文件逐Git blob一致、无missing/difference，保存的source before/after均343files passed；full log **51026d10…**明确261/17.615/OK。这个统一结果消除新修复/组成在固定5.17 CPU版本及Mac四Linux skip上的未测状态，不能追溯改写原260/256pass4skip或原247历史；仍只验证tiny Qwen、CPU emulator、host ownership/schema，**没有actual full-target GPU graph、真实private-pool测量或paired E2E速度结果**，旧RMS/分布/无加速与overlap限制继续保留。记录者停止编辑，交root审核日志。


<a id="full-target-graph-result"></a>
## 2026-10-09 — S30：完整 pretrained target 的同backend真实graph保真通过

**问题与方法。** S28/29的CPU emulator不能证明完整pretrained Qwen在真实设备capture/replay保持RoPE、特征与全部layer KV一致。新协议冻结执行源码 **3829d5355018211f9b8464f00a44bdf55c3779e6**，原HF embedding、全部28 decoder blocks及final norm连同scratch/gather进入graph；prefill、LM head、采样、metadata与commit在capture外。固定Q=(1,4)、K ceilings=(144,144)，contexts依次(128,128)/(129,131)/(130,135)，人工commit(1,3)/(1,4)/(0,0)，最后abort。这些commit是fixture，不是采样接受决定。selected raw blocks[1,7,14,21,26]逐层对实际eager hooks核身份，另核final norm；layer26不是最后block。先三次native eager全部通过，再warmup2+capture1与三次有用真实replay；保留原pooled/per-request atol/rtol0.02及RMS≤0.005，own commit/inactive/lease要求exact。

**准备失败与修复。** CPU准备留证中，首次7项通过；随后helper默认参数 `require_gpu=require_gpu` 在module import时NameError，产生loader error，未进入GPU。修复后最终8项/2.086秒、OS0；该准备使用acabd0f+三overlay，随后归档为3829d53，不能把准备snapshot认作后来完整执行tree。实际GPU仅一次，无retry/fallback/阈值或domain缩减。成功执行及独立释放后，网络传输两次exit255、首次打包mtime警告exit1；另post-release标量读取曾误把list当dict，修正后核验完成。保留原失败，成功resume得到独立verified archive与fresh `complete/`，未混入partial目录、未重跑GPU；post-release只读import生成的一项pyc另记，346执行源码未变。

**直接证据与结果。** 本轮读[公开报告](../reports/full-target-graph-20261009-3829d53/README.md)、[aggregate](../reports/full-target-graph-20261009-3829d53/aggregate.json)、[可复用CPU审计器](../reports/full-target-graph-20261009-3829d53/verify_raw.py)及本地未发布 `output/full-target-graph-gpu-20261009-3829d53/final-handoff.json`；独立核handoff的15项critical及3项public SHA全部一致。正确完整证据使用 `complete/`。source.tar SHA **10d52f29…**；root另核逐字节等于3829d53 Git archive，verified evidence archive SHA **94ae5bc7…**、197727457 bytes。三stages、六observations与60项structural checks通过，worker/controller/外层SSH均OS0，独立identity/ASR/KFD检查确认释放；记录者没有再次live查进程或运行tensor审计。

完整features/logits、28层speculative/committed KV、inactive isolation、rollback、固定地址与commit后consumer lease均有检查。Warmup/capture将Python model/decoder计数增加三次；三次replay改变tokens/positions但计数均不增加，证明所观察边界的真实Python-free replay。原raw CPU audit与root独立重跑均OS0：30 artifacts、3430 checks、1239 pooled/request comparisons，max abs/RMS均0、全部torch.equal及uint8 storage-byte equal。root结果保存在 `root-raw-audit.json`，记录者仅核报告与标量，不冒称亲自加载30个tensor。byte equality是额外描述性观察，不提高原gate阈值；worker原bit_equal仍按torch.equal解释。

**资源口径。** 实测graph私有pool retained **2097152 bytes**，含allocated/active为0的inactive segment；reservation **536870912 bytes**不变。global reserved从1694498816到1692401664、delta **−2097152**，独立于private retained量，不能用净allocated/global delta替代保留显存。combined graph/workspace预算768MiB、workspace cap256MiB、allocator fraction cap6GiB、deadline300秒；这些不是transient/process-wide峰值测量。Supervisor **25.762924秒**含setup/证据I/O，不能当吞吐或speedup。

**结论边界与下一决策。** 通过的是一个有限Q家族、同pinned native backend的完整target eager↔graph保真；hook证明raw-layer身份，不构成独立attention oracle。此前whole-Qwen layer26 RMS失败、cross-backend/sequential law与BF16 endpoint TV限制、S24无加速结果均不解除。没有模型质量、完整sampling distribution无损、paired E2E收益、t−2 capacity graph家族、ZOS或CPU/GPU overlap结论。下一步另审同backend paired请求计时与完整setup/物理工作收费；其CPU准备不冒称已有GPU速度结果。记录者仅维护日志，无GPU/进程/实现/stage/commit/push操作。


<a id="paired-r1-result"></a>
## 2026-10-09 — S31：36批R1同backend配对完整请求完成，speculative慢于target-only

**问题与方法。** S30证明同backend graph保真，不回答加上draft/全shadow与验证后是否更快。依据[冻结配对协议](paired-r1-benchmark.md)，在R1/C64与R1/C256分别对同栈target-only和trained step1280 fixed-γ7/full-shadow，完整生成128输出；每case/arm两warmup、五primary、两diagnostic，共36批，交替臂次序。两臂共享resident target/selected-layer copies/draft weights和graph pools，target-only不执行draft/shadow；这是共同resident条件下的执行归因，不是另行优化的target-only部署。只有Q1/Q8 graph，Q2…7明确native eager尾轮，prefill eager，physical Q不padding。Batch timer含reset/session/admission、全部shadow/q/p/draw、metadata/event等待、尾轮、commit/draft projection/release与final sync；输入准备及startup另列。

**准备与执行身份。** 初始3829d53+三overlay的349文件准备为11tests/2.779秒；root在6b349ba+三overlay的352文件副本另跑11tests/6.024秒/OS0，core非作者审查743项CPU checks通过（非743项unittest），direction源码/协议审查未发现具体blocker。三overlay随后归档为执行commit **192a3578eb71527e02fded28a3beaf487d53859a**。先前15:21保存状态不是launch readiness；本次fresh preflight核binding/source、ASR原身份ready/nonbusy、KFD仅ASR与free25250160640 bytes后，仅执行授权一次，不以旧PID/log推live状态。

**完成、启动检查与证据。** 本轮直接读本地未发布 `output/paired-r1-gpu-20261009-192a357/scalars/` 的result/samples/runtime/capture-progress、实际exit和independent-release及 `root-scalar-audit.json`，核其四个input SHA一致；独立从samples重算 **36批、4608输出、4059轮**与四项primary pooled rates一致。Root额外核36 identities、全shadow/实际graph/完整输出和启动检查；这是scalar audit，未重验大setup tensors。worker/controller/SSH OS0、未timeout，独立release确认owned identities消失、原ASRready/not busy/KFD仅ASR，free恢复25250160640 bytes。随后完整archive已回收并fresh解包为 `complete/`：root-provenance-verification留证102307761 bytes、SHA **68bb8239…**，source.tar逐字节等192a357 Git archive、352source逐hash一致、全部scalars与complete一致。Core首次 `independent-scalar-audit.py` 曾对整数buckets调用len，修正仅审计脚本、原实验evidence未变，不算GPU失败；root的scalar审计一次OS0，无此错误。随后core与root分别实际运行[公开raw审计器](../reports/paired-r1-benchmark-20261009-192a357/verify_raw.py)，均OS0：2 artifacts、36 saved tensors、126 numerical comparisons、404 assertions，maxabs/RMS0、resident exact。Root结果保存在 `root-setup-raw-audit.json`；原scalar audit仍只支持scalar范围，本记录者仅核留证，未亲自加载tensor。

Q1/Q8各一次same-native eager↔first replay启动核验通过：原.02/.02/RMS≤.005下selected raw/final/logits及28层scratch KV标量maxabs/RMS0、resident隔离通过、真实replay Python counters不增。每graph私有poolretained **2097152 bytes**，各reservation仍512MiB；workspace41646512 bytes、resident/scratch73400320 bytes，不能拿retained代替全部resident/process峰值。startup **21.627453秒**（validation1.320902秒）另收费，supervisor **195.083493秒**含setup与I/O，不作为primary吞吐。

**实际性能负结果。** 每cell五primary各合计640输出；pooled rate按sum(tokens)/sum完整batch wall计算，不平均单批rate。

| Prompt长度 | Target-only tok/s | Full-shadow spec tok/s | Spec/target | 冻结vLLM tok/s | Target/vLLM | Spec/vLLM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 44.804932 | 23.103708 | 0.515651 | 131.264056 | 0.341334 | 0.176009 |
| 256 | 44.181720 | 19.274350 | 0.436252 | 127.339898 | 0.346959 | 0.151361 |

Speculative吞吐仅同栈target-only的51.57%/43.63%，两臂也都低于保留的vLLM强baseline；没有加速。Spec主轮455/530，实际graph轮435/500、eager尾轮20/30，graph rows3480/3965、尾rows95/135；target-only各635轮全部Q1 graph。Spec accepted draft180/105、proposed3120/3570，draft/round **0.395604/0.198113**，accepted/proposed **5.7692%/2.9412%**。这是synthetic prompt和empty-EOS固定预算下的本次轨迹工作量，不是自然development接受率，不能把quality32或TF overlap转作本次收益预测。Graph覆盖多数轮仍慢，不单凭覆盖率推加速，也不凭这两个cell定位某一个算子为唯一瓶颈。

**边界与下一决策。** 本段无运行gate失败，失败的是性能收益假设；旧whole-Qwen layer26 RMS、cross-backend/sequential law与endpoint TV限制继续保留，同backend启动保真不解除它们。36批是两个R1 case，不是完整六case、多请求serving frontier或all-graph；TTFT/SLA、scheduler、t−2 capacity、ZOS/CPU-GPU overlap未由本次证明。下一步用已测负结果拆分shadow/FP64概率/metadata/commit及新栈target成本，再决定优化与finite physical-B family政策，保留强baseline和完整收费；不将新gate/局部kernel时间改写成E2E改善。[公开报告](../reports/paired-r1-benchmark-20261009-192a357/README.md)、[aggregate](../reports/paired-r1-benchmark-20261009-192a357/aggregate.json)及可复用审计器已完成，raw审计与scalar/归档身份分别留证；记录者未操作GPU/remote/进程/实现、未commit/push或读取final test。


<a id="paired-profile-cpu-repair"></a>
## 2026-10-09 — S32：配对profile的两项P1修复，Linux CPU验证与独立复审完成

**问题与方法。** S31的speculative吞吐仅同栈target-only的51.57%/43.63%，多数轮走graph仍不能解释完整请求慢在哪里。[配对profile协议](paired-r1-profile.md)沿用36批、两case/两arm、每批128输出和原算法/RNG；warmup与primary直接调用原batch，只有diagnostic增加完整session的host spans/GPU events及最多四条CPU+ROCm trace，拟拆分target/head、FP64概率、host等待与spec增量。计时区分inclusive/exclusive与区间union，不叠加嵌套阶段或重叠的CPU/GPU时间；缺失关联保持unknown，不以CPU-only trace替代。预算保留1800/1790秒、free guard8GiB、allocator6GiB、每graph reservation512MiB、combined1280MiB与workspace64MiB；新增每trace256MiB/合计1GiB、RSS8GiB采样abort与100000 spans上限，不用RLIMIT_AS。

**初始准备与实际失败。** 首轮6项CPU测试为5 pass/1 error：fixture对per-layer KV tuple调用clone，触发AttributeError，是测试夹具错误，非模型或GPU失败；修正后依次9、10、12项通过，原12项日志为3.314秒/OS0。静态修正还覆盖O(N²)汇总、monitor错误留证/rename竞争、round/stage关联、startup失败保留与精确trace数。随后独立审查实际跑出两项P1并要求GPU前修复：无效null/负数/bool ID、缺pid/tid的None相等和先滤掉无CPU父链的重复runtime，会制造假关联；零ID的合法语义未获证明，也不能当有效链。另一个stdlib反例用PyDLL.sleep持GIL约1.003686秒，原线程monitor虽设0.025秒周期，实测最大检查间隔1.008387秒，未在持GIL期间持续采样；这是CPU契约反例，不是AMD profiler故障。

**修复与真实Linux检查。** 新关联规则只接受正JSON整数ID及有效pid/tid，先对全部runtime检查唯一性，再要求唯一包含的CPU父链，每条未解活动仍unknown。Diagnostic改由独立stdlib进程采样，以PID/start ticks与预绑定pidfd锁定worker，ready/stop握手、停前限额检查和正常OS0必需；留证写失败也执行绑定worker的abort，原supervisor回收后代，primary不新增monitor。Mac为15项/3.246秒/OS0，其中13 pass、两项真实Linux /proc+pidfd测试skip；冻结副本在隐藏GPU的Linux环境实际15项/12.622秒全过、worker/controller/SSH OS0。两秒持GIL实测区间2.000383秒，总41次采样，严格start<t<end为**39次**，最大gap0.0520248秒、正常monitor OS0。另一用例把partial trace写成11 bytes超过10-byte测试限额：通过的真实测试断言绑定worker被SIGKILL、monitor退出86并被reap，gil-end未写，failed结果保留sample_count28和partial；结束留证确认五个owned进程身份均消失、原ASR身份ready/nonbusy、KFD仅ASR、free25247080448 bytes。这些是当时CPU检查的释放证据，不是未来launch readiness。

**独立复审与来源核验。** Direction复审冻结五文件，19项trace反例/对照与8项模拟Linux syscall的monitor边界通过，结论为两个原P1无剩余blocker；EOF、坏命令、身份改变与写失败的模拟检查不冒称真实内核实验，也未重复完整suite或运行GPU。记录者核其四个evidence SHA与handoff SHA **dd80144e…**、五文件当前/冻结SHA一致，独立从markers/samples重算上述39次与gap；Linux archive SHA **aeff6612…**一致。原Git base **0f4c1554…**的archive逐字节匹配，加五overlay后360文件逐manifest一致。Root首次source audit曾OS1，因为误把仅含五overlay的source目录当作完整解包树；纠正重建后OS0，错误属于审计脚本的路径假设，不属于源码、CPU测试或GPU实验。初始准备/失败审查留在 `output/paired-profile-implementation-20261009/`、`output/paired-profile-independent-review-20261009/`，修复/Linux与复审分别留在相应的 `-v2/`；均为本地未发布证据。

**当前决定与边界。** 五文件修复已归档为commit **f0b0268775a51e33fde3f374aef5e29c2fed4ec2**，root报告已push；本节记录截止于CPU准备交接：通过范围是CPU准备、资源监控契约及两个P1的独立复审，不提供真实ROCm correlation能力、算子瓶颈归因或收益结论。Root随后授权core在该immutable source与fresh preflight后执行一次GPU profile，执行结果另记，不把启动当完成。旧whole-Qwen layer26 RMS、cross-backend/sequential law、endpoint TV与S31负收益继续保留；多请求、finite physical-B/capacity家族、t−2、ZOS及CPU/GPU overlap目标未被本轮缩小。本轮journal唤醒失败后由sol_data临时独占记录，root审核提交；记录者只核本地源码/标量/哈希并维护文档，未改实现、运行测试/GPU、stage/commit/push或读private/final-test样本。


<a id="paired-profile-trace-limit"></a>
## 2026-10-09 — S33：唯一GPU profile在trace导出触及文件限额，未完成完整timeline

**执行与失败。** 在S32修复commit **f0b0268775a51e33fde3f374aef5e29c2fed4ec2**、fresh source/binding及ASR/KFD/free guard核验后，core仅执行授权一次原36批profile。四项diagnostic0完整session与observer恢复均成功；首个diagnostic1（C64/spec）底层完整batch已结束，停在trace_export，独立monitor观察单个 `trace.json.partial.tmp` 为 **282942823 bytes > 268435456-byte cap** 后abort。回收时文件为 **298398246 bytes**，这是采样检查间的overshoot；RSS最大 **4249055232 bytes < 8GiB**，不是本次触发项。worker OS **−9**、controller/SSH **1**、timed_out=false、result failed，不能写成完整profile成功或timeout/OOM。未产生通过导出/解析/关联验证的最终trace；原ROCTracer duplicate flow start:4警告保留，但abort早于关联检查，警告不等于关联gate失败，也不据残缺文件推测ROCm correlation能力或具体GPU瓶颈。

**已留存范围。** `samples.jsonl`只有 **32/36批 = 8 warmup + 20 primary + 4 diagnostic0**，完整保存输出4096 tokens；已结束但导出失败的diagnostic1未计入这32批。每cell五次primary均保留，pooled rate重算为C64 target/spec **45.040683/24.316672 tok/s**，C256 **44.604033/20.737556 tok/s**，仍无加速，不能把新诊断当E2E改善。Root的scalar审计核同路径outputs/work/decisions一致；记录者独立重算样本/输出数和四项rate。Diagnostic0可保留host inclusive/exclusive spans与GPU stream event区间作为有限成本证据，event含dispatch gaps且不是kernel-active时间；CPU/GPU重叠不相加。Diagnostic1虽留spans/completed batch，完整CPU+ROCm timeline及可验证kernel关联仍缺失。

**来源、回收与释放。** 本地未发布证据在 `output/paired-profile-gpu-20261009-f0b0268/`，完整归档 **123145901 bytes**、SHA **8f5dd443…2383d**。Root核55个early scalar文件逐字节与complete一致，记录者读取审计留证；未重跑setup raw tensors或验证残缺timeline。记录者核source.tar逐字节匹配f0b0268 Git archive，pre/post-source均360文件通过、controller post-input passed（worker被SIGKILL后未写该留证）；直接读取independent-release确认controller/worker/monitor共七个owned identities消失，原ASR身份ready/nonbusy、KFD仅ASR，free **25252777984 bytes**。这些是失败后的释放观察，不作为未来运行许可或launch readiness；原失败result、partial文件与已完成samples保留，未重复launch。[公开失败报告](../reports/paired-profile-20261009-f0b0268/README.md)与[聚合证据](../reports/paired-profile-20261009-f0b0268/aggregate.json)保留失败口径，S31仍是已完成的配对E2E结果。

**有限分解与下一决策。** Direction对四个diagnostic0的spans/completed-batch独立推导，记录者核三份review证据及八份输入SHA一致。每cell仅一次diagnostic0，较primary mean有6.6%–19.3%观察扰动，以下不从primary计时抵扣：spec整段tail rounds占wall **6.22%/7.95%**，即假设全部消失，剩余5.891/6.603秒仍高于diag target的3.189/3.060秒，不能把补Q2–Q7 graph当作主差距解释。Draft-propose父区间约48% wall，与其子阶段不相加；host exclusive包含同步等待/observer成本，不称纯CPU计算。共同host候选_model_signature在target每轮prepare/submit/finish共381次，exclusive1.053/0.856秒，需先做CPU配置序列化、tensor枚举、元数据/_version扫描的成本拆分；建议仅在封闭同步round研究显式immutable-model lease，验证parameter/buffer替换、in-place版本、config/layer/training变更及各提交边界的fail-closed，公共异步入口保留原检查，不能简单缓存signature。该建议尚未执行，也不保证低接受率spec胜出；不扩大trace限额或重复GPU launch。完整timeline、kernel-active归因及跨栈差距仍未证明，旧RMS/law/TV限制与多请求、capacity/t−2、ZOS/overlap目标不变。记录者仅核本地标量/哈希并写日志，无remote/GPU/实现修改或commit/push。


<a id="signature-cpu-safety"></a>
## 2026-10-09 — S34：signature的CPU/meta成本拆分完成，跨边界缓存否决，生产实现不改

**问题与安全否决。** S33发现重复_model_signature的host候选成本，本轮先问能否减少检查而保留现有拒绝语义。Direction在Git **d9e282d0e7ce5982a5d14bf08804f5a8c9948c6f**上用真实小HF CPUReplayEmulator检查before-prepare、prepare→submit、submit→finish及同步wait期间另一线程in-place变更：四边界×原检查/缓存旧signature共**八组**，原四组均拒绝，缓存四组均放行并返回features。requires_grad=false不禁止写入，公开model/tensor别名和stream检查也不提供可强制的immutable lease；同步等待仍允许host线程变更，因此保留prepare/submit/finish三个fresh检查，否决跨边界signature cache。该结论仅保留原拒绝语义，原.data写入可绕过_version的局限未解除。两次初始collector错误（预期失效后读取healthy-only lengths、将内部tensor当mapping）修正后最终OS0，属于证据collector，不算生产实现失败；记录者核六证据、三source与handoff SHA **fe8c9594…**一致，留证在 `output/signature-safety-review-20261009-d9e282d/`。

**方法、成本与决定。** Core仅构造BF16 Qwen3-0.6B meta对象（28层、310 parameter tensors、2 buffers、596049920元素），不加载权重/forward或初始化GPU；每组件50调用×7 repeats、warmup10，固定seed随机顺序。Mac用架构匹配synthetic config，AMD读pinned config，机器/版本/config差异不混池，meta data_ptr为零。原full signature中位为Mac **422.633µs**、AMD **1601.134µs**；仅单次调用内复用当前dtype/device字符串的候选为 **418.322/1560.543µs**，median ratio **0.989798/0.974649**。AMD原范围1522.485–1804.966µs、候选1359.685–1771.143µs明显重叠；有限样本未证明稳定优化，不据重叠作统计“全是噪声”证明，决定**不改生产signature**。AMD独立组件enumeration867.150µs、config-to-dict+repr100.717µs、metadata/version274.086µs；各组件有自己的调用/分配开销，不加和成full成本，也不从真实graph diagnostic或primary时间抵扣。

**执行与证据。** 唯一AMD CPU/meta窗口为上述Git base加benchmark/test两overlay，source前后364文件通过，三GPU visibility变量为空、GPU initialized=false；measurement/focused test/worker/SSH实际OS0、无timeout，1项focused test **5.437秒**，核original/recomposed/candidate exact tuple、alias/mixed dtype与CPU/meta device及buffer/config/training变更。日志中的四条(null)启动行保留，不补推原因。Root逐字节核Git base archive、overlay与364 manifest，并重算12组件×7 repeats的median/min/max/ratio；记录者读取该审计，独立重算两平台标量及核公开identity的30项evidence SHA、公私result副本一致。独立release确认三个owned identities消失、原ASR身份ready/nonbusy、KFD仅ASR、free25252777984 bytes；无新GPU实验。[公开报告](../reports/model-signature-cpu-20261009/README.md)、[安全否决审查](../reports/model-signature-cpu-20261009/safety-review.json)与[身份/证据哈希](../reports/model-signature-cpu-20261009/identity.json)，本地原始 `output/signature-cpu-20261009/` 保留source、实际exit、审计与释放留证。

**当时状态与下一决策。** 记录者另按root授权仅做一次18:59:03（UTC+8）只读核验：8768真实listener的/proc PID/start ticks与原ASR一致、HTTP200 ready/nonbusy，可读KFD仅ASR，free25252777984 bytes；286个/proc fd无权限保留，不推所有进程都可见或未来launch readiness。时间戳与SHA留在 `output/amd-readonly-20261009T105902Z/`，无launch/kill/GPU/训练。关闭这项host小修分支，保留三个变更检查，回到finite-family/多请求admission与真实physical-B：layout只容纳实际admitted anchors/proposals，原prior freeze仍先于当前RNG，不以padding补零score；后续hot graph/eager家族须受原内存预算约束。此为下一方向，未执行新GPU gate；S33完整timeline缺失、旧数值/law/TV与paired负收益均未解除，capacity/t−2、ZOS及CPU/GPU overlap仍待完整系统证据。记录者仅核证据/写日志，未改实现、运行测试或stage/commit/push。


<a id="query-family-cpu-foundation"></a>
## 2026-10-09 — S35：finite physical-B family的CPU基础完成，保留兼容失败与审查修正

**问题与方法。** S34未采用signature缓存；本轮回到实际工作量，先让同一固定physical-B程序容纳不同ordered Q，而不按最大Q补dummy proposals。[QueryFamily基础](query-families.md)固定R/B/maxQ/Kcapacity/maxK，immutable transaction保存本次实际Q、context与slot incarnation，要求每Q为正整数、sum(Q)=B且各项/总K不超界；offsets、positions、cuQ/cuK、commit与feature消费均用实际Q，预留K尾部不计入可见长度。有限完整tuple一次注册shared metadata/gather arena，各family用固定形状view，不事后扩展或lazy capture；backing显存计费一次，graph I/O与每graph private reservation仍分别计费，不声称共享graph私有pool。Pending transaction和feature lease禁止过早复用；相同data_ptr不替代writer/program/view身份，三个fresh model-signature边界保留。

**CPU覆盖与composition边界。** 六项新测试以真实tiny Qwen和CPUReplayEmulator，对(1,5)/(3,3)/(5,1)、请求重排、context增长与small→large→small B，核raw features/logits及全部layer KV与独立exact eager一致；覆盖inactive resident、incarnation、writer/view drift、未完成/已完成cancel及workspace预算。Native spy核固定family maxQ/maxK参数，而非本次actual maxima，只证明host参数传递，不证明ROCm算子/capture；emulator每次仍执行Python。真实PackedSpeculativeSession在**手工allocations/fixed_budget模式**对照exact Bucket，q/p、决策、RNG、outputs、target/draft KV相同，含零allocation和明确eager/graph-family选择；未接通current-confidence、full-shadow CapacityRoundDriver、t−2或STS驱动的family选择，不把此composition写成完整调度集成。

**实际失败、窄修复与验证。** 初始6项/.108秒与既有persistent 34项/.590秒/OS0后，冻结full suite实际**302项/5.904秒/OS1、errors3/skips6**：给旧exact Bucket eager也加execution_kind=explicit_eager，违反原paired benchmark缺省契约，两项benchmark与一项observer测试均报Unexpected eager tail or undeclared graph。这是真实兼容错误，不改写成已过，也未放宽benchmark；修复只给QueryFamily eager加标记，恢复exact旧行为。新冻结副本的受影响focused **26项/4.020秒/OS0（skip2）**，完整 **302项/6.727秒/OS0（296 pass、6 Linux skip）**，source before/after370通过。两次freeze/log/实际OS code在 `output/query-family-cpu-20261009/`保留，首轮与修后manifest SHA分别 **2fcde5db…/00d7caa9…**；base为 **1e813cad79601ef7d231f6676388f168e6f2e194**加四module/一test overlay。Root和记录者各核冻结370source逐manifest、当前/冻结五overlay一致，base archive逐字节等Git；记录者核17份CPU证据SHA并读日志，未重跑suite。

**独立审查与静态疑点撤回。** Direction对修后冻结副本独立核370source，在R1/R2/R3共27 transactions×2 layers上核独立gather oracle、metadata/positions/K尾部及abort后committed KV；mixed exact/shared fixture实测unique backing **20646 bytes=charged bytes**，并核abort后feature lease、混合路径拒绝复用与completion-checker异常poison所有family程序，结论限于CPU foundation未发现blocker。先前cancel残留completion疑点来自旧源码，root/记录者实际git show HEAD确认原来就同时清ticket与completion，故撤回，只有新增回归覆盖，**没有本轮cancel生产修复**。Exact Q override的bool/float疑点也非缺陷：冻结Bucket.accepts已有strict type(q)is int，三个独立override均ValueError且未发布transaction，无需再修。Review留在 `output/query-family-independent-review-20261009/`；记录者核五证据/五overlay及handoff SHA **6106b51f…e9c142**一致，不把两项静态疑点计入真实失败数。

**当时状态、边界与下一小步。** 本轮记录者只做一次19:28:44（UTC+8）授权只读核验：真实ASR listener的/proc PID/start ticks与原身份一致、HTTP200 ready/nonbusy，可读KFD仅ASR、free25246564352 bytes，可见实验任务候选为空；284个/proc fd无权限保留，不外推不可读范围或未来launch readiness。时间戳/SHA留在 `output/amd-readonly-20261009T112843Z/`。本轮无remote CPU、GPU、训练或速度实验；通过的是有限physical-B的CPU store/target/strategy基础。下一小步另做CPU桥接既有同步full-shadow CapacityRoundDriver与declared identity physical-B测试profile，覆盖current-score zero underfill、nonuniform Q、roster churn与拒绝时不耗RNG，保留freeze-before-draw/private-source校验，分别报告reserved K与actual B；该桥接尚未完成。旧RMS/law/TV、S31负收益、强vLLM baseline与S33完整timeline缺口不解除，native variable-Q、capacity曲线、ZOS/多stream overlap和收益仍待证据。记录者未改实现、运行测试或stage/commit/push。
