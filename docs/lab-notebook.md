# 实验日志

最近直接证据核对：**2026-10-08 22:46（UTC+8）**；其后进度由协调者转达 agent 报告，以下明确注明。本文持续追加；旧结论若被修正，保留原结论并说明修正依据。历史实验与实时进程状态分开记录。

研究问题：冻结 Qwen3-0.6B target 后，并行 DSpark 草稿能否比带 KV cache 的 target-only greedy 更快地产出完全相同的 token？训练可运行、loss 下降、回退输出一致，各自只回答这个问题的一部分。阶段门槛见[实验计划](experiment-plan.md)。

公开链接只指向聚合报告。生成的 prompt、回答、精确 token 轨迹、机器路径、权重、优化器状态及凭据不进入日志。`output/` 是 **未发布的本地证据**。已有实验早于首个项目提交，实际身份由当时 manifest 中的源码、配置、数据、模型哈希绑定；后来的 Git 基线不被描述成启动这些历史实验的 commit。

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
