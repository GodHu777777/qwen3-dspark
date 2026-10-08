# 扩充数据 CPU 准备（2026-10-08，运行标签 20261009）

已准备 1280 条唯一 prompt：1024 train、128 validation（dev）、128 final test。
生成前已排除全部 64 条 pilot 输入，包括被长度截断拒绝的输入；与 pilot 重叠为 0。
本阶段没有 GPU 生成。原始 prompt 与精确 token IDs 保留在 ignored 数据目录，
公开文件只有计数、长度分布、来源与哈希。

- [选择汇总](selection-summary.json)：split、排除来源、长度分布、源 revision 与 SHA-256。
- [执行身份](source-identity.json)：实际准备脚本/共享数据代码、配置、tokenizer 与运行库身份。
- [CPU 测试日志](data-tests.log)：8 项全部通过；AMD 同样 8 项通过。

源分片哈希已重新核对，仍与 pilot 一致。完整选择使用原子目录发布；原参数
`prepare.py --resume` 核对后复用相同的选择。配置文件只增加一个空白换行的
恢复负向测试被 `Selection resume identity mismatch` 拒绝，未改变原选择。
旧 pilot 也通过新版 audit，仍为 49 train / 7 validation / 8 rejects。

生成后训练用 `records.jsonl` 只包含 train + validation；最终 test 的 records、
导出与拒绝记录单独放在 `final-test/`。该目录权限 700、文件 600，lock 声明禁止
用于 checkpoint/policy 选择。权限与声明是同用户工作约定，**不是访问隔离**；
完整生成 batches 仍包含 test。只允许对 test 做格式/完整性审计，调参仅用 train/dev。
Astra core 已独立审查三路导出和恢复身份，未发现阻止生成的泄漏/恢复问题。

这些文件记录实际 CPU 准备源码哈希；准备时数据脚本尚未提交，不能用后续 commit
替代执行身份。正式生成前须提交并定点部署源码，先在独立目录做三路小型生成、
导出、审计和恢复 smoke，然后才在新输出目录生成完整 1280 条。

## 实际生成 smoke 与正式运行启动

正式生成 pipeline 版本为 `d5e1538960afded86d487eab71908ce2c7531a7e`，部署前
比对全部 5 个脚本的 Git blob、本地与 AMD SHA-256 完全一致。每个 split
机械选前 2 条，共 6 条，未根据内容作选择。

- [Smoke 首次生成汇总](smoke-summary.json)：6 条全 EOS 完成，2 train / 2 dev / 2 test，2,074 token。
- [Smoke 审计](smoke-audit.json)：三路身份/来源/token/EOS/预算与输出哈希全部通过。
- [恢复对照](smoke-resume-equivalence.json)：原命令重复启动，所有导出 SHA-256 不变，再审计通过。
- [生成执行身份](generation-source-identity.json)：提交、配置/选择/源码/模型哈希与 runtime。

恢复进程会重写 summary 中本次 invocation 的耗时和显存，故首次生成指标从保留的
原 generation log 提取，不把恢复进程的 0 秒当作实际生成时间。完整 1280 条
生成在上述 smoke 全部成功后启动；此版本报告只确认启动，尚无完整数据完成结果。
生成样本与 raw batch 保持 ignored，最终 test 禁止用于选 checkpoint/policy。


## 正式生成完成与最终结构审计（2026-10-09）

上述准备、smoke 与启动记录是当时的阶段结论。随后原 generator 和 runner 自然结束；
正式 generation、full audit 和 runner 退出码全部为 0，六个阶段记录均 exit 0。
记录者直接核实终态，并用与冻结 commit Git blobs SHA-256 相同的四个脚本，
独立调用 stdlib `audit_run`；结果与 runner 的审计完全一致，没有加载模型或启动 GPU。
160 个 batch payload/checksum 全部匹配，1280 条 batch 记录与导出记录逐项一致。

| Split | 输入 | 接受 | 拒绝 | 截断 | 模板 token 不匹配 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 1024 | 932 | 92 | 69 | 23 |
| Validation / dev | 128 | 119 | 9 | 5 | 4 |
| Final test | 128 | 119 | 9 | 6 | 3 |
| 合计 | 1280 | 1170 | 110 | 80 | 30 |

原始生成 finish reason 为 EOS 1200、length 80；30 条 EOS 完成回答仍因模板 token
不匹配而拒绝。全部 1170 条接受记录终止于 EOS 151645。没有按答案正确性过滤，
也没有为消除拒绝而重生成、修补 token 或改变冻结脚本。模板不匹配原因尚未在此
结果中定位；这些记录保持隔离。单分片与长度/模板拒绝仍会限制代表性。

- [正式生成 summary](full-summary.json)：最终输出哈希、生成循环时间与显存记录。
- [完整 audit](full-audit.json)：规范化身份、64 条 pilot 排除、split 互斥、来源、
  token/EOS/预算和 exports SHA-256 检查通过。
- [终态与独立复核](full-completion.json)：各阶段退出状态与独立结构审计口径。
- [正式 generation 身份](full-source-identity.json)：d5e1538、实际 1280 输入的配置/选择/源码/
  模型/runtime 哈希；与上面的六条 smoke identity 分开。私有模型路径替换为标签，
  config SHA 仍绑定原始配置 bytes。
- [split 长度与 batch 聚合](full-aggregate.json)：接受记录的 prompt/output/template 长度
  min、nearest-rank p50/p90/p95/p99、max、sum，及来源/batch 完整性。

训练可用数据是 932 train、119 validation；最终 test 的 119 接受记录只用于后续
冻结方案的最终评估。训练用 records 不含 test。此轮仅做 test 格式、长度、EOS 和
完整性统计，未做质量评估或选 checkpoint/policy。权限/lock 仍是同用户约定。

`generated_tokens=469958` 包含被拒绝的输出；summary 的 `training_tokens=536929`
是三个 accepted splits 的模板总 token，并非 train-only 数量。真正 train accepted
模板总量为 425199、dev 54608、test 57122；精确 output token 分别为
297385、38692、40748。train 最大模板长度 2225，dev 2000；后续 eligibility 与实际
anchor/内存形状仍须由 immutable training preflight 重新核对，不以此替代 GPU gate。
生成 invocation 4853.66 秒、峰值 allocated 3863313920 bytes 是造数据的运行记录，
包含本轮暂停/运行开销，不能据此计算 serving speedup。原始 prompt、回答、token
序列和机器路径未公开；完整生成数据仍在 ignored/远端目录。
