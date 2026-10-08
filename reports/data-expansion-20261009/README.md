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
