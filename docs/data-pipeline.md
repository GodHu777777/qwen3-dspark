# 三路数据生成与隔离

`configs/expand.example.json` 准备 1280 个输入（1024 train、128 dev、128 test）。
这是输入数量，最终有效回答数取决于 EOS、长度和模板审计；不得把输入数写成训练完成数。

复制为 `configs/expand.local.json` 并设置本机模型目录，然后从仓库根目录执行：

```bash
python scripts/prepare.py --config configs/expand.local.json \
  --parquet /path/to/train-00000-of-00006.parquet \
  --exclude-prompts data/pilot/input/prompts.jsonl --output data/expand/input
python scripts/generate.py --config configs/expand.local.json \
  --prompts data/expand/input/prompts.jsonl --output data/expand/generated \
  --source-commit "$(git rev-parse HEAD)"
python scripts/audit.py --run data/expand
```

生成前按规范化后的首轮用户 prompt 去重、排除全部 pilot（含被拒绝回答的输入），
固定源分片、tokenizer、配置和脚本身份。`prepare --resume` 只核验并复用相同选择；
生成命令原样重复可恢复完整批次，载入前检查 checksum，未完成批次重新执行。
不要在恢复中修改脚本、配置、模型或运行库；新实验用新目录。

训练用 `generated/records.jsonl` 只含 train 和 validation（dev）。最终 test 放在
`generated/final-test/`，不用于 checkpoint 或策略选择。目录权限与 lock 声明是
同一用户下的工作约定，不是安全访问隔离，底层 batches 仍包含三路记录。
可以对 test 进行格式和来源完整性审计；模型与策略固定后才进行最终质量评估。

正式长任务前需在独立目录验证三路真实生成、导出、审计和恢复。历史 CPU 选择证据：
[数据准备报告](../reports/data-expansion-20261009/README.md)。所有实际样本与权重
保留在 ignored 目录；Git 只保存脚本、配置模板及脱敏聚合证据。
