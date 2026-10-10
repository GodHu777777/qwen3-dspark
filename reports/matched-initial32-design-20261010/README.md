# Initial32 matched-prefix design evidence

This is preparation for a diagnostic, not a new model result.
The [design](../../docs/matched-initial32-diagnostic.md) fixes the original 32
quality prompts and saved first target outputs, comparing checkpoints 512 and
1280 without sampling new trajectories.

Local CPU metadata inspection confirmed identical prompt/case/initial-output
identity across the two checkpoints, with prompt lengths 32–326 (total 2,972).
The public metadata contains lengths and hashes, not tokens or prompt IDs.

A fresh read-only AMD file check confirmed both checkpoint metadata files and
both draft weight files still match their historical bindings. Each draft file
is 646,774,924 bytes. Installed package metadata matches the historical Torch and
Transformers versions. This used no torch import, model load or GPU initialization;
it did not read target weights or optimizer state. A future launch still needs
current target/tokenizer fingerprints, loaded runtime/backend/device checks,
source deployment identity and a fresh coordinated GPU window.

No runner or new model tests were created or run. Source review preserves the
seven-row draft backbone/head, eight-row block probability conversion and
single-row sequential target path. Numerical controls remain separate from
quality, and this design makes no acceptance-rate or throughput claim.

The [independent review](independent-review.json) approves this design only and
binds the exact document SHA256. Implementation, CPU validation and a fresh live
preflight remain separate steps before any model execution.
