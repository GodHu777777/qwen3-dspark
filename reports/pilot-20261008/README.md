# Initial execution experiment — 2026-10-08

This experiment establishes executable training and reference greedy correctness,
not trained quality or speedup. Raw aggregate metrics and original CPU test output
are committed here; private machine checkpoint paths have been replaced by
`REMOTE_CHECKPOINT`. Generated prompts, answers, token traces and weights are not
redistributed.

- Frozen Qwen3-0.6B; 161,692,161 draft trainable parameters.
- 49 train / 7 validation responses, selected before generation from 64 prompts.
- 8 optimizer steps / 16 microsteps, stopped at 4 and resumed to 8.
- Validation macro loss: 3.421146 → 2.714265.
- Teacher-forced overlap: 0.001838 → 0.012885.
- Target frozen checks passed. Peak allocated training memory about 4.30 GiB.
- Real greedy comparison: 2 prompts, 30 output tokens, exact equality to target-only,
  ZERO draft tokens accepted. The directly tested earlier run and final v2 run had
  identical weight SHA-256 `5df9b6e052aff2a5cee2542e7b3356f75a071b560d56195bed1a729818ae4c04`.

At step 4 the overlap was 0.015046: it declined by step 8 despite a lower total
loss. Final confidence MAE 0.013045 is worse than the zero-constant baseline MAE
0.012885 under this definition. Neither total loss nor apparently small confidence
error establishes useful drafting. Do not use these results in a speedup claim.

The test log includes two environment startup messages `(null): No such file or
directory`; all eight unittest cases subsequently executed and passed. The log is
preserved verbatim rather than silently cleaned.
