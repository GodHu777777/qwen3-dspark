# Expanded training: memory gate and steps 32 / 128

Training uses the frozen `796fecc` tree, Qwen3-0.6B, 32 anchors and gradient
accumulation 8. The original max_steps remains 1280. This report stops at 128;
no checkpoint/policy is selected and no later stage is implied.

The audited development export yields 932 eligible train and 119 validation
rows. Exact train inputs contain 424267 tokens; the largest
observed sequence is 2224 tokens. The corresponding rendered-template field is
2225 because it appends one newline after EOS. Training uses exact generation
IDs through EOS, preserving the existing audited data and identity.

## Execution

| Segment | Cumulative traversals | Optimizer loop (s) | Result timer (s) | Whole process (s) | Peak allocated bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh to 32 | 0.274678 | 20.998 | 36.973 | 61.789 | 5571854848 |
| Resume 32 → 128 | 1.098712 | 60.146 | 76.429 | 107.271 | 5633036288 |

Traversals = optimizer_steps × 8 / 932; fixed-order rows cycle, while anchors
are resampled. Optimizer-loop time is the last segment metric elapsed value.
Result time also includes final validation and checkpoint saving. Whole-process
time additionally includes startup/loading, strict resume and initial validation.
These timings are execution evidence, not a serving throughput benchmark.

The independent memory gate passed two complete accumulation/update cycles on
(2224 tokens, 32 anchors). The second began with 1293537536 bytes of Adam state
resident. Peak allocation was 5636591616 bytes; peak reservation 5827985408 bytes.
All projection/backbone/Markov/confidence gradient checks and target-frozen
checks passed. It wrote no checkpoint and did not warm-start training.

## Fixed teacher-forced validation

| Checkpoint | Macro loss | Macro CE | Teacher-forced distribution overlap |
| --- | ---: | ---: | ---: |
| Initial | 3.385942 | 11.998166 | 0.0857% |
| Step 32 | 2.582735 | 7.180634 | 2.1063% |
| Step 128 | 2.525809 | 6.329202 | 9.3599% |

Each evaluation uses all 119 eligible validation rows and the same anchor seed,
with supervision weights within each row and a macro average across rows. This
is separate from the planned 32-example rollout panel. Overlapping block labels
are not unique token counts. Teacher-forced overlap is not rollout acceptance,
confidence calibration, model quality, numerical losslessness or speedup.
Step-128 confidence MAE is 0.098902; a constant-zero
head would have MAE 0.093599 under these same
weights. Lower loss and higher overlap do not establish a calibrated head.

Both bounded segments exited successfully, preserved the frozen target and
passed strict source/config/data/runtime identity checks. Checkpoint weight and
optimizer/RNG-state hashes match metadata, and saved resume steps match the
checkpoint numbers. The resumed initial validation matches step-32 final:
`True`. Large checkpoints
remain private; aggregate hashes are included for provenance.

## Files and limits

- `source-identity.json`: frozen source, target/tokenizer, development data,
  runtime and path-free training hyperparameters.
- `memory-gate.json`: observed shape, both optimizer cycles and memory evidence.
- `summary.json`: stage results, checkpoint verification and timing definitions.
- `training-metrics.jsonl`: numeric optimizer-step metrics 1–128; elapsed resets
  on resume at step 33.

No machine paths, sample IDs, prompt/answer text or final-test quality are
published. No dataset or manifest was changed to satisfy resume. The empirical
memory result does not guarantee all future allocator/kernel behavior. These
training results do not resolve the existing dynamic BF16 decoding failures.
