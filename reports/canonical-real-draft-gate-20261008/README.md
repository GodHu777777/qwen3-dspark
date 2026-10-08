# Experimental canonical target: real draft gate

This is separate from the earlier target-only oracle probe. It loads the actual
aligned step-128 draft checkpoint and reruns all three original examples with
BF16 target, BF16 AMP draft computation, fixed query width 8 and key capacity 256.
Canonical sequential and speculative routes share their full-prompt prefill and
always project the full padded query through the LM head before selecting rows.

All 3 examples, totaling 70 output tokens, matched **canonical** sequential
greedy output. They matched stock dynamic cached output on only 2/3 examples.
The prior dynamic BF16 gate still fails and has not been overwritten or relabeled.

| Example | Output tokens | Verification rounds | Committed draft tokens | Canonical equality | Stock equality |
| --- | ---: | ---: | ---: | --- | --- |
| Seen training trajectory | 32 | 4 | 28 (7 each round) | yes | yes |
| Dev 1 | 6 | 5 | 0 | yes | yes |
| Dev 2 | 32 | 31 | 0 | yes | no |

This verifies integration of real draft proposals, intermediate target features
and cache rollback under this optional execution policy. It does not demonstrate
held-out draft usefulness, universal floating-point identity or speculative
speedup. Both dev examples accepted zero draft tokens.

The 39-test CPU suite passed before the GPU run, including the five new canonical
tests. Shared dynamic-cache behavior remains unchanged except for routing logits
through an equivalent `predict(features, last_only)` interface.

The authorized controller paused only the verified generator for 35.655 seconds,
then resumed the same process with pidfd. Its state was observed as running and
the generation log grew by 78 bytes. The GPU diagnostic completed in 22.08 seconds
with 2.221 GB peak allocated memory. These are resource records, not benchmark
latency or total VRAM measurements.

Source hashes and executed source snapshots accompany the report. Public
summaries omit real prompt content, prompt identities and generated token IDs;
historical executed sources retain machine-specific paths and process IDs, not
credentials. The controller archive records past authorization and is not a
reusable instruction to pause a process.

Canonical padding remains an experimental numerical control. DSpark's global
logical verification budget and actual physical token work must remain distinct;
padding cost belongs in hardware measurements. A slower canonical baseline
cannot establish improvement over stock decoding.
