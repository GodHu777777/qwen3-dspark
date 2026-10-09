# Trained packed decoder gate, 2026-10-09

One fixed GPU execution on source `8378ea7af51fdb84e369a4b306259e88d8dd22b7`
completed. All75 structural checks and both exact draft isolation pairs passed.
The trained draft's15 normal pooled layer comparisons and35 per-request layer
comparisons passed the unchanged `.02` absolute + `.02` relative element limit
and `.005` RMS limit. Maximum pooled RMS was0.00352586; maximum per-request RMS
was0.00382591. This validates this bounded noncausal draft/integration matrix.
**The earlier whole-Qwen target RMS failure remains failed.** No whole-system,
distribution-losslessness, quality, speedup, async-overlap or graph claim follows.

The actual pretrained Qwen3-0.6B target and exact selected trained step1280 draft
were bound by hashes. Target BF16, draft FP32 trainables/RoPE with BF16 autocast,
Transformers5.17.0 and pinned private ROCm causal/noncausal backends were used.
The complete protocol is in [packed-decoder-gate.md](../../docs/packed-decoder-gate.md).
The Git archive SHA256 is
`401822fccfc59088776707b721f3c671763f81da8ac523de11ffcffacbd0ea7c`;
all260 archived files and bound inputs were checked before and after execution.

## Scope and same-input endpoint differences

Five native target forwards produced140 private flash events; seven draft
forwards produced35 events, including four poison/control forwards. Three normal
draft forwards cover15 layers. Their request denominator is35, because the first
two forward batches contain three requests and the fixed-zero batch contains one.
The target compares73 normal query rows over five stages; only the22 verification
rows have measured logits/probability TV in this experiment.

The independent SDPA reference replays the native path's actual input chunks and
committed lengths. Of22 verification rows,18 have nonzero probability TV; maximum
TV is0.02938953 and no argmax changes occur. Endpoint completion records these
differences and has no new pass threshold. Resident-KV comparisons include
inactive requests and repeated observations of retained prefixes:

| Domain | Component | Comparisons | Nonidentical | Maximum absolute difference | Largest comparison RMS |
| --- | --- | ---: | ---: | ---: | ---: |
| Target | K | 420 | 304 | 4.0 | 0.04842292 |
| Target | V | 420 | 304 | 1.25 | 0.21104112 |
| Projected draft | K | 75 | 55 | 0.14380133 | 0.01812883 |
| Projected draft | V | 75 | 55 | 0.4375 | 0.05705351 |

Each comparison is one request/layer/component at one stage; these are not
independent tokens, and the maximum RMS is not a pooled RMS. Reference projected
draft KV uses the original projection with independent target features. Exact
native old-prefix/inactive-cache preservation and nonidentical cross-backend KV
are separate facts.

## Accounted work, not measured performance

| Round | Draft query/base-head rows | Sampled proposal positions | Markov/head calls | Verification query rows | Gathered draft key rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| shadow_one | 21 | 21 | 7 | 11 | 71 |
| shadow_readd | 21 | 21 | 7 | 7 | 61 |
| fixed_zero | 7 | 2 | 2 | 4 | 30 |

Shadow mode computes the complete block for all three requests even when a
request later gets allocation0. Fixed-zero explicitly skips A's draft but still
computes B's entire seven-position backbone/base head, then samples two positions.
Each normal round invokes five draft attention calls. Across those five layers,
context-plus-block concatenation accounts for2,181,120 /1,873,920 /1,536,000 bytes;
gather output accounts for2,181,120 /1,873,920 /921,600 bytes, respectively.
Fixed-zero still copies resident inactive context before gathering. These are
logical allocation/copy volumes, not measured device memory traffic or latency.
The three rounds accepted zero of14 selected proposal tokens on synthetic input;
this is not a model-quality estimate.

## Evidence and execution limits

The worker, launcher, controller and monitored SSH handle all exited0. There was
no timeout, retry or fallback. Peak allocated memory was3,492,600,832 bytes under
the6 GiB allocator cap; correctness instrumentation took35.456 seconds. All four
owned processes were independently confirmed gone, ASR retained its recorded
start time and was ready/not busy, KFD had only ASR, and free VRAM returned to
25,241,243,648 bytes. Timing is not a benchmark.

The already-running external controller used bare PIDs when the review identified
its PID-reuse weakness. Start times were captured after launch and independently
checked at release; no termination signal was needed. Successful release does
not prove that controller safe under PID reuse. The source experiment was not
changed or rerun. Future experiments must use the tested identity-aware supervisor.

A GPU-hidden CPU audit verified25 private artifact hashes, all15 pooled and35
request-layer gate calculations,1,022 endpoint scalar comparisons,73 hidden
query rows,22 probability rows, both exact poison proofs and78 recorded
probability-distribution rows. Maximum CPU/GPU TV-statistic delta was7.05e−9;
recomputed FP32 oracle maximum absolute delta was1.24e−5. These arithmetic
diagnostics do not change the original GPU verdict or thresholds.

The first CPU preparation test run had an autocast-oracle mistake; review fixed
it before this immutable GPU source. The executed oracle explicitly disables
autocast, and all15 saved oracle tensors are FP32. Historical logs remain private.

- [summary.json](summary.json): separate result states, denominators and release.
- [numerical.json](numerical.json): every layer, endpoint/KV and structural scalar.
- [work.json](work.json): per-round work counts and acceptance counts.
- [source-runtime.json](source-runtime.json): source/input identities and runtime.
- [cpu-audit.json](cpu-audit.json): independent saved-tensor verification.

Prompts, generated token traces, probability tensors and machine-specific paths
are excluded from these public aggregates. `system_pass_claimed` remains false.
