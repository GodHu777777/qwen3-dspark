# Native eager DSpark versus vLLM — 2026-10-09

The native eager full-shadow implementation is substantially slower than the
matched target-only vLLM baseline: **7.06%–12.26% of vLLM output throughput**, or
**8.16–14.17 times its batch time**. There is no measured speedup. One target and
one trained step 1280 draft completed all 54 preregistered batches, with actual
worker/controller/outer-shell exit 0 and independently verified GPU release.

## Matched primary results

Both implementations consumed the same exact synthetic prompt IDs and seeds:
R∈{1,2,4}, supplied prompt C∈{64,256}, exactly 128 outputs per request, temperature 1,
ignore-EOS, no template expansion, neutral penalties and no prefix reuse. Each
cell has five primary batches. Rates are pooled output tokens divided by pooled
batch wall seconds, with no discarded samples or average-of-rates calculation.

| R | C | Native tok/s | vLLM tok/s | Native/vLLM throughput | Native/vLLM batch time | Native median batch s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 64 | 16.092 | 131.264 | 12.26% | 8.157× | 7.960150 |
| 1 | 256 | 13.928 | 127.340 | 10.94% | 9.143× | 9.121911 |
| 2 | 64 | 23.686 | 252.165 | 9.39% | 10.646× | 10.744646 |
| 2 | 256 | 24.032 | 246.567 | 9.75% | 10.260× | 10.692894 |
| 4 | 64 | 36.408 | 515.998 | 7.06% | 14.173× | 13.962428 |
| 4 | 256 | 33.848 | 475.260 | 7.12% | 14.041× | 15.092377 |

[Primary metrics](primary-metrics.json) preserve every duration, total, mean,
median and range. [Matched comparison](matched-comparison.json) retains full
precision. Both rates were recomputed independently from each implementation's
original samples. The earlier [vLLM report](../vllm-offline-benchmark-20261009-700bfa6)
records its single BF16 engine, `FULL_AND_PIECEWISE` graph capture and default
`ROCM_ATTN` with its internal Triton paged-attention path. Replay was not
independently instrumented. This is a comparison of complete implementations;
it does not isolate the causal effect of speculative decoding.

Native batch timing includes fresh cache/request construction, target reset,
batched admission and first-token sampling, every full seven-position shadow,
private-confidence copying and validation, target verification, original FP64
q/p sampling checks, commit/crop/projection and final synchronization. Token/RNG
preparation and model loading are outside batch timing and recorded separately.
Admission emits the first of 128 tokens. Finished requests remain resident and
continue contributing to relevant cache work until batch completion. The
pre-draw prefix is always `max(0,min(7,remaining−1))`; ell0 still computes a full
shadow. No capacity planner, graphs or asynchronous overlap are integrated.
Primary request latency and TTFT remain null; batch time is not divided by R.

## What the counts establish

The following per-batch counts repeat identically across all five primary
repeats. Across all nine batches per cell, each request also has the same native
output hash. This is same-path repeat evidence, not cross-backend equivalence.

| Case | Batched verification rounds | Active request-rounds | Accepted draft tokens | Selected proposals | Committed tokens per active request-round |
| --- | ---: | ---: | ---: | ---: | ---: |
| r1-c64 | 91 | 91 | 36 | 624 | 1.396 |
| r1-c256 | 106 | 106 | 21 | 714 | 1.198 |
| r2-c64 | 108 | 200 | 54 | 1351 | 1.270 |
| r2-c256 | 106 | 210 | 44 | 1416 | 1.210 |
| r4-c64 | 112 | 382 | 126 | 2587 | 1.330 |
| r4-c256 | 119 | 423 | 85 | 2868 | 1.201 |

After admission, sequential one-token decoding would need 127 batched steps for
this fixed output budget. Native used 91–119 verification calls, a 6.3%–28.3%
reduction in that call count, while paying for seven-position drafting and
multi-row verification. This arithmetic measures modest accepted-token benefit
on these synthetic trajectories. It is not a timing estimate for a native
sequential implementation, and these inputs do not estimate natural-language
acceptance or quality. [Work metrics](work-metrics.json) give the exact counts.

The previous [local round profile](../packed-profile-20261009-700bfa6) found large
nested target-append and proposal host spans at fixed R2/C128. That identifies
regions to investigate, but it cannot apportion this growing-context end-to-end
gap: host spans include GPU synchronization, and GPU event intervals can include
dispatch gaps. This run contains no isolated host/kernel cost measurement.
Persistent KV storage, graph-friendly execution and reduced synchronization are
engineering hypotheses, not measured explanations or promised speedups.

A same-backend native target-only control would distinguish the native execution
cost from the incremental draft/verification cost more directly. It has not been
run. Neither this result nor the earlier local profile justifies attributing the
entire gap to the drafter, host synchronization, attention or missing graphs.
Future controls must preserve the declared FP64 sampling law and input boundaries;
that control remains a separate future experiment.

## Separate diagnostic observations

Two diagnostic batches per cell followed all primary measurements. Native
clocks share a batch-submission start and end at the host-observed complete
batched admission or each request's finishing round. These differ from vLLM's
individual `add_request` start clocks; do not interpret them as identical service
latency measurements. They include sampling/context work and observation costs,
not isolated prefill. Primary measurements remain the throughput source.

| Case | Request observations | Median native diagnostic TTFT ms | Median native diagnostic completion s |
| --- | ---: | ---: | ---: |
| r1-c64 | 2 | 59.282 | 7.967737 |
| r1-c256 | 2 | 54.054 | 9.257634 |
| r2-c64 | 4 | 56.442 | 10.135949 |
| r2-c256 | 4 | 55.028 | 10.546887 |
| r4-c64 | 8 | 60.022 | 12.305200 |
| r4-c256 | 8 | 61.382 | 13.824833 |

[Diagnostic metrics](diagnostic-metrics.json) preserve mean/min/max and
observation scope. Requests within one batch are correlated; these small pools
do not establish serving-tail percentiles.

## Source, runtime and release

Execution used immutable Git source
`0c36b03416311c0ca529d10ff0a10663ebd407fd`, archive SHA-256
`ad8ee63bb232b847657d85a12fb1d6703c32f415d69060fe6738fe4c153eaafd`.
All 307 archived files were verified before and after execution, along with
actual target/tokenizer, step 1280 checkpoint and exact shared workload bindings.
The complete immutable CPU suite passed 212 tests in 13.648 seconds with GPUs
hidden before this one GPU execution. No retry, panel reduction or backend
fallback occurred; the original 1800-second limit remained fixed.

Torch 2.12.0+rocm7.2, Transformers 5.17.0 and the pinned native ROCm ATen path ran
on gfx1201. One model load/setup took 18.152811 seconds outside batch timing.
Supervisor lifetime was 639.693503 seconds. The observed maximum whole-operation
allocator peak was 2,459,580,928 bytes, reserved 2,728,394,752 bytes, below the 6 GiB
cap. These peaks begin before each target reset and include inherited prior-batch
KV and prepared inputs; they are not independent per-cell steady-state peaks.
The 124 health samples saw maximum global VRAM use 12,178,845,696 bytes, including
the preserved ASR; sampling does not establish a continuous maximum.

Actual worker, controller and outer execution all exited 0, without timeout or
supervisor cleanup signals. A separate live check found worker/controller/shell
absent, the preserved ASR with the same start identity, ready and idle, the only
KFD owner, and VRAM restored exactly to the prior 8,967,499,776 bytes used.
Fresh source/input verification also passed. See [runtime audit](runtime-audit.json)
and [source identity](source-identity.json).

## Independent audit and limits

The [stdlib verifier](independent-verify.py) imports no benchmark implementation.
It reconstructs all 54 schedule identities and all 5,778 rounds: 12,708 active
request-rounds, 99 rounds containing ell0 and 621 rounds with inactive resident
requests. It checks pre-draw allocations, full-seven-position shadows, actual
verification B, acceptance/commit/cache accounting, 128 outputs per request,
diagnostic bounds, primary null clocks, all six pooled rates, saved process
release and source/input integrity. [Verification](verification.json) records
the results. A separate root audit independently checked the scalar results and
matched the archive byte-for-byte against Git; see [root verification](root-verification.json).

[Scalar samples](scalar-samples.jsonl) preserve all 54 batch records, output
hashes, per-request diagnostics and round-count summaries. The original 16 MB
round-level samples remain in private evidence, hash
`4fac289b931017aeaf65a43723aa4413e299656d64ac8dc30ed4b57d374b3883`.
No prompt/output token traces or private machine paths are published.

The earlier whole-Qwen target RMS gate **remains failed**. Native and vLLM use
different numerical and sampling implementations; equal inputs and seeds do not
establish equal distributions. This fixed synthetic offline result is neither
an arrival-load serving frontier nor speculative verification SPS, and it does
not establish a quality result, distribution equivalence or a complete DSpark
scheduler implementation.
