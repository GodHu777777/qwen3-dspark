# Whole pretrained Qwen/KV gate: structural pass, layer numerical failure

The single authorized actual-Qwen3-0.6B run finished its complete fixed matrix.
Structural/content checks passed, but the mandatory normal-layer numerical gate
**failed**. All endpoint comparisons completed. No threshold, backend, case or
source was changed, and no GPU retry was run.

| Result state | Observed result |
| --- | --- |
| Execution | Completed; no timeout; worker/launcher/controller OS exit 1 |
| Structural/content | Passed all 68 recorded checks |
| Normal-layer attention | Failed 3 of 112 layer comparisons; 5 of 252 request comparisons |
| End-to-end comparison | All 67 query rows compared against both reference paths |
| System/equivalence/speed claim | None |

Source is `7d1fcdbd4f3415b1b63bb803965d441545319487`; archive SHA256 is
`8353b060b792106b3f96c2a71699cc300f05dd051ef6a22eda64ce3127034603`.
All 220 archive files were checked before and after execution. The actual
pretrained model, tokenizer and generation manifest fingerprints were verified
before loading. Protocol SHA256 is
`d83dfd50e9b0d371e8bf3538cdc7cedc906136ed88b27c855085df15cf41238d`.
The source was immutable and the fresh standard-library dry-run and CPU runtime
schema checks passed before the live resource guard.

## Complete fixed matrix and the numerical failures

The 28-layer model executed 8 native forwards, each with exactly 28 observed
`aten::_flash_attention_forward` operators: 224 callbacks total. The four normal
forwards supplied actual BF16 Q/K/V and outputs to 112 independent FP32 MATH
bottom-right SDPA checks, both pooled and separately for each request. Dense
packed and independent-request SDPA baselines used separate actual model copies.
The fixed stages, synthetic tokens, crop/exit/re-add operations and poison pairs
are retained in [protocol.json](protocol.json).

Every elementwise comparison passed the original `atol=.02, rtol=.02` rule.
The fixed RMS limit `.005` failed at zero-based decoder layer 26:

| Stage / request | Actual RMS vs FP32 | Nearest-BF16 rounding RMS vs same FP32 oracle |
| --- | ---: | ---: |
| prime / B | 0.0051563464 | 0.0047095117 |
| cached_tail / A | 0.0051409812 | 0.0049002298 |
| cached_tail / B | 0.0056739500 | 0.0051657100 |
| crop_exit_readd / A | 0.0051131269 | 0.0044782015 |
| crop_exit_readd / B | 0.0058588637 | 0.0050888052 |

The prime and crop stage pooled RMS values passed; their per-request checks
caught the failures. Cached-tail pooled RMS was 0.0056172290 and failed as well.
The fourth normal stage, crop-to-zero, passed every layer. All 3 failing layer
Q/K/V/output/oracle/cumulative-length artifacts were retained privately before
continuing the finite matrix. [layer-attention.json](layer-attention.json) keeps
all 224 observations and the original fixed-limit decisions.

A subsequent **CPU-only diagnostic**, requested after the result, rounded each
saved FP32 oracle to its nearest BF16 value. This is a BF16 representability
lower bound relative to that saved reference. It already exceeds `.005` for
2 of the 5 failed request comparisons. For the other 3, the lower bound is below
`.005` while the actual output exceeds it. Native outputs differ from the nearest
rounded oracle, so the failures cannot all be attributed solely to an impossible
BF16 threshold. This diagnostic neither changes the gate nor establishes a
replacement acceptance rule. CPU FP32 MATH reconstruction differs from the saved
GPU FP32 oracle by at most 0.0000534058 in these artifacts; it is reported
separately, not substituted into the original result.

## Structural isolation and end-to-end differences

Actual model tokens/RoPE positions and marker-derived gather indices matched the
independent expected mapping. Every append retained actual old KV prefixes and
inactive caches exactly. Crops retained exact KV/metadata subsets, removed marker
rows disappeared, and the re-added request started with a fresh marker at position
zero. In both poison pairs, request A's per-layer Q/K/V, hidden, logits and all
resident KV were byte-identical. Poison amplitudes do not receive the ordinary
numerical acceptance threshold. [structural.json](structural.json) preserves the
checks and evidence hashes.

| Native output vs reference | Dense packed | Independent per request |
| --- | ---: | ---: |
| Compared query rows | 67 | 67 |
| Nonzero TV rows | 48 | 47 |
| Maximum TV | 0.0229268524 | 0.0532455264 |
| Argmax changed rows | 1 | 3 |
| Maximum absolute logit difference | 0.3125 | 0.46875 |

Final hidden, selected intermediate features, every resident per-layer KV and
full-vocabulary logits were saved before metrics. [end-to-end.json](end-to-end.json)
contains request/stage differences; there is no cross-model numerical pass
threshold or losslessness claim. These are synthetic cache/attention fixtures,
not a quality benchmark or a performance experiment.

## Independent audit and release

CPU rechecking hashed all 9 private tensor artifacts, reproduced 1,398 endpoint
scalar comparisons, and verified both exact poison proofs. It also recomputed
134 probability rows from saved logits. CPU/GPU float64 arithmetic yields a
maximum TV-statistic difference of 2.0468124e-8. Two initial CPU audit attempts
stopped at an overly strict audit-only probability equality assertion; those logs
are preserved privately. The final audit reports these arithmetic differences
without assigning a new equivalence threshold. Original GPU metrics and all gate
verdicts are unchanged. [independent-cpu-audit.json](independent-cpu-audit.json)
includes per-request rounding diagnostics and every probability recomputation.

Peak allocated memory was 3,808,959,488 bytes, below the fixed 6 GiB allocation
cap; the pre-load 8 GiB free guard passed. The controller recorded 45.78 seconds
including model loading, oracles, instrumentation and persistence, which is not a
speed measurement. All four owned PIDs disappeared; KFD returned to the preserved
ASR process alone, and ASR remained ready and not busy. [runtime.json](runtime.json)
records before/after resource snapshots and actual OS exit provenance. Public
files contain scalar evidence and hashes; raw tensors and private paths stay out
of this report.
