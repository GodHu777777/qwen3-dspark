# Full64 native local cost profile, 2026-10-09

One immutable `700bfa60746f1315baaeec5ead28cc5ee553ae52` execution completed all64
ordered allocations at R=2, resident R=2, committed C=(128,128), gamma7. All128
warmups,320 primary samples and64 separate diagnostic samples are retained.
The primary round median across320 observations is94.978 ms, with a range of
78.388–153.418 ms. This is a frozen-cache local allocation profile, **not** an
end-to-end throughput benchmark or a complete capacity-scheduler cost curve.

The measured path includes full shadow proposal, private confidence copying,
capability validation, verification, original FP64 q/p checks and commit/crop.
Real capacity search/calibration/history maintenance is absent. Both the report
and lookup explicitly reject use as a `CapacityRoundDriver` profile. The reported
local rate is `1 / median round seconds`; neither B/time nor output tokens/s is
substituted. The earlier target numerical failure remains unchanged.

## Fixed domain and same-B observations

All vectors in `{0,...,7}²` were measured. Every repetition swept the whole domain,
rotating the canonical order by17 times its index and reversing odd repetitions.
Warmup, primary and diagnostic phases were separate. Each primary cell has five
observations; none were filtered as outliers. Context/RNG/KV restoration and
exact-content checks occur outside timing, with median restoration cost9.574 ms.
Snapshot preparation, model load and evidence writes are also excluded.

| Logical B | Ordered ell cells | Smallest–largest cell median, ms | Median ratio |
| --- | ---: | ---: | ---: |
| 2 | 1 | 85.024–85.024 | 1.000 |
| 3 | 2 | 89.010–93.949 | 1.055 |
| 4 | 3 | 92.178–92.966 | 1.009 |
| 5 | 4 | 94.104–101.923 | 1.083 |
| 6 | 5 | 91.274–94.236 | 1.032 |
| 7 | 6 | 88.831–99.854 | 1.124 |
| 8 | 7 | 91.323–97.951 | 1.073 |
| 9 | 8 | 91.215–105.638 | 1.158 |
| 10 | 7 | 92.645–98.583 | 1.064 |
| 11 | 6 | 93.080–102.755 | 1.104 |
| 12 | 5 | 95.240–106.082 | 1.114 |
| 13 | 4 | 92.934–104.734 | 1.127 |
| 14 | 3 | 92.985–93.612 | 1.007 |
| 15 | 2 | 90.282–90.671 | 1.004 |
| 16 | 1 | 93.316–93.316 | 1.000 |

These are descriptive observed spreads, not proof of a causal layout effect.
At B9, the post-hoc smallest-median cell ell=(0,7) spans86.513–96.968 ms; the
largest-median cell ell=(3,4) spans90.597–118.258 ms. The ranges overlap. Paired by
repetition index, the latter is slower in5/5 repetitions, but differences range
from0.646 to24.954 ms (ratios1.007–1.267). The cells were selected by their observed
extreme medians, so this is not an unbiased inferential test.

The same-B total cost also includes different realized commit work: those B9
cells accept0 versus1 draft token and commit2 versus3 output tokens. Noise, time
drift and that work difference remain possible contributors. The audit preserves
the predeclared balanced-versus-left/right-concentrated contrasts for every B,
including all five paired differences, without inventing a transfer threshold.
All primary decision-count records repeat identically within each exact cell.

## Separate diagnostic regions

| Region | Observations | Median host span, ms | Median GPU stream interval, ms |
| --- | ---: | ---: | ---: |
| Full shadow proposal | 64 | 33.129 | 33.182 |
| Draft backbone, inside proposal | 64 | 11.074 | 11.129 |
| Private-source verification and commit | 64 | 61.121 | 61.145 |
| Target append, inside verification | 64 | 43.461 | 43.512 |
| Target head, inside verification | 64 | 0.129 | 0.780 |
| Target crop, per request call | 128 | 1.651 | 1.699 |
| Project committed draft context | 64 | 4.388 | 4.447 |

The diagnostic complete-round median is95.092 ms. Its median remainder outside
the two top-level host spans is1.069 ms, including policy construction, observer
cost and final synchronization. These are nested regions: do not sum every row.
GPU event intervals include dispatch gaps and synchronization effects, and are
not isolated kernel durations. Host head-call time can be shorter than its GPU
interval because enqueueing is asynchronous. No cross-pass subtraction establishes
an exact host-versus-device split.

Target append and proposal work outside the draft backbone are the main regions
to investigate next. This coarse profile does not establish whether attention,
MLPs, metadata, repeated host synchronization or copying dominates inside them.
Persistent KV/layout buffers and capture-friendly deterministic target subgraphs
are candidates for measured comparison, with the synchronous oracle retained.

## Execution, audit and limits

Worker, controller and monitored SSH handle exited0; there was no timeout,
fallback or retry. Peak allocated memory was2,268,122,112 bytes and peak reserved
memory2,348,810,240 bytes. The identity-aware supervisor's107.010-second lifetime
includes setup, restore and evidence work and is not serving throughput. All
three owned processes were independently absent after exit; preserved ASR kept
the same start identity, was ready/not busy, and was the only KFD owner. Free
VRAM returned to25,241,243,648 bytes. All288 archived files and bound inputs were
unchanged after execution.

A local stdlib audit independently recomputed all512 sequence identities,
per-cell statistics, rates, actual query/commit work and same-B contrasts. Root
also independently checked the source archive against `git archive 700bfa6` and
the sample denominators/statistics. [samples.json](samples.json) is an exact copy
of the original scalar worker result, SHA256
`ca8c40af1fa7d4cd504d50beeca992ff7d505af5bb7bd2aeae5cde3092eca1e2`.
It contains no input/output token traces, probability tensors or private paths.

- [summary.json](summary.json): timing scope, counts and memory aggregates.
- [samples.json](samples.json): every warmup, primary and diagnostic observation.
- [cpu-audit.json](cpu-audit.json): recomputed work, same-B pairs and uncertainty.
- [source-identity.json](source-identity.json): source/input/runtime fingerprints.
- [execution.json](execution.json): actual exits and release.

These measurements cover one frozen R/context/input state. They do not validate
growing-context, request-churn or other-R transfer. A future frozen cost model can
make explicit predictions outside this table, but requires independent validation
and reported error; it cannot relabel those predictions as measured cells. This
local rate must not be directly compared with vLLM's end-to-end output tokens/s.
