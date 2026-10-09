# Original R2 categorical fallback A/B

Immutable source: `c0d55d23c13c6514a825aacf90dc074f8bd3a05c`. A is the unchanged original sampler; B is an opt-in fixed-shape last-positive-index candidate. Production defaults remain A. One bounded, observer-free 20-batch execution used four warmups and four reversed-order A/B pairs per arm.

Predeclared practical screen: **not passed**. Both arms must independently reach pooled B/A throughput ≥1.02 and B must be faster in at least three of four pairs. This is a practical rule, not a confidence interval.

| Arm | A tokens/s | B tokens/s | B/A throughput | B faster pairs | Screen |
| --- | ---: | ---: | ---: | ---: | --- |
| target_only | 81.261790 | 80.510255 | 0.990752 | 2/4 | fail |
| fixed_gamma7_full_shadow | 35.170647 | 34.495830 | 0.980813 | 1/4 | fail |

The candidate misses both predeclared arm gates. Retain the original production sampler and close this optimization branch without retry or threshold adjustment. The observed pooled differences are about −0.92% (target-only) and −1.92% (gamma7); four pairs do not establish statistical significance or universal regression.

Each rate is 1024 output tokens divided by the sum of four complete-session wall times. Pooling excludes warmups. The fixed order is AB/BA/BA/AB per arm, with arm order alternating. Binding/restoring the direct sampler aliases and common raw-state collection happen outside the timer; all actual sampler validation, scalar checks and normal session work remain timed. No internal observer, extra phase fence, signature wrapper or profiler is installed.

| Arm | Pair | Order | A wall s | B wall s | B/A throughput |
| --- | ---: | --- | ---: | ---: | ---: |
| target_only | 0 | AB | 3.146729 | 3.277292 | 0.960161 |
| target_only | 1 | BA | 3.161460 | 3.119764 | 1.013365 |
| target_only | 2 | BA | 3.115390 | 3.191723 | 0.976084 |
| target_only | 3 | AB | 3.177670 | 3.130098 | 1.015198 |
| fixed_gamma7_full_shadow | 0 | AB | 7.181306 | 7.401213 | 0.970288 |
| fixed_gamma7_full_shadow | 1 | BA | 7.240454 | 7.218560 | 1.003033 |
| fixed_gamma7_full_shadow | 2 | BA | 7.352379 | 7.402413 | 0.993241 |
| fixed_gamma7_full_shadow | 3 | AB | 7.341049 | 7.662562 | 0.958041 |

All ten same-arm A/B batches, including warmups, have identical raw output arrays, final RNG bytes and complete round work/decisions/execution. The device gate compared 15 reconstructible original/candidate outcomes, including CDF rounding fallback, zero support, callback mutations and error order; all passed before warmups. An independent CPU audit reloaded both retained large-vocabulary A/B tensor pairs and matched them exactly to each other and their reconstructed input laws.

Independent scalar auditing checked 2330 rounds, 5120 outputs, source provenance (404 files) and every pooled/pair calculation. Startup raw tensor auditing passed 2203 checks and 756 reductions under unchanged .02/.02/.005 limits, max absolute error/RMS 0.0/0.0. Prefill/growth prefix model execution was not independently replayed.

The pinned AMD GPU-hidden affected suite passed all 77 tests (33.472 s; wrapper 41.475 s). Worker/controller/SSH exited 0; independent release passed and ASR remained ready. No timeout, retry or trace export occurred. Original memory limits and all three fresh signature checks remained intact.

The historical same-backend primary rates remain target-only 80.6661, gamma7 34.9868 and zero admission 31.5652 output tokens/s. The separate-stack vLLM reference remains 246.5667. Current A/B rates are the controlled evidence for this helper change; no ratio against a historical rate is used to screen it. Prior cross-backend numerical-law limits, and lack of confidence/hardware-aware scheduling evidence, remain.

- [Aggregate, protocol and provenance](aggregate.json)
- [All 20 scalar samples](scalar-samples.jsonl)
- [Candidate and protocol](../../docs/categorical-fixed-shape.md)
- [Experiment notebook](../../docs/lab-notebook.md)

Raw tokens/RNG/probabilities, tensors, machine paths and process identities remain in ignored private evidence.

## Next decision (not executed)

Close the sampler candidate branch. The next bounded task is read-only CPU reanalysis of the existing natural quality32 trajectories at steps 128/512/1280, after verifying their original private round artifacts. Examine first-position selected p/q acceptance risk by prompt and fixed output-progress groups. These natural trajectories are a different population/backend from this synthetic benchmark. Missing raw evidence must remain an explicit gap; do not regenerate trajectories, extend training or access the locked final test to fill it.
