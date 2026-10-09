# R2 full-shadow inner-service diagnostic

Immutable source: `db9d61c4decb56090993ba9c1dc8d4063ec72763`. One bounded six-batch run; this diagnoses costs and preserves the earlier primary throughput results.

Predeclared hypothesis classification: **inconclusive**. The hypothesis requires FP64 law plus draw to occupy at least half of the parent propose service in both orders, with both observed/plain wall ratios inside [0.95, 1.05]. Crossing the threshold between orders or exceeding the wall-ratio bound is inconclusive.

| Pair order | Plain s | Observed s | Observed/plain | FP64 law + draw / propose |
| --- | ---: | ---: | ---: | ---: |
| plain → observed | 7.230388 | 7.982689 | 1.104047 | 41.65% |
| observed → plain | 7.211030 | 8.184522 | 1.135000 | 41.17% |

Both wall ratios exceed the preregistered 1.05 ceiling. Although both observed fractions are below 0.50, this result is **inconclusive**, not a clean rejection of probability-service dominance in the unperturbed path.

Actual output token arrays, final RNG bytes and all round work/decisions match exactly across all four non-warmup batches. Two pairs cannot isolate observer cost from run variation; the 5% bound is a diagnostic rule, not a confidence interval.

| Child category | Pair 0 service s | Pair 1 service s |
| --- | ---: | ---: |
| backbone | 1.182033 | 1.197803 |
| base_head | 0.093200 | 0.105775 |
| markov_confidence | 0.415325 | 0.433132 |
| fp64_law | 0.676411 | 0.612055 |
| categorical_draw | 0.945883 | 1.020715 |
| proposal_copy | 0.014502 | 0.014769 |
| Child pre-boundary drains | 0.125523 | 0.122796 |
| Parent residual | 0.442377 | 0.458402 |
| Owning propose total (inclusive) | 3.895254 | 3.965446 |

Each parent envelope equals disjoint child services plus child pre-drains plus residual. Do not add parent and children. These are deliberately synchronized service measurements; body times can include waits. Signature spans remain nested in target service, and none of the three checks was removed or cached.

Independent scalar audit covered 657 rounds, 9720 child records and 393 source files. Raw setup audit passed 2203 checks/756 numerical reductions; maximum absolute error and RMS were 0.0/0.0 under unchanged .02/.02/.005 limits. Prefill/growth prefix execution was not independently replayed.

The pinned AMD GPU-hidden related suite passed all 47 tests (42.099 s; wrapper 44.172 s), including the seven new inner-observer cases. Worker/controller/SSH exited 0, independent release passed, and ASR remained ready. No timeout, retry or trace export occurred.

Earlier primary R2 results remain target-only 80.6661, fixed γ7 34.9868, zero admission 31.5652 output tokens/s; separate-stack vLLM remains 246.5667. No inference speedup has been demonstrated. Phase subtraction cannot establish a realizable gain; any optimization needs its own complete-session A/B.

- [Aggregate and provenance](aggregate.json)
- [Six scalar samples](scalar-samples.jsonl)
- [Outer intervals](outer-stages.jsonl)
- [Inner intervals](inner-stages.jsonl)
- [Parent partition accounting](parent-partitions.jsonl)
- [Protocol](../../docs/shadow-inner-r2-profile.md)
- [Notebook](../../docs/lab-notebook.md)

Raw tokens/RNG/tensors, machine paths and process identities remain private ignored evidence.

## Next decision (not implemented)

Stop further profiler subdivision. Prepare one small categorical-sampling candidate: replace dynamic positive-support `nonzero` index construction with a fixed-shape last-positive reduction. This is a source-level opportunity with a tractable equivalence check, not a bottleneck established by these disturbed timings. Preserve FP64 CDF/searchsorted, RNG order, validation and exceptional behavior, including probability mutation by an RNG callback. Only after CPU equivalence and source review should the candidate enter a bounded observer-free, repeated target-only and gamma7 complete-session A/B. No optimization or new A/B result is part of this report.
