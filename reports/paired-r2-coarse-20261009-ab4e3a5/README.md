# R2 bounded coarse-stage diagnostic

Immutable source: `ab4e3a5f818185adb5d413f5da43a3465c609349`. One **six-batch** native AMD run completed under the fixed 300-second supervisor. This is diagnostic evidence, not a new primary throughput benchmark or an optimization.

## Complete-session controls

| Arm | Plain wall (s) | Coarse wall (s) | Observed/plain ratio |
| --- | ---: | ---: | ---: |
| target_only | 3.165032 | 3.216314 | 1.016203× |
| fixed_gamma7_full_shadow | 7.350081 | 7.478367 | 1.017454× |

Both pairs retain exactly equal actual output token arrays, final per-request RNG state bytes and complete work/decision records. One pair per arm cannot isolate instrumentation overhead from run variation. No diagnostic time is subtracted from a primary measurement to infer speedup.

## Disjoint observed service costs

| Category | Target-only (s) | Fixed γ7 (s) | γ7 observed wall share |
| --- | ---: | ---: | ---: |
| full_shadow_propose | 0.000000 | 3.412737 | 45.63% |
| target_service | 2.729951 | 2.590843 | 34.64% |
| fp64_probability_service | 0.386378 | 0.834723 | 11.16% |
| commit_projection_release | 0.029670 | 0.518412 | 6.93% |
| Prior unassigned pre-boundary drain | 0.016910 | 0.015334 | 0.21% |
| Unassigned remainder | 0.053405 | 0.106318 | 1.42% |

Full shadow is the largest observed γ7 category (3.412737 s, 45.635% of its observed wall), followed by target service (2.590843 s). The shadow category includes its nested backbone, heads, FP64 laws, draws and copies. Outside-shadow probability time does not represent all probability work. Target service includes preparation, graph/eager completion, head and nested signature checks. These inclusive call-site boundaries cannot identify which shadow sub-operation is responsible.

Original signature checks remain intact: 381 calls / 0.758828 host seconds for target-only, and 294 / 0.607391 for γ7. Those durations are already inside target service; they are not extra costs to add. Prior safety findings against unsafe signature caching still apply.

Body timings can contain GPU waits. The absence of a kernel timeline prevents conclusions about CPU/GPU utilization or kernel-active bottlenecks. Synchronized service costs guide the next investigation; they do not establish an available speedup.

## Validation and retained limits

Independent scalar auditing checked all 699 rounds, 2087 nonoverlapping stage records, per-round/session remainder accounting, raw semantic equality and all 386 immutable source files. Independent setup tensor auditing passed 2203 checks and 756 pooled/per-request reductions, with max absolute error/RMS zero under the original .02/.02/.005 limits. Prefill/growth token arrays were not independently saved or replayed by this audit.

The pinned AMD GPU-hidden CPU suite passed all 40 related tests (34.599 s). Worker/controller/SSH exited 0, independent release passed and ASR remained ready. No timeout, retry, Kineto or Chrome trace occurred. Original graph/workspace/allocator limits remained unchanged.

The previous five-repeat primary R2 rates remain target-only 80.6661, fixed γ7 34.9868 and zero admission 31.5652 output tokens/s; the separate vLLM reference remains 246.5667. No new speedup, natural held-out quality, confidence-admission or hardware-aware scheduling claim is established.

## Evidence

- [Aggregate, protocol and provenance](aggregate.json)
- [Six sanitized scalar samples](scalar-samples.jsonl)
- [Disjoint coarse stage records](phase-records.jsonl)
- [Protocol and observer boundaries](../../docs/paired-r2-profile.md)
- [Preserved R2 primary comparison](../paired-r2-benchmark-20261009-17da0b4/README.md)
- [Experiment notebook](../../docs/lab-notebook.md)

Raw token/RNG states, tensors, process identities and machine paths remain in ignored evidence.
