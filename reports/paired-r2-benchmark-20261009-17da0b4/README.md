# R2/C256 complete-session three-arm result

Immutable source: `17da0b4cd7829d94d80720bc215d22583285e827`. One bounded AMD run completed all **27 batches**, followed by independent scalar and raw setup audits. No speculative speedup was observed.

| Arm | Output tokens/s | Relative to native target-only | Relative to vLLM |
| --- | ---: | ---: | ---: |
| Target-only | 80.6661 | 1.0000× | 0.3272× |
| Fixed γ7, full shadow | 34.9868 | 0.4337× | 0.1419× |
| Zero admission, full shadow | 31.5652 | 0.3913× | 0.1280× |

The separate matched vLLM R2/C256 reference remains **246.5667 output tokens/s**. Each reported arm rate is 1,280 outputs divided by the five complete primary batch durations. Same weights were resident for all arms. Full-session timers include reset/admission, all shadow and sampling work, verification, eager tails, commit/projection and final synchronization. Setup and input preparation are reported separately.

Fixed γ7 accepted **220 / 7080 selected proposals** (3.1073%), or 0.415094 draft tokens per global round and 0.209524 per request-round. It generated 7350 shadow positions. Verification used 490 graph rounds and 40 eager rounds, including 10 R1 tails. Zero admission generated 8890 shadow positions while selecting none; accepted/selected is null.

Fixed γ7 delivered 1.1084× zero-admission throughput on this workload, but only 0.4337× target-only. Different proposal RNG changes trajectories: this comparison neither isolates draft latency nor demonstrates confidence-head or scheduler benefit.

## Validation and boundaries

Both hot graphs (R2/B2 and R2/B16) passed initial C256/256 and grown C368/375 checks. Independent CPU reconstruction passed 2203 checks and 756 numerical reductions, with max absolute error 0 and RMS 0, under unchanged .02/.02/.005 limits. Selected raw layers, final norm, logits, all-layer scratch KV, metadata and saved resident bytes were checked. Prefill/growth token arrays were not separately saved; the audit does not independently replay prefix execution.

Worker, controller and SSH exited 0; no timeout or retry. Independent process/KFD/ASR release and all 380 source hashes passed. Before execution, the pinned AMD GPU-hidden CPU suite passed all 34 related tests (26.201 s).

The two graphs retain independent 512 MiB reservations; other declared shapes execute eagerly. The five primary repetitions retain the predeclared arm-position imbalance. This synthetic workload is not a natural held-out acceptance evaluation. Prior cross-backend numerical-law limits and R1 slowdowns remain. No Profile/SPS, confidence admission, hardware-aware scheduling or overlap benefit is established.

## Evidence

- [Aggregate, protocol and provenance](aggregate.json)
- [27 sanitized scalar samples](scalar-samples.jsonl)
- [Frozen experiment protocol](../../docs/paired-r2-benchmark.md)
- [Preserved strong vLLM baseline](../vllm-offline-benchmark-20261009-700bfa6/README.md)
- [Experiment notebook](../../docs/lab-notebook.md)

Raw tokens/tensors, process identities and machine paths remain in ignored local evidence.
