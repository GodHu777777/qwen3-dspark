# Paired profile stopped at the trace-size limit — 2026-10-09

Execution source: `f0b0268775a51e33fde3f374aef5e29c2fed4ec2`.
**The complete profiling experiment failed.** The single authorized run saved
32 of 36 scheduled samples: all eight warmups, all twenty primary samples, and
all four diagnostic-0 sessions with host spans and GPU stream events. No complete
GPU timeline passed export and correlation validation. No retry, CPU-only fallback,
threshold change or shortened replacement run was performed.

## Exact failure and retained work

The first diagnostic-1 session (R1 C64, speculative arm) finished its full
128-output batch, restored its wrappers and saved `completed-batch.json` and
`spans.json`. Its output/work/decisions independently match the same-path primary
sample. During trace export, the independent monitor observed the profiler's
`trace.json.partial.tmp` at **282,942,823 bytes**, above the predeclared
**268,435,456-byte (256 MiB)** per-trace threshold. It killed only the bound worker
through its PID/start-ticks-verified pidfd. This extra completed batch was not
appended to `samples.jsonl`; the successful sample count remains 32. The other
three diagnostic-1 sessions were not attempted.

The preserved partial file is 298,398,246 bytes: 29,962,790 bytes over the
threshold, reflecting sampled-abort overshoot and additional writes before the
kill took effect. The threshold was neither relaxed nor applied retroactively.
RSS at the abort was 4,236,587,008 bytes; maximum observed RSS was 4,249,055,232,
below the separate 8 GiB threshold. The failed monitor's 117 saved sampling rows
have a maximum recorded gap of 0.493350 seconds. The triggering observation is
preserved separately in `resource-failure.json`.

The actual worker exit is **-9 (SIGKILL)**, controller exit **1**, and SSH exit
**1**; the supervisor did not time out. The monitor recorded
`Diagnostic resource abort: trace_size_limit`. Its own OS exit was not separately
recorded by the killed worker, so the configured monitor exit code is not presented
as an observed OS result. No worker Python error file exists because SIGKILL
prevented its exception handler from running.

A raw `ROCTracer produced duplicate flow start: 4` warning is also preserved.
The trace-size abort preceded correlation validation; the warning alone does not
establish a correlation-gate failure, usable GPU attribution, or a completed
profiler-capability result. The partial trace is retained without treating it as
a validated common timeline.

## Completed primary measurements

These are five complete primary measurements per cell from an otherwise failed
profile run. Primary calls used the unchanged original batch function with no
new observer, profiler, event collector or monitor process. Each pooled rate uses
640 output tokens over those five complete wall times. Diagnostic times are
excluded. They do not convert this run into a successful profiling experiment.

| Case | Native target-only tok/s | Native speculative tok/s | Spec / target | Frozen vLLM tok/s |
| --- | ---: | ---: | ---: | ---: |
| r1-c64 | 45.040683 | 24.316672 | 0.539882 | 131.264056 |
| r1-c256 | 44.604033 | 20.737556 | 0.464926 | 127.339898 |

The trained speculative arm remains slower than its matched target-only control.
The earlier completed [192a357 paired benchmark](../paired-r1-benchmark-20261009-192a357/README.md)
remains the completed end-to-end result; these new repetitions do not replace it.
The [frozen vLLM reference](../vllm-offline-benchmark-20261009-700bfa6/README.md)
is retained, with its different execution/probability stack and no matched
operator trace. Cross-run differences are not a graph-only causal speedup or an
exact attribution of the gap to vLLM.

## Usable diagnostic scope

All four diagnostic-0 sessions completed their full 128-output budget, matched
same-path primary work/output/decisions, restored original callables and completed
the independent monitor's stop/OS0 handshake. Their saved host span trees and
GPU stream-event intervals are available for bounded native-stack analysis.
Inclusive host spans contain child spans; exclusive durations subtract the child
interval union. GPU event intervals can include dispatch gaps and host-induced
idle time; they are not GPU kernel-active time. Nested stage totals, CPU waits
and overlapping GPU intervals must not be added together. Graph Q1/Q8 and explicit
eager Q2–Q7 tails retain separate round identities and counts.

Independent analysis of the **single diagnostic-0 session per cell** found:

| Host diagnostic observation | C64 | C256 |
| --- | ---: | ---: |
| Entire speculative eager-tail rounds | 0.390824 s | 0.570009 s |
| Tail share of speculative diagnostic wall | 6.22% | 7.95% |
| Target-only model-signature calls | 381 | 381 |
| Target-only signature exclusive host duration | 1.052963 s | 0.855957 s |
| Speculative draft-propose inclusive host duration | 3.044420 s | 3.444537 s |

These diagnostic walls were 6.6%–19.3% above their primary means, so their
components cannot be subtracted directly from primary timing. Even removing
entire observed tail rounds would leave speculative diagnostic wall above the
target-only diagnostic. Repeated model-signature validation is a concrete host
cost candidate, warranting a CPU-only cost/safety experiment with an explicit
immutable-model lease; the observation does not authorize simply dropping checks.

The event caution is visible in target-only completion waits: host waits total
about 1.39/1.42 s, while their event intervals are only 0.0178/0.0142 s because the
start event queues behind preceding GPU work. Submit event intervals are about
1.41/1.44 s. Those intervals overlap and cannot be added or interpreted as isolated
kernel durations. The signature event interval can likewise reflect stream gaps.

No kernel/aten attribution or complete CPU/GPU common-timeline conclusion is
claimed from those event diagnostics. The prior whole-Qwen numerical gate,
distribution-equivalence limitations, R1-only scope and missing capacity scheduler
remain unchanged. No raw-tensor audit was repeated for this failed profile run.

## Evidence, integrity and release

`aggregate.json` retains the completed primary wall times/rates/coverage, frozen
references, failure thresholds and observations, four diagnostic-0 statuses,
source/input identities and release/audit summaries. Raw token/text traces and
machine-specific bindings stay private under
`output/paired-profile-gpu-20261009-f0b0268/`.

Root independently checked the exact Git archive, all 360 source files, all 32
sample identities and 128-output counts, and same-path outputs/work/decisions.
All 55 early scalar files match the complete evidence archive byte-for-byte. A
separate scalar check confirms the extra saved diagnostic-1 batch's invariant;
it is still excluded from the completed sample count. Controller post-input
integrity and post-run source hashes pass. Worker post-input integrity was never
written after its SIGKILL and is not implied.

Independent release inspection found the original controller, worker and all
five recorded monitor identities absent. The preserved ASR PID/start ticks remain
unchanged, ready and not busy; KFD is owned only by ASR, with 25,252,777,984 bytes
of free VRAM. All evidence, including the oversized partial, was recovered after
release without another GPU run. The verified archive is 123,145,901 bytes,
SHA-256 `8f5dd4433bafabdf3555be3b80069827af36c899a3e987904041feaaf4d2383d`.
The raw partial SHA-256 is
`c6f78317525859ad9c3f33e6e321bb953fb883a1b0ff2b86040b9508c15b9fcc`.
