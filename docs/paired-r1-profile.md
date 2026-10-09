# Bounded paired R1 diagnostic profile

`profile_paired_r1.py` observes the frozen complete paired workload using the
original `benchmark_paired_r1.py` factory, batch and algorithms unchanged. This
stage follows the negative end-to-end result at execution source `192a357`.
No performance or GPU profiler support result is implied by CPU tests.

The panel is exactly two cases (R1 C64/C256), two arms, and 2 warmup + 5 primary +
2 diagnostic repeats per cell, with all 128 outputs and the original alternating
pair order. Primary/warmup calls invoke the original batch directly: no new
observer wrappers, profiler, events, resource-monitor process or collector.
Existing original integrity witnesses and supervisor overhead remain. Only a
diagnostic context installs wrappers; every original callable is invoked once
and restored in reverse order even on failure. Any restoration failure aborts.

Diagnostic 0 records a host span hierarchy and GPU stream event intervals.
Diagnostic 1 records a full-session CPU+ROCm GPU timeline (four traces maximum).
Graph Q1/Q8 and explicit eager Q2–Q7 tails retain their original labels/work.
Selected raw layers, FP64 law, original RNG draws, proposal/acceptance/residual,
full-shadow7 even with allocation0, commit/projection and thresholds are unchanged.

Stages include reset/admission, bucket/chunk/layout metadata, graph model signature
and pointers, submit/wait, target/head, logits-to-law, probability validation,
categorical/RNG, accept/reject/residual, KV commit/release, draft backbone,
Markov/confidence heads and context projection. Private confidence D2H and tensor host-materialization
ranges expose the existing copies within their parent function; uninstrumented source-validation work remains in its parent's
exclusive host duration. No full-vocabulary tensors are retained by the observer.

Host inclusive and exclusive durations are reported separately. GPU events are
stream intervals and include dispatch gaps; they are not kernel-active time.
Within-name GPU intervals are unioned. Never add nested stage totals or CPU waits
to overlapping GPU time. The trace summary uses actual GPU event timestamps and
kernel names, with positive JSON-integer CPU/runtime/GPU correlation IDs. Missing/null/bool,
nonpositive, noninteger and unverified sentinel IDs stay unknown; CPU/runtime
PID/TID must be positive integers. Uniqueness is checked across all runtime
records before CPU-chain filtering, so a duplicate ID with an unmatched runtime
cannot manufacture a unique chain. Each correlated activity
retains its IDs, CPU operator/runtime names, and observed round/stage ancestry.
Per-round Q/context/execution-kind accompanies a union of its correlated GPU
activities; unknown activity remains unassigned even when timestamps overlap a
round. Round totals therefore describe correlated coverage, not all GPU work. Missing or ambiguous operator
associations remain unknown. A named Python range does not prove an individual
GPU kernel's semantics. Full-target graph internals can lack aten associations.
At least one real kernel and a usable CPU->runtime->GPU correlation chain inside
the single synchronized complete-session range are required; otherwise profiling
fails explicitly, with no CPU-only substitute.

The unchanged GPU bounds are supervisor1800s/cooperative1790s, 6GiB allocator cap,
8GiB free guard, 512MiB reservation per graph, 1280MiB combined graph/workspace and
64MiB workspace limits. A diagnostic-only independent stdlib process checks the bound worker’s Linux
RSS and trace files every0.25s (independent of the worker’s GIL): RSS threshold8GiB, each trace256MiB, total1GiB,
at most4 traces, at most100,000 scalar spans per diagnostic. These are sampled
abort thresholds with possible overshoot, not virtual-address-space limits or a
promise of zero temporary excess. No RLIMIT_AS is used. On RSS/file/deadline or
monitor inspection failure, it writes the observed values and phase/stage, then
rechecks worker PID/start ticks and sends SIGKILL through its already-bound
Linux pidfd, even if failure logging itself fails. The monitor exits86; the
original parent subreaper owns cleanup and independently checks ASR/KFD release.
Trace bytes already on disk and prior samples remain. A killed profiler may have
no recoverable trace still buffered in memory; the tool never claims otherwise.
The monitor has an explicit readiness handshake and a checked stop/OS0 handshake;
limits are checked before acknowledging stop. Observed sampling gaps above1s
fail the run. Only the bound worker may be killed; the original supervisor reaps
its descendants. Stack, shapes and memory profiling are disabled. No event truncation or retry.

Each diagnostic checks its output hashes, decisions, work, contexts and coverage
against its same-path primary reference outside batch timing. Failure keeps the
completed batch (if available), partial observation/error and previous samples,
but does not produce a completed profiling claim. CPU tests additionally compare
actual per-request RNG state and final target KV before/after instrumentation.
Diagnostic overhead never enters the five-primary pooled throughput.

Default CLI operation only binds source/inputs with stdlib. Binding includes the
original runner, new runner/observer/independent monitor/tests/docs, exact profile protocol and the
explicit new worker entry. Execute uses the new script in its child command;
that child temporarily routes only the isolated original worker's bind/run entry
functions to the new runner. It never dispatches the old main executable.

```sh
python scripts/profile_paired_r1.py --dry-run --model /path/to/model \
  --generation-manifest /path/to/manifest.json --checkpoint /path/to/step-001280
```

A GPU launch requires the separate established immutable-source review and fresh
ASR/process preflight. No launch is performed by this document or CPU validation.
This trace can explain this native stack's measured work; absent a matched vLLM
operator trace, it cannot assign the inter-stack gap by subtracting aggregates.
