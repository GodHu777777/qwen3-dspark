# Packed execution: measured cost and graph preparation

This is the CPU design following the bounded
[trained packed decoder result](../reports/packed-decoder-gate-20261009/README.md).
No profiling run, new timing result, graph capture or further GPU authorization
is implied. Native target numerical differences remain known; performance work
must retain that limitation. The synchronous CPU implementation remains the
semantic reference while measured costs determine the first optimization.

## Measurement questions and units

The current planner maximizes `expected_progress * Profile.sps[B-1]`.
Consequently an empirical `sps` for that objective must be **completed rounds per
second**, `1 / T_round`, with all draft and verification costs included. Substituting
`B / T_round` or output tokens/second would multiply progress twice. Report
target-only verification service time separately; it is not a complete-round
capacity curve. Likewise vLLM output tokens/second is an end-to-end baseline,
not a speculative-verification SPS curve.

Two measurements answer different questions:

1. Frozen-cache local profile: compare actual execution cost across declared
   verification allocations at the same input state. It supplies shape/cost
   evidence, with snapshot/reset costs separately visible.
2. Fixed-output trajectories: execute complete real requests, including admission,
   full shadow work, sampling, rejected work, commit/crop, host scheduling and
   completion. Compare equal prompt/output workloads against vLLM and report
   measured goodput and latency. Local profile rates are not this throughput.

## Domain before curve

`async_capacity.Profile` currently stores only per-request context interval,
discrete B, physical buckets and an SPS array. It has no resident-request count,
total-KV or allocation-layout guarantee. Its current guards must not be described
as permitting measured curves to transfer between those domains.

Every profile observation must bind source/checkpoint/runtime/backend, execution
mode (eager/graph), probability policy, shadow/fixed mode and at least:

- Active R and resident R separately, with canonical request-slot order.
- Exact context vector C, total active/resident context and request capacity.
- Exact proposal-prefix vector ell; `q_i = 1 + ell_i`, logical `B = sum(q_i)`.
- Physical query rows, per-request key lengths, attention pair domain,
  total resident KV bytes and gathered/copied KV bytes.
- Full shadow width, per-position head shapes, graph bucket/capacity if any,
  realized accepted/committed lengths and resulting projection/crop work.

Equal R, B and even total C need not imply equal cost. Varlen attention includes
`sum_i q_i * (C_i + q_i)`, and the current cache/gather path also copies inactive
resident context. Equal per-request contexts with balanced versus concentrated
ell isolate allocation-shape effects; permuting long prefixes onto unequal
contexts exposes context/allocation interaction. Keep these comparisons paired.

The first formal local profile fixes R=2, resident R=2, C=(128,128), gamma7,
no churn and eager full-shadow execution. It predeclares all64 vectors in
`{0,...,7}²`, covering every integer B from2 through16, intermediate layouts and
both concentration directions. Every cell remains in the output table even if
the300-second bound leaves it unmeasured. Balanced and left-concentrated vectors
form a26-cell diagnostic subset for presentation only; they do not reduce the
formal measurement domain or justify extrapolation to unmeasured allocations.

Report per-cell distributions and paired latency ratios/differences, not just a
single fitted line. Do not decide a transfer tolerance after seeing favorable
measurements. If B-only cost is inadequate, retain ell/layout in the lookup, or
use an explicitly conservative per-B cost over a completely measured finite
domain. A maximum over two tested layouts is not a bound on untested layouts.
Do not add interpolated B values. A missing cell is missing evidence.

Broader measurements use separate domains for R∈{1,2,4} and equal per-request
contexts C∈{64,256}; unequal-context and inactive-resident cases follow explicitly.
Changing R or crossing the bound context/layout domain invalidates lookup. Churn
requires a compatible measured domain or an explicit cold start/reprofile; a
fresh planner alone does not supply missing measurements. Keep this admission
wrapper separate from the existing toy `Profile` until the measured schema and
residuals justify the interface change.

## Timed boundaries and experimental controls

Load the immutable target/draft once per authorized window. Use synthetic
prompts, actual trained weights and the proved execution path. Reconstruct exact
native cache/RNG/request state before a local-profile cell; bind the snapshot
identity and verify restoration outside timing. Record preparation/restoration
cost and peak memory separately. No CPU/GPU tensor dump, profiler, independent
model replay or per-layer numerical oracle runs inside throughput samples.

For the first local diagnostic, propose two discarded warmup rounds and five
measured repetitions per cell. Every repetition sweeps all64 cells; rotate the
canonical order left by17 times the repetition index and reverse odd repetitions.
Warmup, primary and diagnostic phases remain separate. This fixed ordering
reduces confounding of B with elapsed run time without claiming to eliminate drift. Save every sample, including outliers and incomplete cells. A300-second
worker ceiling and6 GiB allocation cap bound the proposed first profile; reaching
the limit yields a partial table, never invented cells or an automatic retry.
Use the tested identity-aware external supervisor, live ASR/device ownership
guards and a fresh separately authorized window.

Use two instrumentation passes over the same frozen cases:

- Primary service-time pass: synchronize immediately before/after the complete
  round and measure wall time. Count all validation, metadata, layout, probability
  operations, host copies, allocation, KV changes and history maintenance that
  remain in the real path. Do not hide a removed expensive check by leaving it
  outside timing while still depending on it for correctness.
- Diagnostic pass: record GPU events around coarse draft backbone/base head,
  Markov/confidence/probability work, target verification/head and committed-KV
  projection/copy, plus host regions and allocated/reserved peaks. Explicit stage
  synchronizations and event instrumentation can change execution; report their
  round total separately rather than substituting it for the primary pass.

`CapacityRoundDriver.step` already charges synchronized coarse stages but adds
stage synchronizations. Its current measurement is the cost of that synchronous
implementation. Use it as the starting reference, then distinguish any lower-
synchronization timing path explicitly. Report unaccounted wall-time remainder;
never add overlapping GPU event durations and host time as if disjoint.

The allocation sweep must use exact predeclared ell, not let fresh confidence
draws choose which B gets measured. Charge the actual host allocation-validation
path used by that sweep and label it. Separately time the real planner/driver
trace; a frozen-allocation model microbenchmark cannot silently stand in for
full planner cost. Save realized commit lengths because projected-KV/crop work
depends on acceptance. Do not force acceptance to make measurements favorable.

## Comparable fixed-output workload

Coordinate one shared synthetic workload manifest with the vLLM baseline owner:
R∈{1,2,4}, prompt lengths64/256, output budget128, fixed per-request seeds,
temperature1, ignore EOS, exact token IDs and no additional chat templating.
Hash tokenizer/model/source and bind whether lengths include every supplied
token. The manifest owner is the baseline agent; both runners consume identical
requests. These are synthetic throughput cases, not quality examples.

Report prefill separately and complete admission-to-final-output wall time;
use equal output budgets and count actual output tokens. Engine/model construction
and graph capture are separately reported setup costs. Warmups/repeats reuse one
engine/model load and require fresh request state. Native speculative trajectories
and vLLM need not generate equal tokens from equal seeds; acceptance and resulting
context growth are reported, not suppressed. Ordinary vLLM decode generally has
one query per active request, so its observed scheduler B and actual batch shapes
must remain distinct from speculative `R + sum(ell)`.

This offline fixed-batch comparison precedes any arrival-load frontier or
production-serving claim. Synthetic prompt lengths alone cannot establish a
natural-workload performance distribution.

## Optimization selected by observations

Current code provides concrete candidates, not a conclusion about dominance:

| Current mechanism | Measured cost to inspect | Candidate change |
| --- | --- | --- |
| Context/block `torch.cat`, gather, repeated cache crop | Copy bytes, allocations, KV stage time | Preallocated target/draft storage and indexed writes |
| `PackedLayout.build`: `unique`, `nonzero`, `tolist` | Host stalls, layout time, repeated metadata kernels | Maintain host-owned slot/length metadata; fill stable device buffers |
| Per-position/request `bool`, `item`, probability validation and CDF draws | Synchronization count and probability/host time | Batched device decisions with one deliberate host boundary |
| Full-vocabulary FP64 q and observation copies | Memory peak, bandwidth and clone cost | Preserve actual-q ownership with bounded buffers or immutable views |
| Repeated small model/head launches | Event/launch timeline, complete-round time | Capture fixed-domain backbone/verification subgraphs |

Choose the highest measured whole-round contributor first. Optimization must
preserve actual-q retention, pre-token confidence, request RNG ownership, commit
prefixes, cache isolation and fail-closed semantics. Changing probability dtype
or deleting checks is a semantic change and needs its own evidence; it is not a
free graph-enabling refactor. Keep the CPU synchronous oracle and small content
checks for each changed boundary, without rerunning the previous large gate as
a substitute for measuring the performance change.

## Graph-friendly execution boundary

The current path is not graph-ready: dynamic allocation, Python request loops,
host-extracted lengths/decisions and changing storage addresses prevent simply
wrapping `session.step` in capture. A staged design should:

1. Reserve separate target and mixed-dtype draft KV storage per slot/capacity;
   make gather/index/cumulative-length/input/output buffers persistent. Preserve
   FP32 draft K and BF16 V representations instead of globally casting caches.
2. Maintain explicit logical lengths and incarnation/slot ownership. Commit
   writes must be transactional at the externally visible boundary; scratch
   verification tails cannot become visible to draft context or other requests.
3. Capture deterministic model subgraphs first, with fixed buffer addresses and
   declared R/query/KV capacities. Keep scheduler, RNG and accept/reject outside
   capture initially and charge their synchronization/host cost.
4. Prove the private ROCm operator accepts the intended capture/replay contract
   for those static buffers before claiming graph support. Capture failure stays
   a recorded unsupported bucket; no unreported fallback.
5. Compare eager and replay on the same supported domain, reporting logical and
   physical query/KV work. Padding or dummy rows must preserve visibility and
   positions and be charged. Capturing a bucket must not replace the paper's
   variable verification budget by an unreported fixed computation.

Only after measured subgraph benefit should the design consider device-side
sampling/acceptance or overlap. Each later optimization requires full-round
wall-time evidence; an isolated fast kernel or successful capture is insufficient.

## CPU-prepared runner

`scripts/profile_packed_decoder.py` implements the full64 local experiment. Its
stdlib default binds the actual target/step1280 checkpoint, shared workload file,
complete local protocol and source/supervisor hashes. The local snapshot uses the
first128 tokens and original seeds of the shared `r2-c256` requests. Admission
commits128 target/draft context tokens and emits a separate latest anchor; this
is not an end-to-end request with the256-token shared prompt.

The runner preserves exact cache tensors, markers, request metadata and per-request
RNG state, with restore and exact-content checks outside each timed sample. The
primary pass retains two warmups and five samples per cell; a separate diagnostic
pass retains one sample per cell. GPU stream event intervals and host spans may
be nested and must not be added as disjoint kernel costs. Every sample records
actual acceptance/commit lengths, all existing logical/physical work and its
restoration cost; GPU samples additionally record allocated/reserved peaks.

The measured round includes full shadow proposal, private confidence host copies,
capability validation, target verification, q/p validation and cache commit/crop.
It does **not** execute the real capacity search/calibration/history driver, and
explicitly labels that omission. Its rate cannot yet be substituted into the
planner as a complete scheduler-inclusive cost. Completed exact cells alone
support `lookup`; domain mismatch, missing or partial cells raise. Lookup rejects
any requested use as a `CapacityRoundDriver` profile, and both report and returned
statistics explicitly set `capacity_round_driver_profile_eligible=false`.

Formal execution reuses `guard_vllm_smoke.supervise` and `ASRGuard`, including
process start-time identity, subreaper cleanup and ASR/KFD checks. A300-second
supervisor deadline and290-second cooperative worker boundary preserve partial
data without shrinking the declared matrix or retrying. This preparation has
no GPU timing result.

Snapshot/restore is private to this single-owner profiling runner. Restoring an
outstanding capability or failed session is rejected. Numeric epochs/nonces may
repeat after restore, but old proposal handles still fail exact object-identity
validation; CPU regression tests exercise that boundary.
