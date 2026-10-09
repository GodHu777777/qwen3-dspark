# R2/C256 three-arm complete-session diagnostic

`scripts/benchmark_paired_r2.py` reuses the R1 paired runner through explicit
optional interfaces. Original R1 protocol, emitted default schema and observer
entry points remain unchanged. No production sampling/cache/planner code changes.

The sole case is the original shared `r2-c256`: two original 256-token requests,
original independent seeds, 128 output tokens each, temperature1, empty EOS set,
and the unchanged FP64 sampling law. Three arms retain the same loaded target,
trained step1280 draft and two graph pools:

- `target_only`: target-only sampling, no draft execution.
- `fixed_gamma7_full_shadow`: full seven-position shadow per active request,
  then verify at most seven proposals, clipped by remaining output budget.
- `full_shadow_zero_admission`: still generate all seven positions per active
  request, but verify zero proposals through existing FixedPrefix/private-source
  validation. Shadow proposal RNG changes the subsequent output trajectory.

The zero arm measures aggregate overhead of this full-shadow execution path;
its difference from target-only is not isolated drafter latency or a matched
trajectory experiment. There is no measured SPS/Profile, current-confidence
admission, t−2 capacity policy, calibration change or overlap claim.

This E2E decision precedes another cost table: target-only speed1/V omits
full-shadow costD and host costH, while this synchronous path delivers expected
outputs at E/(D+V+H). Even a commonD cannot be dropped from this ratio when
choosing B. The old fixed-C128 profile excludes planner/history and cannot
represent R2 growing, unequal contexts or R1 tails.

## Finite execution plan and setup

Declare 23 verification families once: R2/B2…16 and R1/B1…8. Only R2/B2 and
R2/B16 have graphs, with actual Q=(1,1) and (8,8). Other families, including every
R1 finish tail, execute explicitly eager inside the timer. Actual Q rows are
never padded. Shared family metadata/gather storage is charged once; each graph
has independent I/O and a private pool. Graphs are never captured on a miss.

Two additional exact workspaces support ordinary admission Q=(256,256), C=(0,0),
and setup-only growth Q=(112,119), C=(256,256). Slots=2, context capacity384,
scratch query capacity512, max buckets25. Verification families use maxK383,
Kcapacity383×R and maxQ=min(8,B−R+1). Setup first prefills original prompts,
captures each hot family at C=(256,256), and validates native eager against
replay. An eager transaction commits actual growth to C=(368,375); both graphs
are validated again there before any timed sample. This includes B16 with a
request reaching K383. Selected raw layers, final norm, logits and every layer's
scratch K/V are compared pooled and separately per request, with unchanged
atol.02/rtol.02/RMS≤.005. Raw outputs are saved before numerical assertions.
Capture and both aborted verification paths must preserve resident KV. Real GPU
replays must leave model/all-decoder Python counters unchanged. CPU emulation
executes Python and is never labeled real GPU execution.

Memory/time bounds remain two512MiB graph reservations,1280MiB combined
workspace/graph budget,64MiB workspace cap,6GiB allocator cap,8GiB prefree guard,
1800-second supervisor and1790-second cooperative deadline including setup.
Completed captures, failing pool snapshots, completed samples and raw setup
failure artifacts are retained. No automatic retry or changed budget is allowed.

## Schedule and accounting

Each arm has2warmup,5primary,2diagnostic batches:27 total. Enumerate the nine
case replicates in phase order, with global ordinal0…8, then cyclically rotate
canonical arm order left by ordinal mod3. Across all nine replicates each arm
occupies each position three times. Five primary repeats cannot be perfectly
balanced: position counts are target-only[2,2,1], fixed-γ7[1,2,2], zero[2,1,2].
This imbalance is declared before measurements and is not adjusted afterward.

The complete timer includes fresh reset/session, admission and first sampled
token, all draft/shadow work, original q/p checks and draws, host work, target
graph/eager submission and waits, commit/projection/release, finished R1 tails
and final synchronization. Input tensor/RNG preparation, loading, setup validation
and capture are separately reported. Scalar serialization follows the timer.
Whole-operation memory includes shared loaded weights/pools, prepared inputs and
inherited resident KV. Diagnostic host-observed latencies are separate from
primary measurements; they are not kernel timings.

Each round retains active requests, actual allocations and physical rows,
execution kind, model counters, full work and sampling decisions. Per-batch
metrics distinguish accepted draft tokens, selected proposals, all shadow
positions and request-round counts. Zero selected proposals has a null
accepted/selected ratio. Completed batch output is256 tokens, and admission
plus committed round outputs must equal that count. Each primary arm rate is
1280 divided by its five complete batch durations; all27 ordered identities
are required for completion. R1 eager coverage is included in these durations.

The existing strong vLLM `r2-c256` reference remains separate. Binding verifies
its source/workload/model identity and recomputes five256-output scalar samples:
246.56672736039107 output tokens/s. Native FP64 execution differs from vLLM;
this is a stack comparison, not target-law equivalence. Prior numerical-law
limitations and R1 slowdown remain unchanged. New device execution follows
reviewed frozen source and coordinated resource checks; preparation and CPU
tests alone provide no pretrained native performance result.
