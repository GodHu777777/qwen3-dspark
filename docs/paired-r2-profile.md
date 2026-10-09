# Bounded R2 coarse-stage diagnostic

`scripts/profile_paired_r2.py` is a separate six-batch diagnostic over the original
R2/C256 workload,128 output tokens per request and the two existing arms
`target_only` / `fixed_gamma7_full_shadow`. It reuses the complete R2 factory,
startup raw numerical validation, finite23-family domain, B2/B16 graph pools,
explicit eager tails, original trained weights and FP64/RNG law. No production
sampling/cache/graph code or default R1/R2 protocol/schema is changed.

This follows the measured R2 result: fixed gamma7 reached43.37% of the same-stack
target-only throughput, while target-only reached32.72% of the frozen vLLM rate.
The next decision needs bounded service-cost evidence before another capacity
curve. This diagnostic does not optimize anything or construct SPS/Profile,
change confidence admission, calibration, training or final-test data.

## Fixed schedule and scope

The exact six complete batches are:

1. warmup target-only
2. warmup fixed-gamma7
3. plain target-only
4. coarse-sync target-only
5. coarse-sync fixed-gamma7
6. plain fixed-gamma7

Each arm has one complete plain control and one deliberately synchronized
observation. These are diagnostic samples, not new primary throughput repeats.
The supervisor is300 seconds with a290-second cooperative deadline including
loading and the original complete R2 startup validation. Other limits remain
8GiB prefree,6GiB allocator cap, two512MiB graph reservations,1280MiB combined
budget and64MiB workspace cap. No Kineto, Chrome trace, timeline, profiling retry
or GPU sampler is created. Each observed batch bounds stage and signature scalar
records separately at8192 and complete rounds at254.

## Disjoint outer callable boundaries

The observer temporarily wraps original callables and does not reimplement any
model, probability, random draw or cache operation. Every recorded stage has its
actual callable, admission/round callsite and round index. Only the outermost
wrapped callable receives synchronization boundaries. Nested coarse wrappers
return the original callable directly without a nested fence or extra stage.

- **Full shadow proposal:** the entire speculative `propose` call. It owns its
  nested backbone, heads, Markov work, FP64 laws/draws, validation and copy work.
  Its probability work is not also charged to the probability category.
- **Target service:** strategy prefill/verify, target predict, and the separately
  invoked admission head. Verification includes actual metadata preparation,
  graph prepare/submit/completion or eager forward. Prefill owns its internal
  target commit. Predict owns its nested head, so only admission produces an
  outer head record.
- **FP64 probability service:** outside proposal, actual logits-to-probability,
  categorical and verify-proposal calls. Each owns its nested law validation,
  accept/reject, residual and RNG work. No per-op GPU trace is inferred.
- **Commit/projection/release:** outside prefill, actual strategy commit,
  draft append-committed projection and strategy feature-release calls.

These category names describe observed call sites, not totals for every similar
operation in the system. Private confidence D2H/capability checks before target
verification, reset/session creation, loop/record work, token materialization,
final synchronization and instrumentation gaps stay in the explicit remainder.

For every outer callable, record the wall time of its **pre-boundary drain**,
then its host body and **post-boundary drain**. Pre-drain may wait for previously
unassigned queued work and is reported separately. Body plus post-drain is the
current synchronized service interval, including the small boundary bookkeeping
gap between body return and the post-drain. That gap is retained separately;
service−body−post-drain is not forced to zero. Fence wall time includes waiting and API
overhead; it is not an isolated measurement of synchronization overhead.

The accounting identity is complete batch wall = sum of disjoint pre-drains +
sum of disjoint synchronized services + explicit unassigned remainder. Round
start/end intervals additionally split that remainder into unassigned round and
session portions without adding those portions a second time. The body
and post-drain are components of service, never added a second time. The existing
`_model_signature` checks retain host call counts/spans associated with the owning
stage, without added fences. Their spans are already inside parent costs and
must never be added to the outer totals or used to remove/cache those checks.

Fences change queueing, cache behavior and possible CPU/GPU overlap. These costs
rank work in the observed serialized execution only; they are not kernel-active
times or an unchanged original critical path. Report each observed/plain complete
batch time ratio and delta as the observed batch difference. With only one
plain/coarse pair per arm, this difference includes instrumentation effects and
possible run-to-run variation; it is neither precisely isolated observer overhead
nor a statistical significance result. Do not subtract stage
costs from primary measurements to manufacture a speedup.

## Runtime evidence and invariance

Every batch invokes the unchanged complete R2 batch function. Observer enter and
restoration are outside its timer; wrappers and fences run inside the observed
batch timer. All three phases use the same post-timer collector. It reads actual
output token arrays and final per-request RNG state bytes/digests without drawing,
forwarding or changing state. These remain private evidence, not public reports.

Runtime comparison uses the full semantic dictionaries: raw tokens, RNG bytes,
all per-round decisions/Q/work/execution counters, output hashes, actual shadow
work, prefill work and final lengths. Hashes supplement direct equality rather
than replace it. Both complete control/observed rows are persisted before an
invariance failure aborts the run, regardless of their order. Failed batches
retain partial stage/signature/round records and wrapper restoration status.
After an observation error, wrappers stop recording and transparently allow
original cleanup calls until restoration. No partial batch counts toward the required six. Root independently recomputes
scalar accounting; this runner is not a new numerical-law oracle.

CPU tests execute real tiny-Qwen/draft128-token sessions for all six batches,
compare target and draft KV as well as runtime semantics, exercise R1 eager tails
and signatures, reject RNG-only drift with unchanged output hashes, and verify
restoration/partial evidence on operation failure, record limits and deadlines.
CPU results do not establish native GPU service costs. Any native run uses reviewed
immutable source and the coordinated resource guards; no remote execution is part
of this preparation.
