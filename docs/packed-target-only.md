# Batched target-only session control

`packed_target_sampling.PackedTargetOnlySession` is a multi-request target-only
control for comparing the same target execution backend with speculative
sampling. It owns no draft model or draft cache, and performs no proposal,
confidence, shadow, acceptance or residual-distribution work. This is a CPU
correctness implementation, not a throughput result or replacement for the
frozen strong vLLM baseline.

The constructor requires an explicit target strategy:

```python
from dspark_qwen.packed_sampling import RequestSpec
from dspark_qwen.packed_target_sampling import PackedTargetOnlySession
from dspark_qwen.target_strategy import FiniteTargetBuckets, PersistentTargetStrategy

# target is an empty PersistentQwenTarget with explicit CPU or pinned native
# backend. Both lists contain caller-declared finite Bucket objects.
strategy = PersistentTargetStrategy(target, FiniteTargetBuckets(
    prefill=prefill_buckets, verification=anchor_only_buckets,
))
session = PackedTargetOnlySession(target, target_strategy=strategy,
                                 temperature=temperature, eos_ids=eos_ids)
session.admit({request: RequestSpec(prompt_ids, output_budget, request_rng)})
result = session.step(active_request_names)
```

`RequestSpec`, `logits_to_probabilities`, `sample_categorical`, stop validation
and temperature validation are the existing shared APIs. Target model forward
precision stays unchanged; categorical probabilities use the original explicit
float64 rule. Each request has its own supplied RNG. The independent single
request oracle remains `cached_sampling.cached_target_sample`; it is not changed
or wrapped with another probability implementation.

Admission validates finite bucket availability before adding residents or
drawing tokens. One target prefill processes all prompts. Only each request's
final hidden row reaches the original LM head, in one R-row call. The first token
counts toward the positive output budget and can immediately finish on EOS or a
one-token budget. Features are released after that head and all first-token draws.

`step(active)` requires unique unfinished request names, in caller order. Every
request contributes exactly one query: its newest emitted anchor. The strategy
preflights an exact ordered `(1, ..., 1)` bucket at the current context lengths,
then performs one batched target verify and one R-row head projection. Each
request consumes exactly one categorical draw. All decisions precede the one
target commit, whose count is one anchor row per active request. Features are
then released and request outputs/stop state are published. The cache is always
`prompt + output[:-1]`, even after EOS or budget completion. Inactive residents
keep their KV unchanged. Finished requests can be removed and re-admitted under
fresh incarnations; unfinished requests may also be removed explicitly.

Missing buckets are clear pre-execution errors with no model forward or RNG
draw. Unknown, duplicated or finished active requests are rejected. Execution
errors permanently invalidate the session, with strategy-owned scratch abort,
feature release and reset attempted; cleanup failures remain available as
`cleanup_error`. No RNG or partially completed device-write rollback is claimed.

The same interface accepts eager execution or an already registered and captured
graph target. Graph registration/capture is explicit setup owned by the caller;
this module neither captures on a request-path miss nor changes the graph family.
The target's owner-stream and feature-lifetime contract still applies. This
control has no draft consumer, so release follows the head/sampling/commit work.
GPU entry requires the same explicit pinned target backend marker as the
speculative session; choosing a backend is not evidence of native correctness.

Returned work includes actual target physical/logical Q/K and capacity costs,
one head call, head rows, categorical draw count and explicit zero draft/shadow
work. `target_probs` returns actual per-request FP64 laws for numerical auditing.
Admission work is available through `last_prefill_work`. All request preparation,
head/probability work, decisions, commit and release belong inside any later
whole-call timing. This module does not add a benchmark or report speedup.

For paired E2E measurement, use matching target weights/backend, forward dtype,
prompt/output-budget/EOS policy and declared setup costs. Record selected feature
layer IDs: a target configured to capture raw selected layers still performs
those copies even though this session does not project them through a draft.
Use matching selected-layer configuration to isolate speculative/draft overhead,
or explicitly report a different optimized target-only configuration. Equal
seeds do not require target-only and speculative token identity because their
draw consumption differs. Preserve each algorithm's original sampling law and
measure emitted output tokens and whole request/round cost.

The CPU tests use actual four-layer tiny-Qwen models and compare emitted tokens,
every sampled probability row, complete RNG traces and every layer's committed
KV against independent target-only/reference-prefix forwards. They exercise
unequal prompt lengths and output budgets, partial active sets, EOS at admission
and decode, lifecycle, finite-bucket refusal, R-row admission projection,
single-query decode batching and permanent invalidation after failures. They do
not establish full-model native numerical equivalence, graph capture support,
performance, calibrated capacity scheduling or CPU/GPU overlap.
