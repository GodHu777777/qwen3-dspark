# Packed proposal and multi-request verify/commit: CPU reference

`dspark_qwen/packed_sampling.py` connects the packed draft backbone, batched Markov
proposal heads, one packed target verification forward, and committed-feature
projection. It is a synchronous correctness reference. CPU fixtures passed;
actual trained-draft noncausal GPU and integrated native decoder execution remain
unverified. The earlier whole-Qwen target RMS gate remains failed.

## Request and cache lifecycle

Construct a `PackedSpeculativeSession(target, packed_draft, temperature=1.,
eos_ids=..., amp=...)` with empty compatible target/draft request pools. The target
may be the explicit native `VarlenPackedTarget`; CPU tests inject a causal target
oracle and the separate noncausal draft oracle. GPU session construction fails unless both target and draft kernels identify
the explicit pinned backends; the known-failing historical public target default
cannot silently enter this session. No runtime fallback is introduced. A formal
GPU runner must separately bind source/runtime/target/checkpoint identities.

`admit({name: RequestSpec(input_ids, max_new_tokens, rng)})` admits new requests
with nonempty prompts, positive output budgets and caller-owned per-request RNGs.
It performs one packed prefill, selects each request's final hidden row and runs
**one R-row target LM-head projection** for initial sampling. It does not project
whole prompts into vocabulary logits. All committed prompt features are projected
into draft KV in one batch. Finished/EOS requests remain resident until explicitly
removed. Reusing a removed name receives a new monotonically increasing session
incarnation and new target/draft markers.

At every completed boundary, both caches contain the committed prefix excluding
the newest emitted token (the next anchor). `remove(name)` releases one request.
Inactive requests are not verified and retain byte-identical persistent KV.
Model execution, draw, observer-copy or commit failures invalidate the session,
clear both cache pools and refuse later operations. Random draws are not replayed;
this API does not advertise transaction rollback after a model-side failure.

## Proposal before allocation

`propose(active, proposal_limits=None, mode='shadow')` defaults to a complete
shadow proposal of `block_size` tokens for every declared active request. Each
request's whole bidirectional draft backbone still executes even when its
subsequent allocation is zero. Shadow limits must be positive; a later planner
must explicitly account for history refresh at zero allocation, including a
capacity of one target anchor per request. A smaller declared positive proposal
limit affects serial heads/draws, not the whole-block backbone width.

The separate `mode='fixed_budget'` baseline freezes lengths before any proposal
draws and caps them by the remaining output budget. Only this explicit mode
permits zero limits to skip a request's draft backbone. `step(allocations)` is a
convenience wrapper for this fixed-budget baseline. It is not the future
current-confidence allocator.

A proposal batch executes one flattened backbone, one flattened base LM-head
projection, and one Markov embedding/projection plus confidence head per proposal
position across all requests still needing that position. Python request loops
sample from per-request RNGs; they never invoke a request's model or head.
Raw confidence is computed before the current token is sampled. Actual sampled
q rows are retained in the unchanged `float64_softmax_normalize_cdf_v1` law.

Every `IssuedProposal` carries `request`, `incarnation`, session mutation `epoch`,
`cache_length`, `nonce`, `proposal_limit`, `mode` and a `TensorProposal` observation
copy. This session epoch advances on successful admission, verification commit or
removal; it is distinct from a planner's consecutive round index and must not be
used as the planner's t-2 clock. The session privately retains the original
proposal tensors. Modifying the public observation copy cannot replace actual q,
tokens or original confidence used by verification/policy validation.

## One target verify, legal prefix, one draft commit

`verify_commit(handles, allocations, allocation_policy=...)` validates exact
issued-object identity, current epoch/incarnation/cache length and one-time use.
Stale, reused, superseded, re-added or foreign capabilities are refused before a
target forward. Each allocation must be an integer prefix length bounded by the
issued proposal limit and remaining output budget. Verification slices the
**internally retained actual q** without resampling or renormalizing its rows.

All selected requests, including allocation zero, enter one packed target append
of `[old anchor, selected proposal prefix]` and one packed target logit projection.
The existing stochastic acceptance/residual/bonus/EOS/budget implementation then
runs per request. After each request's target cache is cropped to its committed
prefix, all committed target features enter one `append_committed` projection.
Only old anchor and newly committed predecessors enter caches; rejected tails and
the newest output token are excluded.

The allocation contract must be explicit:

- `fixed_before_proposal` requires fixed-budget handles and exactly their declared
  lengths; it cannot disguise post-draw allocation changes as fixed admission.
- `external_nonanticipating` is a caller responsibility declaration, **not** proof
  that arbitrary planner code is valid.
- A typed object may implement `validate_allocation(context, handles, allocations)`
  and return a proof-record dictionary. A rejection occurs before target work.

Typed validation receives a snapshot of session epoch, unfinished request roster
with incarnation/cache length/remaining budget, exact seven-field proposal
identities, and immutable CPU tuples of **private original** confidence logits.
The public observation copy is not a source of score provenance. The explicit
host-copy value count is included in work accounting. A planner adapter can bind
frozen calibration, historical capacity and current score/allocation evidence to
this snapshot. The session records its proof separately and always leaves
`planner_causality_proven=False`; structural capability checks alone are not a
causality theorem. Historical absolute K, actual verification rows, physical
buckets, churn/replanning and a consecutive t-2 epoch are planner responsibilities.

## Work and validation scope

Reports preserve proposal/backbone/copy work for every contributing proposal
batch, including shadow work later allocated zero; full-block base LM-head rows,
serial Markov head rows, target query rows and committed-context projection work
are distinct. A verification of only part of a previously generated batch still
records that batch's already-spent work. These counters are not a time or speedup
measurement. GPU synchronization, probability checks, capability observation
copies, float64 full-vocabulary q retention and per-request target crop copies
remain substantial. There is no asynchronous overlap, graph replay, allocator
optimization or equal-work serving benchmark in this implementation.

CPU validation uses actual tiny target/draft models. It checks multi-request
rollouts against independent single-request cached sampling and reconstructs
actual per-layer target/draft KV from each committed prefix after every round.
Separate tests compare batched actual q and pre-token confidence to the unchanged
single-request Markov implementation, inspect module/head call counts, verify
R-row prefill projection, exercise shadow zero allocations, capability rejection,
private confidence provenance, typed policy refusal, request churn, inactive KV
isolation and fail-closed projection errors. BF16 autocast fixtures retain FP32
trainables/RoPE and mixed projected K/V representation. Controlled acceptance
fixtures exercise every rejection position, full acceptance and EOS commit
branches while still running real batched models; they test integration cache
handling, not a new independent proof of the existing acceptance law.
