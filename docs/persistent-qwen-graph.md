# Prepared full-target native eager and graph execution

`persistent_qwen_target.py` now exposes an opt-in pinned native eager path and
an explicit full-model graph lifecycle. `persistent_qwen_graph.py` supplies the
CUDA/ROCm graph backend and a separately named CPU replay emulator. This is
implementation and CPU preparation, not a completed native full-target GPU test
or speed result. The original target adapters, CPU attention oracle and sampling
law remain intact.

## Actual captured computation

The graph body calls the original owned `model.model`, with persistent token IDs,
explicit request-local positions and a fixed mask mapping. It therefore includes
embedding, every original decoder layer's QKV/QK normalization/RoPE, scratch KV
copies, capacity gathers, pinned attention, output projections, residuals, MLPs
and final norm. Selected-layer hooks emit copies of raw decoder block outputs
into fixed context slices. A separate buffer stores the final-normalized hidden
state. Raw selected final-block output is never substituted with final norm.

A capture-specific HF Cache emits deterministic tensor writes/gathers directly.
Its Python layer/feature sets check program coverage during capture; those sets
are not used as evidence of work on a subsequent real replay. The ordinary eager
Cache continues to use the existing checked layer-by-layer store API.

The original LM head remains outside this graph and uses the actual hidden rows.
Session admission retains its R-final-hidden-row projection instead of projecting
all prompt rows. Proposal sampling, q/p calculations, confidence validation,
acceptance, metadata preparation, transaction checks and resident commit are
outside the graph. This implementation introduces no asynchronous scheduler.

## Explicit choices and finite registration

CPU callers still pass `test_kernel=test_only_persistent_sdpa`. GPU callers must
explicitly select `native_backend=rocm_varlen.BACKEND` on a BF16 GPU model with
Hq16/Hkv8/D128 and Transformers 5.17.0. The real `PinnedRocmVarlenKernel` validates
the pinned device/runtime/schema and remains available as `_varlen_kernel` for
the existing session guard. Only this separate capacity path calls its private
operator; the old wrapper's exact-K contract is not relaxed.

Graph execution additionally requires `graph_backend=TorchGraphBackend()`,
positive `max_graph_buckets` and `graph_byte_budget`. Tests instead explicitly
select `CPUReplayEmulator()` on CPU; it executes Python on each replay and has no
GPU capture claim. There is no automatic backend or eager fallback.

All buckets are registered explicitly. `max_buckets` and
`workspace_byte_budget` bound the total workspace registry. Each graph registration
also declares `reserve_bytes` for graph-owned retained allocations. The combined
`graph_byte_budget` charges **all registered workspaces + fixed graph input/output
buffers + all declared graph reservations**, and graph count is separately bounded.
The CUDA backend synchronizes capture setup and reads the allocator snapshot
scoped to `graph.pool()` with `include_traces=False`. It sums segment `total_size`,
including inactive blocks retained by that private pool, and rejects totals above
the bucket reservation. Unsupported/malformed snapshots, foreign devices and
shared pools fail closed; there is no live-tensor allocated-delta fallback.
Global reserved bytes before/after/delta are reported separately and need not
equal private-pool bytes. A budget failure preserves measured/reservation bytes
in its message and attaches `memory_accounting` plus the original `pool_snapshot`
to the exception for the bounded runner's durable failure report. This is not a cap on
transient capture peak or the entire process: a separately bounded GPU runner
must still enforce its process allocation cap. Static model/resident/scratch
allocations are not mislabeled as graph workspace. No LRU recapture, lazy bucket
creation or capture-on-miss occurs in the request path.

```python
# The model/runtime/device and resource limits must be selected by the caller.
target.register_bucket(prefill_bucket)
target.register_graph_bucket(verify_bucket, reserve_bytes=declared_graph_bytes)
prompt_features = target.prefill(prompts, bucket=prefill_bucket)  # eager
# Consume admission head rows and selected features here.
target.release_features(prompt_features)
target.capture_graph(example_chunks, bucket=verify_bucket)      # explicit setup
```

Capture uses current committed prefixes and real query inputs but changes only
scratch and output buffers; it aborts its setup transaction without committing.
Two side-stream warmups precede one actual GPU graph capture. Setup waits for the
capture stream. The first useful request replay is a distinct submission with a
fresh transaction and generation. Model/config/layer selection, parameter and
buffer versions/addresses, and persistent allocation pointers are checked before
reuse. Unknown or changed shapes and signatures are rejected.

## Prepare, submit, finish, consume

1. `prepare_graph_verify(chunks, bucket=...)` validates exact ordered Q, resident
   incarnations, context bounds and idle output ownership. It calls `pool.begin`
   outside capture to copy cumulative lengths, positions, indices and masks, then
   copies query token IDs into fixed input storage. It returns a fresh exact-object
   ticket bound to the transaction and executor.
2. `submit_graph(ticket)` replays that registered graph once and records a
   backend-owned completion event bound to the **exact receipt object**. The
   captured program must match that receipt's exact registered writer. The
   receipt binds transaction, incarnations and generation; equal integer
   generations from different pools confer no ownership. Duplicate/copied/foreign
   submissions are rejected, while sharing a backend across targets is legal.
3. `finish_graph(completion, wait=True)` waits for and checks the owned event,
   validates current identities/addresses, publishes complete scratch readiness
   through the pool capability, and returns a new `PackedFeatures` object. With
   `wait=False`, an unfinished event leaves the submission pending and raises;
   no layers become ready. Missing or failing event operations do not publish
   features. `verify` composes these three steps synchronously for graph buckets.
   `verify_eager` provides the same-backend eager control explicitly.
4. The caller performs unchanged head/probability/acceptance operations, then
   `commit(features, prefix_counts)` once all decisions are known, and consumes
   selected features in the draft projection. Graph output storage remains leased
   after target commit. `release_features(features)` ends that lease only after
   the last consumer has queued its reads on the owner stream.

The target records one owner CUDA stream. Prepare, replay, finish, head, commit,
abort and release reject a different current stream. **All external head/draft
feature consumers must also run on that same owner stream.** Python consumer
return is not claimed to mean GPU completion: same-stream ordering places later
output overwrites after the consumer's queued reads. Switching consumers to a
separate stream without explicit event ownership is unsupported. No graph output
may be retained or read after release; later replay intentionally overwrites its
fixed buffers. Target APIs reject released/copied/foreign features.

A pending lease blocks new verification, reset and request-slot changes even if
commit has completed. Eager features have independent allocations, so release
checks ownership without imposing that graph reuse restriction. Session strategy
releases after original admission head + draft prefill, or target commit + draft
append_committed. This is a single-pending/single-stream baseline, not overlap.

## Pool external-write contract

The pool registers an exact writer capability with its bucket, the complete
ordered layer signature, fixed scratch addresses and a trusted backend completion
checker. A prepared receipt binds that writer to the exact current transaction,
slot incarnations and a fresh pool-owned generation. Eager and external writes
cannot mix in one transaction.

Only a submitted receipt whose backend-owned event confirms that exact receipt can
publish the complete layer-ready set. Copied, stale, foreign, unsubmitted,
unfinished or pointer-drift receipts fail. An unfinished external write cannot
be aborted and reused. This contract trusts the registered fixed write program;
it is not an independent proof that every intended GPU kernel wrote every layer.
Actual full-target numerical/KV validation remains necessary.

`cancel_graph(ticket)` cancels an unsubmitted transaction, or a submitted one only
once the pool observes completion. A checker returning false leaves work pending
and the store healthy; a checker exception poisons both store and target even
if the checker later recovers. After features are published, use
`abort(features)` and then release. Capture/replay/event uncertainty calls
`invalidate`, poisoning the pool and executor instead of silently retrying eager.
Invalidation allows a held feature lease to be released but does not permit reset
or buffer reuse. A failed resident commit also remains a poisoned store, not an
atomic rollback guarantee. Source/device failure recovery requires reconstruction.

## CPU evidence and next real-device boundary

Thirteen graph-specific tests use real four-layer tiny Qwen models. They compare
all-layer KV, selected raw outputs, final norm and logits with the explicit eager
path through two growing contexts and partial/full commits. They check exact
addresses, independent inactive residents, abort, fresh feature ownership, lease
extension through a real `PackedSpeculativeSession` and `DSparkDraft` projection
for two rounds, stale/copy/foreign receipts, equal-generation cross-pool events,
wrong captured writers, completion not ready versus checker exceptions, capture
failure, inactive private-pool reservation accounting, registry budget rejection,
model/buffer drift and owner-stream checks. The emulator is used only to execute the same host
lifecycle and actual tensor body; it cannot verify that Python does not run on
real GPU replay or establish capture support/performance.

The first combined local run exposed two preparation issues: message-case drift
broke an existing rejection assertion, and routing ordinary eager prefill directly
to `verify_eager` bypassed the public verify boundary used by the session's lease
fixture. The corrected path preserves ordinary public verify, while explicitly
keeping prefill eager for graph buckets. Failure details and subsequent logs are
in `output/persistent-qwen-graph-cpu-20261009`.

Independent review of the first frozen candidate found cross-pool event aliasing,
live-allocation undercount of graph pools, and unpoisoned cancel checker exceptions.
The original review/counterexamples are preserved under
`output/persistent-qwen-graph-review-20261009`; repair tests and source manifests
are separate from the original handoff under
`output/persistent-qwen-graph-cpu-20261009`. A read-only hidden-GPU inspection of
the pinned AMD Torch 2.12 API confirmed support for scoped `memory_snapshot` and
`CUDAGraph.pool`; this inspected source, not a device allocation/capture result.

The prior first-layer native probe passed Q=(1,4), context ceilings=(144,144),
with growing contexts (128,128), (129,131), (130,135). A future full-target protocol
should begin with that finite verification family, actual pretrained weights,
explicit eager prefill, selected layers [1,7,14,21,26], and paired native eager
versus captured full-target outputs/KV. It must retain original numerical limits,
raw selected versus final norm distinction, partial-commit cache semantics and
original failure reports. A separate immutable bounded runner and authorization
must establish those GPU facts; this document does not report an execution.

The exact ordered Q shape still depends on current prefix allocations. It is not
selected solely by t−2 capacity K, and a passing fixed-family graph would not
complete the bounded R/physical-B/max-Q family, capacity scheduler or CPU/GPU
overlap. The observed native end-to-end gap remains unchanged until a new measured
implementation improves it.
