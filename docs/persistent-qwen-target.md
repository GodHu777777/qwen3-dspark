# Persistent full-Qwen target: explicit CPU integration candidate

`persistent_qwen_target.py` connects the persistent KV transaction store to a
complete real HF Qwen3 model forward. It reuses the original embedding, every
decoder layer's QKV/QK normalization/RoPE/attention output projection/MLP, final
model norm, and LM head. It does not duplicate the model implementation, change
sampling probabilities, or replace the existing target oracles.

The current constructor requires a CPU model and an explicit CPU attention test
kernel. GPU use is refused. The separate first-layer capacity-tail/capture probe
provides evidence about its own operator subgraph; it is not a full-target native
or graph result. This adapter is eager, opt-in and not yet connected to
`PackedSpeculativeSession`.

## Forward and storage boundary

A custom HF `Cache.update` adapter receives each real layer's new, already
RoPE-applied K/V. It copies them to the corresponding independent scratch layer,
gathers the committed prefixes plus scratch into persistent per-bucket staging,
and returns those K/V views to the same HF attention layer. A separate attention
callback consumes the real Q and request-local cumulative lengths. The explicit
CPU oracle performs causal attention within each request and excludes unused
capacity. The layer then continues through its original output projection,
residual paths, MLP and later layers. All layer KV remains scratch until commit.

The model receives explicit per-request position IDs from the transaction, such
as `[C_A, C_A+1, C_B, C_B+1, C_B+2]`, and an explicit mask mapping handled by the
callback. No flattened resident-token count is used as each request's position.
Generic HF mask inference is rejected by the cache adapter. Its `get_seq_length`
only describes the packed committed population, not request-local positions.
The cache explicitly reports `is_compileable=False`; an empty HF cache-layer
list must not accidentally imply compilation or graph readiness.

Selected decoder block hooks are registered once when the adapter takes ownership
of the model. They capture raw block outputs, including the raw final block if
selected. `PackedFeatures.context` concatenates those selected outputs in layer-ID
order. `PackedFeatures.last` is the distinct final-normalized model output;
`predict` uses the original LM head on those actual Q rows. This distinction is
covered by a real tiny-Qwen test that selects the last decoder block as well as
earlier blocks. The intended real model selection `[1,7,14,21,26]` can be supplied
as `layer_ids`; it is not hard-coded to the tiny fixture's indices.

## Explicit transaction interface

Construct the target with the owned model, selected layer IDs, resident-slot and
context/query capacities, and an explicit CPU test kernel. Register a finite set
of `Bucket` objects before use. For example, a small prompt bucket may declare
Q=(3,5,4), C ceilings=(0,0,0), while verification uses exact Q=(2,3) with bounded
growing context ceilings. A separate Q=(1,) bucket handles an active subset.
The work field `committed_write_policy` describes the scratch-only contract; it
is not a claim that every production forward compares all resident bytes. The
CPU tests perform the actual pre/post K and V comparisons.

Physical Q remains the chosen actual query shape; the adapter does not pad every
round to a global maximum. Storage and staging capacities remain visible in work
records, including the inherited per-bucket workspace allocation.

- `add_request(name)` creates a fresh slot incarnation.
- `prefill(chunks, bucket=...)` requires empty admitted requests, runs the full
  model, and commits every prompt query. It returns real full-model features.
- `verify(chunks, bucket=...)` runs all layers into scratch and returns a
  `PackedFeatures` capability. Committed `lengths` and resident K/V remain unchanged
  while the result is outstanding. A second forward, removal, reset or admission
  is rejected during this transaction.
- `predict(features, last_only=False)` applies the original LM head. It retains
  the existing request-to-logits layout. As in the old `PackedTarget`, this
  helper projects all Q rows even when `last_only=True`. Future session admission
  must preserve its existing optimization of selecting only R final hidden rows
  before the LM head; do not route admission through a full-prompt vocabulary
  projection. No projection shape or probability law is changed here. No
  confidence, temperature or sampling operation is added to the target.
- `commit(features, committed_query_lengths)` accepts only the exact outstanding
  features object and one prefix count per verified request. Counts refer to
  verified input queries, including the latest anchor; they are not the number of
  accepted draft tokens. The pool validates all counts before any committed write.
- `abort(features)` discards the scratch transaction without changing committed
  prefixes. A failed model forward also aborts its scratch-only transaction, so a
  later attempt can reuse the original committed prefixes.
- `request_kv(name)` returns detached audit copies in the existing target's
  per-layer `[1,Hkv,C,D]` format. `remove_request` invalidates that incarnation.
  `reset` releases requests while retaining allocated bucket addresses; `close`
  removes permanent hooks and restores the original model backend after pending
  work has been resolved.

A device/copy failure during an actual resident commit can poison the underlying
pool; this is not an atomic rollback guarantee across layers. Ordinary forward
errors occur before commit and preserve the prior resident state. This target
alone does not provide a joint transaction with the draft cache or recovery from
arbitrary device failure.

The model is exclusively owned while this adapter is installed. Do not share it
with another target or run unrelated forwards concurrently. The callback checks
that the attention module belongs to the active adapter. Slot handles and
transactions inherit the storage module's exact-object/incarnation checks;
features also require exact object identity for commit or abort.

## Subsequent sampling integration

This is deliberately not an append/crop drop-in. Existing `PackedSpeculativeSession`
admission uses `target.append` and expects an immediately committed prompt;
verification also uses `append`, then separately crops each request. A later
explicit integration must route admission through `prefill`, verification through
`verify`, and replace the per-request crop loop with one target `commit` after
all decisions are known. Returned per-request selected features can supply the
same committed draft-context projection as before. Invalidating a session must
resolve any outstanding target scratch transaction before reset.

That integration must retain actual q, the existing probability policy, request
RNG ownership, EOS/output-budget handling, and the established cache invariant
that committed context excludes the latest emitted anchor. None of those
sampler changes is made in this module or inferred from its CPU tests.

## Deterministic full-target graph boundary remains unfinished

The eventual target subgraph can start from persistent token IDs, explicit
positions and fixed-shape metadata, then execute the original layers, scratch
writes, attention and final outputs. Metadata preparation, capability validation,
sampling and commits must stay outside it. This eager implementation still uses
Python `Cache.update` state changes, selected-layer hook bookkeeping, concatenated
inputs/features and ordinary HF outputs. Stable KV addresses alone do not make
those operations a replay-safe full-model graph API.

In particular, Python `Cache.update` and hook bookkeeping execute during capture
but do not execute again on graph replay. A future implementation needs explicit
external transaction staging/feature-ownership publication, persistent output
buffers, and stream/event readiness before commit; it cannot reuse the current
Python `_staged` set as proof that a new replay completed. That boundary remains
unimplemented and is not hidden behind a graph flag.

The initial exact ordered-Q bucket family supports finite CPU integration tests.
It is not selectable solely from t−2 global K before current confidence determines
ell. A later bounded R/physical-B/max-Q family, dynamic cumulative values,
allocation padding accounting and double-buffered metadata ownership are separate
requirements for the intended scheduling overlap. Do not equate the first-layer
probe, this full-model eager adapter or fixed addresses with that completed system.

## Local evidence

Four actual tiny-Qwen tests cover all four model layers' K/V, selected raw block
features, final norm and full/last-row logits against independent per-request HF
forwards. They include unequal contexts, complete prefill, two growing-context
verification rounds with partial/full commits, invalid and copied capabilities,
abort, a layer-one forward failure and retry, inactive-resident KV preservation,
bit-identical output/feature/logit isolation when an inactive request is changed,
slot reuse, explicit bucket/input rejection, and stable allocations across reset.

The local run uses existing Torch 2.11.0 and Transformers 5.4.0 on CPU with GPU
visibility disabled. It establishes this adapter's CPU integration semantics and
small-model numerical agreement, not compatibility with the pinned native
Transformers 5.17.0 runtime, full pretrained-model equality, GPU dispatch, graph
replay, serving throughput or acceleration. No environment was installed or
modified; no remote CPU/GPU work was performed. Private logs and source hashes
are in `output/persistent-qwen-cpu-20261009`.
