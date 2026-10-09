# Persistent full-Qwen target: explicit transactions and bounded graph path

`persistent_qwen_target.py` connects the persistent KV transaction store to a
complete real HF Qwen3 model forward. It reuses the original embedding, every
decoder layer's QKV/QK normalization/RoPE/attention output projection/MLP, final
model norm, and LM head. It does not duplicate the model implementation, change
sampling probabilities, or replace the existing target oracles.

The default test path requires a CPU model and explicit CPU attention oracle.
An optional `native_backend` selects the pinned BF16 ROCm capacity operator,
with a real `PinnedRocmVarlenKernel` runtime/schema guard and no fallback. An
additional explicit graph backend and finite budgets enable the prepared
full-model graph path in [persistent-qwen-graph.md](persistent-qwen-graph.md).
The separate first-layer capacity-tail/capture result remains evidence about
its own operator subgraph, not this full-target implementation. No full-target
GPU execution or speed result is implied. `PackedSpeculativeSession` uses its
explicit persistent target strategy; existing append/crop targets remain default.

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
context/query capacities, and either an explicit CPU test kernel or the pinned
native backend. Register finite buckets within count/workspace byte budgets. Register a finite set
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
- `verify(chunks, bucket=...)` uses a registered captured bucket when explicitly
  configured, otherwise the explicit eager backend. `verify_eager` is the
  same-backend eager control. Both run all layers into scratch and return a
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
- `release_features(features)` releases graph output ownership after commit or
  abort AND after the draft has consumed selected features. The eager path checks
  exact feature identity but has no reusable-output lease. A graph lease prevents
  new forwards, reset and slot changes even after target commit.
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

## Explicit sampling integration

This target is not an append/crop drop-in. `PersistentTargetStrategy` routes
admission through `prefill` and verification through `verify`, and replaces the
per-request crop loop with one target commit after all decisions are known.
It releases features after the unchanged R-final-row admission head and draft
prefill, or after target commit and draft append_committed. Failure cleanup tries
abort, release and reset independently, then poisons uncertain storage rather
than reusing it. Actual q, FP64 probability policy, RNG ownership, EOS and output
budget semantics remain in the original sampler. Target committed context still
excludes the latest emitted anchor.

## Prepared full-target graph boundary

The implementation now has explicit prepare, submit and finish operations,
fixed token/position/metadata buffers, a capture-time HF cache interface,
selected raw-block copies and a distinct final-normalized output buffer. The
actual original embedding, QKV/QK normalization, RoPE, attention, MLP, all decoder
layers and final norm are emitted by the full HF model call during capture.
The LM head, sampling, metadata validation and commits remain outside.

Python cache/hook bookkeeping does not recur during real graph replay. Instead,
a registered fixed program binds its complete layer signature, scratch addresses,
transaction and submission generation. Submission matches the exact captured
writer; backend-owned completion binds the exact receipt rather than a potentially
colliding per-pool generation number. It confirms readiness
before the pool permits commit. This is a trusted backend completion contract,
not an independent proof that each kernel wrote correctly. Captured raw/final
outputs stay leased until the last consumer releases them. The initial native
implementation requires one owner stream for target, head, draft consumers,
commit and release. It does not promise overlap or double buffering.

Only explicitly registered, explicitly captured buckets can replay. Graph count,
workspace bytes and retained private-pool segment bytes (including inactive
blocks) are bounded; global reserved deltas are recorded separately. A not-ready
cancellation remains pending, while completion-checker exceptions poison the
target and store. Missing buckets,
address/model signature drift, failed events and capture failures never silently
fall back to eager execution. See the graph document for lifecycle, budget and
failure details. The CPU replay emulator executes Python and is explicitly
labeled; it is not evidence of full-target GPU graph capture.

The exact ordered-Q family is still not selectable solely from t−2 global K.
A later bounded R/physical-B/max-Q family, padding accounting and double-buffered
metadata ownership remain separate work for intended scheduling overlap.

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
small-model numerical agreement, not full pretrained-model equality, GPU dispatch, graph replay, serving throughput
or acceleration. That initial local result is in
`output/persistent-qwen-cpu-20261009`. Subsequently, immutable `f46e63f` passed the
complete 230-test suite in 13.647 seconds on the existing AMD Torch 2.12.0+rocm7.2 /
Transformers 5.17.0 runtime with GPUs hidden and actual SSH/test exit 0. All 334
archived files were hash-verified; no tested API incompatibility was observed.
Evidence is in `output/persistent-qwen-pinned-cpu-20261009-f46e63f`. No environment
was changed and that check did not execute GPU work. Later graph preparation
tests and their distinct emulation boundary are documented separately.
