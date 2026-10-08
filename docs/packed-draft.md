# Packed draft backbone: CPU-prepared noncausal varlen path

`dspark_qwen/packed_draft.py` provides an inference-only multi-request draft
backbone using the existing `DSparkDraft` weights/modules. It does not modify the
training model or immutable training/whole-Qwen snapshots. Native noncausal GPU
execution remains unverified; CPU fixture success is not a native correctness,
quality or speed result.

## Interface and ownership

Construct `PackedDraft(draft.eval(), native_backend=DRAFT_BACKEND)` explicitly on
the pinned ROCm device. This selects `rocm_aten_no_window_noncausal_pinned_v1`.
There is no implicit backend or automatic dense fallback. CPU tests instead
supply `test_kernel=test_only_noncausal_varlen` without a native backend.

- `add_request(name)` gives a new request a never-reused marker and zero context.
- `append_committed({name: features})` accepts nonempty `[1,new_tokens,features]`
  target-feature chunks for any subset of resident requests. The caller supplies
  only committed features, including an old anchor but excluding the newest
  emitted anchor; rejected verification tails must be excluded before this call.
- `crop(name,length)` retains an exact projected-KV prefix. Cropping to zero,
  removal and re-adding the same external name are supported.
- `backbone({name: anchor_token})` accepts one `[1,1]` anchor per active request and
  returns flattened hidden rows plus request spans, context lengths and work
  counts. `features.for_request(name)` returns that request's whole draft block.
  Inactive resident requests do not enter the gathered attention domain.
- `request_kv(name)` returns the actual per-layer projected context tensors for
  independent inspection. Proposal block KV is transient and never appended.

The wrapper owns mutable projected-context KV and marker/position metadata. It
reuses model modules without changing them. Inference is `no_grad`; callers keep
trainable parameters and RoPE buffers FP32 and use BF16 autocast with BF16 frozen
target embedding/head modules. Projected K and V retain the existing model representation separately: with
FP32 RMSNorm weights, K may remain FP32 while V is BF16. The wrapper preserves
those cache dtypes and explicitly applies the active autocast dtype at the
attention boundary, matching the existing SDPA behavior; it does not downcast
model weights or RoPE buffers. Final norm output may remain FP32. Each cache
component precision must remain consistent across appends/backbone. Concurrent mutation of the same wrapper or its
underlying model is unsupported.

## One flattened backbone, fully visible own block

All incoming committed chunks are validated before any projection, concatenated
once, and run through one context FC/norm and one RoPE call. Each draft layer
projects the entire new context once. Persistent cache changes occur only after
all projections, candidate metadata allocations and candidate-state validation
complete. Projection or late metadata failure cannot partially commit a request.
Crop likewise constructs and validates its whole candidate state before commit. Request-local positions continue from each separate committed length.

All active anchors form one flattened embedding and RoPE call. At each layer the
implementation performs one input norm, one Q projection, one block K/V
projection, one noncausal varlen attention call, one output projection and one MLP
across the complete batch. Python request loops only build metadata, input rows
and gather indices; they never execute a request's draft model separately.

Each request's gathered K/V consists of its committed context followed by its own
**entire** draft block. Every query sees every one of those keys, including later
block positions. The kernel therefore uses `is_causal=False` and no window limits.
Using the target's bottom-right causal operator would change the draft model.
`PackedLayout` is reused for request-local geometry only; the separate native
kernel explicitly selects noncausal semantics and inherits the existing exact
ROCm runtime/schema/BF16 input guards. The previous target tensor or whole-Qwen
gates do not validate this noncausal kernel.

Backbone width is always `spec.block_size` per active request. Selecting a short
proposal prefix after backbone does not save its computation, because the draft
block is bidirectional. Admission length is deliberately not a backbone argument.
An engine may explicitly skip draft computation for a target-only request, but
must exclude that request from this call and account for the skip separately.
This interface does not yet implement batched Markov heads, stochastic draws,
accept/reject verification, global allocation or asynchronous scheduling.

## Content and physical work checks

The cache preserves all prior projected KV bytes on append and uses exact
request-marker/position subsets on crop. Removed markers never reappear after
request-name reuse. Backbone is read-only with respect to persistent context,
including if its attention operator fails.

Work reports include full block query rows, gathered K rows, resident context
rows, per-layer attention/MLP calls and the logical sum of `q_i*k_i` pairs.
They also count actual K/V concatenation and gather elements/bytes across layers,
and attention-boundary cast elements/output bytes.
The current implementation concatenates resident physical context with all block
KV before gathering, so inactive context still contributes to that copy cost.
Appending committed chunks likewise records old-plus-new cache-concatenation
volume. These are allocation/copy-domain counts, not measured memory traffic or
latency. A paged storage implementation can change these costs later without
silently attributing their removal to the current path.

CPU tests compare actual layer KV and final hidden values to independent
single-request cached and full draft backbones. They cover differing request
positions/order, zero context, multiple append/crop cycles, crop-to-zero,
exit/re-add, inactive caches, same-shape active/inactive poison, explicit full
block visibility, module/kernel call counts, transaction failure retention and
BF16 autocast with FP32 trainables/RoPE. A native adapter spy checks the literal
noncausal/no-window call without loading a GPU. No GPU was used for these tests.

CPU development evidence retains the initial BF16 fixture failures caused by an
incorrect assumption that projected K and V always share a dtype. Inspecting the
unchanged original cached backbone established the mixed cache representation
and attention autocast boundary. After preserving those original representations
and adding explicit boundary casts, all 10 tests passed. The AMP fixture compares
against original cached SDPA with fixture-only `.02/.02` tolerance; it is not a
replacement for any formal GPU gate or a distribution-equivalence assertion.
Late metadata-allocation and crop-candidate failures also preserve every old
cache/metadata reference and marker identity.
