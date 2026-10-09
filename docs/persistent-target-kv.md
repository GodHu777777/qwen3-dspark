# Persistent target KV: independent storage candidate

`dspark_qwen/persistent_target_kv.py` implements fixed-address resident target KV,
independent verification scratch, and registered attention staging buffers. It
has CPU tensor and real tiny-Qwen KV tests. It is not connected to the decoder,
a native attention implementation, a captured GPU graph or an overlap result.
Existing target oracles, sampling law and benchmark code remain unchanged.

## Why this step follows the measured profile

The complete 700bfa6 profile covers R=2, C=(128,128), all 64 ordered allocations.
Its 320 primary round times have median 94.978 ms. Diagnostic host spans have
median 43.461 ms in target append, 33.129 ms in full shadow proposal, 61.121 ms in
verification/commit (which includes target append), and 1.651 ms per target crop.
These are nested host/stream regions, not additive isolated GPU kernel times.
The largest same-B spread of cell medians is B=9, 91.215–105.638 ms (1.158×).
The raw result SHA-256 is
`ca8c40af1fa7d4cd504d50beeca992ff7d505af5bb7bd2aeae5cde3092eca1e2`.

Current target append constructs GPU marker/position tensors, runs dynamic
`unique_consecutive`/`nonzero`/`tolist` layout extraction, validates values, extends
HF DynamicCache by concatenation, and gathers active KV every layer. Acceptance
then gathers all resident layers during crop. This dependency structure supports
first separating committed state from speculative work and making its addresses
stable. The profile does not show how much of 43.461 ms is recoverable by graphs,
or establish that the new gather implementation is faster.

The new implementation deliberately uses ordinary fixed-size tensor gathers and
selection as a testable data-movement reference. It gathers both committed and
scratch candidates for every capacity row. This can do more traffic than the
old exact-size gather. A fused source-selection/gather or paged native attention
may later remove that traffic; no speedup is assumed for this candidate.

## Interface and ownership

- Construct `PersistentTargetKV(layers, slots, context_capacity,
  max_query_tokens, heads, dim, dtype, device)` with keyword arguments.
- `add_request(name)` returns a store-minted `Slot` capability. Request name,
  physical slot and monotonically increasing incarnation are bound; exact object
  identity also rejects copied, forged and cross-store handles.
- `load_prefix(slot, keys, values)` copies real model KV `[L,C,Hkv,D]` once into an
  empty slot. Keys must already include the model's position/RoPE transformation.
- Register `Bucket(query_lengths, context_ceilings, source)`. Shapes are immutable
  tuples; `source` identifies the declared applicability, not performance proof.
- `begin(active_slots, bucket)` fixes current lengths, query positions and source
  indices in persistent metadata. Slots may be a reordered active subset;
  inactive requests remain resident. Admission/removal/another transaction is
  refused until commit or abort.
- For each model layer, `stage_layer(tx, layer, new_keys, new_values)` copies only
  verification KV `[sum(Q),Hkv,D]` into scratch. `prepare_attention(tx, layer)`
  gathers prefix plus scratch into the bucket's reusable K/V arrays and returns
  those arrays, `cu_query`, `cu_key`, and query-local positions.
- `commit(tx, committed_query_lengths)` copies only the caller-selected prefix of
  each request's verification queries into resident KV. Counts include the latest
  anchor. For k accepted draft tokens followed by a replacement/bonus, the usual
  committed query count is k+1, not k. Sampling, EOS and output-budget logic remain
  the caller's responsibility. Every layer must have staged before commit.
- `abort(tx)` drops the scratch transaction with no committed writes. A stale or
  copied transaction cannot be reused. `remove_request` releases a slot logically;
  stale bytes are never made visible to a fresh incarnation.

The resident arrays are `[L,slot,Ccapacity,Hkv,D]`; scratch is
`[L,max_query_tokens,Hkv,D]`. They have different allocations. Attention staging
is per bucket and reused across layers; consumers must finish with a layer's
views before the next layer overwrites them. Caller-owned model output tensors
are copied at the staging boundary. `request_kv` returns detached copies for
content auditing. This single-owner prototype does not make buffers private from
arbitrary Python mutation and does not support concurrent transactions.

Validation of a commit finishes before any resident write. A device/copy failure
during resident bootstrap or commit poisons the pool rather than claiming atomic
rollback. Failed metadata preparation publishes no transaction: a retry rewrites
all metadata, while earlier transaction capabilities remain invalid. Python
errors during scratch staging leave committed state unchanged; abort is allowed.

## Growing contexts and graph boundary

A bucket fixes the **ordered query lengths**, active count and per-entry context
ceilings. Actual context lengths, slot assignments, incarnations, positions,
source-index contents and cumulative key lengths may change within those bounds.
Physical Q is exactly logical Q; no maximum-Q padding erases the allocation
choice. Context capacity is real allocated work: `work(tx)` separately records
logical active K, physical K capacity, padding, inactive resident K, gather rows,
resident/scratch/workspace bytes and causal pairs. Thus R=2, ell=(0,7) and (7,0)
can have separate shape buckets even though both have B=9.

Each registered bucket currently owns six capacity-shaped K/V staging tensors
(two source gathers and one result, for both K and V), plus metadata. Registration
is explicit and finite: the module never enumerates all query vectors, and
`work(tx)` reports the workspace bytes for the selected bucket. Registering every
ordered vector over large R would cause a combinatorial memory expansion; a
production graph cache needs a bounded selection/eviction policy and an explicit
total workspace budget before such registration. Do not infer that this prototype
has solved graph-cache sizing.

The measured C128 domain is an evidence label. It does not prohibit growing
contexts: storage bucket ceilings can be declared independently, and any cost
prediction using them needs separately frozen applicability and validation. This
module neither predicts cost nor exposes a measured capacity/SPS profile.

The candidate deterministic target-layer boundary is:

1. Outside capture, prepare immutable transaction metadata and copy token IDs,
   positions, source indices and cumulative lengths into persistent input arrays.
2. In a future fixed-shape target subgraph, compute Q/K/V with explicit positions,
   write new K/V into scratch, run fixed-size source gathers/selection, then native
   varlen attention with exact Q and actual cumulative key lengths, followed by
   the deterministic projections/MLP. Selected hidden features need persistent
   outputs for later draft-cache projection. Full model weights remain constant.
3. Outside capture, sample/validate using the existing probability law. Only then
   commit the selected scratch prefix. Graph replay must never commit speculative
   KV into resident storage before that decision.

The current private ROCm wrapper **cannot directly consume this capacity-tail
layout**: it checks tensor K length equals the sum of actual key lengths, and
its layout validation reads GPU values into Python. The current HF target also
performs dynamic cache and hook operations. A separate graph-specific adapter
must keep the validated runtime/schema pin, move host checks to the preparation
boundary, and verify native support for `cu_key[-1] < Kcapacity` with fixed
`max_seqlen_k`. Native operator support and allocation-free replay behavior have
not been established. CPU SDPA slicing demonstrates intended attention semantics,
not ROCm capability. Never pass capacity padding as if it were valid context.

The next bounded native compatibility experiment, after separate authorization,
should reuse one small query bucket at two growing contexts with identical buffer
addresses. It must compare eager exact-length native outputs against capacity-tail
native outputs before attempting capture, poison unused capacity to verify it is
unread, then replay with changed positions/cumulative lengths and compare outputs
and committed KV. A failure preserves the current private wrapper and old oracle;
it must not silently reinterpret padded capacity as actual K or claim capture.
This is the concrete missing capability check, not another full quality sweep.

The exact ordered-Q bucket is a storage correctness choice, not yet the shape
abstraction required by two-step capacity scheduling. At t−2, the global capacity
K may be known while the current per-request ell vector still depends on current
confidence. Consequently this implementation cannot preselect the exact-Q graph
from K alone and has not implemented zero-overhead scheduling. The interface must
not force a permanent one-graph-per-ell policy.

A future bucket family could instead bind active R, physical B, maximum query
length, and context ceilings while taking actual ordered query lengths through
persistent cumulative-length tensors. Such a family requires native support for
dynamic cumulative values under fixed physical tensors, sufficient max-Q bounds,
and explicit handling of admission, finished requests and budgets that make actual
Q smaller than reserved B. Any extra physical queries must be isolated from real
requests and reported as padding, never silently hidden in logical B. Those
conditions remain unverified; this module does not weaken them by reusing an
exact-Q bucket for a different shape.

CPU scheduling/GPU execution overlap remains a separate dependency problem. The
CPU may prepare future t−2 capacity while a current deterministic target graph
executes, but buffers cannot be overwritten until their consuming stream event
completes. The single-transaction API intentionally does not yet promise such
concurrency. A later double-buffered metadata/input workspace needs two distinct banks,
producer/consumer ownership and explicit stream events; a CPU producer may fill
only a bank whose previous GPU consumer has completed. Commit hazards on shared
resident KV and the corresponding query/scratch bank must also be ordered. These
conditions can be added around the separated committed/scratch state without
claiming that stable addresses alone provide scheduling overlap.

## CPU evidence

Eight tests exercise stable pointers across growing contexts, exact query work,
ordered allocations, active subsets with inactive residents, scratch/commit
isolation, zero/partial commit, abort, rejected stale/forged/copied/cross-store
handles, stale transactions, invalid decisions, injected metadata-copy failure,
indexed device-alias canonicalization, and independent variable-prefix attention. A real tiny Qwen forward supplies all
layers' KV; partial commits are compared with separate complete-prefix forwards.

The 2026-10-09 local run used existing Torch 2.11.0 and Transformers 5.4.0 with CPU
tensors and CUDA/HIP/ROCR visibility disabled. This differs from the native pinned
Torch 2.12.0+rocm7.2 / Transformers 5.17.0 environment; it establishes storage
semantics, not pinned-runtime model equivalence or native graph behavior. No
package was installed and no remote CPU/GPU operation was performed. Private
verification and source hashes are in `output/persistent-target-kv-cpu-20261009`.

Run in an existing CPU environment with Torch and Qwen3-capable Transformers:

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python -m unittest discover -s tests -p test_persistent_target_kv.py -v
```

Device-alias regression: the first CPU `cpu:0` reproduction allocated tensors on
`cpu` while preserving the indexed string in the store, so valid same-device KV
was refused. The constructor now uses `keys.device` after the first allocation as
the canonical device for validation and subsequent buffers. An actual indexed-CPU
bootstrap/stage/commit test covers this without requiring GPU access. CUDA alias
behavior is addressed by the same allocation-derived normalization but has not
been executed in this local test. The original seven-test log and the reproduced
failure are retained beside the new eight-test result.
