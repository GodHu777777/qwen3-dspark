# Finite actual-query families: CPU foundation

`QueryFamily` permits different ordered per-request query lengths in one fixed
physical-B program. This is an opt-in CPU-tested foundation over the existing
target and sampling lifecycle. No native GPU family capture, performance curve,
capacity-driver connection or CPU/GPU overlap is established by these tests.

```python
from dspark_qwen.persistent_target_kv import QueryFamily

small = QueryFamily(request_count=2, query_tokens=3, maximum_query_length=2,
                    key_capacity=24, maximum_key_length=16, source='declared B3')
large = QueryFamily(request_count=2, query_tokens=6, maximum_query_length=6,
                    key_capacity=32, maximum_key_length=16, source='declared B6')
target.register_families((small, large))  # once, before any use of these families
target.register_graph_bucket(large, reserve_bytes=declared_private_pool_bytes)
# Explicit setup with real resident prefixes and matching query chunks:
target.capture_graph(example_chunks, bucket=large)
# (1,5), (3,3), (5,1) all select large; metadata changes, physical B stays six.
features = target.verify(chunks, bucket=large)
target.commit(features, accepted_input_prefix_counts)
# Consume raw selected features in the draft on the owner stream, then:
target.release_features(features)
```

The example uses the target's explicitly selected backend. CPU tests use
`CPUReplayEmulator` and `test_only_persistent_sdpa`; that emulator runs Python on
every replay and cannot demonstrate native graph support.

## Transaction and storage contract

Family identity fixes R, physical B, maximum Q, total K capacity and maximum K.
`pool.begin(handles, family, query_lengths=actual_ordered_q)` requires positive
integer lengths with `len(Q)==R`, `sum(Q)==B`, `Q_i<=maxQ`, `C_i+Q_i<=maxK` and
`sum(C)+B<=Kcapacity`. No reservation is filled with dummy query rows. Existing
exact `Bucket` declarations and `begin(handles, bucket)` remain supported.

Actual ordered Q is stored in the immutable transaction together with exact slot
incarnations and contexts. Cumulative query/key lengths, positions, scratch
offsets, spans, commit prefixes, causal-pair work and feature consumers all use
that transaction's Q. Native attention receives the **fixed family** maximum Q
and maximum K, even when the actual maximum is smaller. Its cumulative lengths
exclude neutral K capacity tail. The native host-call spy validates these fixed
arguments on CPU; it does not validate the native operator's generalized cases.

One explicit tuple registers the entire family arena. Metadata and six gather
K/V staging tensors allocate once at the maximum declared R/B/Kcapacity; each
family receives stable, correctly shaped prefix views. No extension, resizing,
automatic family enumeration or capture-on-miss occurs later. Existing exact
buckets retain independent workspaces and may coexist with the family arena.

The workspace budget charges the maximum backing allocations once. Work records
retain `bucket_workspace_bytes` for the current view's logical extent and report
`shared_arena_allocated_bytes` separately. `workspace_bytes` is the target's
total allocation charge. Graph input/output buffers are still independently
allocated at each program's actual B, and every graph still reserves and charges
its own private-pool bytes. No graph-private-pool sharing is introduced. Choose
only declared hot families for graph registration; other registered families
execute explicitly eager and report that execution kind. A registered graph
that has not been captured refuses verification, without falling back.

The single pending transaction and owner-stream feature lease prohibit switching
families while work is pending or features remain in use. Commit alone does not
release a graph feature lease. Writer authority binds its exact registered
object, family, workspace object/view layout and scratch pointers. Same backing
addresses do not authorize another family's captured program. Exact receipts,
events, generations, slot incarnations, failure poisoning and the three fresh
model-signature validation boundaries remain in force.

## Session composition and remaining work

`FiniteTargetBuckets` accepts finite mixtures of exact buckets and families. It
selects the smallest compatible key capacity, breaking ties by declaration
order. `PersistentTargetStrategy` registers the declared family tuple once when
needed. Explicit pre-registration is also supported; all selected families must
already belong to that arena. This preserves admission and allocation preflight,
the original FP64 q/p law, actual proposal provenance, request RNG consumption,
joint target commit and draft projection before feature release.

Six focused tests cover real tiny-Qwen outputs/logits and all-layer KV against
exact eager controls through `(1,5)/(3,3)/(5,1)`, reordered requests, context
growth and small-large-small physical B; inactive residents and incarnation
reuse; fixed native launch arguments; actual backing and graph accounting;
writer/view refusal and unfinished/completed cancellation; and a real
`PackedSpeculativeSession` comparison of q/p, RNG traces, outputs and both target
and draft KV, including zero allocations and eager/graph family selection.

The session test uses `step(manual_allocations)` and its `fixed_budget` proposal
mode. It is not a full-shadow capacity-driver composition test. It does **not** connect
`CapacityRoundDriver`, historical t−2 capacity selection, calibrated confidence
ranking or a measured physical profile to family choice. That causal integration
and the distinction between reserved K and actual B need their own validation.
The current slice neither constructs 74 private graph pools nor changes STS,
training, final-test access, numerical-law claims or scheduling overlap. A later
native gate must validate unequal Q distributions and maxima below launch bounds,
stable metadata replay, exact outputs/KV, private-pool accounting and explicit
eager cases before any full-session performance experiment.

## Local validation record

Execution source is base `1e813cad79601ef7d231f6676388f168e6f2e194` plus five
source/test overlays, recorded in
`output/query-family-cpu-20261009/freeze-repair.json`. All 370 frozen files matched
before and after the final complete CPU run: 302 tests, 296 passed and six Linux
process/monitor tests skipped on macOS, 6.727 seconds, actual process exit 0.
This is local Python 3.14 / Torch 2.11 / Transformers 5.4 CPU evidence, not a
pinned AMD runtime validation. GPU visibility variables were empty.

The initial six new tests and 34 persistent tests passed. The first full run then
found three related compatibility errors: adding an eager execution-kind field
to exact buckets violated the frozen paired benchmark's existing contract. The
repair adds that field only for `QueryFamily`; the original exact-bucket protocol
and its assertions remain unchanged. All 26 affected benchmark/profile tests
passed after repair (two Linux skips), followed by the complete run above. The
original failed run, repaired logs, actual exits and both frozen versions remain
under `output/query-family-cpu-20261009/`.

Independent review found no blocker in this frozen CPU foundation. Its separate
R1/R2/R3 oracle verified 27 transactions and two layers each, including exact
gathers, offsets, positions and neutral K tail. Mixed exact/family workspace
backing measured 20,646 bytes, equal to the allocation charge. Additional checks
covered lease retention through abort, mixed exact/family reuse after release,
cancel-checker failure poisoning and strict integer actual-Q overrides. Evidence:
`output/query-family-independent-review-20261009/handoff.json`. No GPU or remote
operation was performed for this implementation or review.
