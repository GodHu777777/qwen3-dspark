# Finite actual-query families: CPU foundation

`QueryFamily` permits different ordered per-request query lengths in one fixed
physical-B program. This is an opt-in CPU-tested foundation over the existing
target and sampling lifecycle. The synchronous full-shadow capacity driver is now connected and CPU-tested
below. Native GPU family capture, measured capacity costs and CPU/GPU overlap
remain unverified.

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

The original six-test foundation uses `step(manual_allocations)` and its
`fixed_budget` proposal mode. Its result alone does not establish full-shadow
capacity integration. The later bridge below tests that separate causal path;
a measured physical profile remains outstanding.
The foundation neither constructs 74 private graph pools nor changes STS,
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


## Full-shadow capacity bridge: CPU validation

`CapacityRoundDriver` composes with `PersistentTargetStrategy` and the declared
families through the existing interfaces: freeze historical t−2 capacity before
current draws, generate all shadow positions, bind current confidence, recheck
private confidence data, select actual Q, then verify/commit and record history.
Only an early identity physical-B profile guard was added to production code.
The guard runs at construction and step entry; the concrete engine never pads
actual B to reserved K. General physical mappings remain valid in the pure planner.

Five bridge tests use actual tiny-Qwen/draft/head execution with synthetic SPS
curves. Their exact-bucket eager control also generates full shadow proposals,
then verifies the same allocations. They compare q/p, decisions, RNG draws,
outputs and all target/draft KV; a separate full-prefix HF cache checks committed
content. Calling `step(fixed_budget)` would consume different RNG and is not the
control used here. These are untrained CPU fixtures, not quality measurements.

The positive fixture produces these successive actual shapes:

| Round | Active R | Reserved K | Actual Q | Execution |
| --- | --- | --- | --- | --- |
| 0, 1 | 2 | 2 | (1, 1) | Explicit eager, full shadow still generated |
| 2 | 2 | 5 | (3, 2) | CPU replay emulator |
| 3, after removing one request | 1 | 4 | (4) | Explicit eager |
| 4, new incarnation with one output left | 1 | 1 | (1) | Explicit eager, three shadow positions |

The zero-score fixture fixes the real confidence head to finite −1000 logits
before cache/source binding. With a synthetic rising SPS curve, rounds 2 and 3
reserve K=8 but execute B=2, while still generating six shadow positions. This
verifies reservation/actual-work separation without changing scores after draws.
It does not establish trained head calibration or sensible measured hardware costs.

Invalid physical profiles are rejected before draws, including public profile
replacement. Missing B5 after two successful cold rounds, changed public
confidence and foreign capabilities fail after full-shadow proposal draws but
before target verification RNG/forward/commit. Partial rounds cannot be retried.
Public token/probability observation tampering does not affect computation from
privately retained proposals. Historical digests bind observed shadow duration;
independently timed runs need not have equal history digests.

Private frozen evidence: `output/capacity-query-family-cpu-20261009/`, base
`8bee244fbacae3fd093d24c1b2512f5ab3d807c6` plus `async_round.py` and the new
`tests/test_capacity_query_family.py`. All 372 frozen files matched before and
after the complete CPU run: 307 tests, 301 passed and six Linux-only skips,
7.058 seconds, process exit 0. The five focused bridge tests passed in 0.238
seconds. These elapsed times describe testing, not inference throughput.
Native variable-Q replay, measured capacity costs, multi-request end-to-end gains
and actual CPU/GPU overlap remain next gates.
The earlier same-stack slowdown and separate strong vLLM baseline are unchanged.

Independent review found no blocker in this frozen synchronous CPU scope. Four
additional cases checked changed t−2 profile identity, out-of-range context,
incomplete shadow proposals and profile replacement/restoration. The first two
reject before drawing; incomplete shadow rejects after its two proposal draws
per request, without target work. Review evidence is retained under
`output/capacity-query-family-independent-review-20261009/`.
