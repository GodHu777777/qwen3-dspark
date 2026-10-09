# Optional persistent target in packed sampling

`PackedSpeculativeSession(..., target_strategy=...)` explicitly selects the
persistent target lifecycle. Omitting it constructs `AppendCropTargetStrategy`,
which uses the existing target append/crop operations, backend and numerical
behavior. The independent cached, canonical and dense numerical oracles and the
float64 probability implementation are unchanged.

The CPU candidate can be constructed as follows (model, draft and request RNGs
must already satisfy the normal session contract):

```python
from dspark_qwen.persistent_qwen_target import (
    PersistentQwenTarget, test_only_persistent_sdpa,
)
from dspark_qwen.persistent_target_kv import Bucket
from dspark_qwen.target_strategy import FiniteTargetBuckets, PersistentTargetStrategy
from dspark_qwen.packed_sampling import PackedSpeculativeSession

target = PersistentQwenTarget(
    target_model, layer_ids=(0, 2), slots=2, context_capacity=64,
    max_query_tokens=9, test_kernel=test_only_persistent_sdpa,
)
buckets = FiniteTargetBuckets(
    prefill=(Bucket((4, 5), (0, 0), 'explicit two-prompt CPU protocol'),),
    verification=(
        Bucket((4, 4), (48, 48), 'explicit two-request full verification'),
        Bucket((2, 1), (48, 48), 'explicit partial/zero allocation'),
        Bucket((1, 1), (48, 48), 'explicit anchor-only verification'),
    ),
)
strategy = PersistentTargetStrategy(target, buckets)
session = PackedSpeculativeSession(target, packed_draft, target_strategy=strategy)
```

The caller declares finite tuples for both phases. Each bucket has exact
**ordered** query lengths and per-entry context ceilings. Selection uses the
smallest compatible key capacity, with declaration order breaking ties. There
is no implicit shape enumeration, maximum-Q padding or dynamic registration.
Admission, different active subsets, changed allocations and last-budget rounds
must have explicit compatible shapes. A missing bucket is a clear error.
All registered buckets must fit the target's resident and scratch capacities.
Their context ceilings are storage bounds, not measured hardware throughput or
a causal scheduler capacity estimate. Existing typed policy validation and its
request identities, real confidence provenance and output-budget bounds remain
in force; this strategy supplies no scheduler proof.

The provider also accepts explicitly declared `QueryFamily` values with fixed
R/physical-B/maxQ/Kcapacity/maxK and per-transaction actual Q. Their metadata and
gather workspaces share one predeclared arena, while graph input/output buffers
and private reservations remain separate. Selection still requires exact actual
physical B with no proposal padding. See [query-families.md](query-families.md)
for the CPU session composition evidence and the unconnected capacity-driver and
native validation boundaries.

Admission checks bucket availability before adding requests or consuming RNG.
It calls target prefill, selects only each request's final hidden row, and makes
the original single R-row LM-head projection. It then samples each first token
and appends the selected raw target block features to draft context KV. Only
the last hidden state used for logits is final-normalized.

Verification forwards each active request's newest emitted anchor followed by
its allocated proposal prefix. All per-request decisions are computed, in the
original order and using the privately retained actual q and original RNGs,
before one target commit. For an accepted prefix followed by a replacement or
bonus, committed input rows are the anchor plus accepted draft tokens. Their
count equals the number of newly emitted tokens. If EOS or budget stops inside
the accepted prefix, the newest emitted token remains the next excluded anchor.
The identical selected prefix of raw features extends draft KV. Both committed
caches therefore remain `prompt + output[:-1]`, including for finished requests.
Inactive residents retain their exact KV. Removal releases the slot; re-admission
gets a fresh request and slot incarnation, invalidating old proposal handles.

Fixed-budget `step()` checks the known query shape before draft forward or
proposal draws. In full-shadow mode, draft proposals and their real RNG draws
already exist before allocation. The subsequent bucket check occurs before
target forward or verification draws; it cannot undo those shadow draws.
Bucket refusal leaves existing request/cache/proposal state available for a
valid allocation. Zero allocation still consumes the collected full shadow and
verifies exactly one anchor row, without changing the actual q of other requests.

Feature ownership extends through draft projection: strategy release happens
after admission's R-row head and draft append, or after verification's target
commit and draft append. A target exposing `release_features(features)` receives
that call; the original eager CPU adapter without this method needs no action.
This ordering is tested with a fake CPU lease over actual model features, and
does not by itself demonstrate a graph or asynchronous event lifetime.

Execution failure invalidates the session permanently. Cleanup attempts pending
scratch abort, feature release and target reset in that order, and always clears
draft caches. A target that cannot be cleaned is still unreachable through any
public session operation; the cleanup error is retained on the failed session.
If the target supplies an explicit invalidator, cleanup failure also invokes it.
No rollback of consumed RNG, partial device writes, or joint target/draft commit
is promised. This also covers admission failures after requests were added.

Work records preserve the underlying target's physical/logical Q, active K,
capacity K/tail, inactive resident rows and staging/memory counts. Persistent
strategy adds phase, provenance, declared bucket count, selection rule and the
single commit's per-request counts including anchor. Bucket selection, commit,
feature release and draft projection all execute inside the session call; they
must be included in any later whole-round timing. Allocation and registration
happen during explicit strategy construction and must be reported as setup cost.

`tests/test_persistent_sampling.py` uses complete real tiny-Qwen forwards,
independent per-request HF cache reconstruction and real draft KV projection.
It compares same-seed legacy/explicit append-crop/persistent sessions, actual
p/q, emitted outputs and RNG draw traces; exercises all rejection indices using
legal controlled uniforms under the actual distributions; and covers acceptance,
EOS/residual/bonus, output budget, inactive residents, lifecycle, finite bucket
refusal, admission failures, cleanup failures and feature release ordering.
Probability laws and proposal decisions are never mocked by these tests.

The session change does not enable a full-target native backend, change the GPU
entry guard, implement full-model graph replay, overlap work, establish BF16
cross-kernel equality, or demonstrate speedup. Separate native adapter work must
retain its own execution evidence and admission requirements.
