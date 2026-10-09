# Full-shadow inner-service diagnostic

`scripts/profile_shadow_inner_r2.py` adds a bounded diagnostic over the existing
R2/C256 trained gamma7 full-shadow path. It reuses the R2 factory, both hot target
graphs, all eager tails, complete startup numerical validation, unchanged FP64
law/validation and per-request RNG. Only the new script/tests/document are added;
production code and previous benchmark/profile defaults remain unchanged.

The six complete batches are target-only warmup, gamma7 warmup, gamma7 plain0,
gamma7 inner-observed0, gamma7 inner-observed1, gamma7 plain1. Each request emits
128 outputs on the original prompts and seeds. The two compared pairs reverse
plain/observed order. All loaded weights, both graphs and resident allocations
remain shared. The supervisor/cooperative limits remain300/290 seconds; memory
limits remain8GiB prefree,6GiB allocator, two512MiB private graph reservations,
1280MiB combined and64MiB workspace. No trace, Kineto, optimization, training,
calibration, changed validation, SPS/Profile, new admission or retry is introduced.

## Owning parent and disjoint child service

The existing outer observer still records whole-propose synchronized service,
target service, external probability service and commit/projection/release. Its
model-signature wrappers remain host-only; all three current checks still execute.
The new child observer is active only inside an owning full-shadow propose:

- one packed draft backbone;
- one shared base LM head, identified by the propose callsite;
- seven calls each to Markov embedding, Markov projection and confidence;
- seven batched FP64 logits-to-probability calls;
- seven categorical draws per active request;
- one observation-copy helper per active request.

The shared head's admission/target calls are not base-head children. Nested
probability/helper calls within an already observed child execute unchanged but
are not fenced or counted again. CPU test collection is separate from the runtime
observer and cannot create proposal-copy records. Concatenation/stacking, Python
assembly, validation outside these child callables and other uncovered work stay
in the explicit parent residual.

Each child retains its prior-work pre-drain, function host body, post-drain and
small boundary-bookkeeping gap. Child synchronized service is body through the
end of post-drain. The owning parent denominator is its synchronized-service
interval after the outer pre-drain: body through post-drain, including bookkeeping.
It excludes the outer pre-drain. For each propose, parent service equals child
pre-drains plus disjoint child services plus explicit residual. Parent and child
ledgers must never be added. Body/post/gap are components, not extra totals.

Fences deliberately alter queueing and overlap. This is serialized observed
service attribution, not kernel-active time or the unchanged critical path. The
plain/observed whole-batch ratio includes instrumentation and possible run-to-run
variation. It is not precisely isolated overhead or a statistical confidence test.
No stage subtraction can establish speedup.

## Record bound and domain guard

The original batch always processes every unfinished request, admits one initial
output, and commits at least one additional output per active request per round.
Thus128-output requests permit at most127 rounds, including R1 finish tails.
Before each observed propose, guards require original resident r0/r1, original
active order with every unfinished request, budget128, prompt length256,
gamma7/shadow, no explicit reduced proposal limits, temperature1 and no EOS.
The runner/binding also check exact original prompt/seed/sampling data. Actual
round128 fails before another proposal rather than relying on an older254 cap.

R2 children per round are1 backbone +1 head +21 Markov/confidence +7 laws +14 draws
+2 copies =46. R1 tails have38. At most127×46=5842 children fit the8192 child cap.
Outer admission has8 records and each R2 round at most10: outer cap usage≤1278.
At most3×127=381 signatures remain separately counted. These three lists contain
at most7501 records;127 round summaries and127 parent partitions bring their
combined bounded rows to7755. Outer/child/signature guards are independent; no
extra diagnostic guard records or failure-cleanup records expand this domain.
First failure stops collection and lets unchanged cleanup pass through until all
wrappers are restored. Partial parent/child evidence survives failure.

## Semantic gate and decision

All plain and observed rows must match full raw output tokens, final RNG bytes,
round decisions/Q/work/execution and final lengths, including across both pairs.
Each completed row is persisted before comparison rejection. The common token/RNG
collector runs after the timer for all phases. Private raw evidence is not included
in public reports. CPU tests additionally compare every round's actual full-shadow
proposal tokens/q/confidence and resident target/draft KV exactly, covering varying
Q and R1 tails while caches remain populated. Nested wrapping cannot skip existing
validation or the three fresh target signature checks.

For each observed batch, independently sum FP64-law and categorical-draw child
service, then divide by the owning propose service sum. Reconstruct this fraction
from raw child/parent records. Child coverage is checked against actual R and full
seven-position work; no missing child is silently assigned zero.

The predeclared diagnostic hypothesis is that probability-law plus draw service
occupies at least50% of propose in both observed batches. Classification first
requires both observed/plain whole-batch ratios within inclusive[0.95,1.05]:

- Both fractions≥0.50: **supported** in this diagnostic domain.
- Both fractions<0.50: **not supported**.
- Fractions fall on opposite sides, or either ratio is outside the allowed range:
  **inconclusive**, with order disagreement or disturbance identified.

A below-threshold result is distinct from opposite-order disagreement. Semantic
failure aborts the experiment; it is never reclassified as timing uncertainty.
The5% acceptance rule is not a confidence interval. Any optimization selected
later needs its own repeated complete E2E comparison. Existing numerical-law
limits, same-stack slowdown and separate strong baseline remain unchanged.
