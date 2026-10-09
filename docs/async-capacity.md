# CPU two-step capacity reference

`dspark_qwen.async_capacity` separates a historical capacity decision from current
prefix admission. It leaves `scheduler.py` and its literal synchronous
first-non-improvement rule unchanged. This module executes no model, CUDA stream,
graph, target verification or request commit; it is not evidence of overlap or
serving speed.

A planner round is a consecutive integer starting at zero. It is distinct from
the packed sampler's mutation epoch, which can advance several times per round.
`freeze(epoch=..., session_epoch=..., roster=..., profile=...)` must run before any
current proposal draw. The caller then runs a complete shadow proposal for every
active request, calls `allocate(ticket, conditional_scores)`, and finally records
all full shadow rows using `finish`. Conditional scores follow the ticket's
canonical request order. Rows contain pre-token conditional probabilities, not
already cumulative scores. The caller must use a frozen calibration transform.

`freeze` uses only the sealed frame exactly two planner rounds earlier to search
all feasible discrete historical target sizes, including every SPS cliff. For
each size it maximizes expected prefix progress by cumulative-score rank, then
maximizes expected progress times SPS. Exact objective ties choose smaller K.
Historical departed requests still participate; no new request borrows a previous
incarnation's score row. Historical remaining output budgets cap useful prefixes at
`min(gamma, remaining−1)`. With acceptance-prefix length A, actual progress is
`min(1+A, remaining)`: its expectation is `1 + sum(P(A >= j))` only through
`j <= remaining−1`. The final budget position has no bonus-token benefit and is
excluded from both historical capacity candidates and current verification
admission, while full shadow proposals still contain all gamma positions. An
independent rejection-path enumeration checks this rule, including remaining=1. Zero-score extensions remain in this historical full search, so an
unusual rising SPS curve can reserve them. Every discrete SPS value and physical
bucket must be supplied; no interpolation is performed.

The local cold-start policy is target-only admission for the first two rounds,
with **full shadow draft on both rounds and every subsequent round**, including
zero-extra-capacity rounds. The history therefore becomes informative after two
rounds without a target-only deadlock. `finish` requires exactly R × gamma observed
shadow positions and positive elapsed shadow time for a nonempty roster. These
are caller-supplied work observations, not independently instrumented timings;
they must include the actual complete draft work in later system measurements.
This policy is an explicit reference choice, not a claimed paper-exact startup
implementation.

Current R and remaining output budgets are known before proposal draws. They
clamp historical absolute K to current feasibility. A capacity too small for R
baseline tokens is rejected for an outer admission scheduler to handle. Churn
and any clamp are explicit `replanning` reasons. New request incarnations must be
session-wide monotonic; retired incarnations cannot return. Profile identity
includes its complete SPS curve, physical buckets and a conservative context
interval. Each current request's cache + anchor through cache + anchor + gamma
must fit that interval. A different t−2 profile or incompatible context is
rejected; a caller can explicitly start a fresh planner and pay cold-start costs.
This is not a silent change of profile or evidence that a graph shape stayed fixed.

Within frozen K, current admission sorts cumulative conditional scores by score,
then request name, incarnation and position. Ties always put an earlier own
position before its later positions. Only positive scores enter the current
allocation. Consequently these three quantities differ and are reported separately:

- Historical absolute K and current reserved K, decided before current draws.
- Actual logical target B = R + sum(prefix lengths), potentially below reserved K.
- Physical bucket B for that actual logical shape, potentially larger than either.

The reservation also records its physical bucket. Sparse/zero scores may change
the actual physical bucket; this reference does not launch a padded reservation
or claim graph replay. Profile SPS must charge real bucket execution costs when
later measured; this module's synthetic test curves are not hardware evidence.

The causality test intervenes on simulated sampled x_j by varying only later
conditional scores c_(j+1:) while keeping c_(1:j) and other requests fixed. The
admission of own position j remains unchanged. Changing its pre-token c_j itself
may legitimately change admission. Full historical search is safe with respect
to these current interventions because the selected history is t−2. Numeric inputs alone do not prove the model produced pre-token scores.
`Calibration` freezes the temperature tuple and source-artifact SHA before draws.
`bind(ticket, issued_handles)` creates an allocation hook that rechecks the exact
handle objects, seven identity fields, cache lengths, budgets and mutation epoch.
During `verify_commit`, it recomputes scores from the sampler's privately retained
original confidence logits, not mutable public observation copies. A mismatching
score digest or allocation is rejected. The returned evidence is serializable;
it binds numeric score provenance but does not itself certify a lossless model
law or pre-token architecture.

`async_round.CapacityRoundDriver` executes the actual order on a packed session:
freeze → complete shadow propose → bind → validate/verify/commit → finish. It
rejects previously issued outstanding proposals and poisons itself on any partial
round failure. Planner epochs and model mutation epochs stay distinct. Full round
wall time includes calibration, host copies, hashing, private-source validation,
model work and explicit device synchronization. Per-stage durations are reported;
none of these synchronous costs may be hidden from a later end-to-end comparison.
The executable order strengthens the call-order contract but is still a
synchronous reference, not asynchronous buffer management or graph overlap.

Run the CPU fixtures with GPUs hidden:

```sh
CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
python3 -m unittest tests.test_async_capacity tests.test_async_round tests.test_scheduler -v
```

The independent exhaustive oracle enumerates allocations for jagged SPS curves.
Other tests cover exact t−2 selection, interventions, ties, zero underfill,
churn/new incarnations, budget clamps, empty/growing rosters, stale/foreign tickets,
profile/context changes, cold-start work and insufficient baseline capacity.

The initial hidden-GPU run passed 20 tests, including two real tiny-Qwen driver
tests. Three consecutive rounds reconstruct and compare the complete committed
target/draft KV contents, retain full shadow work during both cold rounds, and
use frame 0 at round 2. This uses synthetic SPS and test-only dense kernels;
it is neither a native ROCm numerical gate nor a throughput result.
