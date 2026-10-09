# Opt-in fixed-shape categorical fallback candidate

`dspark_qwen/experimental_categorical.py` contains an independent candidate.
Production `tensor_sampling.sample_categorical` is unchanged and remains the exact
oracle. Importing the experimental module does not replace any production alias.
No speed or native support claim follows from the CPU gate.

The candidate preserves original probability checks, one-row rejection, FP64
`cumsum`, `searchsorted(right=True)`, `_uniform` evaluation and validation order,
and final `where`. It changes only fallback-index construction for a nonempty
one-dimensional tensor after the RNG callback: integer positions, a positive
support mask, and a fixed-shape maximum return the last positive position.
The mask is recomputed after the callback, never cached. Existing normalization,
CDF arithmetic, generator/device/dtype and draw ordering remain unchanged.

An RNG callback can mutate the referenced probabilities after their initial
validation. If it changes dimensionality or makes the tensor empty, the candidate
uses the original `nonzero(...)[-1, 0]` expression. This preserves unusual existing
behavior, including the first-coordinate result after a two-dimensional reshape.
If a nonempty one-dimensional callback clears all positive support, the reduction
produces -1 and the original expression raises its original IndexError. Invalid
uniform results are still rejected before fallback computation. The scalar check
can synchronize a native device; its cost belongs inside any later complete
session measurement. There is no probability/index cache or new graph family.

`tests/test_experimental_categorical.py` compares returned tokens, dtype/device,
input tensor values/metadata, exceptions including messages, RNG count/state and
callback order against the original. Cases include small exhaustive quarter-grid
laws, exact CDF ties and adjacent floats, one-hot and sparse laws, tiny/subnormal
support, noncontiguous views and vocabulary151936, tolerated sum below1 with zero
tail, invalid inputs/uniforms, and callbacks changing support/shape/dtype.

Real tiny-Qwen CPU sessions compare target-only and full-shadow gamma7 A/B for
all128 outputs/request on original R2/C256 inputs. Every proposal's complete
q/tokens/confidence, every round's actual target p/decisions/work, populated
all-layer target/draft KV, outputs and final RNG bytes must match. Natural
nonuniformQ, R1 tails and rejection/residual paths are retained. A separate real
shadow session with budget2 explicitly covers admission1 followed by remaining1,
because that branch is not guaranteed by the original seeded tiny-model rollout.
The extra short session is CPU correctness evidence, not a changed E2E workload.

Only after CPU correctness and independent review may a separately frozen,
observer-free20-batch target/gamma A/B runner be prepared. Native equivalence and
complete-session measurements remain separate gates. Production default remains
the original sampler until evidence and an explicit integration decision exist.

## Observer-free complete-session runner

`scripts/benchmark_categorical_ab_r2.py` is an opt-in experiment with the same
stdlib `bind`/dry-run and bounded supervisor/worker interface as previous R2 runs.
A directly binds the unchanged original sampler; B directly binds the candidate.
All three imported aliases (`tensor_sampling`, `packed_sampling`,
`packed_target_sampling`) are switched before the batch and restored after it,
including on exceptions. The timed sampler has no wrapper. No coarse/inner
observer, profiler, stage fence or signature wrapper is installed. All original
fresh target signatures, two hot R2 graphs, eager tails, numerical setup checks,
common post-timer semantic collection and complete-session synchronization remain.

Before warmups,15 bounded categorical cases run on the actual device using separate
seeded generators. They cover boundaries, zero tails, CDF rounding overflow,
mutable support/shape/no-positive errors, validation order and the151936-element
native reduction domain. The large sparse case uses a noncontiguous stride2 view.
Each original/candidate result is compared exactly, including actual post-callback
probability tensors, returned token/device/dtype/shape, exceptions and RNG states.
Private `categorical-gate/result.json` stores reconstructible case specifications,
results, source identity and comparison status. The two large A/B probability
pairs are also saved as private `.pt` files with SHA256 references for independent
review; small inputs/results remain inline. A failure retains completed evidence
and prevents every warmup/measurement. CPU gate runs are explicitly labelled
`cpu_emulator_not_native`. They cannot establish native support.

The fixed20 complete batches are targetA,targetB,gammaA,gammaB warmups followed by
four paired blocks per arm. Variant order is AB/BA/BA/AB; arm order alternates
target,gamma then gamma,target. Every batch uses original R2/C256 prompts/seeds and
128 outputs/request. All same-arm semantic dictionaries, including warmups, must
match exactly. Complete raw rows are saved before rejection; any semantic failure
aborts and produces no performance aggregate. Partial/deadline runs are not pooled.

For each arm report four paired wall ratios, four full-session walls per variant,
and pooled1024 outputs divided by summed wall time per variant. The practical
screen passes only if BOTH arms have pooled B/A throughput at least1.02 and B is
strictly faster in at least3 of4 pairs. This is not a statistical confidence bound.
A complete valid miss retains A and records no demonstrated useful E2E gain.
Historical frozen vLLM rates retain their original denominator and limitations.
The same300/290-second supervisor/cooperative bound includes setup and gate;
8GiB prefree,6GiB allocator, two512MiB graph reservations,1280MiB graph budget and
64MiB workspace are unchanged. There is no retry or automatic production rollout.
