# Matched initial32 diagnostic: design only

Status: S49 design proposal following the [six-probe result](../reports/matched-prefix-probes-20261010/README.md).
No runner, model load, GPU execution, new sample or training is part of this turn.
The question is whether the opposite step512→1280 changes seen in two states
extend across the **original32 quality initial states**. This is development-panel
coverage, not a causal explanation, untouched holdout, checkpoint selection or speed test.

## Inputs and present evidence

Use only the original quality ordinal0–31, in original order, at steps512 and1280.
Each state is its exact saved prompt plus its already emitted first target token.
Do not retokenize, regenerate that token, select states by outcome, or extend history.
A local stdlib check of both recovered `run.json`, `private-outputs.jsonl` and
`private-rounds.jsonl` found exact case/seed/prompt/first-token equality for all32,
prompt lengths32–326, and exactly one round0 with seven saved proposals per case
in each checkpoint. Bind original panel, source, result/worker success and raw
round/output hashes as in the six-probe analysis; no later raw round is a quality row.

The historical bindings pin these checkpoint identities:

| Step | Draft weights SHA256 | Metadata SHA256 |
| --- | --- | --- |
| 512 | `667d2dd6e8ad7d11e2115e936af1b6cf7e44357d68e3412f5c9ef433f26690ae` | `30d8813eaa5a4db7a50ddbf164f17b8c045e79ae38434e378aa8e703bc699ddb` |
| 1280 | `d6f21ab3af5187ed181efdbde54118a9ada21b458d9957515a17d6b3693d0de4` | `8bb0c5113d02decac5b8a64c585284dd1664f1a5f88f13014713ec1facc7ecc5` |

They refer to `train-expanded-20261009-796fecc/step-000512` and `step-001280`;
full private paths remain in the original bindings. Both share panel file SHA256
`b5119c7f83c1f6d0ea08b58eb795bad2d1475fb6e4b71ece46ed9ae1fe00ee1e`,
target fingerprint, vocabulary151936, layer IDs `[1,7,14,21,26]`, five draft layers,
block size7, Markov rank256 and mask token151669. Read the complete bound draft
configuration, rather than reconstructing it from these selected fields.

Historical runtime is Torch2.12.0+rocm7.2, Transformers5.17.0, HIP7.2.53211,
AMD Radeon Graphics, native SDPA; target weights/forward BF16, trainable draft
parameters FP32 under BF16 autocast, temperature1 and the original
`float64_softmax_normalize_cdf_v1` adapter with no filtering. This adapter's explicit
normalization is the original definition of a newly computed law; never repair or
renormalize a retained historical law for comparison.

The current `model.py`, `cached_target.py`, `cached_decode.py` and
`tensor_sampling.py` are byte-identical to the recovered a278e5a collector copies.
Root's fresh read-only dependency check, OS exit0, additionally found both remote
checkpoint directories and confirmed the four hashes above. Draft weight files
are each646,774,924 bytes; metadata sizes are4474/4475 bytes. Installed package
metadata reports Torch2.12.0+rocm7.2, Transformers5.17.0, safetensors0.8.0 and
Python3.12.14. Evidence is private
`output/matched-initial32-design-20261010/dependencies.json` and its waited
`dependency-inspect.exit.json`. This check imported no torch and read no target
weights or optimizer state; package metadata is not a loaded runtime/device check.

Remaining preflight gaps are current target/tokenizer fingerprint, actual loaded
HIP/driver/device/backend, deployed source and a coordinated free GPU window.
Before execution, revalidate checkpoint metadata/weights and bind those remaining
identities. Reuse weights-only loading; never load optimizer/resume state or modify
its metadata. An unavailable/mismatched identity is a preflight gap, not permission
to choose another checkpoint/environment. Today's read-only process inventory is
not future GPU readiness; verify again immediately before any separately reviewed run.

## Minimal computation, preserving historical shapes

One target model, one draft instance, one state at a time. Freeze execution order:
512 ordinals0/1 →1280 ordinals0/1 →512 ordinals2–31 →1280 ordinals2–31.
This requires four weights-only loads, with both files rehashed each time; the four
anchor states count toward the panel and are never recomputed. Reset all target/draft
caches between states and after weight replacement. Do not retain projected draft
KV across checkpoints.
Use existing classes and probability adapter directly; no new decode engine,
canonical/packed/varlen backend, tokenizer, generation loop or sampler is needed.

For each checkpoint and each original state, run steps1–5 inside `torch.no_grad()`
with `model.eval()` and `draft.eval()`. Preserve the original BF16 autocast scope
for every draft context projection, backbone, confidence/Markov head and LM head
operation, including the manually computed row0 path; do not rely only on the
`backbone_cached` decorator. Target operations retain their original loaded BF16
dtype and inference scope. No trainable operation may retain an autograd graph.

1. Construct `CachedTarget(model, draft.spec.layer_ids)`. `prefill(ids)` receives
   exactly one unpadded original prompt `[1,N]`, as in `cached_speculative_sample`.
   It returns the prompt context features; target cache length is N. Do not prefill
   prompt+anchor together: that changes the original execution shape.
2. Construct fresh `DraftContextCache(draft)` and, under the original BF16
   autocast, call `append(features.context)` once. Its per-layer projections depend
   on this checkpoint. Assert both cache lengths equal N: the saved anchor is
   excluded from projected context. No generated completion feature is loaded.
3. Set anchor to the saved first target token `[1,1]`. Under the same autocast,
   call `draft.backbone_cached(anchor, cache.layers, N)`, preserving the complete
   anchor+six-mask block `[7,H]`; project **all seven** hidden rows through
   `draft.lm_head`, yielding `[7,V]`. For k=0 retain the original `[1,rank]`
   `markov_embedding(anchor[0])`, original row0 confidence calculation, and
   `base[0:1] + markov_projection(embedding)`, followed by the original FP64
   probability adapter on `[1,V]`, then select row0. Save q0 before any draw; do not call `_sample_stochastic`
   or sample a proposal. Cutting the backbone or LM-head to one row would be a
   different numerical experiment even though only q0 is scored.
4. Append the fixed original `[anchor]+seven_saved_round0_proposals` as `[1,8]`
   to that target and project all eight hidden rows using `target.predict(features)[0]`.
   Pass the complete logits matrix `[8,V]` through the original FP64 probability
   adapter before selecting row0, as in `cached_sampling`; validate all eight laws.
   Retain only row0 logits and FP64 p_block0. These are **historical-suffix block replay**
   laws, not a new accepted trajectory. The suffix differs across checkpoints;
   it is causally future to row0 but can affect BF16 execution numerics. No verifier,
   accept/reject draw, cache commit or later-position quality calculation follows.
5. Release the block target/cache/features. Construct a fresh
   `CachedTarget(model, ())`, prefill the same `[1,N]` prompt, append only the saved
   anchor `[1,1]`, and use `predict(features, last_only=True)[0]` followed by the
   FP64 adapter on that `[V]` vector. This matches `eval_stochastic_gate.compare_probe` row0, including its
   fresh prefill and single-row head. Save p_seq0. Release this cache before the
   next state. The sequential pass must not reuse a cropped block cache.

For every ordinal, freeze **p_ref = newly computed step512-pass p_seq0**. The
step1280-pass p_seq0 is a numerical-repeat control, never a replacement reference.
Both checkpoint q rows are compared with exactly that saved reference vector.
Historical step128 sequential vectors exist for only two states and are not the
reference for this new32-state panel. Proposed work count is64 draft block forwards,
128 target prefills,64 target eight-token appends and64 target one-token appends;
no warmup, retry or hidden repeated forward is included.

## Historical anchors and validity gates

Keep row0 raw confidence as a private compatibility check against its original
round value exactly, preserving its call order before the Markov projection; it is not a
quality metric, calibration fit or admission rule.

The saved probes at ordinals0/1, steps512/1280 can check exact q0, p_block0 and
p_seq0 replay, selected-token gathers, and original row0 probability/logit numerical
controls. They cannot check new draft logits or intermediate features (not saved),
other30 historical full q/p vectors, a dense training q, or later-state behavior.
Saved selected p/q for the other30 are useful narrow gather checks, not full-law
reproduction. Store new and historical comparisons privately without publishing
selected probabilities, tokens, IDs or seeds.

Before starting any ordinal2–31 work, require all four checkpoint/state anchors
to pass against their historical tensors. Check all64 new q0 and p_block0 gathers
against the original raw selected probabilities at each original first proposal
token exactly; this is a narrow replay check, not a full-law quality metric. Proposed strong replay gate: require
byte equality of each of the three FP64 probability vectors, exact retained p
logits/maxima/argmax and selected gathers; original TV and mean logit reductions
retain absolute1e−12 reconciliation. Save differences and raw evidence **before**
evaluating the gate. A lawful but changed vector is numerical drift, not corrupt
probability data; it nevertheless stops this proposed historical-replay experiment
as historical replay numerical mismatch/inconclusive; it does not invalidate the
original evidence. This is a compatibility gate, not a theoretical BF16 equality
guarantee or a quality significance threshold. No tolerance relaxation, favorable reference replacement, retry,
or automatic alternative backend follows. Review this exact gate before a runner
is written; historical BF16 cross-path inequality does not by itself fail it,
because each newly replayed path is compared with its own historical path.

Every new probability row must be CPU-retained FP64 `[151936]`, finite in[0,1],
with mass error≤1e−12. Corresponding target logits are finite BF16 with exact
original shapes. Validate all eight transient block probability rows created by
the `[8,V]` adapter call; only row0 contributes to quality or retained full laws. Record cache lengths and physical forward/head
shapes. Bind input/source hashes before and after; preserve real parent/worker exits.
Any failure ends the attempt, retaining validated partial records. Identity/schema
mismatch or an illegal probability law is an evidence gap; lawful but unequal
selected gathers or raw confidence, like unequal historical full vectors, are
historical numerical mismatch/inconclusive, not raw-data corruption. Coverage or
resource failures retain their explicit partial/gap reason. Partial coverage is
never silently reweighted.

## Report quality and numerical controls separately

For each of32 states and each checkpoint, report full-vocabulary
`O_ref=sum min(p_ref,q)`, `O_seq=sum min(p_seq,q)` and
`O_block=sum min(p_block,q)`. Record masses, L1 and the identity
`O=(mass_p+mass_q-L1)/2`; interpret1−O as rejection probability only under
normalized laws. Preserve mass residuals rather than clipping results.

Report step512→1280 ΔO_ref for every state, its exact sign, and TV(q512,q1280).
Give the equal-state mean of32 overlaps/differences, median and range, and counts
of positive/zero/negative changes, alongside the complete32-row table. No token,
length, acceptance or trajectory weighting; do not hide disagreement behind a mean,
construct a confidence interval from vocabulary entries, or invent a gain threshold.
The panel was previously used for development and is not an independent test set.

Separately report teacher byte equality, TV(p_seq1280,p_ref),
TV(p_block,p_seq) and TV(p_block,p_ref), plus historical replay differences.
For each own/reference overlap shift use
`TV(p,p_ref)+abs(mass_p-mass_ref)/2+1e−12`; for the paired-change shift use the
sum of the two bounds. Recompute the checkpoint-change sign under both saved new
sequential teachers, retaining mixed signs as numerically sensitive. Block laws
remain path controls. Neither q drift nor p drift is itself evidence of improvement.
This design does not equate its common reference with an ideal numerical teacher.

## Bounds, artifacts and stop

Proposed caps, frozen before implementation/execution: one coordinated GPU worker,
≥8GiB free before loading;6GiB PyTorch allocator limit,8GiB host RSS,1GiB additional
private output,300s wall/290s cooperative limit. Also stop if whole-device used VRAM
rises by more than8GiB above a fixed pre-load baseline. This is a conservative
whole-card pressure threshold including ASR/other-process changes, not worker-only
VRAM attribution. Missing telemetry is a preflight gap. Never reset that baseline
or stop ASR/other jobs to bypass the threshold. A supervisor terminates/reaps only
its own worker on a cap; baseline, telemetry and allocation peaks are retained.
These are caps, not predicted use. No automatic
increase after a failure. Keep model/checkpoints in place; no weight transfers or
base-environment changes. Root must first verify a free window and telemetry support.

Retain only q0/p_block0/p_seq0 and row0 target/draft logits per completed state,
plus scalar checks; stream private CPU rows to disk and discard transient tensors.
The192 FP64 probability vectors alone are233,373,696 bytes; row0 logits add well
under100MiB. Do not save all7/8 vocabulary rows, model features or KV caches.
Hash private tensor/scalar artifacts and produce a public ordinal/checkpoint-only
summary, source/runtime/input identity record, validation status and limitations.

Stop after one bounded report, including an inconclusive/gap report. All32 valid
states are required for the equal32 summary. No new trajectory, training, data
addition, loss ablation, final-test read, STS change, checkpoint reselection or speed
claim follows automatically. Implementation and any GPU run remain separate reviewed
steps; this document alone is not a runner or evidence that the live preflight passed.
