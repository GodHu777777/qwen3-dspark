# Matched-prefix first-position diagnostic: design only

Status: proposed next CPU evidence step after S47, not an implemented or executed
experiment. This document is based on source at `b40f8a58bc589a9af5735167c93dec12e0c2af3b`
and the original quality collector snapshot `a278e5a`. No probe transfer, tensor
load, model forward, runner, training, generation or final-test access occurred
while preparing it. No production default changes are proposed.

The smallest useful next step is to recover **six existing full-probability probe
files: original quality ordinals0 and1, round0, at steps128,512,1280**, and compare
only their first probability rows on CPU. Do not start a new32-state forward grid,
a121-state progress panel or a seven-position exposure experiment. The two
ordinals were selected for numerical probes before the historical rollouts, not
from S47 acceptance outcomes. They are a diagnostic of two states, not a sample
large enough to represent quality32.

## Established facts and the remaining evidence gap

[S47](../reports/natural-first-risk-20261009/README.md) reconstructed8,158 valid
first-position selected pairs. At1280, pooled alpha was0.372736 and26/32 prompt
means were below0.5. Selected alpha at one sampled token is not the state quantity
`sum_v min(p(v),q(v))`. Public teacher-forced overlap0.373430 is also a different
population, position weighting and numerical law; the similar numbers do not
establish a small training-versus-rollout gap.

The recovered32 prompt identities, exact prompt tokens and seeds match across
all three checkpoints. Root separately checked that all32 initial target-output
tokens match as well. See the [public initial-state identity check](../reports/matched-prefix-design-20261010/initial-state-identity.json)
and its [scope record](../reports/matched-prefix-design-20261010/README.md).
Consequently the two proposed round0 states have the same semantic prefix across
checkpoints: original prompt followed by the same one initial target token.
This fact does not establish equality of the associated floating-point laws.

The original collector records both selected probes as collected. Its source
saved actual full q, block p and subsequent fresh-sequential p. Those tensors
were not in S47's36 recovered files, and root's local inventory found no matching
`.pt` payloads. Their current remote existence, exact byte counts, completeness
and hashes remain unverified. Metadata saying “collected” does not replace that
check. Recovery failure is an evidence gap, not permission to regenerate them.

What the current scalar evidence cannot answer:

- Full-vocabulary first-position overlap or total variation at any of the8,158
  states; selected p/q cannot reconstruct unselected support.
- Whether a later draft checkpoint is better on exactly the same full state and
  target law; S47's later-round prefixes and horizons differ.
- Whether target numerical-path variation changes the apparent draft comparison.
- Whether a training objective, training-data composition, capacity, insufficient
  optimization or generated-prefix distribution causes the remaining mismatch.
  Even the six proposed probes cannot attribute these mechanisms.

## First q and training alignment: inspected source

These model/loss/target/train/data/cache files in the current checkout are
byte-identical to their copies in the recovered historical collector source.
Line references below identify the inspected symbols; historical collector line
numbers are stated separately where its later interface changed.

| Component | Concrete source evidence | Implication for position0 |
|---|---|---|
| Training tokens and anchors | `data.py:6–18`, `tensors:24–27`, `select_anchors:30–36` | Audited prompt+completion+EOS; no partial truncation. Anchor a is selected so token a+1 is an assistant token. The expanded config uses32 anchors/sequence, sampled without replacement, then sorted. |
| Target features | `target.py:FrozenTarget.capture:18–36` | Concatenated outputs of selected decoder blocks feed the draft; final-normalized hidden states feed target supervision. Hooks use zero-based block indices. Target is frozen. |
| Draft visibility | `model.py:block_mask:12–19`, `DSparkDraft.backbone:101–119` | Context positions j<a are visible; each block contains anchor token at position0 and six mask tokens, with bidirectional attention within that synthetic block. Actual future completion tokens are not supplied to its backbone. |
| Training prediction/previous token | `model.py:DSparkDraft.forward:121–132` | Block position k predicts token a+k+1. Markov previous is recorded token a+k. At k=0 it is exactly the anchor token. |
| Teacher alignment | `train.py:forward_loss:19–28` | Teacher hidden at `label_positions−1` predicts the same label. At k=0 this is h[a] predicting x[a+1], not h[a−1]. |
| Cached state | `cached_sampling.py:110–137`, `cached_decode.py:DraftContextCache.append:26–35` | Both caches exclude the newest emitted anchor. Initial context is the prompt; only committed target feature chunks enter the draft cache thereafter. |
| Cached backbone | `model.py:project_context_kv:140–146`, `backbone_cached:149–164` | Draft context projection is checkpoint-dependent. It uses the same learned projections with positions strictly before the anchor; the only current block again contains anchor+six masks. |
| Actual sampled first q | `model.py:propose_stochastic_cached:172–191`, `_sample_stochastic:193–208` | One full block backbone and base LM head are computed; k=0 adds `markov_projection(markov_embedding(anchor))`, then the FP64 probability adapter. q0 is captured before drawing y0. |
| Actual first p | `cached_sampling.py:127–130`, `cached_target.py:append:62–88`, `predict:115–119` | Target appends anchor+all proposed tokens. The anchor row predicts proposal token0; its full p0 is a block-verification law. Future proposal tokens do not belong to its semantic causal prefix, although execution shape can change numerics. |

The first proposal logits are `lm_head(h0) + markov_projection(markov_embedding(anchor))`.
Embedding/LM head alias frozen target modules (`model.py:98–99`); h0 depends on
the draft checkpoint. Confidence does not change q or admission in this collector.

**There is no direct teacher-versus-sampled previous-token substitution at k=0.**
Both paths use the same anchor. That within-block Markov substitution starts at
k≥1, where training uses the recorded preceding token and rollout uses the
preceding draft draw. Generated-history distribution shift can still affect the
anchor/context, but it is a different hypothesis. The initial matched states
also share prompt context, avoiding the later-round committed-history mixture.
Logical alignment does not guarantee BF16 dense/cache numerical equality.

The expanded training objective is0.1 CE +0.9 probability L1 +1.0 confidence BCE
(`configs/train-expanded.example.json:19–23`; `losses.py:6–27`). Valid assistant
positions have weights exp(−k/4), normalized within the selected blocks of each
example. Teacher and draft logits are converted to FP32 for softmax/L1, and
confidence targets are detached `clamp(1−L1/2,0,1)`. CE is on the recorded next
token. Confidence loss can still send gradients into shared hidden/Markov
features; detached targets do not mean confidence training is isolated from q.
`train.py:109–126` averages accumulated microstep gradients; `evaluate:31–41`
macro-averages per-example metrics. All119 dev rows were monitored during the
completed training schedule; these probes are not an untouched holdout.

Runtime laws instead use FP64 temperature scaling, softmax and an explicit final
normalization (`tensor_sampling.py:37–48`), retaining BF16 forward logits. This
arithmetic distinction and population/position weighting prevent equating the
reported training overlap with S47's sampled alpha. No new objective ablation is
part of the proposed CPU probe analysis.

## Proposed next-turn recovery and validity gate

Recover only `private-probe-case0-round0.pt` and
`private-probe-case1-round0.pt` from each of the three original bound collection
directories. Resolve directories through the already audited run/deployment
provenance, never a search for a similarly named new run. Preserve existing
files; save a fresh private recovery inventory with source path, byte count,
current SHA256 and waited transfer OS exit. Verify transferred hashes. As with
S47, current raw hashes are not historical precommitted payload hashes.

The historical `collect_rollout.py:150–156` saves `actual_q`, `block_probs`,
`block_logits`, `proposal_tokens`. Its `eval_stochastic_gate.py:compare_probe`,
lines297–329, adds `sequential_probs`, `sequential_logits` and
`semantic_committed_output_prefix`. The sequential route prefills the same
prompt, appends the saved committed output prefix, then scores row0 before any
proposed token is appended. Later rows condition on checkpoint-specific proposal
tokens and are deliberately excluded from quality comparison.

A future CPU-only process would use device visibility disabled,
`torch.load(map_location='cpu', weights_only=True)`, and no model/tokenizer/checkpoint
loading. Require all six artifacts; do not substitute a third prompt or another
round. Validate each against its run binding, original source hashes, checkpoint
identity, case ordinal, full raw round, output prefix and published probe metrics:

1. Original protocol, prompt identity and seed are unchanged; the selected probe
   is case0 or1 at round0, with emitted-before=1 and full proposal length7.
2. Proposal token vector exactly equals the raw round; saved committed output
   prefix equals the original output's first token. Prefix tokens must match
   across checkpoints, rather than relying only on a prefix hash.
3. `actual_q` has shape `[7,151936]`; `block_probs` has `[8,151936]`;
   `sequential_probs` contains all8 rows of that vocabulary. Require FP64,
   finite entries in[0,1] and sum error≤1e−12 for each saved probability row;
   no epsilon, clipping, re-softmaxing or renormalization. Check corresponding
   logits/token tensor shapes and completed sequential payload, not just row0.
4. Gathering row0 at that checkpoint's original first proposal token exactly
   reproduces its saved selected_q[0] and selected_p[0]; match raw confidence/
   labels only where those fields exist, without claiming absent q logits were
   saved. Actual q and saved probability rows are authoritative.
5. Recompute original numerical-probe control metrics from saved tensors,
   including row0 TV/max-probability/logit differences/argmax agreement.
   No later-position quality analysis is added. Predeclare absolute tolerance1e−12
   for FP64 probability reductions and mean absolute logit-difference reductions;
   require exact max differences, argmax booleans, shapes and selected gathers
   from the retained elements. CPU/device reduction differences are preserved,
   not grounds to relax checks after inspection.

A missing file, incomplete payload, wrong prefix/binding, invalid law or failed
exact gather is an evidence gap. Unequal valid p vectors across numerical paths
are instead a measured control result, not a missing-data failure. No transfer
or validation listed here has been performed in this design turn.

## Prespecified row0 comparisons and numerical controls

For each of the two states independently, let q_c, p_block,c and p_seq,c be the
saved first rows for checkpoint c∈{128,512,1280}. Fix **p_ref=p_seq,128** before
looking at full-probe outcomes. It is the earliest checkpoint's already saved
sequential law, not a newly calculated ideal target or a reference chosen for
the best result.

Report the following six-state rows and adjacent checkpoint differences:

- `O_block,c = sum_v min(p_block,c(v), q_c(v))` and
  `O_seq,c = sum_v min(p_seq,c(v), q_c(v))` retain each saved native pairing.
- `O_ref,c = sum_v min(p_ref(v), q_c(v))` compares draft laws against one
  identical target vector on that fixed state. For normalized laws,1−O is the
  rejection probability under a q draw; stored-law mass errors remain explicit.
  Cross-check `O=(sum(p)+sum(q)−L1)/2` and record the mass residual
  `O−(1−L1/2)=(sum(p)+sum(q))/2−1`. Do not equate this expected acceptance
  with the single historical accepted/rejected outcome.
- Record exact byte-equality of all p_seq rows, TV(p_seq,c,p_ref),
  TV(p_block,c,p_seq,c), TV(p_block,c,p_ref), and adjacent TV(q_c,q_previous).
  A q change is not itself an improvement. Keep the saved sampled-alpha result
  next to full overlap only as a single-draw observation.

For normalized probability vectors,
`|O(p,q)−O(p_ref,q)| ≤ TV(p,p_ref)`; the change between two own-pair overlaps
versus their common-reference change is bounded by the sum of the corresponding
two TV terms. For stored rows accepted within1e−12 of unit mass, report their
actual mass errors and include the small mass-defect correction
`|sum(p)−sum(p_ref)|/2` in each bound, plus the declared reduction reconciliation
tolerance. Do not silently assume mathematically exact mass1 from a tolerance.

If sequential p is byte-identical across checkpoints, the shared-teacher control
is direct. If not, O_ref remains a defined fixed-vector comparison, while the
own-pair result retains numerical-path uncertainty. Check the sign of adjacent
q-overlap differences under each already saved same-state p_seq reference,
without selecting a favorable reference or adding new forward paths. Mixed signs
are **numerically sensitive**, not evidence of a robust native improvement.
Report per-state values even when the two states disagree; an optional two-state
equal mean cannot conceal the disagreement. There is no invented quality-gain
threshold or statistical-confidence claim.

## What would follow from the result

| Observation after validity checks | Defensible conclusion | Still unresolved |
|---|---|---|
| Both states improve under p_ref and the saved-teacher sensitivity controls agree | Later q fits these two fixed initial states better, separating this comparison from a changed rollout-prefix population | Generality across32 prompts, later progress, source of residual mismatch, and any training intervention |
| Common-law improvement is absent or states disagree | S47's aggregate trajectory improvement is not demonstrated as uniform improvement on these two states | It does not refute S47 or prove that trajectory composition caused its change |
| Own-pair and common-law comparisons differ within measured target-law variation, or sensitivity signs flip | Numerical-path choice limits the quality inference | Whether a new controlled model experiment would resolve it; no automatic dense/cache grid |
| All overlap values remain low | A full-vocabulary mismatch exists on those valid state/law pairs | Data mismatch versus capacity/objective/optimization/exposure bias; confidence retuning is not established as a remedy |
| Probe recovery or identity checks fail | The smallest existing-evidence route is blocked; report the exact gap | This protocol contains no replacement generation, wider state selection or new forward |

This phase cannot compare training-prefix and rollout-prefix distributions,
compute per-loss gradient conflicts, verify dense-versus-cached draft equality,
or diagnose k≥1 Markov exposure. Those require different evidence and a separately bounded
design/review. Only after this narrow result should a further bounded hypothesis be
designed; no automatic training extension, data addition, policy tuning,
checkpoint reselection or throughput experiment follows.

## Resource bound, outputs and stop

Proposed allowance, to be frozen before any later execution: six exact files,
≤64MiB each and≤384MiB transferred, a single CPU analysis process with at most
one full payload loaded at a time,≤2GiB RSS,≤256MiB additional private output,
and300s wall/290s cooperative analysis limit. These are conservative hard caps,
not measured file sizes or runtime predictions. The schema suggests roughly
197MB total payload, but actual sizes must be checked before transfer; an
oversized/missing artifact stops preparation without increasing the cap. No
GPU/model process, shared generation-file read or final-test access is needed.

Retain only required row0 copies after each file is validated; at most six q rows
and twelve p rows persist, with scalar provenance records. Preserve original
payloads and exact before/after input hashes. Public output contains ordinal0/1,
checkpoint, full-law metrics, reference/control labels, validation status and
limits. Tokens, IDs, seeds, selected probabilities and full tensors remain
private. Record real exits and incomplete evidence; do not rerun with relaxed
checks or replace a failed case. End after one report, including a gap report.

Existing context: [S47](../reports/natural-first-risk-20261009/README.md),
[quality128](../reports/expanded-quality128-20261009/README.md),
[quality512](../reports/expanded-quality512-20261009/README.md),
[quality1280](../reports/expanded-quality1280-20261009/README.md),
[completed training](../reports/expanded-training-step1280-20261009/README.md),
[collector contract](rollout-collection.md). The numerical-law limitations and
failed sampler-optimization branch remain unchanged.
