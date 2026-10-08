# Experimental gates

The project asks: for a frozen Qwen3 target, when does parallel speculative
execution commit more exact greedy tokens per second than cached target-only
decoding? Confidence and resource scheduling must improve that measured decision.
An implementation milestone does not by itself establish a performance gain.

## 1. Learnability before scale

Keep the original NeMo architecture as the reference configuration. First train
on a tiny fixed set and inspect per-position top-1 agreement, CE/L1, confidence,
and actual consecutive accepted draft tokens. Use fixed anchors for diagnosis.
A proposed diagnostic target is >=90% first-position top-1 on fixed training
anchors and mean >=3 accepted draft tokens on the training prompts. Failure
triggers alignment/optimization investigation. Passing says nothing about
held-out quality. Target-greedy trajectories avoid confusing a sampled token
with the teacher's greedy choice.

## 2. Held-out quality

Split normalized prompt identities before generation. Prevent overlap with the
pilot and retain regeneration provenance. Dev is available for checkpoint and
policy selection; final test stays locked until those choices are fixed.
Report prompt-level confidence intervals, accepted length distributions, first
position agreement, and EOS/length-capped cases. Approximately 100 independent
test prompts and thousands of output tokens are a useful first measurement,
not a guarantee of broad coverage. Inspect truncation and prompt-length bias.

Do not select solely on total training loss. In the pilot, overlap declined
between steps 4 and 8 despite decreasing loss. Teacher-forced probability overlap
is neither greedy rollout acceptance nor speedup. Confidence must beat constant
baselines and be evaluated on rollout prefixes before controlling verification.

## 3. Cached execution and a cost bound

Implement incremental target and draft context caches. Test rejection/crop,
all-accepted bonus, EOS and length limits against reference greedy token output.
Keep attention position and cache-length invariants explicit. The full-recompute
reference must remain available as a correctness oracle.

Measure fixed draft lengths 1 through 7 and cached target-only. For draft length
k, accepted prefix A and per-token cached target cost t, the necessary break-even
condition is:

    E[A + 1] * t > draft_cost + verify_cost + control_and_cache_cost

Handle EOS and output-budget truncation separately. Even perfect acceptance
cannot help if `(k + 1) * t` is below the measured round cost. The 161M draft is
large relative to a 0.6B target; approximately 78M parameters are in the two
rank-256 Markov matrices. This is a testable cost constraint, not proof that the
architecture must fail. Preserve the reference result before any rank/layer ablation.

## 4. Confidence and hardware-aware scheduling

Do not skip target verification of committed tokens based solely on confidence.
The present head learns distribution overlap, while the deployed reference is
greedy. Measure reliability and Brier score for the actual acceptance event;
check conditional prefix survival instead of multiplying uncalibrated scores.

Compare target-only, fixed 7, the dev-selected best fixed length, a simple cost
lookup policy, and any learned policy on untouched test prompts. The controller
must be allowed to turn speculation off. Vary context length and batch/load;
state a latency or throughput objective before tuning resources. AMD/NVIDIA
comparisons require actual measurements on both devices, never extrapolation.

## Reporting and resume-worthy claims

Use the same target, precision, prompts, output lengths and decoding semantics;
include warmup, synchronized timing, repeated interleaved trials, prefill/decode/
end-to-end latency, p50/p95, throughput and peak memory. Bootstrap at prompt level.
A practical target is >=1.1x median end-to-end speedup with interval lower bound
above 1, but publish a well-supported negative result if the measurements fail.

Keep source commit, configuration, data hashes, runtime versions, raw aggregates
and reproduction commands with each experiment. Commit verified milestones and
push without rewriting existing history. Do not publish generated examples,
credentials, weights or optimizer states by accident. A CV description must use
only measured claims; current evidence supports a working training/correctness
prototype, not a fast inference engine.
