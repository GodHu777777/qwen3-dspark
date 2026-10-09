# Frozen step1280 STS: fit44 then eval43

All 44 fit prompts and 43 eval prompts completed on the same frozen source,
checkpoint, selection manifest, panel and sampling protocol. Fit produced
3,157 blocks; eval produced 3,280. The fit
artifact was frozen before eval launched. Worker, launcher and controller OS exits
were all 0 for both groups, with no timeout; both CPU fit and eval exited 0.

On eval43, STS ECE is worse than the unscaled head at positions
**1, 2, 5** and worse than
the fit-only constant at positions
**1, 2, 3, 4, 5**.
STS Brier is worse than the unscaled head at positions **1–6**, while both the
STS and unscaled head have lower Brier than the fit-only constant at all seven
positions.
The complete per-position results are retained below; fitting ECE is not presented
as held-out performance. A near-constant predictor can have low ECE; this alone
does not show that the head is useless. Lower head Brier supports probability
prediction on this panel, without establishing scheduler benefit. The original
frozen STS remains the paper-method experimental branch, with unscaled and
fit-only constants retained as baselines. These eval results are not used to
change the checkpoint, temperature grid, sampler or admission policy.

## Frozen fitting procedure

The previously selected step1280 checkpoint was collected with temperature 1,
no filtering, full proposals up to seven tokens, a 128-token output budget,
native BF16/SDPA target forward and actual float64 q. Source commit is
`034064bf8fea7f67c9039c2dd103b65ca12e803a`. Original manifest cases and seeds remain
unchanged; fit and eval prompt IDs and token-prompt hashes are disjoint. Quality32
is not a calibration population; final test remains locked.

CPU STS uses the predeclared 61-point grid `2**(i/10), i=-30..30`, 20 equal-width
bins, the documented endpoint clamp, and sequential minimization of cumulative
prefix ECE while earlier temperatures stay frozen. Selected temperatures are
`0.933032992, 0.812252396, 0.812252396, 0.659753955, 0.659753955, 5.65685425, 1.51571657`. They are frozen before eval,
and eval performs no temperature search, checkpoint choice or prevalence fitting.

The comparison constant for position j is the cumulative-prefix acceptance rate
at that position estimated only from fit44 effective labels. It is applied
unchanged to eval43; it is not a product of conditional prevalence estimates.
An unsupported fit position would remain explicitly unavailable rather than being
filled with zero or eval prevalence. All methods here use the same effective
label population and 20-bin metric definition.

## Prompt-held-out evaluation

| Prefix position | Unscaled ECE | Frozen STS ECE | Fit-only constant ECE |
| --- | ---: | ---: | ---: |
| 1 | 0.064531 | 0.065574 | 0.025194 |
| 2 | 0.032885 | 0.034139 | 0.015349 |
| 3 | 0.025359 | 0.023183 | 0.015649 |
| 4 | 0.012696 | 0.011781 | 0.005735 |
| 5 | 0.006182 | 0.007553 | 0.007272 |
| 6 | 0.003416 | 0.003276 | 0.004826 |
| 7 | 0.002097 | 0.001255 | 0.003084 |

| Prefix position | Unscaled Brier | Frozen STS Brier | Fit-only constant Brier |
| --- | ---: | ---: | ---: |
| 1 | 0.159953 | 0.160217 | 0.221589 |
| 2 | 0.083799 | 0.084626 | 0.115460 |
| 3 | 0.039423 | 0.040300 | 0.050194 |
| 4 | 0.018455 | 0.019060 | 0.024082 |
| 5 | 0.009345 | 0.009751 | 0.011232 |
| 6 | 0.005385 | 0.005542 | 0.006002 |
| 7 | 0.002766 | 0.002752 | 0.002872 |

| Prefix position | Fit events / labels | Eval events / labels | Eval block coverage | Eval prompt coverage | Fit-only constant |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1120 / 3157 | 1081 / 3280 | 100.00% | 43 / 43 | 0.354767 |
| 2 | 465 / 3137 | 432 / 3251 | 99.12% | 43 / 43 | 0.148231 |
| 3 | 213 / 3115 | 170 / 3224 | 98.29% | 42 / 43 | 0.068379 |
| 4 | 94 / 3093 | 79 / 3204 | 97.68% | 42 / 43 | 0.030391 |
| 5 | 57 / 3068 | 36 / 3184 | 97.07% | 42 / 43 | 0.018579 |
| 6 | 33 / 3044 | 19 / 3159 | 96.31% | 42 / 43 | 0.010841 |
| 7 | 18 / 3023 | 9 / 3135 | 95.58% | 42 / 43 | 0.005954 |

Count refers to observed effective cumulative-prefix labels. Rejection-tail
positions that were verified remain zero labels; accepted EOS includes itself
and removes later positions. Unproposed budget tails are absent. Fit had
0 zero-block prompts and eval had 0;
all were retained in completed prompt counts and coverage. No fake negative
blocks were inserted. Event counts, especially the final positions, describe
sparse positives; rounds within a prompt are correlated and no independent-round
confidence interval or broad calibration guarantee is asserted.

| Method | Effective labels | Count-weighted mean position ECE | Label-weighted Brier |
| --- | ---: | ---: | ---: |
| unscaled | 22,437 | 0.021307 | 0.046296 |
| sts | 22,437 | 0.021254 | 0.046745 |
| fit_prevalence_constant | 22,437 | 0.011117 | 0.062615 |

The ECE column is an average of per-position ECEs weighted by label counts, not
ECE recomputed after pooling positions into common bins. Full bin counts,
predicted/observed means and coverage remain in [eval-metrics.json](eval-metrics.json).
[fit-metrics.json](fit-metrics.json) is explicitly fitting-population evidence.

## Execution and independent checks

Fit proposed/verified 21,657 positions with
21,637 effective labels; eval proposed/verified
22,465 with 22,437 effective labels.
The two groups retained 5,181/5,128 output tokens
and 8/6 EOS prompts respectively. Each GPU group
had a fresh live process/KFD/ASR/VRAM check and a hard 1200-second bound. All four
owned PIDs per group were gone afterward, KFD was held only by preserved ASR,
and ASR remained ready/not busy. No retry or partial fit occurred. Peak allocated
memory was 2,381,026,304 bytes for fit and
2,474,884,608 bytes for eval.

Before and after each group, archive/source, target, training config, selection,
checkpoint and panel hashes were verified. The collector launcher waited the
worker's OS exit, the controller waited the launcher's, and an outer shell
captured the controller's actual OS exit. The shell itself is not reported as a
separately waited exit. The GPU groups and CPU fit/eval are not timing benchmarks.

[verify_sts.py](verify_sts.py), SHA256
`63846cd43301168ac852201fd9a4dc4ac243c52d7ed59459c69babecb7d9c76f`, verifies all 223
archived files and execution evidence, rechecks full raw collection counts/labels,
confirms prompt disjointness and frozen-artifact ordering, and independently
recomputes sigmoid/cumprod, fit prevalence, ECE and Brier. It requires the private
run directory; prompts, token traces and raw artifacts are not published here.
[verification.json](verification.json) records the passing scalar results.

Frozen fit artifact file SHA256:
`77dc61e54e82c4dbfda302743efe5d126584efd2c501e02fa3738737d99111e3`.
Eval report file SHA256:
`a3149fb6c34bd414f5a8724906cb6097eb4ad8c1865f7b695fbadfd52d027854`.
[source-identity.json](source-identity.json) retains all public source, selection,
checkpoint and artifact fingerprints. Neither private artifact is included in this
report. [runtime.json](runtime.json) and [collection-audit.json](collection-audit.json)
retain scalar resource and group evidence.

Eval43 is prompt-held-out from STS fit only: all 119 development prompts were
already teacher-forced monitored. Native block/sequential probability differences
from prior quality probes remain unresolved. These measurements do not establish
answer quality, distribution losslessness, causal scheduler integration, engine
overlap or serving speed. The historical quality runs remain bound to their own
`a278e5a` source. No further GPU workload was started after eval43.
