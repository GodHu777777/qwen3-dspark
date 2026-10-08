# Expanded training: authorized step128 → step512 segment

This separate report preserves the earlier [step32/128 report](../expanded-training-20261009/README.md) unchanged. The immutable training source remains `796fecc0d9142c1a733ad6df47b105fab4ee9f05`; subsequent packed/varlen development is outside that training source. The fixed configuration still specifies 1280 maximum steps; this authorized segment stopped at 512, with no automatic continuation.

| Check / metric | Result |
| --- | --- |
| Process exit / timeout | 0 / false |
| Strict checkpoint checks | Both file hashes, identity, resume step and all 62 optimizer states verified at 512 |
| Source / config | 112 source files and config bytes unchanged |
| Dev loss, step128 → step512 | 2.525809 → 2.191069 |
| Dev CE | 6.329202 → 4.807052 |
| Teacher-forced distribution overlap | 0.093599 → 0.287295 |
| Confidence MAE | 0.098902 → 0.184832 |
| Cumulative train traversals | 4096 / 932 = 4.394850 |
| Peak allocated memory | 5,632,856,064 bytes |

Validation covers all 119 eligible dev rows, fixed evaluation anchor seed, weighted within each row then macro averaged across rows. The resumed initial validation exactly equals step128 final validation. The 932 train rows contain 424,267 exact input tokens. Teacher-forced overlap is not measured speculative rollout acceptance, output quality or speedup; final test remains outside these evaluations.

Confidence targets change with the draft. The rise in raw MAE alone therefore does not establish worsening confidence. At step512 the constant-zero baseline MAE equals mean overlap (0.287295), versus learned MAE 0.184832. Constant-mean/median baselines and real rollout calibration have not been measured.

Timing boundaries are distinct: the optimizer loop including Python/logging took 240.048189 seconds; training result elapsed was 256.275666 seconds, including final validation and checkpoint saving after initial validation; the child process wall time was 289.109379 seconds, including startup, identity checks, loading/resume and both validations. These are execution timings, not serving benchmarks.

`summary.json` holds scalar results and timing definitions; `checkpoint-verification.json` holds strict integrity checks and checkpoint hashes; `source-identity.json` binds the immutable source, config, model and development data. `training-metrics.jsonl` contains only steps129–512. Private process logs, machine paths, generated text, weights and optimizer state are excluded. No checkpoint selection, calibrated scheduling or BF16 losslessness is claimed.
