# Expanded training: authorized step512 → step1024 segment

The strict resume completed at step1024 with training-child and runner OS exit 0,
no timeout, and no automatic continuation to 1280. All 62 optimizer states and the
resume step are 1024; both checkpoint files match their recorded SHA256. Frozen
source `796fecc0d9142c1a733ad6df47b105fab4ee9f05`, all 112 archived files, config
bytes and checkpoint/run identity remain unchanged. The first 512 metric records
match the earlier evidence byte-for-byte, and steps 1–1024 are continuous/finite.
The resumed initial dev result equals the step512 final result exactly.

| Dev metric | Step512 | Step1024 |
| --- | ---: | ---: |
| Loss | 2.191069 | 2.064108 |
| CE | 4.807052 | 4.370612 |
| Teacher-forced distribution overlap | 0.287295 | 0.353486 |
| Confidence MAE | 0.184832 | 0.208603 |

Validation covers all 119 eligible dev rows with the same evaluation anchor seed,
weighted within rows and then macro averaged. Training has 932 eligible rows and
424,267 exact input tokens. The cumulative 8192 microsteps correspond to
8.789700 train traversals. These teacher-forced values do not replace the
separate fixed-quality32 rollout panel or establish answer quality or speedup.
Confidence targets evolve with the draft: constant-zero MAE at this step is
0.353486, versus learned MAE 0.208603;
this alone does not establish calibrated confidence. Constant-mean/median
baselines and real rollout calibration remain unmeasured.

Optimizer-loop elapsed (including Python/logging) was
323.608606s. Training-result elapsed was
338.225756s, including final validation and checkpoint saving
after initial validation. Child-process wall time was 368.694960s,
including startup, checks, loading/resume, both validations and saving. Peak
allocated memory was 5,632,856,064 bytes. These are execution
and resource measurements, not serving performance. The runner waited the
training child's OS exit; the outer shell captured the runner's OS exit. The
shell wrapper itself is not reported as a separately waited third exit.

The original configuration still has max_steps 1280; the current intermediate
checkpoint does not complete that plan or freeze checkpoint selection. No 1280
segment was started by this controller. Final test remains locked; STS fit/eval,
quality selection and performance gates remain separate.

`summary.json` records scalar results and timing scopes;
`checkpoint-verification.json` preserves strict checks and hashes;
`source-identity.json` preserves frozen source/data/config/runtime identity;
`training-metrics.jsonl` exports only steps 513–1024. Earlier step32/128/512 and
quality reports are unchanged. Private machine paths, samples, process logs,
weights and optimizer state remain outside the public report.
