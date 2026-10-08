# Expanded training: authorized step1024 → step1280 segment

The strict resume completed at step1280 with training-child and runner OS exit 0,
no timeout. The planned max_steps1280 training schedule is complete. All 62 optimizer states and the
resume step are 1280; both checkpoint files match their recorded SHA256. Frozen
source `796fecc0d9142c1a733ad6df47b105fab4ee9f05`, all 112 archived files, config
bytes and checkpoint/run identity remain unchanged. The first 1024 metric records
match the earlier evidence byte-for-byte, and steps 1–1280 are continuous/finite.
The resumed initial dev result equals the step1024 final result exactly.

| Dev metric | Step1024 | Step1280 |
| --- | ---: | ---: |
| Loss | 2.064108 | 2.029412 |
| CE | 4.370612 | 4.252141 |
| Teacher-forced distribution overlap | 0.353486 | 0.373430 |
| Confidence MAE | 0.208603 | 0.214376 |

Validation covers all 119 eligible dev rows with the same evaluation anchor seed,
weighted within rows and then macro averaged. Training has 932 eligible rows and
424,267 exact input tokens. The cumulative 10,240 microsteps correspond to
10.987124 train traversals. These teacher-forced values do not replace the
separate fixed-quality32 rollout panel or establish answer quality or speedup.
Confidence targets evolve with the draft: constant-zero MAE at this step is
0.373430, versus learned MAE 0.214376;
this alone does not establish calibrated confidence. Constant-mean/median
baselines and real rollout calibration remain unmeasured.

Optimizer-loop elapsed (including Python/logging) was
160.130239s. Training-result elapsed was
176.858068s, including final validation and checkpoint saving
after initial validation. Child-process wall time was 209.207282s,
including startup, checks, loading/resume, both validations and saving. Peak
allocated memory was 5,632,856,064 bytes. These are execution
and resource measurements, not serving performance. The runner waited the
training child's OS exit; the outer shell captured the runner's OS exit. The
shell wrapper itself is not reported as a separately waited third exit.

The original max_steps1280 training schedule is now complete. This does not
freeze checkpoint selection or complete the overall project. No quality or STS
job was started by this controller. Final test remains locked; fixed-panel
quality, STS fit/eval and performance gates remain separate.

`summary.json` records scalar results and timing scopes;
`checkpoint-verification.json` preserves strict checks and hashes;
`source-identity.json` preserves frozen source/data/config/runtime identity;
`training-metrics.jsonl` exports only steps 1025–1280. Earlier step32/128/512/1024 and
quality reports are unchanged. Private machine paths, samples, process logs,
weights and optimizer state remain outside the public report.
