# Expanded step1280: final planned checkpoint quality32 comparison

All 32 frozen quality prompts completed with worker, launcher and controller OS
exit 0 and no timeout. Accepted draft tokens per verification round changed from
**0.4958 at step512 to 0.6930 at step1280**,
with 28 prompts improving, 4 declining and 0 unchanged.
The original max_steps1280 training schedule is complete. This is native-protocol
rollout acceptance evidence, not output-quality, speedup or losslessness evidence.
It does not automatically select a final checkpoint or initiate STS collection.

| Quantity | Step128 | Step512 | Step1280 |
| --- | ---: | ---: | ---: |
| Completed prompts | 32 | 32 | 32 |
| Output tokens | 3,874 | 3,941 | 3,856 |
| EOS prompts | 5 | 4 | 5 |
| Verification rounds | 3,268 | 2,620 | 2,270 |
| Proposed positions | 22,415 | 17,960 | 15,563 |
| Target-verified proposal positions | 22,415 | 17,960 | 15,563 |
| Attempted acceptance uniforms | 3,842 | 3,901 | 3,815 |
| Effective cumulative-prefix labels | 22,415 | 17,955 | 15,546 |
| Accepted draft tokens | 581 | 1,299 | 1,573 |
| Accepted draft tokens / round | 0.177785 | 0.495802 | 0.692952 |
| Actual committed tokens / round | 1.175643 | 1.491985 | 1.684581 |
| Verification query rows, excluding prefill | 25,683 | 20,580 | 17,833 |

Temperature 1, no filtering, full proposals up to block 7, 128-token output budget,
native BF16/SDPA and actual float64 q remain unchanged. All 30 worker modules,
panel bytes, full manifest, 32 ordinal cases and seeds, draft configuration and
target/data fingerprints match both earlier collections. Fresh stdlib dry-run
comparison allowed exactly checkpoint path, weight hash, metadata hash, manifest
path and binding hash changes. A heavyweight-import blocker also passed before
execution. Source archive contains 161 verified files from commit
`a278e5a5d7d74a1f77f8dcd475700959ee158cb0`.

Against step512, macro mean of prompt-level accepted draft tokens per round changed
from 0.528944 to
0.749224, and median paired
difference was 0.200878. Accepted token
counts ranged from 8 to 79.
EOS outcome changed on 3 prompts. Same prompts and
seeds do not imply equal generated prefixes or output horizons across checkpoints.
Rounds are correlated; no independent-round confidence interval or speedup ratio
is inferred. [per-prompt.json](per-prompt.json) preserves every ordinal case and
comparisons against both step128 and step512, including declines.

| Prefix position (one-based) | Step128 events / labels | Step512 events / labels | Step1280 events / labels | Step1280 rate |
| --- | ---: | ---: | ---: | ---: |
| 1 | 519 / 3268 | 806 / 2620 | 843 / 2270 | 37.1366% |
| 2 | 58 / 3247 | 291 / 2602 | 395 / 2257 | 17.5011% |
| 3 | 4 / 3223 | 118 / 2580 | 184 / 2235 | 8.2327% |
| 4 | 0 / 3202 | 44 / 2563 | 84 / 2221 | 3.7821% |
| 5 | 0 / 3181 | 20 / 2547 | 41 / 2205 | 1.8594% |
| 6 | 0 / 3159 | 12 / 2529 | 17 / 2188 | 0.7770% |
| 7 | 0 / 3135 | 8 / 2514 | 9 / 2170 | 0.4147% |

Accepted-prefix-length histogram for lengths 0 through 7 is `1427, 448, 211, 100, 43, 24, 8, 9`.
There were 9 fully accepted seven-token blocks
out of 2174 seven-token proposals.
24 blocks accepted the whole proposal;
15 of these reached the output budget and emitted no bonus draw.
The run made 9 bonus draws and 2242 residual draws.

EOS ended 5 prompts: 4 accepted
draft EOS, 1 residual EOS and 0
bonus EOS. Accepted EOS truncation excluded 17
later verified positions from cumulative-prefix labels. Effective label counts
therefore use their own denominators. Attempted uniforms equal accepted draft
tokens plus rejected rounds; committed tokens after the 32 initial draws equal
accepted draft tokens plus residual and bonus draws, totaling
3824. Independent raw-round checks verified
prefix labels, counters, finite shapes and cache commit deltas.

The two preselected numerical probes (cases 0/1, round 0) yielded
16 rows, of which 16
had nonzero block-versus-fresh-sequential TV. Maximum TV was
**0.0927704844638265**, with
0 argmax changes. Earlier maxima were
0.045764141753646695 at step128 and
0.05288759555199574 at step512. These probes concern
different sampled prefixes; they are not controlled checkpoint effects on the
target. Execution checks do not establish native BF16 sequential-target-law
losslessness.

Source archive SHA256 is `669b48cd2a11693422f34b798607dbae307bafd6de0af2d2b461fdf3afc636bf`;
panel manifest is `fd8efdd09f58c20a24e4c248532bb1460282b6eadb2e73d51754e9acaeeaa2ec`;
step1280 binding is `2868d52abcc228f929a5dc5ef1a7e4772b2d5cf8e4ae6456e1bd93b92ecf6d7b`.
[source-identity.json](source-identity.json) preserves source, checkpoint, panel and
input hashes; [protocol.json](protocol.json) matches earlier protocols exactly.

Peak allocated memory was 2,266,962,944 bytes. The worker's OS
exit was waited by the launcher, the launcher's by the controller, and the shell
captured the controller's OS exit. The shell itself is not claimed as a separately
waited exit. All four owned PIDs were absent after completion, KFD was held only
by preserved ASR, and ASR remained ready/not busy. Resource snapshots and full OS
exit provenance appear in [runtime.json](runtime.json).

This collector is a correctness reference with FP32 draft parameters and BF16
autocast, float64 probability arithmetic on the tensor device, synchronous scalar
checks, diagnostic copies and JSON writes. Its work counts or wall time do not
establish serving speed. A future serving path needs measured costs and a strong
target-only engine baseline at matched context lengths, requests and load.
Checkpoint selection, STS fit44/eval43 and system performance gates remain
separate. All development rows were already used for teacher-forced monitoring;
final test stays locked. Private outputs, record/token IDs and vectors remain
outside this report. No extra GPU job was started by this collection.
