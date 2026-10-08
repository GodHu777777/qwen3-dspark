# Expanded step128: quality32 native stochastic rollout

One authorized collection completed all 32 frozen quality prompts. Worker,
launcher and controller each exited 0. All observed finite, cache, probability,
EOS and budget checks passed. Actual draft acceptance remains low: **581 draft
tokens accepted across 3,268 rounds**, or **0.1778 accepted draft tokens/round**.
This is a native-protocol quality measurement, not a speed claim or proof that
native BF16 block sampling matches fresh sequential target sampling.

## Actual rollout and denominators

| Quantity | Observed |
| --- | ---: |
| Completed quality prompts | 32 |
| Sampled output tokens | 3,874 |
| Prompts ending with EOS | 5 |
| Verification rounds | 3,268 |
| Proposed positions | 22,415 |
| Target-verified proposal positions | 22,415 |
| Attempted acceptance uniforms | 3,842 |
| Effective cumulative-prefix labels | 22,415 |
| Accepted draft tokens | 581 |
| Verification query rows, excluding prefill | 25,683 |

Verification queries include old anchor plus proposal (`n+1` rows per round).
First rejection makes the later verified prefix labels known zero even though
those positions have no attempted acceptance uniform. None of this run's EOS
was accepted draft EOS: all five were residual draws. Therefore no verified tail
was removed by accepted-EOS truncation.

| Accepted draft prefix length in a round | Round count |
| --- | ---: |
| 0 | 2,749 |
| 1 | 461 |
| 2 | 54 |
| 3 | 4 |
| 4–7 | 0 |

The seven rounds accepting their entire proposed block were budget-truncated
blocks; there were no fully accepted seven-token blocks. Budget boundaries mean
committed tokens per round are not mechanically `1 + accepted draft/round`:
3,842 tokens were committed after the 32 initial draws, and seven potential bonus
draws were absent at the budget boundary. The corresponding average is
`3842/3268 = 1.1756`, with no timing implication.

| Prefix position (one-based) | Accepted-prefix events / effective labels | Rate |
| --- | ---: | ---: |
| 1 | 519 / 3,268 | 15.8813% |
| 2 | 58 / 3,247 | 1.7863% |
| 3 | 4 / 3,223 | 0.1241% |
| 4 | 0 / 3,202 | 0% |
| 5 | 0 / 3,181 | 0% |
| 6 | 0 / 3,159 | 0% |
| 7 | 0 / 3,135 | 0% |

[per-prompt.json](per-prompt.json) contains all 32 ordinal-case aggregates,
including output/EOS, rounds, all four denominators, accepted draft counts and
accepted-prefix-length histograms. Every prompt accepted at least one draft
token; counts ranged from 1 to 31. No prompt text, token IDs, record IDs or full
probability vectors are exported. Rounds are not asserted to be independent
statistical replicates; no confidence interval is claimed.

## Frozen numerical probes and identity

Quality cases 0 and 1, round 0 were selected before collection. All 16 `n+1`
comparison rows had nonzero total variation against fresh sequential replay on
the same semantic prefix; maximum TV was **0.045764141753646695**. No argmax
changed. Execution checks passing do not remove these numerical differences.

- Source commit: `a278e5a5d7d74a1f77f8dcd475700959ee158cb0`.
- Git archive SHA256: `669b48cd2a11693422f34b798607dbae307bafd6de0af2d2b461fdf3afc636bf`.
- Frozen panel manifest: `fd8efdd09f58c20a24e4c248532bb1460282b6eadb2e73d51754e9acaeeaa2ec`.
- Panel file bytes: `b5119c7f83c1f6d0ea08b58eb795bad2d1475fb6e4b71ece46ed9ae1fe00ee1e`.
- Binding: `df2366c25b0ef9c2e72136ef1dd32b1cb4c092ba0412ab58bbb137f1004e7a24`.

All 30 executed package hashes equal the independent deployed package and the
corresponding Git blobs. [source-identity.json](source-identity.json) binds the
actual checkpoint, target/tokenizer, development records and generation evidence.
[protocol.json](protocol.json) records temperature 1, no filtering, actual float64
q, native BF16/SDPA, full block7 capped only by remaining 128-token budget, fixed
per-prompt seeds and quality32/fit44/eval43 grouping. This run uses expanded
step128, not the earlier pilot aligned128 checkpoint.

## Resource isolation and scope

Peak allocated GPU memory was 2,267,426,816 bytes. Before and after collection,
`/dev/kfd` had only the existing ASR process, ASR was HTTP-success/ready/not busy,
and used VRAM was 8,962,183,168 bytes. Desktop render/card occupancy was recorded
privately and left intact. Collector processes exited. [runtime.json](runtime.json)
contains explicit exit and pre/post resource evidence; wall time is not used as
a performance result. Private samples and bounded probe tensors remain remote.

Quality32 is for checkpoint comparison. The reserved fit44/eval43 have not been
collected or fitted by this run. Every development prompt had already participated
in teacher-forced monitoring, so later eval43 is held out from STS fit only.
Final test remains locked. No step512 collection was started in this window.
