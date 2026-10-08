# Expanded step512: quality32 checkpoint comparison

All 32 frozen quality prompts completed, with worker, launcher and controller exit 0
and no timeout. Under the same collector source, panel, per-prompt seeds and
sampling protocol as [expanded step128](../expanded-quality128-20261009/README.md),
accepted draft tokens per verification round increased from **0.1778 to 0.4958**.
This ratio improved on 31 of 32 prompts. This is broader native-protocol rollout
acceptance improvement, not output-quality, speedup or losslessness evidence.
Step512 remains an intermediate checkpoint in the unchanged 1280-step training
plan; this report does not freeze the final checkpoint or start later training.

| Quantity | Step128 | Step512 |
| --- | ---: | ---: |
| Completed prompts | 32 | 32 |
| Output tokens | 3,874 | 3,941 |
| EOS prompts | 5 | 4 |
| Verification rounds | 3,268 | 2,620 |
| Proposed positions | 22,415 | 17,960 |
| Target-verified proposal positions | 22,415 | 17,960 |
| Attempted acceptance uniforms | 3,842 | 3,901 |
| Effective cumulative-prefix labels | 22,415 | 17,955 |
| Accepted draft tokens | 581 | 1,299 |
| Accepted draft tokens / round | 0.177785 | 0.495802 |
| Actual committed tokens / round | 1.175643 | 1.491985 |
| Verification query rows, excluding prefill | 25,683 | 20,580 |
| Macro mean of prompt-level accepted draft / round | 0.176533 | 0.528944 |

The comparison keeps temperature 1, no filtering, full proposals up to block 7,
128-token output budget, native BF16/SDPA and actual float64 q. All 30 executed
package files equal source commit `a278e5a5d7d74a1f77f8dcd475700959ee158cb0` and the
step128 worker. Panel file, complete manifest, ordinal prompt identities and seeds,
draft configuration and target/data fingerprints match. Checkpoint and its
metadata hashes are bound separately. The earlier step512 CPU dry-run used an
older source inventory; a fresh dry-run on the exact step128 archive was required
and passed before this collection.

Per-prompt accepted-draft/round improved for 31 prompts and declined for 1, with
median paired difference 0.321621. Macro means confirm the direction without
weighting long trajectories more heavily. Accepted token counts ranged from 5 to 70.
The same prompt and seed do not imply identical generated prefixes across
checkpoints: EOS outcome changed on 3 prompts and output horizons differ. Counts
and work rows therefore must not be turned into a speedup ratio. Rounds are
correlated; this report makes no independent-round confidence interval claim.
[per-prompt.json](per-prompt.json) retains every ordinal case and its step128
comparison, including the decline.

| Prefix position (one-based) | Step128 events / labels | Step512 events / labels | Step512 rate |
| --- | ---: | ---: | ---: |
| 1 | 519 / 3268 | 806 / 2620 | 30.7634% |
| 2 | 58 / 3247 | 291 / 2602 | 11.1837% |
| 3 | 4 / 3223 | 118 / 2580 | 4.5736% |
| 4 | 0 / 3202 | 44 / 2563 | 1.7167% |
| 5 | 0 / 3181 | 20 / 2547 | 0.7852% |
| 6 | 0 / 3159 | 12 / 2529 | 0.4745% |
| 7 | 0 / 3135 | 8 / 2514 | 0.3182% |

The step512 accepted-prefix-length histogram for lengths 0 through 7 is
`1814, 515, 173, 74, 24, 8, 4, 8`. There were 8 fully accepted seven-token blocks,
out of 2,515 seven-token proposals. Long accepted prefixes remain uncommon.
Seventeen blocks accepted their entire proposal; nine of these reached the token
budget without a bonus draw. Eight bonus draws occurred.

Four prompts ended with EOS: three residual EOS draws and one accepted draft EOS.
That draft EOS was at proposal position 2 of a seven-position verified block:
its first 2 prefix labels remain, while 5 later verified positions are excluded.
Thus effective labels 17,955 differ from verified positions 17,960; attempted
uniforms 3,901 equal accepted 1,299 plus 2,602 rejected rounds. Actual committed tokens
after the 32 initial draws total 3,909. This is not simply rounds plus accepted
counts: the accepted-EOS block and nine budget-boundary blocks omit a bonus draw.
Independent raw-round checks validated these labels, all counters and cache deltas.

The two preselected numerical probes (quality cases 0/1, round 0) yielded 16 rows.
All 16 had nonzero block-versus-fresh-sequential TV, with maximum
**0.05288759555199574** and no argmax changes. Step128 maximum was 0.0457641418;
these values concern different sampled prefixes and are not a controlled
checkpoint effect on the target. Passing execution checks does not establish
native BF16 sequential-target distribution losslessness.

Source archive SHA256 is `669b48cd2a11693422f34b798607dbae307bafd6de0af2d2b461fdf3afc636bf`;
panel manifest is `fd8efdd09f58c20a24e4c248532bb1460282b6eadb2e73d51754e9acaeeaa2ec`;
step512 binding is `41d39d07cd6c8a55efab5ea2ddd75e39f55e7a570e3de264ae7c278379db4b56`.
[source-identity.json](source-identity.json) holds source, checkpoint, panel and
input fingerprints; [protocol.json](protocol.json) is unchanged from step128.

Peak allocated memory was 2,267,403,776 bytes. Before and after the collection,
only the preserved ASR process held the compute-device handle; ASR was ready and
not busy, and used VRAM was 8,962,183,168 bytes. Desktop render occupancy remained
recorded and untouched. All three collector PIDs exited. [runtime.json](runtime.json)
contains OS exits and resource provenance; this run is not a performance benchmark.

The comparison supports reviewing the next bounded training segment, not changing
temperature, data, seeds or block policy to manufacture acceptance. The frozen
training maximum 1280 has not yet been reached. Final checkpoint selection and
STS fit44/eval43 collection remain pending; all development prompts were already
used for teacher-forced monitoring, while final test remains locked. Private
sample outputs, record IDs, token IDs and full vectors are excluded from this
report. No additional GPU work was launched by this collection.
