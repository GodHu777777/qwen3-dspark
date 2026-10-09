# Native variable-Q full-target graph capability

Execution source: `a4942afd308e720ab36a399ef677b37821c23e1d`.
The single bounded run passed all six native eager reference states, two graph
captures, five useful graph replays and one explicit eager state. Worker,
controller and SSH exited 0. An independent live check confirmed resource release
and the preserved ASR service. The original numerical thresholds were unchanged.

This is a finite same-backend target capability result. **It is not a speed,
trained acceptance, measured SPS or asynchronous overlap result.**

## What passed

The [frozen protocol](../../docs/query-family-graph-probe.md) uses pretrained BF16
Qwen3-0.6B, Torch 2.12.0+rocm7.2, Transformers 5.17.0 and the pinned gfx1201
attention backend. Two active requests and one inactive resident share the target
KV pool. B3 and B5 have separately captured full-target programs and share one
metadata/gather arena; B2 executes eagerly.

| Case | Execution | Ordered requests | Actual Q | Ordered context lengths |
| --- | --- | --- | --- | --- |
| 0 | B3 graph | r0, r1 | (1, 2) | (128, 128) |
| 1 | B5 graph | r0, r1 | (1, 4) | (129, 129) |
| 2 | B2 eager | r0, r1 | (1, 1) | (130, 132) |
| 3 | B5 graph | r0, r1 | (2, 3) | (131, 133) |
| 4 | B5 graph | r1, r0 | (4, 1) | (135, 132) |
| 5 | B3 graph, abort | r0, r1 | (2, 1) | (133, 137) |

Fixed native maxQ/maxK exceed each state's actual maxima. Actual query rows sum
to B without padding. The middle eager execution overwrites shared metadata;
subsequent graph execution restores the correct inputs. Reversing request order
also changes slot/gather mapping. Artificial commits exercise transaction
mechanics and are not sampled acceptance decisions.

Each capture executed the model and all 28 decoder Python boundaries three times
(two warmups and capture). All five useful replays left those counters unchanged;
the explicit eager state incremented each once. The graph includes the original
embedding, decoder body, final norm, scratch/gathers and raw selected features.
Prefill, LM head, sampling, capacity planning and commits remain outside capture.

## Independent evidence checks

Before GPU execution, the immutable 375-file archive passed 18 related CPU tests
on AMD with all three GPU visibility variables empty: 3.628 seconds, exit 0.
The separate local preparation and independent CPU review remain documented in
notebook S37; they are not native execution evidence.

The completed device run recorded 12 ordered observations and 144 structural
checks. Raw selected layers `[1,7,14,21,26]`, final norm, logits, all 28 layers'
scratch and committed KV, inactive isolation, exact own commit/abort, stable
addresses and post-commit/abort feature leases passed. Layer 26 is not the last
Qwen block. Eager hooks witness raw-layer identity, not independent attention
correctness.

A separate CPU audit of the saved evidence passed 5,584 checks and 3,612 numerical
reductions over 65 tensor artifacts, with no missing checks and actual exit 0.
It reconstructed ordered metadata, token slices, commit/abort and inactive
contents. All compared tensors were equal under `torch.equal` and byte comparison;
observed max absolute error and RMS were 0. This is descriptive: the frozen
cross-path limits remain absolute/relative 0.02/0.02 and RMS <= 0.005, pooled and
per request. No threshold was tightened or relaxed after execution.

## Memory and provenance

Each graph's distinct private pool retained 2,097,152 bytes, including inactive
segments with allocated/active bytes zero. The original reservation remains
512 MiB per graph. Shared arena storage is 3,605,722 bytes; total registered
workspace including eager prefill is 6,967,476 bytes. The two reservations plus
separate graph I/O total 1,073,840,192 bytes. Pool retained bytes, workspace
storage and reservation are different quantities.

Limits remained 1280 MiB combined budget, 64 MiB workspace cap, 6 GiB allocator
cap and a single 300-second worker window. Supervisor elapsed time was 32.0443
seconds including setup and evidence I/O; it is not throughput. No timeout,
retry, training or profiler export occurred. Allocator accounting does not
measure total process/driver or transient peak memory.

The 365,270,999-byte evidence archive passed its remote/local SHA256 check.
`source.tar` equals the execution commit's Git archive, and all 375 deployed
files matched before and after execution. The [aggregate](aggregate.json)
contains numerical scalars, work counts, memory records and source/evidence
hashes. Private raw tensors, command lines, process identities and machine paths
remain in `output/query-family-native-20261009/`, not this public report.

## Limits and next decision

The passing domain is R2, the declared B3/B5 graph families and B2 eager state.
It does not certify arbitrary request counts, shapes or contexts. Both reference
and replay use the same native attention backend; previous cross-backend RMS,
sequential-law and BF16 endpoint differences remain unresolved.

The [same-backend paired benchmark](../paired-r1-benchmark-20261009-192a357/README.md)
still shows speculative decoding slower than target-only. The separate
[vLLM baseline](../vllm-offline-benchmark-20261009-700bfa6/README.md) is unchanged.
This probe executes no trained draft or confidence-driven allocation. The next
measurement must establish actual family costs and then complete multi-request
end-to-end comparisons, charging full shadow work, explicit eager cases, host
planning, synchronization and setup. The earlier CPU bridge's synthetic SPS
curves cannot supply those hardware costs.
