# Paired R1 end-to-end benchmark — 2026-10-09

Immutable execution source: `192a3578eb71527e02fded28a3beaf487d53859a`.
One authorized GPU run completed all 36 batches: two native arms × two R1
contexts × (two warmups, five primary measurements, two diagnostics). Each batch
has one request and exactly 128 output tokens. Worker, controller and SSH exited
0; independent identity, ASR health, KFD and VRAM checks confirmed release.

The trained step-1280 fixed-γ7 full-shadow arm is **slower than the matched native
target-only arm** in both measured cases. The frozen vLLM reference remains faster
than either native arm. Rates below pool 640 output tokens over the five complete
primary batch wall times; they are offline throughput, not serving latency.

| Case | Native target-only tok/s | Native speculative tok/s | Spec / target-only | Frozen vLLM tok/s | Target / vLLM | Spec / vLLM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R1 C64 | 44.804932 | 23.103708 | 0.515651 | 131.264056 | 0.341334 | 0.176009 |
| R1 C256 | 44.181720 | 19.274350 | 0.436252 | 127.339898 | 0.346959 | 0.151361 |

The [frozen vLLM report](../vllm-offline-benchmark-20261009-700bfa6/README.md)
uses the same workload identity but a different execution/probability stack; it
is a strong retained performance reference, not evidence of equal target laws.
Comparison with an older eager native run cannot attribute the whole change to
GPU graphs: this run also uses a persistent target, reused workspaces and a new
paired protocol. No graph-only causal speedup is measured here.

## Matched execution and timing

Both arms keep the same original Qwen3-0.6B target, trained draft weights and two
graph pools resident. Target-only performs no draft cache construction or draft
execution. Speculative execution retains all seven shadow proposal positions,
even when the remaining output budget selects fewer proposals or zero. Actual q,
FP64 validation/sampling, per-request seeded RNG, acceptance, residual and bonus
logic are unchanged. Selected raw layer IDs are [1,7,14,21,26].

Q1 and Q8 use actual full-target GPU graphs; Q2–Q7 use explicitly labelled native
eager tails. Prefill is eager. Graph replay leaves all 29 model/decoder Python
counters unchanged; eager tails increment each once. LM head and sampling remain
outside the captured target body. Verification K capacity is 383, committed
context capacity 384, scratch capacity 256, with ten workspaces.

Wall time includes fresh session/reset, prefill and first draw, every full round,
shadow work, graph waits/eager tails, metadata, sampling, commit/projection/release
and final synchronization. Input/RNG preparation, evidence serialization/hashing
and cold setup are outside. Cold setup took 21.627453 seconds with exactly one
target load and one draft load. Every sample accounts for one admission token
plus 127 round-committed tokens; final contexts are 191 and 383.

| Primary speculative totals | C64 | C256 |
| --- | ---: | ---: |
| Accepted draft tokens / selected proposals | 180 / 3,120 | 105 / 3,570 |
| Accepted / selected | 5.7692% | 2.9412% |
| Accepted draft tokens per round | 0.395604 | 0.198113 |
| Round-committed tokens per round | 1.395604 | 1.198113 |
| Actual graph rounds / all verification rounds | 435 / 455 | 500 / 530 |
| Actual graph query rows / all verification rows | 3,480 / 3,575 | 3,965 / 4,100 |
| Eager tail rounds / query rows | 20 / 95 | 30 / 135 |

Target-only uses 635 Q1 graph rounds per case across its five primary samples.
Acceptance denominators above are selected proposals; the speculative arm still
computes seven shadow positions in every round. Repeated output hashes match
within each arm/case across its nine samples. These are timing repetitions of the
same request/seed, not nine independent quality examples.

## Setup fidelity and memory

Before timing, Q1 and Q8 each compare native eager against first replay at the
unchanged elementwise absolute/relative limits 0.02/0.02 and RMS ≤ 0.005. A separate
CPU audit of both saved artifacts (36 tensors) recomputed 126 comparisons:
five selected raw layers, final norm, logits, and all 28 layers' scratch K/V per
shape. All pass with maximum absolute error and RMS 0. Original resident K/V
before/after tensors remain exactly unchanged. `torch.equal` and uint8 storage
byte equality also hold; these are descriptive diagnostics, not stricter
numerical acceptance thresholds.

Each graph's private pool retains 2,097,152 bytes, including inactive segments;
allocated/active sizes in those segments are zero. Each graph keeps its original
536,870,912-byte reservation. Q1 global allocator reserved bytes increase by
56,623,104; Q8 by 18,874,368. These global deltas are separate from private-pool
retention and cannot substitute for it. Workspace bytes are 41,646,512;
registered graph accounting is 1,073,852,488 bytes; resident/scratch storage is
73,400,320 bytes. The frozen budgets remain 1,280 MiB combined graph/workspace,
64 MiB workspaces, 6 GiB allocator cap and 8 GiB minimum free VRAM. Private-pool
accounting does not measure transient/process-wide peak memory; per-sample
allocator peaks are retained separately in `aggregate.json`.

## Evidence and independent audits

`aggregate.json` preserves the five primary wall times per cell, recomputed
coverage and rates, acceptance counts, source/input fingerprints, numerical
summary, memory accounting and release/provenance outcomes. It contains no raw
prompt/output or per-round traces. Private originals are under
`output/paired-r1-gpu-20261009-192a357/`.

Root's independent scalar audit checked all 36 identities and 4,059 rounds;
a second scalar audit passed 50,324 assertions. The CPU raw-tensor audit passed
404 assertions. Root independently verified the exact Git archive, all 352
extracted source hashes, and byte identity between the early scalar recovery
and the complete archive. The verified evidence archive is 102,307,761 bytes,
SHA-256 `68bb82399cab6b4686090b015ee3613f7bf7baae0676b4eb9dff8966bde872c4`.
The independent release check found both owned process identities absent, the
original ASR identity ready/not busy, KFD owned only by ASR, and 25,250,160,640
bytes of free VRAM. Packaging/transfer occurred after experiment completion;
no GPU retry occurred.

`verify_raw.py` reproduces the saved-tensor CPU checks from an evidence root
containing `complete/`, `archive-remote.log` and `evidence-verified.tar.gz`:

```sh
HIP_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' \
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
  python reports/paired-r1-benchmark-20261009-192a357/verify_raw.py \
  --evidence-root /path/to/private/evidence --output /path/to/audit.json
```

This finite same-backend setup fidelity check does not repair the earlier
[whole-Qwen layer-26 RMS gate failure](../whole-qwen-varlen-gate-20261009/README.md)
or establish sequential/cross-backend distribution equivalence. This is only
the R1 C64/C256 subset, not the full six-case panel, an all-graph decoder,
a capacity scheduler, asynchronous overlap or an arrival-load serving result.
