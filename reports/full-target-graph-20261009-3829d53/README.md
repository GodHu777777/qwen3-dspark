# Full pretrained native target graph capability — 2026-10-09

Immutable execution source: `3829d5355018211f9b8464f00a44bdf55c3779e6`.
This report concerns the complete original HF Qwen3-0.6B target body under the
pinned native attention backend, with eager prefill and LM head outside capture.
The bounded run completed all six observations: three native eager states before
capture, then three actual GPU graph replays. Worker, controller and outer SSH
exited 0; independent process identity and ASR/KFD checks confirmed release.

`aggregate.json` retains every observation's numerical scalars and structural
checks, actual graph-pool accounting, execution/source identity, and independent
raw-tensor audit summary. Raw tensors, machine-specific binding paths and process
command lines remain private under
`output/full-target-graph-gpu-20261009-3829d53/`.

## What was exercised

The fixed Q=(1,4), K ceilings=(144,144) family ran at committed contexts
(128,128), (129,131), (130,135), with artificial query commits (1,3), (1,4),
(0,0). The last state explicitly aborted. Selected raw block IDs [1,7,14,21,26]
were checked individually against their actual eager layer hooks, separately
from the original final norm. Features, logits, all 28 layers' speculative and
committed K/V, inactive resident isolation, rollback, fixed addresses and
commit-to-consumer feature leases were checked.

The graph includes embedding, all original decoder QKV/QK norm/RoPE/attention,
output projections, residuals, MLPs, final norm and scratch/gather operations.
Two warmups plus capture executed Python model/decoder hooks three times. All
three useful replays changed inputs and positions while leaving those counters
unchanged. This establishes actual Python-free replay at the observed boundaries.

Original thresholds remained elementwise absolute/relative 0.02/0.02 and
RMS <= 0.005, pooled and per request. Each path's own commit reconstruction,
inactive bytes and feature-lease reread checks require exact equality. Artificial
commits are fixtures, not sampled acceptance decisions.

## Memory and execution limits

The measured graph private pool retained 2,097,152 bytes, including a fully
inactive segment (allocated/active bytes 0). The unchanged registered reservation
was 536,870,912 bytes. Global reserved bytes moved from 1,694,498,816 to
1,692,401,664 (delta -2,097,152); this is a separate quantity and cannot replace
private-pool accounting. Combined graph/workspace budget was 768 MiB, workspace
cap 256 MiB, allocator fraction cap 6 GiB and worker deadline 300 seconds.
Transient/process-wide peak memory was not measured by private-pool accounting.

The supervisor elapsed time was 25.762924 seconds including setup and evidence
I/O. It is not a performance measurement. Transfer failures occurred only after
successful experiment completion and verified release; the original failure and
packaging-recovery records remain private. They did not trigger another GPU run.
A post-release read-only inspection generated one extra Python cache file; all
346 immutable source files retained their original hashes and the extra file is
recorded separately from execution source.

## Independent CPU audit and reproduction

The raw CPU audit read 30 tensor artifacts, completed 3,430 checks and recomputed
1,239 pooled/request numerical comparisons. All pass the frozen limits with
max absolute error and RMS both 0. All compared tensors also match byte-for-byte
when reinterpreted as `uint8`. This byte check is a descriptive audit result,
not an additional or tightened pass threshold. Original worker `bit_equal` fields
retain their original `torch.equal` semantics. The audit process exited 0.

`verify_raw.py` reads a user-provided evidence root containing `complete/`,
`source.tar`, `expected-source.json` and `ssh.os-exit`. `complete/` is the checked
remote archive extraction, with `run/worker`, supervision and independent release
records. It reads the raw `.pt` tensors on CPU, reconstructs commits and rollback,
checks each raw-layer witness, actual token slices/positions/cumulative lengths,
recomputes pooled/request numerical metrics, and verifies replay counters,
source/artifact hashes and exit/release records. It needs a compatible existing
PyTorch runtime; it neither loads target weights nor executes a model/GPU run.

```sh
HIP_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES=''   PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2   python reports/full-target-graph-20261009-3829d53/verify_raw.py   --evidence-root /path/to/private/evidence --output /path/to/audit.json
```

## Applicability

This is a same-backend native eager versus graph fidelity result for one finite
shape family. The eager cache and captured cache paths differ, but both use the
same native attention backend. Raw layer hooks witness identity; they are not an
independent attention oracle. The earlier
[whole-Qwen layer-26 RMS gate failure](../whole-qwen-varlen-gate-20261009/README.md)
and BF16 endpoint distribution differences remain unchanged.

No sequential/cross-backend distribution equivalence, end-to-end speedup, t−2
capacity-based graph family or scheduling overlap follows from this capability
run. The next performance question requires a separately measured paired
end-to-end workload; this report makes no throughput claim.
