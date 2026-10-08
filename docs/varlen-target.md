# Native varlen target: CPU implementation and proposed GPU gate

The dense `PackedTarget` remains the default and independent oracle. The separate
`VarlenPackedTarget` uses an HF attention callback that preserves Qwen QKV
projection, QK normalization, local-position RoPE and DynamicCache updates. It
passes THD queries, one gathered active KV tensor and int32 cumulative lengths
to one native `torch.nn.attention.varlen.varlen_attn` call per layer. The callback
constructs no dense attention mask; inactive resident KV stays in persistent
state but is excluded from transient attention input.

The target owns its model's attention configuration: do not share that model
with another target or concurrent forward. `PackedLayout` requires each queried
request to be one contiguous query group, keys to be a full request-local prefix,
and queries to be that prefix's terminal suffix. The native call uses
`window_size=(-1,0)` and GQA. No backend error triggers a dense fallback. Native
imports are delayed until execution; the project's minimum Torch requirement
does not promise availability of this experimental newer API.

## CPU evidence

On the existing Torch 2.12.0+rocm7.2 / Transformers 5.17.0 environment with GPUs
hidden, five existing packed tests and five new varlen tests passed in 1.253 s.
Private evidence is `output/varlen-target-cpu-20261009/`. The explicitly injected
CPU-only dense kernel validates layout, bottom-right causality, actual tiny Qwen
hidden/context/logits and every layer's KV, mixed query lengths, inactive KV,
crop/exit/readd/crop-zero, poisoned inactive KV, one forward with one adapter call
per layer, and state clearing after a kernel error. This establishes CPU FP32
adapter semantics, not successful native GPU execution. Work counters distinguish
adapter calls, CPU oracle calls and native calls; existing dense Q×K counts remain
explicitly counterfactual domains when native varlen is selected.

## Fixed small-tensor GPU protocol (not executed)

`python scripts/probe_varlen.py --dry-run` uses only stdlib. Syntax and a guarded
dry-run forbidding Torch/Transformers/package imports passed. The full protocol
and its SHA256 are printed with source hashes. No execution is authorized by the
existence of this tool; coordinate a separate GPU window first.

After approval, `--execute --output <fresh-directory>` starts one bounded worker
using the invoking Python. It creates an independent process group and kills only
that group on the fixed 120-second timeout or controller interruption. No model,
checkpoint, final-test text or environment installation is involved.

All native inputs are BF16, Hq=16, Hkv=8, D=128 (the actual target attention
dimensions), with seed 20261009 and scale 1/sqrt(128). Physical KV starts
interleaved by request; the later cases actually filter this tensor, preserve
retained prefixes and append new rows:

| Case | Query lengths | Active key lengths | Inactive resident keys |
| --- | --- | --- | --- |
| Cached terminal suffix | 1, 8 | 17, 29 | 13 |
| Crop to 8/3, exit old marker3, readd as marker4, append | 3, 2, 1 | 11, 5, 1 | 0 |
| Crop request1 to zero, then append | 2 | 2 | 6 |

Each case compares against two FP32 MATH SDPA oracles on the same BF16-rounded
inputs: independent per-request bottom-right masks, and the original
`PackedTarget._attention_payload` dense mask over all resident keys. The oracles
must agree within atol=2e-6/rtol=1e-5. Native outputs must be finite and satisfy
both elementwise atol=0.02/rtol=0.02 and RMS≤0.005 against each oracle. A zero-Q,
local-position value ramp has an analytic causal mean, checking bottom-right
alignment directly. Poisoning other active requests' QKV must leave the first
request exactly unchanged; poisoning only inactive KV must leave all outputs
exactly unchanged. These thresholds are fixed, with no retry or relaxation.

Every native invocation is CPU-profiled and must contain exactly one
`aten::_flash_attention_forward`; no success through a different backend is
accepted. This is dispatch evidence, not a kernel performance measurement. The
worker records tensor shapes, actual device/runtime, native Python source hash,
operator events, numeric errors, peak allocation and aggregate check time.
It records `gcnArchName` and the read-only ROCm flash-library preference getter;
the latter is explicitly a preference, not independently traced GPU kernel identity.
The controller retains stdout/stderr, PID, source/protocol identity, exit code,
timeout and completion status. A failed native call produces a failed run and
exception evidence, never a CPU/dense fallback success.

A passing probe would establish only support and bounded error for these small
tensors. It would not establish whole-Qwen numerical fidelity, BF16 distribution
losslessness, speculative acceptance, speedup or calibration. Full native model
gates and measured layout/gather/attention/crop costs would still be required.
