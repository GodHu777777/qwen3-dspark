# Native varlen: explicit-window alignment diagnosis

The one authorized six-call diagnostic completed, with worker, launcher and
controller exit 0 and no timeout. **The public wrapper matches upper-left
causality; the private ATen no-window control matches the required lower-right
causality**, on this pinned ROCm runtime and the original first-case shape.
Duplicating KV heads produces bit-identical public outputs, so these observations
do not support GQA grouping as the cause of this particular failure.

The [original native gate](../native-varlen-probe-20261009/README.md) remains
failed. No production adapter or backend default was changed. Successful
diagnostic execution is not a passed production gate or full-model correctness.

## Fixed matrix and results

Inputs reproduce the original CPU seed20261009, FP32 physical K then V then Q
random draw order and BF16 conversion. Q lengths are1,8; K lengths17,29;
physical K includes13 inactive tokens; Hq16/Hkv8, head dimension128. The public
repeated-KV variant duplicates each KV head twice and disables GQA. The private
variant uses `is_causal=True`, `window_size_left=None`,
`window_size_right=None`, scale `1/sqrt(128)` and dropout0. No retry, fallback,
threshold adjustment, backend install or additional native call occurred.

All errors below are versus the independent FP32 MATH SDPA reference. Agreement
requires every element within atol0.02+rtol0.02×|reference| **and** RMS≤0.005.

| Variant / input | Upper-left max abs / RMS | Lower-right max abs / RMS | Agreement |
| --- | --- | --- | --- |
| Public GQA / random | 0.00760746 / 0.000992666 | 3.775065 / 0.674074 | upper-left only |
| Public GQA / zero-Q ramp | 7.45e−9 / 2.48e−9 | 0.328125 / 0.320387 | upper-left only |
| Public repeated KV / random | 0.00760746 / 0.000992666 | 3.775065 / 0.674074 | upper-left only |
| Public repeated KV / zero-Q ramp | 7.45e−9 / 2.48e−9 | 0.328125 / 0.320387 | upper-left only |
| Private no-window / random | 3.774414 / 0.674082 | 0.00428557 / 0.000610631 | lower-right only |
| Private no-window / zero-Q ramp | 0.328125 / 0.320387 | 2.98e−8 / 1.49e−8 | lower-right only |

Each output has18,432 elements. Public random outputs mismatch17,841 elements
against lower-right; public ramp outputs mismatch all18,432. Private controls
have zero mismatched elements against lower-right. Max relative errors, with the
predeclared1e−30 denominator floor, are diagnostic only and do not replace the
fixed elementwise/RMS criteria. [aggregate.json](aggregate.json) retains all
per-request errors, both oracles, analytic fingerprints, operator counts and
output hashes.

For zero Q and V=local-key-position/32, the public outputs are bit-exact BF16
upper-left analytic means: first request0; second request0..7 divided by64.
The private control is bit-exact lower-right: first request0.25; second
request21..28 divided by64. FP32 reference averaging accounts for the tiny
nonzero ramp oracle errors in the table. Both public GQA and repeated-KV outputs
are bit-identical for random and ramp inputs.

Together with the revision-matched source chain documented in
[the diagnostic plan](../../docs/native-varlen-diagnostic-plan.md), this supports
an explicit-window mapping issue on this runtime/shape: literal right-window0
follows upper-left alignment, while the private no-window causal special case
uses lower-right. The matrix does not establish behavior for other versions,
architectures, shapes or full Qwen/KV execution. Exact device-kernel identity was
not independently traced beyond native ATen dispatch and recorded AOTriton
preference.

## Evidence and isolation

- Source commit: `cd317f79a572dc976ddf660f3bcb2d4a4724635d`.
- Git archive SHA256: `3537bf3185e1b5f5ae07f477bb4f14c13db6d97c648c1633900c8e7caad58c40`.
- Diagnostic script SHA256: `ac47f2ba1e7b1a940f3a0e87081b6063bf9d199f51fa9f62ac654a467254962f`.
- Protocol SHA256: `c370831c863364f0da90a658e7bb2935ec937293016a95ba0f5393967bff8de8`.

All31 executed source files match the corresponding Git blobs. A separate
GPU-hidden CPU audit verified49 tensor hashes/shapes/dtypes, reconstructed all
original input tensors exactly, checked six native flash event counts and
bit-exact ramp/GQA comparisons. Raw output tensors and operator evidence were
saved before classification, as verified by source review and an independent CPU
control-flow test. Full tensors stay in private remote evidence; public exports
contain scalar errors, shapes and hashes only.

Torch2.12.0+rocm7.2, HIP7.2.53211, gfx1201, BF16 and AOTriton preference were
recorded. Peak allocated GPU memory was81,909,248 bytes. Pre/post used VRAM was
8,962,183,168 bytes, `/dev/kfd` had only the existing ASR process, and ASR was
ready/not busy. Desktop render/card users were recorded and left untouched.
All three diagnostic processes exited. The120-second limit was unchanged;
reported elapsed time includes profiling/oracles/persistence and is not a
performance benchmark. No model weights, training snapshot or prior report was
modified.

A future explicit pinned private backend must still pass the **entire original
tensor gate**, with unchanged cases/thresholds, and a **whole Qwen/KV gate** before
claiming adapter correctness. This diagnostic does not authorize those GPU runs.
