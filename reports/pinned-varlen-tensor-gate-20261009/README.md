# Explicit pinned ROCm backend: complete small-tensor gate

The single authorized run of `rocm_aten_no_window_pinned_v1` passed all **three
original cases and11 native calls**. Worker, launcher and controller exited0;
the original120-second limit was not reached. Runtime pins matched exactly.
This is a small-tensor gate pass for the explicit private backend. The
[original public-wrapper gate](../native-varlen-probe-20261009/README.md) remains
failed; no whole-Qwen, losslessness, calibration or speed claim follows.

## Original cases and unchanged numerical criteria

The original seed, CPU K→V→Q random draw order, shape schedule, FP32 MATH
per-request/packed bottom-right references, atol0.02/rtol0.02, RMS≤0.005 and
exact isolation criteria were retained. The public protocol JSON SHA256 remains
`08c501349bb59d30a888f1a5669a6c5bcb0bdfbf5f9e894c409fb2e702d49ff9`.
The explicit private protocol has a separate identity binding its runtime and
None-window entry arguments, while referencing that original protocol hash.

| Case | Q lengths | Active K lengths | Inactive K | Per-request max abs / RMS |
| --- | --- | --- | ---: | --- |
| Interleaved cached tail | 1,8 | 17,29 | 13 | 0.004285574 / 0.000610631 |
| Crop, exit and re-add | 3,2,1 | 11,5,1 | 0 | 0.007105827 / 0.000889141 |
| Crop to zero then append | 2 | 2 | 6 | 0.007795095 / 0.001023080 |

The packed FP32 references satisfy the same fixed limits. All three zero-Q
position-ramp outputs are exactly the lower-right analytic means (max/RMS error0).
All three other-request poison checks and both applicable inactive-KV poison
checks are byte-exact. Each of the11 calls records exactly one
`aten::_flash_attention_forward` event. Full counts/errors are in
[aggregate.json](aggregate.json); protocol details are in
[protocol.json](protocol.json).

A separate GPU-hidden CPU audit verified all11 saved native tensor-file hashes,
recomputed the three FP32 bottom-right references, checked the original fixed
limits and compared them to the saved GPU MATH references. It independently
rechecked ramp and poison output equality. Full QKV/cu/output and reference
tensors remain in private remote evidence. Public files contain scalar results,
shapes, event counts and hashes.

## Source, resources and limits

- Executed source: `5281a6af05ddd6b3e80cd3aeded892c61d2c91a5`.
- Git archive SHA256: `233d3f493ce084b18761beccfe34c56f0002a56cd098b9c0c31d1873cbaec38e`.
- Probe script SHA256: `0da7f5b52671257a0d33c08b0e55829132c0a3fa5c5753b0aabd7779f802915e`.

All32 executed script/package files match their Git blobs; worker and launcher
source identities agree. The observed runtime matches the exact Torch/HIP/git,
gfx1201, AOTriton-preference, environment and private-schema pins recorded in
[runtime.json](runtime.json). This is not an independent device-kernel binary
attestation.

Peak allocated GPU memory was82,391,552 bytes. Pre/post used VRAM was
8,962,183,168 bytes; `/dev/kfd` had only the existing ASR process, which was
ready/not busy. Desktop render/card users were recorded and left untouched.
All three gate processes exited. No retry, relaxed threshold, fallback, package
installation, model weight or frozen training-snapshot change occurred. Elapsed
time includes profiling, references and evidence writes and is not a benchmark.

This source preserves the original FP32 error reduction. A sufficiently large
finite BF16 error could overflow that reduction and fail scalar serialization;
raw native tensors are already saved before it. That failure-evidence limitation
did not occur here, and the executed source has not been retrospectively changed.

The remaining whole pretrained Qwen/KV gate must separately check exact cache
content/position/isolation and each normal layer's actual same-QKV attention
against the original fixed numerical limits. End-to-end BF16 hidden/KV/logit/TV
differences must be reported separately. This tensor pass does not authorize or
substitute for that GPU gate.
