# Independent native varlen alignment diagnosis (GPU not authorized)

The original [native gate](../reports/native-varlen-probe-20261009/README.md)
remains a failed numerical gate. This plan neither changes its thresholds nor
replaces its result. The proposed diagnostic must use a new source identity and
fresh evidence directory, with separate GPU authorization before execution.

## Read-only source finding

The installed Torch Python varlen source SHA256 is
`2f5384e0bc8ce371d00a1c09d38ad019517009798e7cb3434f56cf4b9fa351ea`.
Its public wrapper sets `is_causal=True` for `window_size=(-1,0)` (line316), then
also passes the explicit left=-1/right=0 arguments through its internal call
(lines110–126). GQA validates Hq divisibility by Hkv; it does not expand the KV
heads before ATen. No input transpose is missing in the documented THD contract.

The installed Torch version reports Git revision
`7661cd9c6b841b62b7f411aa52ec51f05457263b`. Source at that exact revision gives
this chain:

1. [`attention.cu` lines504–511](https://github.com/pytorch/pytorch/blob/7661cd9c6b841b62b7f411aa52ec51f05457263b/aten/src/ATen/native/transformers/cuda/attention.cu#L504)
   passes window arguments through `aotriton_adapter::parse_window_size`.
2. The installed adapter header lines174–187 converts any negative window into
   `nullopt`, preserving right=0.
3. [`mha_all_aot.hip` lines82–103](https://github.com/pytorch/pytorch/blob/7661cd9c6b841b62b7f411aa52ec51f05457263b/aten/src/ATen/native/transformers/hip/flash_attn/aot/mha_all_aot.hip#L82)
   sets both defaults to the bottom-right special value when causal, then replaces
   either one with its explicit window argument. The result is a bottom-right
   special left value and literal right=0. The varlen forward passes these to
   `CausalType::WindowedAttention` (lines509–512).
4. Installed AOTriton headers identify version0.11.2, revision
   `dd1b68b604b5258ee7a9f7b66ad95e7a82c18065`, and define different top-left and
   bottom-right special values. At that revision,
   [`parse_window` lines43–63](https://github.com/ROCm/aotriton/blob/dd1b68b604b5258ee7a9f7b66ad95e7a82c18065/tritonsrc/masked_load_store.py#L43)
   maps bottom-right right to k−q, but leaves literal right=0 unchanged. The left
   special value maps to q, allowing the full prefix.

This source chain predicts an upper-left causal boundary for the public
`(-1,0)` call instead of the required cached-query lower-right boundary. It is a
specific testable hypothesis, not an observed diagnosis of the failed output.
Installed headers and upstream revision-matched implementation are static
evidence; an actual binary build may differ. Private copies and SHA256 identities
are retained alongside the original probe evidence. No Torch/AOTriton code was
imported or executed for this source analysis.

## Proposed fixed diagnostic matrix

Reconstruct the original first case exactly: CPU `manual_seed(20261009)`, then
generate FP32 physical K `[1,8,59,128]`, physical V with the same shape, then
Q `[9,16,128]`, copying each to GPU BF16 in the original order. Physical keys
interleave requests of length17,29,13; only the first two are gathered.
Q lengths are1,8 and K lengths17,29. Hash all input bytes and layout metadata
before native calls. No model/checkpoint/final-test inputs are used.

Perform exactly six native calls, all predeclared. Each variant runs once; a
numerical mismatch is recorded as a diagnostic observation rather than stopping
the matrix. A backend exception, nonfinite output or timeout ends the process
with a failed diagnostic execution. No automatic alternative backend is allowed.

| Calls | Entry and heads | Input |
| --- | --- | --- |
| 1–2 | Original public wrapper, Hq16/Hkv8, GQA enabled, window(-1,0) | Original random QKV; then zero-Q and V=local-key-position/32 |
| 3–4 | Same public wrapper, Hq16/Hkv16, GQA disabled; KV heads duplicated with repeat_interleave(2) | Same random values; then the same analytic ramp |
| 5–6 | Private ATen flash entry, original Hq16/Hkv8, is_causal=True, both window arguments=None | Original random QKV; then the same analytic ramp |

The private-ATen pair isolates explicit window mapping. It is a separately named
diagnostic variant, not a replacement for the production adapter and not a
fallback after its failure. Preserve scale1/sqrt(128), dropout0, BF16, original
cu lengths, no splits tuning and the original120-second worker timeout.

For every output record elementwise pass count, maximum absolute error, RMS and
maximum relative error against **both** FP32 MATH SDPA masks:

- Upper-left oracle: key position j≤local query index i.
- Lower-right oracle: j≤k−q+i, as required by cached suffix queries.

Scalar error reductions use FP64 to keep even very large finite BF16 failures
serializable; attention oracles remain FP32. Relative-error denominators have a
fixed1e-30 floor, reported in the protocol.

Use the original numerical criteria atol0.02/rtol0.02 and RMS≤0.005 to classify
agreement, with no relaxation. Zero-Q/ramp analytic means are i/64 (upper-left)
and (k−q+i)/64 (lower-right), broadcast across heads and dimensions. The expected
first request is0 versus0.25; for the second request the expected sequences are
0..7 divided by64 versus21..28 divided by64. These values are exactly BF16
representable. Record an additional full-attention analytic mean (k−1)/64 as a
diagnostic fingerprint, without changing the two primary oracles.

The random GQA and duplicated-KV outputs share the same mathematical reference.
Comparing them tests the head-grouping path independently of alignment. Compare
each request separately as well as the whole tensor; a global maximum alone
cannot localize a boundary or mapping failure.

Persist the case identity, cu lengths, input hashes, native event counts and
runtime/backend identity **before numerical classification**. After each call,
write its full scalar errors and a private output tensor before proceeding, so a
later assertion does not discard earlier evidence. Mark diagnostic completion
separately from lower-right correctness. Preserve full stdout/stderr, source and
archive hashes, peak memory, worker exit/timeout, process inventory and ASR
pre/post checks. Desktop render activity stays recorded and untouched; this is
not a performance benchmark.

If public outputs agree with upper-left while the private no-window variant
agrees with lower-right, that supports the explicit-window mapping diagnosis
for this source/runtime/shape. If duplicating KV changes which oracle matches,
investigate GQA independently. If neither oracle matches, retain the outputs and
investigate layout or backend behavior. None of these outcomes upgrades the
original failed gate or establishes full-model correctness.
