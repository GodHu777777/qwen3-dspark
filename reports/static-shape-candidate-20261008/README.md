# Fixed-shape BF16 execution candidate

This is an experimental target-only numerical result on Radeon AI PRO R9700,
Torch 2.12.0+ROCm 7.2 / Transformers 5.17.0. It is **not** a trained-draft gate or
a speedup result. Original failed examples were retained; no epsilon changed
token selection.

The candidate uses query width 8, StaticCache key capacity 256, explicit fixed
causal/valid masks, and an eight-row LM-head projection before selecting output
rows. Canonical sequential and oracle-block routes share the identical full
prompt prefill. This fixes decode tensor shapes/strides, not every possible
source of floating-point variation.

- Canonical sequential and oracle-block outputs matched on all 3 original
  examples, totaling 70 output tokens. This uses oracle proposals, not the draft.
- At both original failing prefixes, padded full/short-block and sequential
  histories produced identical logits: maximum difference 0 in this probe.
- Actual decode SDPA query/key/value/mask shapes and strides had one distinct
  configuration. Dummy suffix changes, poisoning uncommitted KV with +/-1000,
  and buffer-address-preserving rollback checks passed.
- Canonical versus stock cached decoding matched only 2/3 examples. The third
  diverged at output index 10. The candidate therefore does not establish
bit-exact equivalence to stock BF16 greedy execution.

The target-only probe additionally invokes `diagnose_same_prefix` on the saved
native failures and records dynamic-cache four-path results separately from the
canonical results. It does not rerun the native draft evaluator or turn the old
native failed gate into a pass. The helper projects a selected hidden row in
each path, so these diagnostics isolate transformer/cache history effects rather
than reproduce every original verification LM-head matrix shape.

StaticLayer in HF 5.17 does not implement crop. The isolated prototype adjusts
each layer's cumulative-length tensor in place and masks invalid backing slots.
It is a pinned experimental adapter, not a supported generic StaticCache API.
CPU tiny-model checks are separately recorded; their shared prefill excludes the
initial anchor, whereas the real-model probe shares the full prompt prefill.

The authorized diagnostic paused only the identity-checked data generator for
28.893 seconds. A detached controller used pidfd signals, bounded drain checks,
a 210-second child timeout and a finally-block resume. The same process was
observed running afterward and its log grew by 78 bytes. GPU probe elapsed time
was 19.05 seconds with 1.404 GB peak allocated memory; these are diagnostic
resource records, not performance comparisons.

The `.executed.py` files are historical source archives. In particular,
`controller.executed.py` records the past authorization and process identity;
it is not a reusable launcher or authorization to pause another process.

Next required gates: real trained-draft proposals, longer untouched inputs and
capacity boundaries, then matched canonical sequential/speculative timings plus
stock target-only timings. A slower padded baseline must not manufacture the
appearance of a deployment speedup. Stock/canonical numerical differences must
remain visible even if the canonical gate passes.
