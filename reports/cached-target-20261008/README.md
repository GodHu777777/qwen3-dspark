# Cached target baseline and numerical fidelity

This report measures **target-only** eager Transformers decoding and isolated
target verification-block costs on an AMD Radeon AI PRO R9700. It does not
measure draft execution, target intermediate-feature capture, speculative
acceptance or speculative speedup. It is not an optimized-engine benchmark.

Seven pilot validation prompts, 2 warmup runs per prompt, 5 timed repeats, batch
size 1, BF16/SDPA, up to 64 output tokens with EOS stopping. The existing ASR
service remained resident (about 8.58 GB VRAM at preflight). GPU clocks were not
fixed. Raw synchronized timings and all seven prompt lengths are retained in
`timings.json` and `summary.json`; prompt text and token IDs are excluded.

- Handwritten cached greedy matched the explicit pure-greedy HF cached policy
  on all seven prompts: 117 checked tokens in total, up to 24 per prompt.
- Six CPU FP32 tests passed, covering prefill, incremental and block logits,
  intermediate features, every crop boundary including zero, repeated rollback,
  EOS, output limits and isolation from saved generation processors.
- Aggregate end-to-end throughput: 29.9215 tokens/s; remaining-token decode:
  30.0387 tokens/s. Median prefill plus first-logit time: 36.29 ms.
- Median verification time for 1 new target token was 32.90 ms; lengths 2–8 were
  approximately 36.54–38.01 ms. Length means new target inputs; draft length 7
  would require an anchor plus 7 proposals. No draft cost is included.

The verification medians are non-monotone and noisy. These measurements show
possible amortization of target work, not a proven speculative benefit or an
optimal block-length policy. EOS gives different output lengths across prompts;
aggregate throughput is total emitted tokens divided by total measured time.
Peak allocated memory includes the untimed correctness checks and excludes ASR.

## Numerical fidelity remains a separate question

One of seven BF16 full-recompute traces diverged from cached greedy at zero-based
output index 14. A same-prefix diagnostic found a correct cache length (163),
both top-two BF16 margins equal to zero, and maximum logit difference 0.21875 at
that position. Computing only the LM head in FP32 still changed candidate
ordering: differences already existed in the hidden state. The full FP32 model
had no argmax disagreements over the same 24 diagnostic positions, with maximum
logit difference about 0.000116. This supports a floating-point shape/rounding
explanation; it does not establish bit-identical execution across shapes.

No prompt was removed and no epsilon modified greedy selection. The first
full/cache gate stopped the run; the second preserved its failed partial data.
The final gate compares the same cached execution policy to HF. Full/cache
differences are still reported per prompt. Future multi-token speculative
verification versus sequential cached BF16 requires its own numerical gate.

Transformers 5.17 also changed cache crop behavior: `crop(0)` is a no-op.
The wrapper accepts an absolute retained prefix length and uses a negative
removal count, with an explicit no-op when no suffix needs discarding.

`source-identity.json` binds source, model, input records and private raw evidence
hashes. Failed partial timings are not mixed into the successful aggregate.

## Independent integrated cache audit

`cpu-cache-content-audit.py` uses synthetic tiny-model inputs only. Run it from
the repository root with `PYTHONPATH=. python reports/cached-target-20261008/cpu-cache-content-audit.py`.
It checks the actual projected draft KV at every round against a fresh complete
committed prefix, rather than checking only recorded cache lengths.

All 42 cases and 429 proposal rounds matched output tokens and cache lengths:
block sizes 1/3/7, prompt lengths 1/4/9, every rejection offset and full acceptance.
Maximum absolute projected-KV difference was 1.9446e-6, and maximum draft backbone
difference was 7.1526e-7. The audit initially encountered one 1.1325e-6 difference
against a 1e-6 tensor-closeness assertion; it now reports numerical errors directly
while retaining exact token equality. No epsilon changes token selection.
This independently supports the cache bookkeeping and supplies no BF16 fidelity
or speed guarantee.
