# Matched-prefix diagnostic preparation

This is design evidence, not a model or distribution-overlap result.
`initial-state-identity.json` records a CPU comparison of already recovered
quality32 cases and outputs: all 32 initial states are identical across steps
128, 512 and 1280, including prompt tokens and the initial target output token.
No token values, prompt IDs or seeds are exported. Source file hashes bind the
present recovered bytes; the recovery provenance limit from S47 still applies.

The proposed next diagnostic is described in
[the design](../../docs/matched-prefix-diagnostic.md). The six historical full
probability probes have not been recovered or loaded in this preparation. No
model forward, new generation, training, GPU experiment or final-test access ran.
