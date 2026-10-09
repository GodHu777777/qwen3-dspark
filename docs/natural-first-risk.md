# Existing quality32 first-token risk reanalysis

`scripts/analyze_natural_first_risk.py` reads only recovered historical quality32
artifacts for steps128,512,1280 and the original collector source snapshot. It
uses Python's standard library. It does not load a model, follow bound paths to
weights or generation data, regenerate trajectories, fit anything or access test
data. Output directories must be new. Production sampling is unchanged.

For each original round, emitted-before progress starts at1 for the initial
sampled target token and adds every preceding committed round token. The script
independently checks `cache_before - prompt_length + 1`, cache increments, actual
committed output sequence, final output length, block/round identities, prefix
labels and all original run/aggregate counters. Bins are fixed at0–31,32–63,
64–95,96–127; no proposal occurs at progress0. EOS flags come from the original
audited run and round termination records; the analyzer does not reload the
model's generation configuration to reclassify token IDs.

A valid selected first-position pair has finite p0≥0 and finite q0>0. The script
stores p0/q0 and alpha=min(1,p0/q0) in private round output. Alpha is the selected
token's acceptance probability; rejection risk is1-alpha. Binary first-prefix
acceptance is exactly `accepted>=1`, checked against the original prefix label.
Finite selected scalars whose quotient overflows float64 retain alpha1 and a
50-digit decimal ratio representation; they are not discarded as missing.

Public `summary.json` keeps all32 ordinals, per-prompt summaries, pooled round
summaries and equal-prompt means. Equal-prompt means use only prompts with a
nonempty denominator, explicitly reporting available/32 for every metric. The
same summaries apply within each fixed progress bin, with prompts reached and
valid-risk counts. All-label observed rates and observed rates restricted to
valid p0/q0 records are separate. Missing or invalid selected pairs never become
zero acceptance; their reasons and counts remain explicit. Missing files, broken
identities or failed structural checks produce an evidence gap with all32
unavailable ordinals. No replacement data are generated.

Zero-round initial-EOS prompts and EOS/budget coverage remain visible. Reaching
the token budget can coincide with EOS; `budget_reached` denotes output length,
while EOS is the original termination flag. Progress-bin populations differ as
prompts finish, so later-bin summaries are not matched progress experiments.
Per-prompt checkpoint changes preserve declines and unavailable values.

Each checkpoint follows its own sampled prefixes and horizons. Comparisons are
descriptive and cannot identify a causal training-data mismatch or exposure-bias
mechanism. A selected-token statistic is not full-vocabulary per-state overlap.
Rounds within prompts are correlated; no independent-round confidence intervals,
gamma/admission policy search, calibration, checkpoint reselection or speed
projection is performed. Current recovered file hashes are not historical
precommitted raw-file hashes. Parent recovery/provenance evidence separately
checks original published bindings, source identities and aggregates.
