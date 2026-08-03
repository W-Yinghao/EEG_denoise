# Mainline subject-aware residual diffusion

Scientific decision: **generic_residual_diffusion_supported_subject_aware_mainline_not_supported**.

| Question | Passed | Mean effect | 95% interval |
|---|---:|---:|---:|
| H1 DIFF-MATCH vs ONE-STEP-MATCH | True | 0.004507 | [0.002693, 0.006649] |
| H2 DIFF-MATCH vs DIFF-POP | False | 0.000000 | [-0.000001, 0.000001] |
| H2 DIFF-MATCH vs DIFF-SHUFFLED | False | 0.000001 | [0.000000, 0.000002] |
| H3 natural EEG safety | False | — | frozen -0.02 margins |

All six methods used the same query per unit. DIFF-POP, DIFF-MATCH and DIFF-SHUFFLED shared one checkpoint and common random numbers; query EOG and annotations were opened only after outputs were frozen.

Coverage: 16 Klados evaluation source records; 58/59 SGE participant stems produced performance results, with 1 preblocked singleton retained only in the feasibility denominator. Klados is paired source-record evidence; SGE is participant/stem development evidence, not untouched confirmation.

## Six-method compact results

| Dataset | Method | Clean RRMSE | EOG coherence reduction | Non-artifact preservation | Output/input RMS | Latency/window (s) |
|---|---|---:|---:|---:|---:|---:|
| klados | RAW | 1.29481 | — | — | 1 | 0 |
| klados | POP | 0.399022 | — | — | 0.650107 | 0.00994752 |
| klados | ONE-STEP-MATCH | 0.402964 | — | — | 0.651692 | 0.014213 |
| klados | DIFF-POP | 0.398465 | — | — | 0.650454 | 0.990316 |
| klados | DIFF-MATCH | 0.398457 | — | — | 0.650433 | 0.979608 |
| klados | DIFF-SHUFFLED | 0.398602 | — | — | 0.650487 | 0.981361 |
| sgeyesub | RAW | — | 0 | 1 | 1 | 0 |
| sgeyesub | POP | — | 0.323079 | 0.776205 | 0.77251 | 0.00172975 |
| sgeyesub | ONE-STEP-MATCH | — | 0.32567 | 0.773214 | 0.773109 | 0.00327545 |
| sgeyesub | DIFF-POP | — | 0.329127 | 0.678043 | 0.79129 | 0.599277 |
| sgeyesub | DIFF-MATCH | — | 0.329127 | 0.678038 | 0.791291 | 0.598963 |
| sgeyesub | DIFF-SHUFFLED | — | 0.329126 | 0.678011 | 0.791298 | 0.598601 |
