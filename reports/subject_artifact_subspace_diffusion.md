# Support-calibrated artifact-subspace diffusion

Decision: **current_support_calibrated_artifact_subspace_diffusion_not_fully_supported**.

| Question | Mean effect | 95% CI | Supported |
|---|---:|---:|---:|
| Subject calibration: MATCH−POP | -0.00514551 | [-0.011260878070625936, 0.0007053212738483179] | False |
| Specificity: MATCH−WRONG | 0.00286179 | [-0.0030262215657071347, 0.008666526052281975] | False |
| Diffusion point estimate: DIFF−DET | 0.0592242 | [0.03359608827236877, 0.08789631440913759] | True |
| Posterior utility | — | risk–coverage/calibration | True |
| Natural EEG safety | — | frozen −0.02 margins | True |

The model diffuses only bounded rank-two artifact coefficients. The support basis defines query coordinates and reconstruction; query EOG, eye tracking, labels, outcomes, participant identity, and best-of-K selection are absent from inference. Orthogonal-complement consistency means preservation only relative to the estimated artifact basis, not preservation of all neural signal.

## Core methods

| Dataset | Method | Clean RRMSE | EOG coherence reduction | Low-artifact preservation | Continuous-segment PSD distortion |
|---|---|---:|---:|---:|---:|
| klados | RAW | 1.29481 | — | — | — |
| klados | POP | 0.399022 | — | — | — |
| klados | DIFF-POP | 0.476743 | — | — | — |
| klados | DIFF-MATCH | 0.532465 | — | — | — |
| klados | DIFF-WRONG-SAME-CELL | 0.554289 | — | — | — |
| klados | DET-MATCH | 0.591689 | — | — | — |
| klados | DIFF-MATCH-K1 | 0.56133 | — | — | — |
| klados | DIFF-MATCH-NO-IDENTITY | 0.487698 | — | — | — |
| sgeyesub | RAW | — | 0 | 1 | 0 |
| sgeyesub | POP | — | 0.323079 | 0.776205 | 0.382978 |
| sgeyesub | DIFF-POP | — | 0.237468 | 0.897314 | 0.150934 |
| sgeyesub | DIFF-MATCH | — | 0.232322 | 0.889125 | 0.16018 |
| sgeyesub | DIFF-WRONG-SAME-CELL | — | 0.22946 | 0.893773 | 0.15433 |
| sgeyesub | DET-MATCH | — | 0.230069 | 0.868339 | 0.203863 |
| sgeyesub | DIFF-MATCH-K1 | — | 0.234984 | 0.872444 | 0.179377 |
| sgeyesub | DIFF-MATCH-NO-IDENTITY | — | 0.246868 | 0.796261 | 0.357765 |

Coverage: 16 Klados source records and 58/59 SGE stems. Klados is paired source-record mechanism evidence; SGE is real-EEG development evidence, not independent confirmation.
