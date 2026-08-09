# BCI2b same-session subject-aware diffusion replication

Three-seed development replication of the frozen V11.1 method. Deployment is EOG-guided; this is not confirmation. V12 is archived as a technically unvalidated enhancement route and was neither repaired nor screened here.

Decision: `SUBJECT_CONDITIONING_WITHIN_FIXED_DIFFUSION_REPLICATED`.

Coverage: 9/9 BCI2b participants, three same-session units per participant, three fixed training seeds, K=8 and DDIM25. Windows and seeds were aggregated within participant and were not treated as scientific replicates.

## Primary subject-conditioning effects

| effect | mean | median | positive | sign-flip p | descriptive 95% CI |
|---|---:|---:|---:|---:|---:|
| U_P | +0.03473 | +0.03338 | 9/9 | 0.00195 | [+0.02899, +0.04079] |
| U_W | +0.04836 | +0.03862 | 9/9 | 0.00195 | [+0.02987, +0.07220] |
| U_S | +0.46640 | +0.42898 | 9/9 | 0.00195 | [+0.39931, +0.54669] |

| seed | mean U_P | positive | mean U_W | positive |
|---:|---:|---:|---:|---:|
| 20260808 | +0.03654 | 9/9 | +0.04667 | 9/9 |
| 20260810 | +0.03465 | 9/9 | +0.04779 | 9/9 |
| 20260811 | +0.03299 | 9/9 | +0.05062 | 9/9 |

## Absolute method results

| method | paired RRMSE | correlation | delta SNR | EOG attenuation | preservation | covariance | MI kappa |
|---|---:|---:|---:|---:|---:|---:|---:|
| RAW | 0.48851 | 0.89717 | +0.0000 | +0.0000 | 1.0000 | 0.0000 | 0.3528 |
| LINEAR-MATCH | 0.08760 | 0.99304 | +18.0654 | +0.0973 | 0.8123 | 0.0909 | 0.3490 |
| DET-MATCH | 0.11225 | 0.99319 | +12.5072 | +0.1039 | 0.7979 | 0.1248 | 0.3526 |
| DIFF-POP | 0.14823 | 0.98818 | +9.9214 | +0.0987 | 0.7920 | 0.1241 | 0.3402 |
| DIFF-MATCH | 0.11350 | 0.99289 | +12.3898 | +0.1043 | 0.7982 | 0.1237 | 0.3433 |
| DIFF-WRONG | 0.16186 | 0.98415 | +9.6218 | +0.0987 | 0.7926 | 0.1237 | 0.3443 |

DIFF-MATCH does not beat LINEAR-MATCH or DET-MATCH on aggregate paired RRMSE. That comparison is reported transparently and is distinct from the replicated result that matching subject conditioning improves the same fixed diffusion denoiser relative to POP and WRONG contexts.

## Natural safety

EOG attenuation mean/min `+0.1043/+0.0653`, preservation mean/min `0.7982/0.7618`, covariance mean/max `0.1237/0.1660`, MI-band distortion mean/max `0.0719/0.0954`, MI-kappa relative to DIFF-POP mean/min `+0.0032/-0.0199`.

Participant-level reversals (out of 9): nonpositive EOG attenuation 0, preservation below 0.78 1, covariance above 0.15 2, and MI-kappa delta below -0.02 0.

LINEAR-MATCH and DET-MATCH remain transparent comparators, but whether diffusion beats them is not this replication's primary gate. The supported claim is that matching subject conditioning improves a fixed diffusion denoiser over population and frozen cyclic unseen-WRONG contexts.

## Compute and donor sensitivity

K8/DDIM25 used 200 posterior calls, mean latency `0.0166` s/window, and peak allocated GPU memory `177` MB.

Outer-training-seen donor sensitivity: mean/median utility `+0.03976/+0.04114`, 27/27 recipient-seed comparisons positive. Outer-training-unseen: `+0.04876/+0.03398`, 27/27 positive. Only one truly unseen donor exists per fold, so no claim of three unseen donors is made.

The frozen cyclic unseen-WRONG is the primary continuity control. All other compatible donors were scored separately before recipient-level averaging; outer-training-seen and outer-training-unseen sensitivity are reported separately in `multi_donor_sensitivity.csv`.

Recovery record: evaluator job 930791 was invalid because it requested absent K32 outputs; multi-donor jobs 930846/930855/930864 were invalid because the GPU Python 3.9 loader could not execute `zip(strict=True)`. Neither set entered aggregation. The repaired path built evaluator-blind donor operators on CPU and replayed unchanged GPU inference.
