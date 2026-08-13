# V37T multi-sample uncertainty results

The registered analysis generated 16 independent frozen-model draws for every query in all 15
fold/seed cells (1,080 query-cell rows). Every draw entered the aggregate; no target-selected sample
was used. Samples are empirical repeated SDEdit outputs, not Bayesian credible draws.

## Point summaries

| Method | single-draw RRMSE | sample-mean RRMSE | sample-median RRMSE | matched DET RRMSE |
|---|---:|---:|---:|---:|
| V26 CalibSDEdit-MATCH | 0.754498 | 0.754491 | 0.754491 | 0.749071 |
| V26 PopSDEdit | 0.766193 | 0.766166 | 0.766166 | N/A |
| V27 EnergySDEdit-L0.5 | 0.747096 | 0.747096 | 0.747096 | 0.746193 |

The K=16 mean changes L0.5 temporal RRMSE by less than `5e-7` relative to a single draw and remains
slightly worse than matched EnergyDET. Stochastic averaging therefore does not add meaningful point
fidelity at this operating point.

## Coverage and proper-score diagnostics

| Method | empirical 80% coverage | width | interval score | constant-width score | CRPS | error–dispersion rho |
|---|---:|---:|---:|---:|---:|---:|
| V26 CalibSDEdit-MATCH | 0.006212 | 0.004202 | 4.607473 | 4.605679 | 0.461446 | -0.317952 |
| V26 PopSDEdit | 0.009676 | 0.006453 | 4.686514 | 4.683752 | 0.469722 | -0.321681 |
| V27 EnergySDEdit-L0.5 | 0.002866 | 0.001751 | 4.404181 | 4.403430 | 0.440711 | -0.044502 |

Nominal 50/80/90% L0.5 intervals cover only 0.157/0.287/0.355% of sensor-time targets. The empirical
interval score is slightly worse than a matched-average-dispersion constant-width reference.
Participant error–dispersion association is positive for only 6/15 participants for L0.5 (2/15 for
each unenergized V26 diffusion). The lower L0.5 CRPS primarily reflects its better point estimate;
the near-zero spread is not calibrated predictive uncertainty.

L0.5 sample variance is `1.52e-6`. Its support-projector-parallel variance (`1.24e-6`) exceeds its
complement variance (`1.87e-7`), so stochasticity is spatially concentrated in the intended span,
but its magnitude is too small and its error ranking too weak to be useful uncertainty. Identity-row
variance is `1.43e-6` across the 14 participants with identity rows in the fixed panel; the remaining
participant has no selected identity row and is reported N/A.

## Interpretation

The registered K=16 experiment is technically valid but strongly under-dispersed. It does not
support the claim that stochastic draws provide useful uncertainty beyond matched deterministic
estimation. CalibEnergy-SDEdit remains a method-centric attenuation–retention operating point;
uncertainty is exploratory and should not be promoted as a main contribution.

