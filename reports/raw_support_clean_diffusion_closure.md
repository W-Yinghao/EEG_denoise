# BCI2b raw-support clean diffusion: final scientific closure

This is a no-training participant-first reaggregation of frozen outputs. The route is support-EOG-assisted because EOG was used to exclude gross-artifact support patches; query inference itself receives EEG only.

## Frozen labels

- `SUPPORT_EOG_ASSISTED_RAW_TEMPORAL_CONTEXT`
- `MATCH_OVER_STRONG_POP_NOT_ESTABLISHED`
- `DONOR_SPECIFICITY_SUGGESTIVE`
- `MULTISAMPLE_AVERAGING_GAIN_PRESENT`
- `DIFFUSION_OVER_COMPUTE_MATCHED_DETERMINISTIC_NOT_TESTED`
- `RELATIVE_MATCH_VS_POP_SAFETY_PASSED`
- `ABSOLUTE_NATURAL_SAFETY_NOT_ESTABLISHED`

## Effects

- K1 U_P: +0.00703 mean, +0.00636 median, 7/9, two-sided exact p=0.105469.
- K1 U_W_donor_mean: +0.00980 mean, +0.01473 median, 8/9, two-sided exact p=0.089844.
- K1 U_W_donor_median: +0.01050 mean, +0.01500 median, 8/9, two-sided exact p=0.046875.
- K1 DET_minus_DIFF: -0.06472 mean, -0.06249 median, 0/9, two-sided exact p=0.003906.
- K8 U_P: +0.00382 mean, +0.00230 median, 5/9, two-sided exact p=0.226562.
- K8 U_W_donor_mean: +0.00751 mean, +0.00657 median, 8/9, two-sided exact p=0.074219.
- K8 U_W_donor_median: +0.00884 mean, +0.00902 median, 7/9, two-sided exact p=0.027344.
- K8 DET_minus_DIFF: +0.03199 mean, +0.03345 median, 8/9, two-sided exact p=0.007812.
- K8_minus_K1: +0.09670 mean, +0.09790 median, 9/9.

WRONG includes only actual raw-support donor arms. Donors are scored separately before recipient-level mean/median aggregation; donor-recipient pair wins and leave-one-donor-out sensitivity are saved separately.

Checkpoint raw/EMA/optimizer/RNG fields exist and deterministic reload was tested, but interrupted-training resume equality was not actually executed. No full resume-validity claim is made.

The relative MATCH-versus-POP safety margins passed, while absolute natural safety was not established. This development closure neither changes the historical model nor supports a family-wide conclusion.
