# V24 Round B

Three seeds across all five development folds completed for the population anchor,
TemporalEOGNet/PA-EL-DET, and PA-EL-SCAD-K1. Checkpoint selection used validation
participants only, and the population anchor was frozen before training the subject
branch.

Participant-first paired clean-RRMSE utilities (positive favors MATCH or diffusion)
were:

| Contrast | Mean | Median | Positive | Bootstrap 95% CI |
|---|---:|---:|---:|---:|
| DET MATCH−POP | -0.041906 | -0.011188 | 2/15 | [-0.073964, -0.014724] |
| DET MATCH−WRONG | +0.003479 | +0.028187 | 10/15 | [-0.031922, +0.033083] |
| SCAD MATCH−POP | -0.202303 | -0.143192 | 0/15 | [-0.278762, -0.134745] |
| SCAD MATCH−WRONG | +0.021900 | +0.082165 | 11/15 | [-0.059066, +0.091608] |
| SCAD-K1−DET1 | -0.160397 | -0.135616 | 0/15 | [-0.209513, -0.117166] |

K8 and DET8 were not run because K1 was uniformly worse than DET1 at the
participant level; averaging was not used to rescue a failed K1 estimator.
