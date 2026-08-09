# V11.2 score-component audit

Zero-training development audit. `S_K = LINEAR-MATCH + (DIFF-MATCH - DET-MATCH)` is a score-component diagnostic, not a deployable method. K8 and K32 are separate global panels.

Decision: `CURRENT_ARTIFACT_RESIDUAL_SCORE_OBJECT_CLOSED`.

## K8

RRMSE effect mean/median: -0.016484/-0.012479; positive 0/9; P4–P9 0/6. Spectral delta -0.012081; EOG +0.000429; preservation -0.001516; covariance -0.002321; shared-decoder kappa +0.005991.

## K32

RRMSE effect mean/median: -0.011631/-0.008418; positive 0/9; P4–P9 0/6. Spectral delta -0.008083; EOG +0.000392; preservation +0.000143; covariance -0.002358; shared-decoder kappa +0.005824.

The beta_det/beta_score grid is a post-hoc mechanism ceiling only. It did not select a participant-specific operating point or alter V11.1.
