# Clean-neural subject headroom

Routing: `CLEAN_NEURAL_ALIGNED_DIFFUSION_AUTHORIZED`; authorized dataset: `bci2b`. Support EOG was used only to exclude gross ocular support windows and was not a conditioning feature. Query EOG and later-query covariance were opened only by the evaluator.

## BCI2A
Decision: `CLEAN_NEURAL_COVARIANCE_HEADROOM_NOT_ESTABLISHED`. Evaluated 9/9 participants.
- H_P_airm: mean +0.15087, median +0.12087, 6/9, exact one-sided p=0.128906.
- H_W_airm: mean +0.52241, median +0.52904, 8/9, exact one-sided p=0.003906.
- H_P_logeuclidean: mean +0.14964, median +0.08816, 6/9, exact one-sided p=0.091797.
- H_W_logeuclidean: mean +0.48758, median +0.49551, 9/9, exact one-sided p=0.001953.

## BCI2B
Decision: `CLEAN_NEURAL_COVARIANCE_HEADROOM_DETECTED`. Evaluated 9/9 participants.
- H_P_airm: mean +0.19079, median +0.17133, 7/9, exact one-sided p=0.011719.
- H_W_airm: mean +0.30971, median +0.28070, 9/9, exact one-sided p=0.001953.
- H_P_logeuclidean: mean +0.19359, median +0.17122, 7/9, exact one-sided p=0.011719.
- H_W_logeuclidean: mean +0.31052, median +0.28156, 9/9, exact one-sided p=0.001953.
