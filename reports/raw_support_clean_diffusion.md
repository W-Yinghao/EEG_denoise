# Raw temporal support-conditioned clean EEG diffusion

Decision: `RAW_TEMPORAL_SUPPORT_CLEAN_DIFFUSION_ROUTE_CLOSED` on the full BCI2b same-session development cohort.

The diffusion state is the full clean EEG waveform. Query inference sees corrupted EEG and sixteen raw temporal EEG support patches; it never sees query EOG, labels, participant ID, covariance/PSD summaries, or clean targets. Twenty-six of 27 protocol units are eligible; all nine participants remain the scientific denominator.

## Technical validity

The real-data technical gate passed: exact checkpoint/replay, finite gradients for all trainable tensors, support-set permutation invariance, nonzero MATCH/WRONG response, query-shuffle sensitivity, fixed-batch DET/DIFF reconstruction, and outer-training validation improvement over RAW. DET and DIFF each have 943,683 trainable parameters.

## Participant-first effects

- DeltaSA: mean +0.00150, median +0.00067, 5/9, one-sided exact p=0.363281, descriptive 95% interval [-0.00608, +0.00900].
- E_D: mean +0.03199, median +0.03345, 8/9, one-sided exact p=0.003906, descriptive 95% interval [+0.01715, +0.04755].
- E_K: mean +0.09670, median +0.09790, 9/9, one-sided exact p=0.001953, descriptive 95% interval [+0.08341, +0.11035].
- U_P: mean +0.00382, median +0.00230, 5/9, one-sided exact p=0.113281, descriptive 95% interval [-0.00086, +0.00939].
- U_W: mean +0.00710, median +0.00576, 8/9, one-sided exact p=0.041016, descriptive 95% interval [+0.00070, +0.01349].

U_P fails the frozen subject-utility gate (mean below +0.005 and only 5/9 positive); U_W is positive but cannot rescue U_P. Accordingly, no extra training seeds were submitted. E_D and E_K are positive secondary signals for this one seed, not evidence that subject conditioning succeeded.

## Absolute performance and safety

- RAW paired RRMSE: 0.49133.
- DET-CLEAN-MATCH paired RRMSE: 0.43279.
- DIFF-CLEAN-POP-K8 paired RRMSE: 0.40462.
- DIFF-CLEAN-MATCH-K8 paired RRMSE: 0.40080.
- Natural MATCH means: EOG attenuation +0.05157, preservation 0.68766, covariance 0.28695, MI-band distortion 0.07569, MI kappa 0.38722, ERD preservation 0.92388.
- MATCH-minus-POP safety margins: EOG attenuation -0.00546, preservation +0.00872, covariance -0.01186, MI-band distortion -0.00402, MI kappa -0.00318, ERD +0.00206.

## Historical reference

The frozen EOG-guided DIFF25-K8-POP8-R result has participant-first RRMSE 0.15657 over three seeds. It is reported only as a historical reference because it uses query EOG and a different estimator, and is not an information-matched arm of this EEG-only raw-support model.

## Boundary

This is one-seed development evidence. The frozen temporal Conv/set-token/cross-attention clean-conditional instance did not establish unseen-subject utility and is closed under the preregistered stopping rule. This does not imply a diffusion- or personalization-family-wide negative; no additional architectures, patch/token grids, or sampler variants were run.
