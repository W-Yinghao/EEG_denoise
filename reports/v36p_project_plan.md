# V36P project plan

V36P freezes the V34P/V35P exact-head-fiber method family and evaluates it on the external
54-participant, two-session OpenBMI/Lee2019 motor-imagery cohort. The estimand is whether an
exact function-preserving, model-only Fiber-SANDiff channel replicates across participants and
reduces training-exemplar exposure relative to bank-dependent resampling.

The fixed six-fold protocol assigns nine participants to outer test, nine participant-disjoint
participants to Stage-A validation, and 36 to Stage-A training. After EEGNet and full-sampler
epochs are selected, all 45 non-test participants' Session 1 trials are used for exact-epoch
refit. Outer-test Session 1 is the privacy gallery; Session 2 is the privacy query and task test.
No outer-test observation enters selection, normalization, fiber fitting, Gaussian fitting, or
resampling strata.

Methods are RAW, HEAD_ONLY, LEACE, Fiber-OneStep, one fixed class/logit-conditional
Fiber-Gaussian, training-bank Fiber-Stratified-Resample, and model-only Fiber-SANDiff. All strong
fiber channels preserve centered logits, softmax probabilities, fixed-head predictions, balanced
accuracy, and calibration to numerical tolerance. Finite head-aware attacks do not define mutual
information and cannot support privacy below the H-visible boundary.

Training-exposure evaluation reports exact and near copies, nearest training and held-out fiber
distances, nearest-donor concentration, and one registered nearest-training-distance membership
diagnostic. Sixteen independent releases per query are all retained for distribution analysis.
Latency is neither measured nor used. Waveform sealed data, A-track, and `taas_submission/**`
remain untouched.
