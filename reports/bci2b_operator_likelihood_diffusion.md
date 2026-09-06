# BCI2b operator-likelihood diffusion

Decision: `CURRENT_OPERATOR_LIKELIHOOD_GUIDED_INSTANCE_NO_GO`. This development-only experiment freezes the population anchor, deterministic residual, score context, checkpoint, DDIM25 trajectory, and K=8 common noise. Only the support-derived likelihood center and precision change across POP/MATCH/WRONG.

U_P_OP is -0.04441 (median -0.01580; 4/9), U_W_OP is +0.08734 (median +0.10913; 8/9), and G_MATCH is -0.09076. For transparency, frozen CURRENT RRMSE is RAW 0.48851, LINEAR-MATCH 0.08760, DET-MATCH 0.11287, DIFF-POP 0.14869, and DIFF-MATCH 0.11351. MATCH natural metrics are EOG attenuation +0.09801, preservation 0.80618, MI-band distortion 0.07193, covariance 0.10317, and MI kappa 0.33903. Localized backup authorized: `False`. It is not run unless the global subject effects pass but safety deteriorates. DIFF-vs-DET/LINEAR remains transparent secondary evidence, not this subject-awareness gate.

## Failure and recovery record

Job 931737 exposed Python-3.9-incompatible `zip(strict=...)` and was superseded after the compatibility-only fix. Jobs 931741/931742 were superseded technical probes; 931743 established the original technical contract. Job 931799 exposed a missing frozen PSD-floor config and was replaced by 931804/931805. Final padding-identity validation passed in 931847, after which all 27 inference tasks were replayed before the final evaluator. These engineering failures are excluded from scientific aggregation.
