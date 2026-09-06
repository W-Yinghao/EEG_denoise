# V35P project plan

V35P is an evaluation-only correction to V34P. It freezes every EEGNet,
Fiber-OneStep, and Fiber-SANDiff checkpoint and adds only a non-neural population
fiber resampling channel. Strong releases are interpreted as exact channels whose
ideal subject-information boundary is the recoverable centered task head `H`.

Each outer fold uses only its six non-test Session-T fibers to define predicted-class
and centered-logit-norm tertile strata. Outer-test Session T is the attacker gallery;
Session E is the query. Every method is attacked using H, Z, [H,Z], and [H,U] with
both linear and adaptive models. Sixteen independent releases are used only for
distribution diagnostics; no sample is selected by its outcome.

Latency is neither measured nor used in selection or conclusions. No encoder or
neural sanitizer is trained.
