# V36P exact-fiber validation

The OpenBMI binary task head has centered-head rank 1 and a 127-dimensional fiber for every
fold/seed checkpoint. Across 60 registered exact-preservation rows (HEAD_ONLY plus four strong
fiber channels × 6 folds × 2 seeds), there were zero prediction mismatches and zero fixed-head
balanced-accuracy change.

The largest centered-logit error was `3.001104e-7`; the largest softmax probability error was
`6.899204e-8`. Fixed-head calibration changes were at floating-point scale. H was recoverable from
every release to the same numerical tolerance.

Code-path checks establish that OneStep, Gaussian, Resample and SANDiff strong replacement
functions do not receive source U, source subject, or target support. Gaussian and SANDiff deploy
without a training-fiber bank. Resample donors and its class/confidence strata use only outer
non-test Session-1 fibers. No outer-test data entered Stage-A selection or Stage-B fitting.

This establishes exact fixed-function preservation on the external cohort. It does not establish
formal anonymity or mutual-information removal below the frozen-head-visible boundary.
