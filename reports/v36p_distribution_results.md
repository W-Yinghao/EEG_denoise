# V36P conditional distribution results

| Method | Covariance discrepancy | Energy distance | MMD | Variance retained |
|---|---:|---:|---:|---:|
| Fiber-OneStep | 0.9890 | 5.9211 | 0.18080 | 0.0677 |
| Fiber-Gaussian | 0.6566 | 0.5080 | 0.01424 | 1.0077 |
| Fiber-Stratified-Resample | 0.6293 | 0.4813 | 0.01351 | 1.0074 |
| Fiber-SANDiff | 0.6882 | 0.8809 | 0.03033 | 0.8688 |

Fiber-SANDiff clearly avoids the conditional-mean collapse of Fiber-OneStep, retaining `0.8688`
of fiber variance rather than `0.0677`. However, Fiber-Gaussian is better than Fiber-SANDiff on
energy distance, MMD, and variance calibration in all 12 fold/seed cells; covariance discrepancy
favors Gaussian in 7/12 cells. Resample remains slightly best on the aggregate distribution
metrics, but Gaussian is close while requiring no exemplar bank.

Sixteen-release diagnostics tell the same story. Within-H variance was `66.33` for Gaussian,
`66.64` for Resample, `55.98` for SANDiff, and zero for OneStep. Duplicate rates were zero for
Gaussian/SANDiff, approximately `0.0010` for finite-bank Resample, and one for OneStep.

Participant-first retrained-head balanced accuracy was `0.73204` for SANDiff, `0.72991` for
Gaussian, `0.73046` for Resample, and `0.73296` for OneStep. SANDiff-minus-Gaussian was `+0.00213`
(95% participant bootstrap CI `[-0.00231, +0.00667]`, 25/54 positive). This small recoverable-task
difference does not offset Gaussian's consistent distribution advantage.
