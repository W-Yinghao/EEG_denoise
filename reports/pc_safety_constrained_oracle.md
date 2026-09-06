# P-C safety-constrained oracle

Decision: `GO_MINIMAL_SELECTOR`.

The historical unconstrained result is retained as `legacy_unconstrained_oracle_not_safety_constrained`. This diagnostic mixes frozen STRONG-POP and DIFF-MATCH outputs, then reruns the primary evaluator. It is non-deployable because selection uses outcomes.

| Dataset | Coverage | Successful/denominator | Artifact utility (95% CI) | Preservation | PSD utility | Covariance utility |
|---|---:|---:|---:|---:|---:|---:|
| klados | 0.5 | 15/16 | +0.09441 [+0.05180, +0.15379] | +0.03163 | +0.06392 | +0.06448 |
| klados | 0.8 | 14/16 | +0.08810 [+0.04635, +0.14891] | +0.03025 | +0.05865 | +0.06310 |
| klados | 1.0 | 9/16 | +0.08559 [+0.03181, +0.16659] | +0.02989 | +0.06369 | +0.06733 |
| sgeyesub | 0.5 | 48/58 | +0.01483 [+0.00770, +0.02318] | +0.06461 | +0.34016 | +0.03182 |
| sgeyesub | 0.8 | 44/58 | -0.00156 [-0.01250, +0.01088] | +0.08530 | +0.40343 | +0.03978 |
| sgeyesub | 1.0 | 44/58 | -0.05548 [-0.07495, -0.03504] | +0.08565 | +0.40383 | +0.04013 |

## Minimal deployable selector

Status: `completed_single_minimal_deployable_selector`. Inference features use only observed query EEG and calibration support; outcomes are excluded from inference.

| Dataset | Successful/denominator | Coverage | Artifact utility (95% CI) | Preservation | PSD utility | Covariance utility |
|---|---:|---:|---:|---:|---:|---:|
| klados | 16/16 | 0.443 | +0.02492 [+0.01524, +0.03528] | +0.00806 | +0.01698 | +0.01391 |
| sgeyesub | 58/59 | 0.000 | +0.00000 [+0.00000, +0.00000] | +0.00000 | +0.00000 | +0.00000 |
