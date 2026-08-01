# CGDR interpretation audit after `bc68f59`

Date: 2026-08-02 (Europe/Paris)

```text
original_classifier_result: A_limited
post_absolute_baseline_audit: B_geometry_only
formal_G1_status: NOT_RUN_BLOCKED
```

## Immutable historical result

The existing machine-readable result at
`results/cgdr/klados_v4_repaired_mechanism_audit/result_summary.json` is not
modified by this audit. Its original classifier result is retained as
`A_limited`: query-derived oracle geometry under the development-selected M2
sampler improved over the same-sampler POP reference. That was a limited
source-record mechanism comparison, not evidence of a diffusion-specific
advantage.

Formal G1 remains **`NOT_RUN_BLOCKED`**. The 54 Klados items have no verified
participant mapping, source-record independence is not established, and the
query-derived oracle projector is a non-deployable mechanism upper bound.

## Original registered comparisons

The original development comparison, oracle-projector M2 minus same-sampler
POP, had median delta-e_parallel `-0.002689` and median delta-e_perp
`-0.033847` over 8/8 development source records.

On the 16 evaluation source records, oracle-projector M2 minus same-sampler POP
had median delta-e_parallel `-0.004969` (95% descriptive source-record
bootstrap interval `[-0.005291, -0.002745]`) and median delta-e_perp
`-0.040991` (`[-0.048310, -0.035819]`), with 16/16 paired records and no
failures.

Oracle orthogonal subtraction `Qy` minus corrupted identity had median
delta-e_parallel `-1.190589` (`[-1.840268, -0.613063]`) and essentially zero
median delta-e_perp over the same 16/16 records. This is the absolute geometry
baseline that the original classifier did not require A to match.

## Absolute-baseline audit

For every source record, define the following direct method-minus-baseline
difference from posterior-mean waveforms, with direction interpreted per
metric:

`delta = oracle-projector M2 - oracle orthogonal subtraction Qy`.

| Metric | Median delta | Records favoring Qy |
|---|---:|---:|
| e_parallel | `+1.1447468373519638` | 16/16 |
| e_perp | `+1.2269534435292062e-7` | 16/16 |
| RRMSE | `+0.6999285379182630` | 16/16 |
| correlation (higher is better) | `-0.1777859910940312` | 16/16 Qy better |

Thus M2 was worse than `Qy` on all four audited metrics in every evaluation
source record. For the first three metrics, positive differences are worse;
for correlation, the negative method-minus-Qy difference is worse. M2 still
beat POP, but a POP-relative pass cannot establish a
diffusion advantage when the output fails the stronger absolute geometry
baseline. The trained, information-matched deterministic multichannel U-Net
required by the paper was also absent, independently blocking an A decision.
The run had no frozen complete preservation decision spanning e_perp, PSD,
correlation and the trained deterministic comparison, so the independent
all-preservation requirement is incomplete as well.

The corrected post-audit classification is therefore
**`B_geometry_only`**. Geometry is useful; a diffusion-specific benefit has
not been demonstrated.

> Query-derived oracle geometry is useful under hard-Q consistency, but the
> diffusion-generated component is dominated by deterministic oracle
> orthogonal subtraction. Diffusion-specific value is not supported.

## Why M2 is not itself diffusion evidence

M2 applies deterministic final hard Q-consistency. If `Pi` is the selected
projector, `Q = I - Pi`, `x_ddim` is the state before the final consistency
operation, and `y` is the observation, its output is exactly

`x = Pi x_ddim + Q y`.

Consequently `Q(x-y)=0` and `Pi x = Pi x_ddim`. The M2 label identifies a
hybrid output rule; it does not prove that the iterative prior contributes
value beyond `Qy` or a trained deterministic model.

## Corrected classifier contract

Classification A now requires all of the following independently:

1. iterative restoration is better than matched POP;
2. it is noninferior to or better than `Qy`;
3. it is better than a trained information-matched deterministic U-Net;
4. every registered preservation requirement passes.

No sampler ID, including M0--M4, can substitute for these comparisons. If
geometry fails, the result is C. If geometry works but any A condition fails or
the deterministic comparator is missing, the result is B.
