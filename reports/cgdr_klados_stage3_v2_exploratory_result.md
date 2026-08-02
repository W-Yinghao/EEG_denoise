# Klados Stage-3 v2 exploratory result

Status: `completed_descriptive_no_broad_classifier`

Slurm jobs 919827 (three scope-specific U-Net checkpoints), 919830 (eight
development source records), and 919831 (aggregation) completed.  The records
are sim31--sim36, sim44, and sim45.  They are source records rather than
verified independent participants, and this run is neither formal G1 nor G3.

## Main sampler diagnostic

The table below reports medians across the eight development source records.
The query-derived oracle projector is non-deployable.

| Oracle-scope method | e_parallel | e_perp | RRMSE | Correlation | PSD distortion |
|---|---:|---:|---:|---:|---:|
| M1 warm-start DDIM | 1.2882 | 0.2886 | 0.8361 | 0.8547 | 0.8947 |
| Current M2 final-hard-Q | 1.9386 | 0.00000017 | 1.2374 | 0.7737 | 0.4133 |
| M4 stepwise proximal | 5.6889 | 0.0122 | 3.3535 | 0.8040 | 8.3484 |
| Oracle Qy | 1.0000 | 0.00000006 | 0.6213 | 0.9285 | 0.4362 |
| Oracle soft proximal | 0.8365 | 0.00000006 | 0.5240 | 0.9499 | 0.2282 |
| Paired-supervised U-Net | 0.4034 | 0.0083 | 0.2360 | 0.9904 | 0.1261 |

M1 is materially different from and better than current M2 on e_parallel,
RRMSE, and correlation, so the retained M2 failure is not evidence that every
sampler behaves identically.  M1 nevertheless remains worse than oracle Qy and
soft proximal on those metrics and pays much larger e_perp distortion.  M4 is
worse still.  These observations concern the tested clean-only prior and
frozen sampler configurations only.

The U-Net uses paired contaminated-to-clean supervision while M1/M2/M4 use an
unconditional clean prior.  Its strong result is therefore a differently
supervised exploratory reference, not a fair same-supervision G3 comparison.
The prospective operator-conditioned diffusion comparison addresses that gap
with the same paired exposure and a separately disclosed epsilon-prediction
objective.

## Operator-scope caveat

The v2 matching request used population fallback on 25% of the development
records.  Its old method summary groups requested matching rows together and
therefore cannot estimate a pure matching-P0 operator effect.  The result is
retained unchanged, but matching-vs-population claims are not drawn from it.
The prospective fixed-endpoint revision uses one common matching-eligible set
and reports eligible-only matching separately from the all-requested fallback
policy.

## Preserved status boundary

- Klados: `current_M2_no_incremental_value`
- diffusion family: `not_tested`
- formal G1/G3: `NOT_RUN_BLOCKED`
- this result: exploratory sampler and stronger-baseline diagnostic only

Machine-readable outputs remain under
`results/cgdr/klados_stage3_deterministic_scope_isolated_v2/development/`.
