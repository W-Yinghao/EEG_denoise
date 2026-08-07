# SGE-EB-SCORE-ADAPTER-v7

## Route decision

`DIFFUSION_IMPLEMENTATION_OR_OBJECTIVE_INVALID`. The sampler mathematics passed, but no tested objective overfit all three real SGE batches. Stage B therefore remained fail-closed: the EB oracle ceiling, expanded-pair builder, population backbone and subject-deviation score adapter were not run. No v6 seeds were added.

## Diffusion validity

Oracle-v roundtrip relative error ranged from 7.393e-08 to 7.427e-08. Over 20,000 fixed-batch updates per fold, weighted-v mean RRMSE was 0.9255, unweighted-v 0.9232, and epsilon-pred 8101.5; pass counts were zero for all objectives. K convergence for the historical checkpoint was mean paired RRMSE K=1 1.6668, K=8 0.7652, K=32 0.5725; this diagnostic does not alter frozen K=8 results.

## Corrected v6 context controls

WRONG is scored per donor before donor utilities are averaged. Operator-only uses common reliability; deployed comparisons use support-derived MATCH/WRONG reliability and an equal-participant outer-training population reliability.

- deployed_support_reliability U_P: mean -0.017565, median -0.009808, 95% CI [-0.023079, -0.012193], positive 5/58.
- deployed_support_reliability U_W_donor_mean: mean -0.002273, median -0.002369, 95% CI [-0.009079, +0.004379], positive 26/58.
- operator_only_common_reliability U_P: mean -0.017572, median -0.010203, 95% CI [-0.022659, -0.012661], positive 5/58.
- operator_only_common_reliability U_W_donor_mean: mean -0.002280, median -0.000676, 95% CI [-0.009989, +0.005373], positive 27/58.

The inference files and evaluator truth are physically separated for all 58 compatible stems. Trial IDs/time ranges were replayed from raw records and confirm disjoint generator/clean/artifact roles. The historical scientific status is `CURRENT_STATIC_TRANSFER_SUMMARY_INSTANCE_NO_GO / DIFFUSION_OPTIMIZATION_VALIDITY_NOT_ESTABLISHED / DYNAMIC_TRANSFER_SUBJECT_AWARENESS_NOT_CLEANLY_TESTED`.
