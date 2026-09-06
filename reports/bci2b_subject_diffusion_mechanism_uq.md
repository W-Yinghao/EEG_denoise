# BCI2b subject diffusion mechanism and UQ

Development exploration using frozen V11.1 checkpoints; no network was retrained.

B1 decision: `SUBJECT_CONTEXT_IS_CONSUMED_BY_SCORE_MODEL` (small, participant-heterogeneous score-context signal).
B2 decision: `WEAK_UNCERTAINTY_ASSOCIATION_ONLY`.

## Separable intervention

Operator effect G_A mean/median: +0.03348/+0.02911 (9/9 positive). Score-context effect G_C: +0.00107/+0.00281 (5/9 positive; all 3 seed means positive). Positive-synergy interaction I: +0.05520/+0.03028 (9/9 positive).
Descriptive participant bootstrap intervals: G_A [+0.02306, +0.04414], G_C [-0.00479, +0.00676], I [+0.02870, +0.08605].

The intervention keeps the score network frozen. A selects the external EOG-operator reconstruction anchor; C supplies a0_C and r_det_C to the unchanged score network. G_C is much smaller than G_A and heterogeneous across participants, so the label does not imply a large or uniform score effect. I uses the predeclared positive-synergy definition R_MP + R_PM - R_MM - R_PP. It does not alter the A-track claim.

## Probability scope

Paired construction supports `predictive dispersion and error ranking only`. Training seeds were evaluated separately and were not treated as posterior samples. The deterministic comparator is the three-seed DET-MATCH ensemble.

| metric | DIFF K32 | DET ensemble | DIFF-DET |
|---|---:|---:|---:|
| crps | 0.191845 | 0.182674 | +0.009172 [+0.001364, +0.017260] |
| energy_score | 0.258444 | 0.248973 | +0.009471 [-0.000337, +0.019732] |
| mean_rrmse | 0.099591 | 0.089042 | +0.010549 [+0.008825, +0.012649] |
| risk_coverage_auc | 0.092126 | 0.074931 | +0.017195 [+0.011861, +0.023079] |
| uncertainty_error_spearman | 0.083617 | 0.455714 | -0.372097 [-0.553841, -0.183500] |

Proper-score better: `False`; risk-coverage better: `False`; point estimate not materially worse: `False`. Thus positive dispersion-error association alone supports only the weak-association label.

Natural reversal diagnostic: 5/9 participant reversals, uncertainty reversal AUC 0.700. Outer-training-frozen q50/q80 cutoffs achieved query MATCH coverage 0.851/0.974.

Natural-output uncertainty is exploratory; no query outcome trained a gate or selected a threshold.
