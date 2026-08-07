# Fold-local EB robustness v8.1

Primary support is 120 s; 60 s is secondary. Builder, predictor fitting, held-out inference, and evaluator outputs are physically separated.

{
  "status": "completed_eb_robustness",
  "scope": "descriptive_conditional_overlapping_outer_training",
  "compatible_stems": 58,
  "availability_denominator": 59,
  "primary_budget_seconds": 120,
  "means": {
    "full_vs_pop": 0.19031482723468812,
    "fixed_vs_pop": 0.2206604526667268,
    "reliability_vs_pop": 0.21532330555975315,
    "full_vs_fixed": -0.038502024699970314,
    "full_vs_reliability": -0.03536222724031532,
    "full_vs_wrong": 0.29867550588456443,
    "full_vs_shuffled": 0.31280264614709974
  },
  "loso_120_mean_relative_improvement": 0.20475655795960646,
  "loso_120_positive_count": 53,
  "bootstrap_not_run": "predictor_refit_per_replicate_required",
  "leave_one_study_out": "completed_development_robustness_not_confirmation"
}
