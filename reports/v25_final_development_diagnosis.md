# V25 Final Development Diagnosis

This is development/model-building evidence, not confirmation. All summaries use 15 participants as the scientific unit, after aggregating windows, sessions, tasks, folds, and seeds within participant.

- Engineering: `valid`
- Raw support representation: `clear_development_signal` — deterministic MATCH improved clean RRMSE over the exact POP route by 0.00715 (bootstrap 95% CI 0.00118 to 0.01392; 10/15 positive) and over WRONG by 0.01673 (0.00780 to 0.02691; 14/15 positive).
- Strong population comparison: `support_better`
- Diffusion: `deterministic_better` — SetCalibDiff−SetCalibDET utility was −0.05703 (−0.06431 to −0.05008; 0/15 positive). It stayed negative in mild, medium, and severe strata, so there is no observed tail rationale for retaining this implementation.
- Natural trade-off: `artifact_reduction_insufficient` — diffusion MATCH was worse than POP for artifact remaining ratio (−0.08013 utility; 1/15 positive) and preservation (−0.12930; 0/15 positive).
- Next route: `G. remove current diffusion implementation`. Retain the deterministic raw-support result as development evidence, but do not advance this residual-diffusion implementation or open confirmation.

Exact participant effects, seed effects, frozen comparators, training exposure, and latency are in the accompanying CSV files.

The result does not close subject-aware diffusion as a family. It shows that the tested rank-8 learned residual diffusion adds no value after the V25 deterministic support correction, despite the support encoder itself carrying useful development signal.
