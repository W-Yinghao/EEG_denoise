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

## Engineering and governance

- Base: `8dadb508fd2d50a089246c4e11c83b7b7628fa42`.
- Implementation: `13356f9e4454627c00ed7170fe2293e394410ee9`.
- Round A/selection: `fe8c94e7cd582032c713365f7c82217454377bb3`.
- Round B training: `3d92fa8`.
- Paired/natural result-producing commit: `5d014c6`.
- Ledger v1.2 commit: `b4948537123535ff46acdfb190fd3e6725fe3040`.
- Targeted tests: 23 passed (Slurm 937255).
- Clean git-archive import/tests: import passed and 23 tests passed (Slurm 937256).
- Sealed reads: 0. Query EOG/operator/event inference reads: 0/0/0.
- A-track and `taas_submission/**`: unchanged; manuscript was not compiled.
- No confirmation was run and no K8 experiment was used.

Full accepted/failed/recovery lineage is in `reports/slurm/v25_job_ids.txt`; checkpoints remain outside Git and are bound by `results/setcalibdiff_v25/checkpoint_manifest.csv`.
