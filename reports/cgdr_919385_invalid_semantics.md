# Slurm 919385: invalid inference semantics

Slurm job `919385` completed its requested computation, but its inference and
metric outputs are invalid scientific evidence. The observation-state builder
multiplied the full context precision by the subspace attenuation, so the
zero-attenuation endpoint incorrectly removed precision from the orthogonal
complement. Its interval report also mixed seed-level rows with the
posterior-mean output rule.

The independently trained EEGdenoiseNet clean-prior checkpoint remains valid
and reusable: the defect was downstream in observation precision/guidance.
The corrected run keeps the frozen source fold, query, rank, ridge, eligibility
rules, seeds and checkpoint, and writes new results under
`results/cgdr/klados_v4_source_fold_sim45_corrected/`.
