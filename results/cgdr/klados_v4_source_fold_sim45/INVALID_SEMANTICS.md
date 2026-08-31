# Invalid scientific result: Slurm 919385

The files in this directory were produced by Slurm job `919385`. They are
retained only as failure evidence and must not be used as a CGDR result.
Their evidence status is `exploratory_pre_repair_not_gate_evidence`; formal G1
was not run and remains blocked.

The observation-state construction multiplied both the population precision
and the entire context precision by the subspace attenuation. At the zero
attenuation endpoint this incorrectly removed precision in the orthogonal
complement as well as in the calibrated subspace. The run also pooled
seed-level and posterior-mean rows in its confidence-interval summaries.

A corrected run uses the same frozen source fold, query, rank, ridge,
eligibility thresholds, seeds, training budget and existing population-prior
checkpoint. Its outputs are routed to
`results/cgdr/klados_v4_source_fold_sim45_corrected/`.

Only the inference and result metrics from `919385` are invalid. The
independently trained EEGdenoiseNet clean-prior checkpoint is reusable because
the defect was downstream, in observation precision/guidance construction.
