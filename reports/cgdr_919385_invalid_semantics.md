# Slurm 919385: invalid inference semantics

Slurm job `919385` completed its requested computation, but its inference and
metric outputs are invalid scientific evidence. The observation-state builder
multiplied the full context precision by the subspace attenuation, so the
zero-attenuation endpoint incorrectly removed precision from the orthogonal
complement. Its interval report also mixed seed-level rows with the
posterior-mean output rule.

Evidence status is `exploratory_pre_repair_not_gate_evidence`. The
single-source-record exploratory effect-direction check failed; formal G1 was
not executed. No direction computed from this run is gate evidence.

The old T=200, channel-independent EEGdenoiseNet checkpoint is retained only
as an engineering ablation. It is barred from repaired scientific results,
which require a newly trained T=1000, 19-channel Klados population prior. The
historical corrected output remains under
`results/cgdr/klados_v4_source_fold_sim45_corrected/` and is not promoted.
