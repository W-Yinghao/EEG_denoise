# V33P final diagnosis

```text
engineering: valid
training_pool: six non-test participants per fold
checkpoint_selection: deployed full K1/10-step sampler
privacy_vs_RAW: clear reduction
fixed_task_vs_RAW: small heterogeneous cost
full_sampler_vs_single_timestep: small directionally favorable change
SANDiff_vs_one_step: practically equivalent
permitted_privacy_weight_repair: not used
waveform_sealed_reads: 0
```

## Final method positioning

```text
SANDiff and one-step practically equivalent
```

The V32P positive candidate survives in the narrower sense that SANDiff still
defines a useful privacy–utility operating point after full-pool refitting and
deployed-sampler selection. Adaptive subject recall and verification AUROC both
improve relative to RAW for all nine participants. The task trade-off is not
uniformly positive, and SANDiff does not establish superiority over LEACE or
the matched one-step sanitizer.

No small repair was triggered because canonical full-pool SANDiff produced a
large relative adaptive privacy reduction and remained numerically stable. The
appropriate next step, if pursued, is a larger participant cohort—not another
BCI-IV-2a architecture search.

This is development evidence only. It does not establish formal anonymity,
causal nuisance identification, cross-dataset generalization, or clinical
deployment readiness.
