# V38P final diagnosis

Final positioning: **C**. Non-diffusion transport is preferable after the canonical run and the single registered repair.

The implementation is engineering-valid: 12/12 canonical and 12/12 repair cells completed, all 54
participants were outer-tested once, every donor came from the 36-person outer-training bank, K=8
draws were retained without target selection, and method-specific attackers were retrained on
released Session-1 gallery representations before Session-2 testing.

Canonical SARD retained task performance (fixed-head BA 0.734074 versus RAW 0.734537), but did not
transport away source identity under the registered attacker. Adaptive source BA rose from 0.333241
for RAW to 0.502037 for SARD. Participant-first SARD-minus-RAW privacy utility was -0.168796
(95% bootstrap CI [-0.187963, -0.150461], 0/54 positive). SARD also lagged OneStep by -0.147222
[-0.164074, -0.131201], again 0/54 positive.

The distribution result points in the same direction. SARD energy distance was 0.504167, compared
with 0.299691 for OneStep, 0.213304 for the model-only Gaussian, and 0.190002 for the empirical
resampler. SARD produced nonzero stochastic diversity and no near copies, but Gaussian also had no
near copies while providing substantially better privacy and donor-distribution fidelity. The K=8
augmentation differences (+0.000833 versus OneStep and +0.003611 versus Gaussian) were too small to
constitute a clear registered-axis advantage.

The sole allowed repair increased the existing source-adversary weight from 0.1 to 0.5. It changed
no data, donor, encoder, architecture, sampler, attack, fold, or seed setting. Repaired adaptive
source BA worsened further to 0.559259, so the repair was rejected using participant-disjoint
validation and is retained only as a negative ablation. Per the registered V38P rule, this route now
stops; no further model family is opened.

Registered axis effects:

```json
{
  "augmentation_vs_gaussian": 0.0036111111111111205,
  "augmentation_vs_one_step": 0.0008333333333333526,
  "distribution_energy_vs_gaussian": -0.2908629568026574,
  "distribution_energy_vs_resample": -0.3141652507320978,
  "exemplar_exposure_vs_resample": 0.0,
  "fixed_task_vs_raw": -0.0004629629629628873,
  "privacy_vs_raw": -0.16879629629629622,
  "retrained_task_vs_raw": -0.001574074074074061
}
```

Seeds were not treated as biological samples. Sealed reads were zero and the manuscript was
unchanged. No formal anonymity, exact subject-information removal, waveform denoising, or universal
generalization is claimed.
