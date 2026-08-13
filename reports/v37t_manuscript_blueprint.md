# V37T TAAS manuscript blueprint

## 1. Introduction

Motivate unseen-user ocular denoising with query-disjoint calibration. State the narrowed claim:
support-conditioned diffusion supplies a controllable attenuation–retention point and a testable
stochastic output, not a safe or uniquely identified personal operator.

## 2. Related work

Cover EEG ocular denoising, conditional diffusion, calibration/support conditioning, and
uncertainty. Mention LEACE/fiber privacy only as background; keep the V32P–V36P analysis in its
independent paper.

## 3. Query-disjoint support-conditioned diffusion

Define the fold, support/query separation, online paired construction, raw-support encoder,
population and matched/wrong conditions, training/inference auxiliary boundaries, and the fixed
development panel.

## 4. CalibEnergy-SDEdit

Specify the V25 deterministic support anchor, V26 sensor-coordinate warm-start SDEdit, EEG-only
temporal confidence, rotation-invariant support projector, and V27 closed-form final energy. Freeze
L0.5 before the V37T uncertainty analysis.

## 5. Paired denoising and support intervention

Show absolute results alongside MATCH-population, MATCH-mean-wrong, all-donor ranks,
lagged/shuffled controls, and V31 exact support-duration. Interpret support value as heterogeneous
and not uniquely donor-specific.

## 6. Natural attenuation–retention trade-off

Show L0.5/L2/L8 absolute attenuation against low-EOG observation retention, PSD and covariance.
Keep task-valid physiological outcomes unavailable and avoid a composite “preservation” claim.

## 7. Stochastic uncertainty and matched one-step comparison

Report all 16 draws, single/mean/median summaries, coverage/width/interval score, ensemble CRPS,
error–dispersion association, support-span versus complement variance, identity uncertainty, and a
matched-average-dispersion constant reference. Treat EnergyDET as competitive positioning.

## 8. Limitations, privacy, and scope

Discuss development-only evidence, mixed donor specificity, support-state linkage, imperfect
uncertainty calibration, no task-valid physiology, no sealed confirmation, and lack of a compatible
registered strong U-Net on the common panel.

## 9. Conclusion

Conclude only that the frozen operating point provides absolute ocular attenuation with an explicit
retention trade-off; retain stochastic-uncertainty wording only if the registered diagnostics are
informative.
