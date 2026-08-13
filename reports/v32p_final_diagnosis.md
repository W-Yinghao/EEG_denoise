# V32P final development diagnosis

## Outcome

```text
engineering: valid
representation_baseline: established
adaptive_privacy: residual_linkage_remains
utility: weak_but_above_chance
diffusion_positioning: competitive_with_one_step
selected_positive_candidate: SANDiff
selected_strength: nested_validation_selected_per_fold_and_seed
waveform_interaction: deferred_not_comparable
```

SANDiff is selected because its participant-disjoint validation balance was
slightly higher than the matched one-step sanitizer (0.14191 versus 0.14138),
and the corresponding nested outer-test point retained marginally higher
fixed/retrained MI utility with marginally lower adaptive linkage. The
difference is small; selection means “positive method candidate,” not a claim
that diffusion dominates one-step.

The useful positive result is narrower: selective task-conditioned replacement
produced a smooth privacy–utility curve, and strong SANDiff improved the
frontier relative to LEACE and the registered DANN instance. Adaptive subject
accuracy nevertheless remained around 0.65, far above 3-class chance, while
task BA remained around 0.33. This is development evidence, not formal privacy,
cross-dataset generalization, or clinical readiness.

No permitted small repair was used: the canonical model was finite, stable,
and did not show scale collapse. The P100 failure was an environment mismatch;
V100 recoveries used unchanged science. Earlier completed arrays were retained
as superseded when evaluation coverage was corrected. Waveform sealed reads
were zero and manuscript files were unchanged.

## Recommended next action

Freeze this V32P SANDiff candidate and review whether its modest gain and
latency are sufficient for the revision narrative before adding any dataset or
encoder. Do not broaden the method family automatically.
