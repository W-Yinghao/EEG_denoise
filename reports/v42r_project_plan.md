# V42R project plan

V42R is a clean-room joint 46-channel conditional x0 diffusion experiment. It reuses only the
audited fold, paired-episode, transfer-estimation, and metric utilities inherited from V41R. The
population and transfer routes share one checkpoint; the primary estimand is MATCH minus POP.

The active paired resource is the established V19/V24 participant-session counterfactual panel. The
historical Klados v4 archive does not expose a recoverable participant mapping and is not silently
relabeled. Fifteen development participants, five outer folds, and two seeds remain frozen.

Execution order is native single-channel sanity, one 10k engineering pilot, ten fixed 80k cells,
paired interventions, and then frozen natural evaluation. Natural results cannot tune the model.

Execution status: native sanity completed valid in job 941715. Canonical pilot 941726 failed at update
1,905 because AMP produced a nonfinite gradient; the only registered engineering repair disabled AMP.
Recovery 941756 completed all 10,000 updates with finite loss/output, plausible scale, exact identity
initialization, and nonzero transfer gradients. The fixed ten-cell campaign is job 941770. The repair
does not alter the architecture, x0 target, transfer estimator, data split, update budget, sampler, or
scientific metrics.
