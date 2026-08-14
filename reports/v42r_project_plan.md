# V42R project plan

V42R is a clean-room joint 46-channel conditional x0 diffusion experiment. It reuses only the
audited fold, paired-episode, transfer-estimation, and metric utilities inherited from V41R. The
population and transfer routes share one checkpoint; the primary estimand is MATCH minus POP.

The active paired resource is the established V19/V24 participant-session counterfactual panel. The
historical Klados v4 archive does not expose a recoverable participant mapping and is not silently
relabeled. Fifteen development participants, five outer folds, and two seeds remain frozen.

Execution order is native single-channel sanity, one 10k engineering pilot, ten fixed 80k cells,
paired interventions, and then frozen natural evaluation. Natural results cannot tune the model.
