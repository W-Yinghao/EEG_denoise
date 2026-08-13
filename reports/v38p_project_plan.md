# V38P project plan

V38P is a new representation-level transport task, not a rerun of V37T waveform SDEdit or the
V32P–V36P exact-fiber channel. It binds the frozen V36P OpenBMI EEGNet representation and task head,
uses the same six 36/9/9 participant folds and two seeds, and trains a compact donor-residual
diffusion with a genuine one-to-many target.

Execution is fail-closed on participant/session leakage, test entries in the donor bank, support /
query overlap, true query labels in inference or donor matching, checkpoint mismatch, nonfinite
outputs, attacker contamination, and participant aggregation. Small effects and intervals crossing
zero are descriptive, not automatic scientific gates.

Stages are: frozen binding and protocol materialization; baselines; canonical OneStep/SARD training
with one weak/medium/strong validation sweep; optional one-factor repair only if authorized by the
registered engineering-valid failure modes; outer-test privacy/utility/distribution/exposure and
K=8 ensemble/augmentation; participant-first aggregation and A/B/C diagnosis.

The complete canonical run is preserved. Its validation results showed that SARD transport retained
or amplified source leakage, satisfying the registered `stays at the source` repair condition. The
sole repair raises the existing source-adversary weight from 0.1 to 0.5. Dataset, encoder, support
budget, donor rule, architecture, task weight, sampler, K, seeds, folds, and attacker family remain
unchanged.
