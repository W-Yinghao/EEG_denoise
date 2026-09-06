# V34P project plan

V34P makes one change to V33P: the sanitizer is restricted to the exact nullspace
of the frozen centered linear task head. It reuses BCI-IV-2a, the V33P Stage-A and
six-subject full-pool EEGNet checkpoints, the three outer test groups, two seeds,
Session-T-to-Session-E attacks, K=1, and ten reverse steps.

Stage A uses the frozen three-subject EEGNet for participant-disjoint epoch
selection on the other three non-test participants. Stage B loads the corresponding
frozen six-subject EEGNet and refits only Fiber-OneStep and Fiber-SANDiff to the
selected epochs. Outer-test participants are used only after selection and refit.

Hard boundaries are exact function preservation, split integrity, checkpoint
binding, finite output, reproducible sampling, participant-first aggregation, and
zero waveform-sealed reads. No dataset, encoder, attacker family, diffusion family,
or manuscript artifact is added or modified.
