# Invalid for the symmetric conditional-vs-deterministic comparison

Slurm job 919843 started the v3 development-selected-checkpoint protocol before
the fairness audit required symmetric fixed endpoints. Any checkpoints or
partial outputs under this v3 result root are retained as historical engineering
artifacts and must not be consumed by the v4 fixed-6000-update protocol.

Reason: v3 selected the deterministic checkpoint with development clean-target
loss, while the proposed conditional comparator used a terminal checkpoint at a
borrowed update count. This is asymmetric selection on the same records later
used for descriptive evaluation.

Replacement roots:

- deterministic: `results/cgdr/klados_stage3_deterministic_scope_isolated_v4/`
- conditional diffusion: `results/cgdr/klados_stage3_conditional_diffusion_matched_v2/`

No v3 file was deleted or overwritten by this revision.
