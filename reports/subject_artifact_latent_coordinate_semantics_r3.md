# Subject artifact latent coordinate semantics r3

- A0/A1 run SHA: `6199d8d3cfa801cfaa7ceb01a6cb2ab92ea86e65`
- A1 Slurm job: `920904`
- Canonical coordinate: `A_physical = sigma_Z * z_standardized + mu_Z`; `Delta = C_normalized * A_physical`.
- The old formal J2 path already computed its reported physical identity through inverse normalization. The r3 change removes the conflicting standardized-zero helper and corrects/deprecates the legacy `restore()` API.
- The transient V1 weights from job 920825 were not checkpointed. The r3 table therefore distinguishes historical transient-V1 values from the saved full-training-checkpoint recomputation rather than pretending they are the same weights.
- Primary eligibility: `blocked`; high-noise latent-RMSE NO-GO retained.
- Compound eligibility: `blocked`; low-artifact preservation NO-GO retained.
- Unexpected V0/V2/V3 changes: `{"compound_V0_changed": false, "compound_V2_changed": false, "compound_V3_changed": false, "primary_V0_changed": false, "primary_V2_changed": false, "primary_V3_changed": false}`.
- Downstream diffusion status: `not_run_blocked_by_model_validity_gate`; family-wide status remains `not_tested`.
