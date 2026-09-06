# V26 V25 diffusion forensic

The V25 coefficient target is not invariant to equivalent rotations of the learned basis, while its decoded sensor-space artifact is invariant. This makes the old latent target coordinate-unstable.

```json
{
  "basis_rows": 15,
  "interpretation": "sensor-space target is invariant while the learned coefficient target changes under equivalent basis rotations",
  "rotation_fixture": {
    "latent_target_relative_difference": 1.6066668249190112,
    "projector_distance": 1.3103815177083624e-15,
    "rotation_magnitude": 4.509035853473691,
    "sensor_artifact_max_difference": 1.3322676295501878e-15
  },
  "sealed_reads": 0,
  "stage": "R1",
  "status": "PASS",
  "training_rows": 30
}
```
