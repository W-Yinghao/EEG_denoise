# V39A final diagnosis

Final positioning: **C — non-diffusion artifact generation is preferable**.

The canonical diffusion generator was finite, retained nonzero within-context diversity, and achieved the highest severity-recovery correlation; it therefore did not meet the registered engineering-collapse condition for the one permitted repair. Nevertheless, its energy distance was worse than empirical resampling, and its augmented denoiser was worse than the strongest non-diffusion arm on paired temporal RRMSE. The empirical arm also provided the only clear absolute natural attenuation. Under the V39A terminal rule, diffusion method search for TAAS-26-0171 closes.

```json
{
  "best_non_diffusion_energy_distance": 7.222085189819336,
  "best_non_diffusion_generator": "Empirical-Resample",
  "diffusion_generator_energy_distance": 16.267313385009764,
  "engineering": "valid",
  "final_positioning": "C",
  "fold_seed_cells": 10,
  "manuscript_unchanged": true,
  "paired_temporal_utility": {
    "bootstrap_high": -0.0244938170030745,
    "bootstrap_low": -0.0734021485675541,
    "comparator": "Real-Artifact-Augmentation",
    "method": "Diffusion-Augmentation",
    "metric": "rrmse_temporal",
    "panel": "paired",
    "participant_mean_utility": -0.04984193507517558,
    "participants": 15,
    "positive_count": 1
  },
  "participant_coverage": 15,
  "primary_contrast_strongest_non_diffusion": "Real-Artifact-Augmentation",
  "query_eog_inference_reads": 0,
  "repair_decision": "not_triggered_no_registered_engineering_collapse",
  "repair_used": false,
  "sealed_reads": 0,
  "support_interventions": {
    "correct": {
      "natural_attenuation_db": -0.0037949335006826363,
      "paired_rrmse_temporal": 0.9469961147931155
    },
    "mean_wrong_support": {
      "natural_attenuation_db": -0.007616069039991066,
      "paired_rrmse_temporal": 0.9532173024828849
    },
    "population_context": {
      "natural_attenuation_db": -0.026017809973169857,
      "paired_rrmse_temporal": 0.9546588571419067
    },
    "registered_shuffled_support": {
      "natural_attenuation_db": -0.014664342514558726,
      "paired_rrmse_temporal": 0.9490239019619175
    }
  }
}
```

Paired and natural proxy evidence remain separate. Seeds and generated artifacts are not biological samples. No formal posterior, physiological ground truth, unique operator recovery, privacy, or universal artifact claim is made.
