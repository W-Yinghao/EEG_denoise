# V26 Round A

Positive-noise operating point selected with validation natural teacher fidelity primary, preservation secondary, and paired fidelity tertiary; sigma=0 remains the deterministic reference. Natural fraction used validation joint error only.

```json
{
  "status": "ROUND_B_CONFIG_FROZEN",
  "sigma_start": 0.05,
  "ddim_steps": 10,
  "natural_fraction": 0.3,
  "validation_operating_points": {
    "(0.0, 10)": {
      "paired_clean_rrmse": 1.4595078933731807,
      "natural_teacher_rrmse": 1.0160729885101318,
      "natural_preservation": 0.834866880926711
    },
    "(0.05, 10)": {
      "paired_clean_rrmse": 1.4626421332044024,
      "natural_teacher_rrmse": 1.0189594626426697,
      "natural_preservation": 0.8248427654592281
    },
    "(0.1, 10)": {
      "paired_clean_rrmse": 1.4629449593097283,
      "natural_teacher_rrmse": 1.019178956747055,
      "natural_preservation": 0.8241477486129345
    },
    "(0.2, 10)": {
      "paired_clean_rrmse": 1.463012045935708,
      "natural_teacher_rrmse": 1.0193151831626892,
      "natural_preservation": 0.8239700637374578
    },
    "(0.35, 10)": {
      "paired_clean_rrmse": 1.46297832755954,
      "natural_teacher_rrmse": 1.019551306962967,
      "natural_preservation": 0.8240186073259489
    },
    "(0.2, 5)": {
      "paired_clean_rrmse": 1.4626733999442272,
      "natural_teacher_rrmse": 1.0192209482192993,
      "natural_preservation": 0.8246884915715782
    },
    "(0.2, 25)": {
      "paired_clean_rrmse": 1.46323111061065,
      "natural_teacher_rrmse": 1.0194818377494812,
      "natural_preservation": 0.8234297923215005
    }
  },
  "natural_fraction_joint": {
    "0.3": 10.254187822341919,
    "0.5": 10.304340422153473
  },
  "sigma_zero_role": "deterministic_anchor_reference_not_diffusion_candidate",
  "selection_uses_test": false,
  "rationale": "Positive-noise operating point selected with validation natural teacher fidelity primary, preservation secondary, and paired fidelity tertiary; sigma=0 remains the deterministic reference. Natural fraction used validation joint error only."
}
```
