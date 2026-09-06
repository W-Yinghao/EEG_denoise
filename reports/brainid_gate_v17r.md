# BrainID Gate-01R

One-shot nuisance-invariant construct-validity repair. The immutable Gate-01 result remains FAIL; no denoiser, diffusion, I²SB, CacheKV, or identity-guided model was trained.

## Data protocol

- `DATA_PROTOCOL_STILL_VALID`: 15/15 longitudinal participants, 57 channels, 52/52 anonymous controls used only as negatives/nuisance calibration.
- R/T/G remained Day-1/Day-7/Day-80; Day-200 and PhysioMotion sealed signals remained unopened.

## J1 identity-carrier forensic

- Descriptive panel report: `/home/infres/yinwang/denoiseNet_brainid_gate_v17r/reports/j1_forensic_report.md`. No panel was selected; J2 stayed frozen to 0.5–30 Hz.

## M1R physiological verifier

- A-R: physiological AUROC 0.7375 (descriptive CI 0.6770–0.7960), artifact-only AUROC 0.7955, gap -0.0579, EER 0.3430, TAR@FAR5 0.2079, positive 15/15.
- B-R: physiological AUROC 0.8241 (descriptive CI 0.7734–0.8714), artifact-only AUROC 0.7978, gap 0.0263, EER 0.2690, TAR@FAR5 0.3483, positive 15/15.
- Decision: `M1R_PHYSIOLOGICAL_VERIFIER_FAIL`; failed criteria: A-R_auroc, A-R_ci_low, A-R_eer, A-R_tar, A-R_artifact, A-R_gap, A-R_wrong_condition, B-R_tar, B-R_artifact, B-R_gap, B-R_rereference, B-R_dropout10, B-R_wrong_condition.

## M0R no-training actionability

- NOT_RUN because M1R failed; no restoration model or analytical reference intervention was run.

## Gate-01R decision

```json
{
  "M0R_actionability": "NOT_RUN",
  "M1R_verifier": "FAIL",
  "PASS_01R": false,
  "controls_opened_for_negative_calibration": true,
  "data_protocol": "PASS",
  "day200_opened": false,
  "denoiser_or_diffusion_trained": false,
  "failed_criteria": [
    "M1R:A-R_auroc",
    "M1R:A-R_ci_low",
    "M1R:A-R_eer",
    "M1R:A-R_tar",
    "M1R:A-R_artifact",
    "M1R:A-R_gap",
    "M1R:A-R_wrong_condition",
    "M1R:B-R_tar",
    "M1R:B-R_artifact",
    "M1R:B-R_gap",
    "M1R:B-R_rereference",
    "M1R:B-R_dropout10",
    "M1R:B-R_wrong_condition",
    "M0R:not_run_after_M1R_failure"
  ],
  "original_gate01": "FAIL_IMMUTABLE",
  "physiomotion_sealed_opened": false
}
```

This is development evidence. Any failure constrains only the frozen Gate-01R construct/actionability instance and is not a family-wide negative.
