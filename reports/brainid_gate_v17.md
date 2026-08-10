# BrainID-Bridge v17 Gate-01

Development prerequisite audit only. No denoiser, diffusion, I²SB bridge, CacheKV, or identity-loss model was trained.

## Data and frozen protocol

- Data status: `DATA_PROTOCOL_VALID`; R/T/G coverage 15/15; 57 common EEG channels.
- Source sampling was 1000 Hz and frozen modeling sampling was 250 Hz, with 1–45 Hz filtering, common-average rereference, and prestimulus baseline correction.
- The source describes one common acquisition system (Neuracle Neusen series (common source-described amplifier)); device identity was therefore not a varying participant feature. Session predictability was tested separately from embeddings.
- R=Day-1 support, T=Day-7 restoration query, G=Day-80 independent gallery/evaluation, F=Day-200 future sealed.
- Day-200 signals and PhysioMotion sealed participants remained unopened. Only mask geometry from the frozen PhysioMotion development set was transferred.
- The distributed Trigger.txt prose conflicts with the paper and official SampleByTrigger.m; the frozen event mapping follows the latter two: code 1 non-target, code 2 target.
- The source describes 52 single-session controls. Only GroupB reference metadata (380 entries across files) was audited; control signals were not consumed in this Gate-01 implementation.

## M1: independent longitudinal verifiers

- Verifier-A: AUROC 0.8398 (participant-bootstrap descriptive 95% CI 0.7606–0.9111), EER 0.2362, TAR@FAR5 0.6268, rank-1 0.7550, positive margins 14/15.
- Verifier-B: AUROC 0.8466 (participant-bootstrap descriptive 95% CI 0.7559–0.9268), EER 0.2333, TAR@FAR5 0.6268, rank-1 0.7658, positive margins 14/15.
- Decision: `M1_LONGITUDINAL_VERIFIER_FAILED`. Failed criteria: A_artifact_shortcut, B_augmentation, B_artifact_shortcut.
- Verifier-A and Verifier-B are separate implementations and checkpoints. Verifier-B was not imported by training or alpha-selection code.

## M0: no-training identity actionability

- Not run because M1 did not pass; no model training was substituted.

## Gate-01 decision

```json
{
  "M0_actionability": "INSUFFICIENT",
  "M1_verifier": "FAIL",
  "PASS_01": false,
  "data_protocol": "PASS",
  "day200_opened": false,
  "failed_criteria": [
    "M1:A_artifact_shortcut",
    "M1:B_augmentation",
    "M1:B_artifact_shortcut",
    "M0:M0_not_run"
  ],
  "m2_m3_executed": false,
  "physiomotion_sealed_opened": false
}
```

This is development evidence. Failure constrains only the frozen longitudinal brainprint/actionability instance and is not a family-wide negative. Passing Gate-01 would only create a preregistration file; it would not execute a later model.
