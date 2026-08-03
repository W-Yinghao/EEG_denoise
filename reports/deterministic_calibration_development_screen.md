# Deterministic calibration development screen

## Model validity

| Candidate/model | V0/D0 | V1/D1 | V2/D2 | V3/D3 | Eligibility |
|---|---:|---:|---:|---:|---|
| Primary artifact-latent diffusion | True | False | True | True | blocked |
| Compound repair+SDEdit backup | False | False | False | True | blocked |
| Deterministic estimator | True | True | N/A (True) | True | passed |

- Deterministic model validity: `passed`.
- Calibration mechanism: `proxy_improvement_without_mechanistic_support`.
- Residual diffusion validity: `not_run_gate_closed`.
- Diffusion reopen eligible: `false`.
- Primary artifact-latent diffusion remains blocked by high-noise latent RMSE; the compound residual/SDEdit backup remains blocked by low-artifact preservation. Neither result is a family-wide diffusion test.
- Deterministic reverse-trajectory validity is `N/A`, not a fabricated pass.
- All SGE evidence in this round is development/exploratory; confirmation eligibility remains false.

## Development calibration screen

- Automatic route: `proxy_improvement_without_mechanistic_support`; diffusion reopen: `false`.
- Klados paired source-record mechanism (positive means matching is better): standardized latent RMSE utility -0.080159 (95% CI -0.148838, -0.035438); clean-waveform RRMSE utility -0.083456 (95% CI -0.134848, -0.039161). These are 8 source records, not participants.
- Natural SGE matching-population EOG-remaining utility 0.050073 (95% CI 0.022083, 0.075658); matching-control utility 0.099104 (95% CI 0.069979, 0.132732).
- Safety non-inferiority: `False`; exact-cell severe reversal: `True`; output scale safe: `True`.
- Coverage: 56 complete stems / 58 compatible outputs; denominator 59 including blocked `study05/study05_p42`. Incomplete scale-safety stems remain in the feasibility denominator.
- This is development/exploratory evidence. It does not establish a participant-independent confirmation claim.

## Slurm job ledger

```text
A0_initial_failed=920894
A1_initial_dependency_not_run=920895
A0_semantic_tests_passed=920897
A1_checkpoint_recompute=920898
B0_deterministic_D0_D3=920900
B0_sole_identity_repair_router=920901
A1_model_specific_aggregation_recompute=920904
B0_model_specific_D0_D3=920905
B0_model_specific_identity_repair=920906
A0_post_repair_code_tests=920908
B0_repair_passed=920906
A0_B1_training_code_tests=920913
B1_manifest_generation=920914
A0_B3_aggregation_routes=920917
A0_paired_mechanism_routes=920918
B1_full_array_submit_rejected_QOS=no_job_id
B1_eight_worker_training=920919
B1_paired_three_seed_training=920927
B2_eight_worker_SGE_inference=920928
B2_paired_three_seed_inference=920929
B3_calibration_aggregation=920930
A0_current_paired_and_worker_tests=920931
J1_real_Klados_paired_preprocessing=920932
J1_real_Klados_paired_preprocessing_failed_shape=920932
J1_post_failure_dependency_not_run=920933
J1_real_Klados_paired_preprocessing_retry=920935
A0_latest_semantic_route_tests_passed=920938
B1_full_75_task_training_completed=920919
B1_paired_FP16_nonfinite_gradient_failed=920927_2
A0_FP32_recovery_tests_passed=921114
B1_paired_FP32_recovery_training=921115
B2_SGE_eight_worker_inference_recovery_chain=921116
B2_paired_three_seed_inference_recovery_chain=921117
B3_calibration_aggregation_recovery_chain=921120
B1_paired_FP32_recovery_completed=921115
B2_SGE_inference_completed=921116
B2_paired_mechanism_completed=921117
B3_calibration_screen_completed_proxy_without_mechanism=921120
Stage_C_not_submitted_gate_closed=no_job_id
C1_final_442_tests_and_compact_report_passed=921216
```

## Compact result paths

- Coordinate semantics: `/home/infres/yinwang/denoiseNet/results/cgdr/subject_calibrated_artifact_diffusion/revisions/j2_r3_latent_coordinate_semantics/result_summary.json`
- Deterministic D0-D3: `/home/infres/yinwang/denoiseNet/results/cgdr/subject_calibrated_artifact_diffusion/revisions/deterministic_calibration_screen/result_summary.json`
- Calibration screen: `/home/infres/yinwang/denoiseNet/results/cgdr/subject_calibrated_artifact_diffusion/revisions/deterministic_calibration_screen/calibration_screen_summary.json`
- Bootstrap table: `/home/infres/yinwang/denoiseNet/results/cgdr/subject_calibrated_artifact_diffusion/revisions/deterministic_calibration_screen/development/bootstrap_summary.csv`
- Participant effects: `/home/infres/yinwang/denoiseNet/results/cgdr/subject_calibrated_artifact_diffusion/revisions/deterministic_calibration_screen/development/participant_effects.csv`
