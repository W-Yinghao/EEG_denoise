# Subject-calibrated artifact diffusion development

This is a mechanical rendering of the frozen J5 verdict. J6 did not reopen
EEG, EOG, labels, arrays, checkpoints, or metric tables and did not recompute
any scientific statistic.

## Engineering and validity

- J0: passed manifest/availability audit.
- J1: passed real-record context validation.
- J2 model validity: failed.
- J5 computational status: passed_fail_closed_terminal.

- V0: failed
- V1: failed
- V2: failed
- V3: passed

## Scientific go/no-go

- Matching calibration: not_run_blocked_by_V0_V3.
- Diffusion versus matched deterministic: not_run_blocked_by_V0_V3.
- Uncertainty contribution: not_tested.
- Calibration-duration evidence: not_run_blocked_by_V0_V3.
- Topic status: not_yet_testable.
- Protocol decision: inconclusive.
- Confirmation eligibility: false.
- Confirmation blockers: V0_to_V3_not_passed, paired_mechanism_evidence_missing_or_not_passed, matched_uncertainty_comparison_missing_or_not_passed.

The real-EEG evidence scope remains:
`J1 real-record structural validation and one development-SGE V0-V3 validity fold only; no full natural-SGE factorial, paired mechanism comparison, or confirmation evidence`.

The retained pre-round M2 decision is unchanged at
`/home/infres/yinwang/denoiseNet/results/cgdr/diffusion_incremental_decision_v2/result_summary.json`. No confirmation outcome was opened and no
confirmation job was generated.

## Slurm ledger

| Stage | Job ID | Recorded status |
|---|---:|---|
| J0 cpu tests + manifest/target audit | 920700 | failed: 393 passed, 1 route-test scope bug |
| J1 cpu real-record context validation | 920701 | dependency_not_run afterok:920700 |
| J0 retry 1, same science config after route-test-only fix | 920712 | passed: 394 tests |
| J1 retry dependency | 920713 | passed: 5 real exact-cells, query annotations sealed |
| J0 post-merge aggregate tests | 920730 | passed: 410 tests in 27.84s |
| J2 L40S V0-V3 real-SGE validity | 920731 | failed_before_data: PyTorch CUDA memory API rejected unindexed device |
| J0 CUDA compatibility replay | 920785 | passed: 410 tests in 27.16s |
| J2 retry 1, same science config | 920786 | completed_model_validity_failed; diagnostic_pre_v1_identity_repair_routing_not_gate_evidence |
| J0 V1 repair-routing replay | 920824 | passed: 415 tests in 27.47s |
| J2 revision-isolated V0-V3 replay | 920825 | completed_model_validity_failed: V0/V1/V2 failed, V3 passed; no J3/J4 |
| J0 final downstream CPU tests and audit | 920855 | failed: 433 passed, 1 float-tolerance-only test failure |
| J1 revision real-record context validation | 920856 | dependency_not_run afterok:920855 |
| J5 fail-closed participant-stem aggregate | 920858 | dependency_not_run afterok:920856 |
| J6 mechanical finalizer | 920859 | dependency_not_run afterok:920858 |
| J0 downstream CPU retry after tolerance-only fix | 920863 | passed: 434 tests in 26.56s |
| J1 revision real-record retry | 920864 | passed: 25 folds, 5 representative exact cells, query annotations sealed |
| J5 fail-closed aggregate retry chain | 920865 | completed_fail_closed_model_validity_failed; no J3/J4 rows read |
| J6 finalizer retry chain | 920866 | completed_j6_mechanical_finalization; confirmation blocked |
| J0 final full CPU verification | 920870 | passed: 434 tests in 27.41s |
| J5 corrected evidence-scope aggregate | 920871 | completed_fail_closed_model_validity_failed; no J3/J4 rows read |
| J6 corrected final report render | 920872 | completed_j6_mechanical_finalization; confirmation blocked |

## Result paths

- J5 aggregate: `/home/infres/yinwang/denoiseNet/results/cgdr/subject_calibrated_artifact_diffusion/revisions/j2_v1_identity_routing_r2/development/aggregate/result_summary.json`
- Final summary: `/home/infres/yinwang/denoiseNet/results/cgdr/subject_calibrated_artifact_diffusion/revisions/j2_v1_identity_routing_r2/result_summary.json`
- Terminal manifest: `/home/infres/yinwang/denoiseNet/results/cgdr/subject_calibrated_artifact_diffusion/revisions/j2_v1_identity_routing_r2/terminal_manifest.json`
