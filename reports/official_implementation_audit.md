# Official implementation audit

Status: J0 Slurm repository inspection is running. This file records the
interpretation rules that are frozen before execution.

| Method | Claimed source | Required status before numerical comparison |
|---|---|---|
| EEGDfus | `XYH0118/EEGDfus` | Official checkout plus audited native loader/split; rerun on frozen split |
| D4PM | `flysnow1024/D4PM` | Official checkout; deployable and extra-information modes separated |
| Essentia | `NKU-EmbeddedSystem/Essentia` | Official checkout if code matches the EEG paper; otherwise paper reconstruction |
| EEGOAR-Net | `dmarcos97/EEGOAR-Net` | Official checkout and physical-scale output validation |
| DS-DDPM | `duanyiqun/DS-DDPM` | Audit only unless the recognition/domain task can be mapped without changing its claim |
| SGEYESUB | `rkobler/eyeartifactcorrection` | Source-faithful port unless MATLAB numerical parity is demonstrated |
| DeepSeparator | `ncclabsustech/DeepSeparator` | Official checkout; legacy dependency compatibility recorded |
| ICA+ICLabel / ASR / EOGRegression | official toolbox APIs | Exact library/version/config and support/query fit scope recorded |

No repository is called an exact reproduction merely because it clones or
imports. J0 records commit, training/data entry availability, bundled
checkpoints, and concrete blockers. Numerical parity and frozen-split results
are separate evidence.
