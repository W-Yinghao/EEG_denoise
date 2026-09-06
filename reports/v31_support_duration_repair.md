# V31 exact support-duration repair

Frozen checkpoints were replayed on the unchanged V30 common panel with the same queries, K=1, and same per-cell diffusion noise. Windows are chronological, non-overlapping 2 s prefixes; EOG coordinates use only the declared acquisition prefix. The 5 s condition therefore has a 5 s acquisition span but two windows / 4 s effective model exposure. Zero support is the architectural population bypass.

V30 rows remain in the CSV as `historical_invalid_duration_contract`; only `V31_exact_duration_contract` rows are active evidence.

| panel | method | acquisition s | effective s | windows | metric | mean |
|---|---|---:|---:|---:|---|---:|
| paired | V25_SET_CALIB_DET | 0 | 0.0 | 0 | paired_risk | 0.761899 |
| paired | V25_SET_CALIB_DET | 5 | 4.0 | 2 | paired_risk | 0.758827 |
| paired | V25_SET_CALIB_DET | 10 | 10.0 | 5 | paired_risk | 0.752947 |
| paired | V25_SET_CALIB_DET | 30 | 30.0 | 15 | paired_risk | 0.750730 |
| paired | V25_SET_CALIB_DET | 120 | 120.0 | 60 | paired_risk | 0.749741 |
| paired | V26_CALIB_SDEDIT | 0 | 0.0 | 0 | paired_risk | 0.761899 |
| paired | V26_CALIB_SDEDIT | 5 | 4.0 | 2 | paired_risk | 0.762597 |
| paired | V26_CALIB_SDEDIT | 10 | 10.0 | 5 | paired_risk | 0.756065 |
| paired | V26_CALIB_SDEDIT | 30 | 30.0 | 15 | paired_risk | 0.753630 |
| paired | V26_CALIB_SDEDIT | 120 | 120.0 | 60 | paired_risk | 0.752535 |
| paired | V29_PA_SC_CDM | 0 | 0.0 | 0 | paired_risk | 0.816939 |
| paired | V29_PA_SC_CDM | 5 | 4.0 | 2 | paired_risk | 0.816727 |
| paired | V29_PA_SC_CDM | 10 | 10.0 | 5 | paired_risk | 0.816728 |
| paired | V29_PA_SC_CDM | 30 | 30.0 | 15 | paired_risk | 0.816728 |
| paired | V29_PA_SC_CDM | 120 | 120.0 | 60 | paired_risk | 0.816728 |
| paired | V29_PA_SC_DET | 0 | 0.0 | 0 | paired_risk | 0.816026 |
| paired | V29_PA_SC_DET | 5 | 4.0 | 2 | paired_risk | 0.815657 |
| paired | V29_PA_SC_DET | 10 | 10.0 | 5 | paired_risk | 0.815657 |
| paired | V29_PA_SC_DET | 30 | 30.0 | 15 | paired_risk | 0.815657 |
| paired | V29_PA_SC_DET | 120 | 120.0 | 60 | paired_risk | 0.815658 |
| natural | V25_SET_CALIB_DET | 0 | 0.0 | 0 | natural_remaining_ratio | 0.931676 |
| natural | V25_SET_CALIB_DET | 5 | 4.0 | 2 | natural_remaining_ratio | 0.918847 |
| natural | V25_SET_CALIB_DET | 10 | 10.0 | 5 | natural_remaining_ratio | 0.923257 |
| natural | V25_SET_CALIB_DET | 30 | 30.0 | 15 | natural_remaining_ratio | 0.927937 |
| natural | V25_SET_CALIB_DET | 120 | 120.0 | 60 | natural_remaining_ratio | 0.938085 |
| natural | V26_CALIB_SDEDIT | 0 | 0.0 | 0 | natural_remaining_ratio | 0.931676 |
| natural | V26_CALIB_SDEDIT | 5 | 4.0 | 2 | natural_remaining_ratio | 0.926540 |
| natural | V26_CALIB_SDEDIT | 10 | 10.0 | 5 | natural_remaining_ratio | 0.930770 |
| natural | V26_CALIB_SDEDIT | 30 | 30.0 | 15 | natural_remaining_ratio | 0.935079 |
| natural | V26_CALIB_SDEDIT | 120 | 120.0 | 60 | natural_remaining_ratio | 0.945210 |
| natural | V29_PA_SC_CDM | 0 | 0.0 | 0 | natural_remaining_ratio | 1.000704 |
| natural | V29_PA_SC_CDM | 5 | 4.0 | 2 | natural_remaining_ratio | 1.000189 |
| natural | V29_PA_SC_CDM | 10 | 10.0 | 5 | natural_remaining_ratio | 1.000190 |
| natural | V29_PA_SC_CDM | 30 | 30.0 | 15 | natural_remaining_ratio | 1.000190 |
| natural | V29_PA_SC_CDM | 120 | 120.0 | 60 | natural_remaining_ratio | 1.000191 |
| natural | V29_PA_SC_DET | 0 | 0.0 | 0 | natural_remaining_ratio | 0.999447 |
| natural | V29_PA_SC_DET | 5 | 4.0 | 2 | natural_remaining_ratio | 0.998538 |
| natural | V29_PA_SC_DET | 10 | 10.0 | 5 | natural_remaining_ratio | 0.998538 |
| natural | V29_PA_SC_DET | 30 | 30.0 | 15 | natural_remaining_ratio | 0.998538 |
| natural | V29_PA_SC_DET | 120 | 120.0 | 60 | natural_remaining_ratio | 0.998539 |
| natural | V25_SET_CALIB_DET | 0 | 0.0 | 0 | low_eog_observation_retention | 0.797641 |
| natural | V25_SET_CALIB_DET | 5 | 4.0 | 2 | low_eog_observation_retention | 0.770806 |
| natural | V25_SET_CALIB_DET | 10 | 10.0 | 5 | low_eog_observation_retention | 0.777601 |
| natural | V25_SET_CALIB_DET | 30 | 30.0 | 15 | low_eog_observation_retention | 0.778366 |
| natural | V25_SET_CALIB_DET | 120 | 120.0 | 60 | low_eog_observation_retention | 0.780069 |
| natural | V26_CALIB_SDEDIT | 0 | 0.0 | 0 | low_eog_observation_retention | 0.797641 |
| natural | V26_CALIB_SDEDIT | 5 | 4.0 | 2 | low_eog_observation_retention | 0.757446 |
| natural | V26_CALIB_SDEDIT | 10 | 10.0 | 5 | low_eog_observation_retention | 0.765688 |
| natural | V26_CALIB_SDEDIT | 30 | 30.0 | 15 | low_eog_observation_retention | 0.766277 |
| natural | V26_CALIB_SDEDIT | 120 | 120.0 | 60 | low_eog_observation_retention | 0.767493 |
| natural | V29_PA_SC_CDM | 0 | 0.0 | 0 | low_eog_observation_retention | 0.992465 |
| natural | V29_PA_SC_CDM | 5 | 4.0 | 2 | low_eog_observation_retention | 0.992539 |
| natural | V29_PA_SC_CDM | 10 | 10.0 | 5 | low_eog_observation_retention | 0.992540 |
| natural | V29_PA_SC_CDM | 30 | 30.0 | 15 | low_eog_observation_retention | 0.992540 |
| natural | V29_PA_SC_CDM | 120 | 120.0 | 60 | low_eog_observation_retention | 0.992540 |
| natural | V29_PA_SC_DET | 0 | 0.0 | 0 | low_eog_observation_retention | 0.998064 |
| natural | V29_PA_SC_DET | 5 | 4.0 | 2 | low_eog_observation_retention | 0.997789 |
| natural | V29_PA_SC_DET | 10 | 10.0 | 5 | low_eog_observation_retention | 0.997791 |
| natural | V29_PA_SC_DET | 30 | 30.0 | 15 | low_eog_observation_retention | 0.997791 |
| natural | V29_PA_SC_DET | 120 | 120.0 | 60 | low_eog_observation_retention | 0.997792 |
| diagnostic | SUPPORT_ENCODING | 5 | 4.0 | 2 | context_stability_to_120 | 3.447470 |
| diagnostic | SUPPORT_ENCODING | 10 | 10.0 | 5 | context_stability_to_120 | 2.083890 |
| diagnostic | SUPPORT_ENCODING | 30 | 30.0 | 15 | context_stability_to_120 | 1.261539 |
| diagnostic | SUPPORT_ENCODING | 120 | 120.0 | 60 | context_stability_to_120 | 0.000000 |
| diagnostic | SUPPORT_ENCODING | 5 | 4.0 | 2 | projector_stability_to_120 | 1.593760 |
| diagnostic | SUPPORT_ENCODING | 10 | 10.0 | 5 | projector_stability_to_120 | 1.130914 |
| diagnostic | SUPPORT_ENCODING | 30 | 30.0 | 15 | projector_stability_to_120 | 0.743179 |
| diagnostic | SUPPORT_ENCODING | 120 | 120.0 | 60 | projector_stability_to_120 | 0.000000 |
| diagnostic | SUPPORT_ENCODING | 5 | 4.0 | 2 | support_encoding_ms_per_query | 1.310093 |
| diagnostic | SUPPORT_ENCODING | 10 | 10.0 | 5 | support_encoding_ms_per_query | 0.422963 |
| diagnostic | SUPPORT_ENCODING | 30 | 30.0 | 15 | support_encoding_ms_per_query | 0.458390 |
| diagnostic | SUPPORT_ENCODING | 120 | 120.0 | 60 | support_encoding_ms_per_query | 0.795701 |
