# v6 diffusion validity adjudication

The historical v6 outputs are preserved. Their corrected scientific status is:

`CURRENT_STATIC_TRANSFER_SUMMARY_INSTANCE_NO_GO / DIFFUSION_OPTIMIZATION_VALIDITY_NOT_ESTABLISHED / DYNAMIC_TRANSFER_SUBJECT_AWARENESS_NOT_CLEANLY_TESTED`.

The v6 condition encoded a static projector, channel FIR scale and reliability; it did not encode lag/frequency/phase. Historical folds had only 8--52 unique pairs, POP reliability was hard-coded to 1.0, and WRONG waveforms were averaged before RRMSE. Therefore v6 is not described as a valid negative result for dynamic-transfer-conditioned diffusion.

| Diagnostic fold | oracle-v error | selected repair objective | all heldout beat RAW | preservation | PSD distortion | covariance distortion |
|---|---:|---|---|---:|---:|---:|
| study01_layout_01_heldout_00 | 7.427e-08 | none | false | nan | nan | nan |
| study02_layout_02_heldout_03 | 7.421e-08 | none | false | nan | nan | nan |
| study04_layout_04_heldout_01 | 7.393e-08 | none | false | nan | nan | nan |

Decision: `DIFFUSION_IMPLEMENTATION_OR_OBJECTIVE_INVALID`. Stage B authorized: `false`. The old seeds 20260807/20260808 remain unsubmitted.
