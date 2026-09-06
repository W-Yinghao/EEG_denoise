# V41R final diagnosis

Final positioning: **D — base model not established**.

The implementation and evaluation are engineering-valid across 5 folds × 2 seeds and all 15
development participants.  However, the official-semantics channel-wise population route did not
establish valid denoising on the paired participant-specific resource: temporal RRMSE was 0.693946
versus 0.550124 for the contaminated input, and SNR improvement was −8.097552 dB.  Output scale was
finite (participant q99 output/input RMS 0.778939), and the channel-wise port was materially better
than the failed V40R multichannel POP RRMSE of 1.192548, but it still removed useful signal rather
than improving the input.

Consequently, the primary MATCH−POP result cannot diagnose the artifact-transfer hypothesis.  Its
temporal-RRMSE utility was −0.001974 (median −0.002136; 6/15 positive; participant bootstrap 95% CI
[−0.009502, 0.005585]). MATCH was better than the registered WRONG intervention by +0.025006
(12/15; 95% CI [0.006635, 0.044664]), but was indistinguishable from SHUFFLED (−0.000678; 9/15;
95% CI [−0.009010, 0.007283]). The independent ORACLE transfer was modestly better than MATCH, yet
the invalid common backbone prevents interpreting that headroom as deployable denoising evidence.

The official single-channel EEGDfus reproduction remains a separate, reasonable-nonidentical native
benchmark (EOG temporal RRMSE 0.296527, spectral RRMSE 0.302818, correlation 0.953041). Its native
single-channel EEGdenoiseNet protocol is not numerically interchangeable with the participant-held-out
46-channel panel. D4PM remains `official_release_not_runnable`; the audited CNN/DeepSeparator assets
were not silently reimplemented or forced into an incompatible protocol.

Natural evaluation was not launched because POP validity was a preregistered interpretation boundary.
Thus query EOG inference reads and sealed reads remained zero. V41R supports no subject-information
conclusion: the explicit transfer signature was tested only on a base model that failed to denoise this
paired resource.

Next route: freeze V41R as a D-position result. Any future artifact-transfer test must first bind a
population diffusion model that independently improves the identical paired resource; it must not be
presented as a repair selected from these test outcomes.

```json
{
  "engineering": "valid",
  "final_positioning": "D_base_model_not_established",
  "fold_seed_cells": 10,
  "manuscript_unchanged": true,
  "natural_authorized": false,
  "participant_coverage": 15,
  "pop_output_input_rms_q99": 0.7789387549459934,
  "pop_snr_improvement": -8.097551776024806,
  "pop_temporal_rrmse": 0.6939456885370116,
  "population_valid": false,
  "primary_estimand": {
    "bootstrap_high": 0.005585172678499163,
    "bootstrap_low": -0.009502168994707365,
    "contrast": "MATCH-POP",
    "metric": "rrmse_temporal",
    "participant_mean_utility": -0.0019743049206833044,
    "participant_median_utility": -0.0021359045058488846,
    "participants": 15,
    "positive_count": 6
  },
  "query_eog_inference_reads": 0,
  "raw_temporal_rrmse": 0.5501239840736768,
  "run_id": "job_941257",
  "sealed_reads": 0,
  "v40r_pop_temporal_rrmse": 1.192548
}
```
