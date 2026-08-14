# V40R final diagnosis

Final positioning: **C — no support increment**.

The official-code anchor is valid as a `reasonable_nonidentical_reproduction`: official-native EOG
EEGDfus achieved temporal RRMSE 0.296527, corrected spectral RRMSE 0.302818, and correlation
0.953041 on the single-channel source-epoch benchmark. That benchmark is positioning evidence, not
participant evidence.

The local multichannel port was finite after the registered small-batch learning-rate repair and has
participant-level output/input RMS q99 1.073. Its absolute denoising, however, is weak: POP temporal
RRMSE is 1.192548 and MATCH is 1.284681. MATCH−POP temporal utility is −0.092133 (8/15 positive,
participant bootstrap CI [−0.276223, 0.002045]). MATCH is better than WRONG on average, but only 6/15
participants improve on the primary metric; it is also worse than SHUFFLED on average. The 10-second
support point is essentially POP (1.192720), while 30 seconds is worse (1.284681).

Natural evidence does not rescue the method. POP remaining ratio is 4.010855 and MATCH is 4.046576;
both imply negative attenuation. MATCH−POP attenuation utility is −0.026204 dB (7/15 positive, CI
[−0.070949, 0.007087]) and low-EOG retention also falls by 0.003645. These are proxy artifact and
observation-retention outcomes, not physiological preservation.

The support representation is linkable under the lightweight registered audit (mean top-1 0.2933
against 1/15 chance; mean disjoint-half verification AUROC 0.6893), so its 512-byte state should be
ephemeral. This is not an anonymity assessment.

V40R therefore closes this implementation-specific hypothesis: the compact query-disjoint support
adapter did not improve the identical population EEGDfus-MC backbone. It does not imply that
diffusion is generally ineffective and does not alter the frozen V39A interpretation.

```json
{
  "d4pm": "official_release_not_runnable",
  "eegoar_net": "protocol_incompatible",
  "engineering": "valid",
  "final_positioning": "C",
  "fold_seed_cells": 10,
  "manuscript_unchanged": true,
  "natural_attenuation_estimand": {
    "bootstrap_high": 0.007087441797553116,
    "bootstrap_low": -0.07094929870686717,
    "contrast": "MATCH-POP",
    "metric": "artifact_attenuation_db",
    "panel": "natural",
    "participant_mean_utility": -0.02620413657954505,
    "participant_median_utility": -0.005516877622030236,
    "participants": 15,
    "positive_count": 7
  },
  "official_eegdfus": "reasonable_nonidentical_reproduction",
  "output_input_rms_participant_q99": 1.0729995776959176,
  "participant_coverage": 15,
  "primary_estimand": {
    "bootstrap_high": 0.0020448731668695794,
    "bootstrap_low": -0.27622328913196653,
    "contrast": "MATCH-POP",
    "metric": "rrmse_temporal",
    "panel": "paired",
    "participant_mean_utility": -0.09213317031977722,
    "participant_median_utility": 0.002112490353566976,
    "participants": 15,
    "positive_count": 8
  },
  "query_eog_inference_reads": 0,
  "repair_scope": "input_channel_and_multichannel_learning_rate_engineering_only",
  "repair_used": true,
  "sealed_reads": 0
}
```
