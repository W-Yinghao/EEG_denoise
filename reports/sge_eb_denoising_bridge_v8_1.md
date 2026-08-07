# SGE EB denoising bridge v8.1

## Scope

Development exploration only. Historical v8 artifacts and `POPULATION_DIFFUSION_BASE_GATE_FAILED` remain unchanged. The EB label is corrected to `FOLD_LOCAL_OPERATOR_PROXY_HEADROOM_DETECTED`.

## Corrected validity

| fold | sampler error | eval-mode loss reduction | corr >= .99 |
|---|---:|---:|---:|
| study01_layout_01_heldout_00 | 7.63e-08 | 0.9999 | true |
| study01_layout_01_heldout_01 | 7.57e-08 | 0.9999 | true |
| study02_layout_02_heldout_03 | 7.62e-08 | 0.9999 | true |
| study04_layout_04_heldout_01 | 7.40e-08 | 0.9999 | true |

Technical validity: `True`.

## Operator proxy headroom

Primary 120 s FULL-feature effects: `{"fixed_vs_pop": 0.2206604526667268, "full_vs_fixed": -0.038502024699970314, "full_vs_pop": 0.19031482723468812, "full_vs_reliability": -0.03536222724031532, "full_vs_shuffled": 0.31280264614709974, "full_vs_wrong": 0.29867550588456443, "reliability_vs_pop": 0.21532330555975315}`. LOSO mean improvement: +0.2048. Fixed lambda outperformed the full-feature predictor and was therefore retained for the bridge. These are operator-proxy results, not denoising success.

## Denoising bridge

Decision: `BRIDGE_GATE_FAILED_SCORE_LORA_NOT_RUN`. Effects: `{"U_D": 0.009298693802621629, "U_P": 0.01591096818447113, "U_S": -0.004028252429432339, "U_W": 0.024539598160319857}`. Natural safety: `{"covariance_distortion": 0.2519163538241798, "eog_attenuation": 0.02576014523066687, "preservation": 0.7386551333798302, "psd_distortion": 0.15137542287508646}`. Oracle-ceiling versus actual-improvement correlation: +0.9938. Cluster intervals are development-descriptive and are saved in `cluster_descriptive_summary.csv`. Representation-ineligible folds would use the declared RAW fallback; all four folds were eligible.

## Score-LoRA

Decision: `SCORE_LORA_NOT_RUN`. No Score-LoRA training ran because the bridge gate failed. Later-query full evaluation was not authorized.

No result is confirmation evidence or a family-wide conclusion.
