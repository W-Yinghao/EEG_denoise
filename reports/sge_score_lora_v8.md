# SGE score-LoRA v8

## Scientific scope

This is development evidence. The v6 status is corrected to `CURRENT_V6_BACKBONE_AND_K8_ESTIMATOR_NOT_VALIDATED / CAUSE_NOT_FULLY_IDENTIFIED`; its sampler roundtrip passed, but its K=8 end-to-end estimator was not validated. No v6 seeds, sampler/objective repairs, confirmation data, or MobileBCI data were run.

## Fold-local EB operator headroom

Status: `completed_fold_local_eb_headroom`. All 58 compatible stems were evaluated; the availability denominator remains 59. Operators and normalization were re-fitted inside each outer fold. The deployable lambda predictor used outer-training support features only.

| support | deployable relative improvement | 95% fold-cluster CI | positive stems |
|---:|---:|---:|---:|
| 30s | +0.0602 | [-0.0170, +0.1303] | 39/58 |
| 60s | +0.1167 | [+0.0702, +0.1600] | 47/58 |
| 120s | +0.1902 | [+0.1212, +0.2449] | 52/58 |

The 60 s and 120 s budgets show development headroom for fold-local empirical-Bayes shrinkage; 30 s is directionally positive but its interval crosses zero. This supports the H/shrinkage operator branch only and does not establish score-space personalization.

## Population diffusion implementation validity and utility

Decision: `POPULATION_DIFFUSION_BASE_GATE_FAILED`. All 3/3 diagnostic folds completed with 32x pair expansion where the legacy fold contained enough participants (256/1664/1664 real paired samples). The analytic roundtrip and high-noise scale checks passed on every fold. Across folds, mean RRMSE was RAW=0.4675, DET-POP=0.4250, and DIFF-POP=0.4114. This aggregate signal does not override the frozen per-fold validity and safety gate.

| diagnostic fold | pairs | RAW | DET-POP | DIFF-POP | fixed-window overfit | preservation | PSD dist. | covariance dist. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| study01_layout_01_heldout_00 | 256 | 0.5890 | 0.6116 | 0.6172 | pass | 0.7155 | 0.2048 | 0.3215 |
| study02_layout_02_heldout_03 | 1664 | 0.4331 | 0.3666 | 0.3635 | pass | 0.7482 | 0.1109 | 0.2435 |
| study04_layout_04_heldout_01 | 1664 | 0.3804 | 0.2970 | 0.2534 | fail | 0.7810 | 0.1048 | 0.1378 |

The base failed because study04 missed the fixed-window x0-MSE threshold (1.215e-4 versus 1e-4), study01 did not beat RAW and failed preservation/covariance safety, and study02 preservation was 0.7482 versus the frozen 0.75 threshold. Therefore population diffusion utility is promising on two folds but implementation eligibility is not established. K=32 remains diagnostic and does not replace primary K=8.

## Score-space subject adaptation

Support-inner decision: `SCORE_LORA_NOT_STARTED`. Later-query decision: `NOT_RUN`. Because the population base gate failed, no participant LoRA parameters were optimized and no later-query subject-adaptation comparison was run. The implemented rank-4 LoRA sits inside frozen U-Net ResBlock score convolutions; it is not the historical output-space or global transfer adapter. Its scientific effect remains untested.

## Evidence boundary

The EB result is support/operator headroom; the three-fold GPU result is population-backbone development validity. Neither is confirmation evidence. The failure closes this population-backbone instance before personalization and is not a family-wide conclusion about diffusion, score-LoRA, or personalization.
