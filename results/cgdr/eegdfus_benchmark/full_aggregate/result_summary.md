# EEGDfus full benchmark aggregate

All eight frozen cells completed. Official-native and strict source-epoch results are reported separately. Comparisons are paired descriptions over the frozen SNR grid, not independent statistical replicates.

The upstream spectral RRMSE remains blocked by the 400-vs-512 denominator shape mismatch. The explicitly named corrected PSD-denominator-shape metric is reported alongside the empty official field.

## Cell means

| Protocol | Noise | Arm | SNR improvement | Correlation | Temporal RRMSE | Corrected spectral RRMSE |
|---|---|---|---:|---:|---:|---:|
| official_native | EOG | conditional_diffusion | 14.5035 | 0.953041 | 0.296527 | 0.302818 |
| official_native | EOG | matched_deterministic | 14.6276 | 0.944086 | 0.318704 | 0.326259 |
| official_native | EMG | conditional_diffusion | 11.1165 | 0.906508 | 0.420541 | 0.668656 |
| official_native | EMG | matched_deterministic | 11.6065 | 0.906748 | 0.405724 | 0.354804 |
| strict_source_epoch | EOG | conditional_diffusion | 15.4822 | 0.959709 | 0.273966 | 0.290797 |
| strict_source_epoch | EOG | matched_deterministic | 15.158 | 0.96403 | 0.261617 | 0.280869 |
| strict_source_epoch | EMG | conditional_diffusion | 10.981 | 0.913213 | 0.394938 | 0.346681 |
| strict_source_epoch | EMG | matched_deterministic | 11.3929 | 0.921276 | 0.377045 | 0.334342 |

## Paired conditional-minus-deterministic descriptions

| Protocol | Noise | ΔSNR improvement | Δcorrelation | Δtemporal RRMSE | Δcorrected spectral RRMSE |
|---|---|---:|---:|---:|---:|
| official_native | EOG | -0.124059 | 0.00895464 | -0.0221764 | -0.0234417 |
| official_native | EMG | -0.49004 | -0.000239762 | 0.0148175 | 0.313852 |
| strict_source_epoch | EOG | 0.324183 | -0.00432013 | 0.0123495 | 0.00992806 |
| strict_source_epoch | EMG | -0.411911 | -0.00806256 | 0.0178934 | 0.0123389 |

EEGdenoiseNet exposes source epochs rather than participant identities; these results cannot support participant-specific or real-EEG deployment claims.
