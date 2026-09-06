# V40R official reproduction

| protocol            | artifact   | arm                   | status                               |   rrmse_temporal |   rrmse_spectral |   correlation |   snr_improvement |   optimizer_updates |   sampler_steps |
|:--------------------|:-----------|:----------------------|:-------------------------------------|-----------------:|-----------------:|--------------:|------------------:|--------------------:|----------------:|
| official_native     | EOG        | conditional_diffusion | reasonable_nonidentical_reproduction |         0.296527 |         0.302818 |      0.953041 |           14.5035 |              208000 |             500 |
| official_native     | EOG        | matched_deterministic | reasonable_nonidentical_reproduction |         0.318704 |         0.326259 |      0.944086 |           14.6276 |              208000 |               1 |
| official_native     | EMG        | conditional_diffusion | reasonable_nonidentical_reproduction |         0.420541 |         0.668656 |      0.906508 |           11.1165 |              344000 |             500 |
| official_native     | EMG        | matched_deterministic | reasonable_nonidentical_reproduction |         0.405724 |         0.354804 |      0.906748 |           11.6065 |              344000 |               1 |
| strict_source_epoch | EOG        | conditional_diffusion | reasonable_nonidentical_reproduction |         0.273966 |         0.290797 |      0.959709 |           15.4822 |              276000 |             500 |
| strict_source_epoch | EOG        | matched_deterministic | reasonable_nonidentical_reproduction |         0.261617 |         0.280869 |      0.96403  |           15.158  |              276000 |               1 |
| strict_source_epoch | EMG        | conditional_diffusion | reasonable_nonidentical_reproduction |         0.394938 |         0.346681 |      0.913213 |           10.981  |              344000 |             500 |
| strict_source_epoch | EMG        | matched_deterministic | reasonable_nonidentical_reproduction |         0.377045 |         0.334342 |      0.921276 |           11.3929 |              344000 |               1 |

The upstream spectral formula is blocked by a 400-versus-512 denominator shape mismatch; the explicitly named corrected PSD-denominator result is used. EEGdenoiseNet has source epochs but no participant identity.
