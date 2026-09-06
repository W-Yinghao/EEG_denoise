# V39A natural development denoising

Natural artifact targets are synchronized-EOG/operator proxy targets. Low-EOG observation retention is not physiological preservation. Paired and natural claims remain separate.

| method                     | metric                        |   participant_mean |   participant_median |   participant_min |   participant_max |   bootstrap_low |   bootstrap_high |
|:---------------------------|:------------------------------|-------------------:|---------------------:|------------------:|------------------:|----------------:|-----------------:|
| Diffusion-Augmentation     | heldout_eog_remaining_ratio   |           1.10681  |             1.0375   |          0.87356  |          2.08205  |        1.0033   |         1.26957  |
| Diffusion-Augmentation     | artifact_attenuation_db       |          -0.003795 |             0.033506 |         -2.12437  |          1.33323  |       -0.442801 |         0.403027 |
| Diffusion-Augmentation     | low_eog_observation_retention |           0.693799 |             0.704278 |          0.575664 |          0.818265 |        0.660033 |         0.725957 |
| Diffusion-Augmentation     | psd_distortion                |           0.487257 |             0.463926 |          0.314677 |          0.609748 |        0.445922 |         0.526557 |
| Diffusion-Augmentation     | covariance_distortion         |           0.290106 |             0.308822 |          0.148552 |          0.401048 |        0.255513 |         0.324785 |
| Diffusion-Augmentation     | output_input_rms              |           0.954904 |             0.939398 |          0.894044 |          1.15087  |        0.931092 |         0.987721 |
| Gaussian-Augmentation      | heldout_eog_remaining_ratio   |           1.26622  |             1.15598  |          0.980014 |          2.3665   |        1.11842  |         1.46578  |
| Gaussian-Augmentation      | artifact_attenuation_db       |          -1.04938  |            -0.833929 |         -3.24132  |          0.208092 |       -1.59708  |        -0.565732 |
| Gaussian-Augmentation      | low_eog_observation_retention |           0.753293 |             0.77598  |          0.636744 |          0.83424  |        0.720741 |         0.782361 |
| Gaussian-Augmentation      | psd_distortion                |           0.442122 |             0.449822 |          0.321615 |          0.575585 |        0.407858 |         0.477985 |
| Gaussian-Augmentation      | covariance_distortion         |           0.220617 |             0.191344 |          0.144569 |          0.354529 |        0.18656  |         0.258516 |
| Gaussian-Augmentation      | output_input_rms              |           0.975672 |             0.98736  |          0.901933 |          1.0121   |        0.958178 |         0.990897 |
| No-Augmentation            | heldout_eog_remaining_ratio   |           1.1098   |             1.01573  |          1.00389  |          2.04009  |        1.02216  |         1.25757  |
| No-Augmentation            | artifact_attenuation_db       |          -0.42004  |            -0.131185 |         -2.95662  |         -0.033583 |       -0.838682 |        -0.148416 |
| No-Augmentation            | low_eog_observation_retention |           0.926058 |             0.927042 |          0.894423 |          0.945054 |        0.918438 |         0.932831 |
| No-Augmentation            | psd_distortion                |           0.194266 |             0.189308 |          0.130585 |          0.345729 |        0.171856 |         0.222861 |
| No-Augmentation            | covariance_distortion         |           0.050836 |             0.045942 |          0.024366 |          0.076603 |        0.04377  |         0.058548 |
| No-Augmentation            | output_input_rms              |           1.00298  |             1.00239  |          1.00148  |          1.00657  |        1.00232  |         1.00377  |
| Real-Artifact-Augmentation | heldout_eog_remaining_ratio   |           0.942607 |             0.911077 |          0.676461 |          1.80862  |        0.836036 |         1.09371  |
| Real-Artifact-Augmentation | artifact_attenuation_db       |           1.58018  |             1.41732  |         -0.275016 |          3.6399   |        0.991661 |         2.18986  |
| Real-Artifact-Augmentation | low_eog_observation_retention |           0.768209 |             0.760939 |          0.698396 |          0.86205  |        0.745036 |         0.791353 |
| Real-Artifact-Augmentation | psd_distortion                |           0.472966 |             0.483647 |          0.298608 |          0.626121 |        0.428873 |         0.51847  |
| Real-Artifact-Augmentation | covariance_distortion         |           0.242948 |             0.240044 |          0.127519 |          0.323976 |        0.212611 |         0.274345 |
| Real-Artifact-Augmentation | output_input_rms              |           0.884987 |             0.882124 |          0.803185 |          0.994522 |        0.857748 |         0.911496 |
| WGAN-Augmentation          | heldout_eog_remaining_ratio   |           2.36571  |             1.69807  |          1.32655  |          6.14436  |        1.70511  |         3.15524  |
| WGAN-Augmentation          | artifact_attenuation_db       |          -4.66159  |            -3.47558  |        -11.532    |         -1.8889   |       -6.13233  |        -3.41107  |
| WGAN-Augmentation          | low_eog_observation_retention |          -0.384902 |            -0.033871 |         -2.79759  |          0.370085 |       -0.943192 |         0.106984 |
| WGAN-Augmentation          | psd_distortion                |           0.375407 |             0.357915 |          0.235641 |          0.530567 |        0.330896 |         0.420281 |
| WGAN-Augmentation          | covariance_distortion         |           0.301927 |             0.295307 |          0.136836 |          0.531779 |        0.242567 |         0.365055 |
| WGAN-Augmentation          | output_input_rms              |           1.59675  |             1.26502  |          1.15851  |          3.34355  |        1.24718  |         1.99494  |

## Secondary support interventions

These controls assess average context sensitivity; they do not identify a unique donor.

| context_condition           |   rrmse_temporal |   artifact_attenuation_db |   low_eog_observation_retention |   psd_distortion |
|:----------------------------|-----------------:|--------------------------:|--------------------------------:|-----------------:|
| correct                     |         0.946996 |                 -0.003795 |                        0.693799 |         0.487257 |
| mean_wrong_support          |         0.953217 |                 -0.007616 |                        0.705928 |         0.484801 |
| population_context          |         0.954659 |                 -0.026018 |                        0.704215 |         0.486826 |
| registered_shuffled_support |         0.949024 |                 -0.014664 |                        0.68783  |         0.489695 |
