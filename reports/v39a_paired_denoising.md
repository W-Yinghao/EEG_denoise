# V39A paired denoising

## Absolute participant-first outcomes

| method                     | metric          |   participant_mean |   participant_median |   participant_min |   participant_max |   bootstrap_low |   bootstrap_high |
|:---------------------------|:----------------|-------------------:|---------------------:|------------------:|------------------:|----------------:|-----------------:|
| Diffusion-Augmentation     | rrmse_temporal  |           0.946996 |             0.562021 |          0.359905 |          5.09854  |        0.546407 |         1.60563  |
| Diffusion-Augmentation     | rrmse_spectral  |           0.631194 |             0.612915 |          0.418522 |          1.07108  |        0.561458 |         0.713468 |
| Diffusion-Augmentation     | correlation     |           0.838336 |             0.858978 |          0.532665 |          0.923313 |        0.786173 |         0.875329 |
| Diffusion-Augmentation     | snr_improvement |          -2.57637  |            -2.37847  |         -5.06005  |         -0.840001 |       -3.08528  |        -2.11492  |
| Diffusion-Augmentation     | artifact_rrmse  |           1.72838  |             1.70021  |          1.38483  |          2.27343  |        1.60739  |         1.85664  |
| Gaussian-Augmentation      | rrmse_temporal  |           0.942166 |             0.536687 |          0.342212 |          4.84415  |        0.560926 |         1.56576  |
| Gaussian-Augmentation      | rrmse_spectral  |           0.718732 |             0.714886 |          0.541302 |          0.989965 |        0.665825 |         0.77775  |
| Gaussian-Augmentation      | correlation     |           0.842887 |             0.873447 |          0.570689 |          0.933585 |        0.796143 |         0.876637 |
| Gaussian-Augmentation      | snr_improvement |          -2.89529  |            -3.05739  |         -4.57089  |         -0.282771 |       -3.43321  |        -2.32649  |
| Gaussian-Augmentation      | artifact_rrmse  |           2.04071  |             1.90107  |          1.5767   |          3.53712  |        1.85564  |         2.29734  |
| No-Augmentation            | rrmse_temporal  |           0.956819 |             0.579583 |          0.299324 |          5.23296  |        0.541981 |         1.63837  |
| No-Augmentation            | rrmse_spectral  |           0.606089 |             0.585107 |          0.388621 |          1.08361  |        0.532526 |         0.695471 |
| No-Augmentation            | correlation     |           0.850307 |             0.867502 |          0.542225 |          0.943541 |        0.797455 |         0.887306 |
| No-Augmentation            | snr_improvement |          -0.834152 |            -0.780587 |         -1.29543  |         -0.481713 |       -0.95296  |        -0.721637 |
| No-Augmentation            | artifact_rrmse  |           1.17241  |             1.1452   |          1.07277  |          1.3313   |        1.1358   |         1.21161  |
| Real-Artifact-Augmentation | rrmse_temporal  |           0.897154 |             0.505194 |          0.336675 |          5.17739  |        0.485649 |         1.576    |
| Real-Artifact-Augmentation | rrmse_spectral  |           0.618671 |             0.594589 |          0.420761 |          1.05539  |        0.55047  |         0.70122  |
| Real-Artifact-Augmentation | correlation     |           0.851143 |             0.873114 |          0.517975 |          0.931299 |        0.795016 |         0.890704 |
| Real-Artifact-Augmentation | snr_improvement |          -2.0402   |            -1.95077  |         -4.97035  |          0.029206 |       -2.60021  |        -1.53495  |
| Real-Artifact-Augmentation | artifact_rrmse  |           1.69427  |             1.66148  |          1.20966  |          2.5175   |        1.54437  |         1.85852  |
| WGAN-Augmentation          | rrmse_temporal  |           1.60745  |             1.12947  |          0.802762 |          5.39633  |        1.13244  |         2.26925  |
| WGAN-Augmentation          | rrmse_spectral  |           0.687157 |             0.670344 |          0.514486 |          1.07228  |        0.62324  |         0.761794 |
| WGAN-Augmentation          | correlation     |           0.695341 |             0.725987 |          0.507724 |          0.787379 |        0.648858 |         0.735869 |
| WGAN-Augmentation          | snr_improvement |          -8.91696  |            -8.05618  |        -13.14     |         -2.81673  |      -10.2849   |        -7.47521  |
| WGAN-Augmentation          | artifact_rrmse  |           4.54105  |             3.60745  |          1.96903  |          8.49268  |        3.5492   |         5.6181   |

## Primary diffusion contrast

Positive utility means diffusion augmentation is better.

| comparator                 | metric          |   participant_mean_utility |   positive_count |   participants |   bootstrap_low |   bootstrap_high |
|:---------------------------|:----------------|---------------------------:|-----------------:|---------------:|----------------:|-----------------:|
| Real-Artifact-Augmentation | rrmse_temporal  |                  -0.049842 |                1 |             15 |       -0.073402 |        -0.024494 |
| Real-Artifact-Augmentation | rrmse_spectral  |                  -0.012523 |                3 |             15 |       -0.023033 |        -0.003802 |
| Real-Artifact-Augmentation | correlation     |                  -0.012806 |                2 |             15 |       -0.018516 |        -0.007118 |
| Real-Artifact-Augmentation | snr_improvement |                  -0.536167 |                2 |             15 |       -0.819102 |        -0.279769 |
| Real-Artifact-Augmentation | artifact_rrmse  |                  -0.034109 |                8 |             15 |       -0.108902 |         0.036359 |

## Severity-stratified temporal RRMSE

| method                     | severity_stratum   |   rrmse_temporal |   artifact_rrmse |   snr_improvement |
|:---------------------------|:-------------------|-----------------:|-----------------:|------------------:|
| Diffusion-Augmentation     | mild               |         0.282049 |         3.09405  |         -7.52495  |
| Diffusion-Augmentation     | medium             |         0.396618 |         1.2237   |         -0.93904  |
| Diffusion-Augmentation     | severe             |         1.62532  |         0.952314 |          0.701294 |
| Gaussian-Augmentation      | mild               |         0.245806 |         3.93245  |         -7.12174  |
| Gaussian-Augmentation      | medium             |         0.420875 |         1.49295  |         -1.86856  |
| Gaussian-Augmentation      | severe             |         1.6658   |         1.09749  |         -0.195558 |
| No-Augmentation            | mild               |         0.163101 |         1.47719  |         -2.00994  |
| No-Augmentation            | medium             |         0.400997 |         1.07949  |         -0.476191 |
| No-Augmentation            | severe             |         1.76378  |         1.02659  |         -0.184167 |
| Real-Artifact-Augmentation | mild               |         0.269398 |         3.12663  |         -7.31564  |
| Real-Artifact-Augmentation | medium             |         0.367207 |         1.16915  |         -0.503234 |
| Real-Artifact-Augmentation | severe             |         1.48552  |         0.84208  |          1.86637  |
| WGAN-Augmentation          | mild               |         1.04569  |         9.33543  |        -16.3392   |
| WGAN-Augmentation          | medium             |         1.01012  |         2.92521  |         -7.6831   |
| WGAN-Augmentation          | severe             |         2.23366  |         1.54453  |         -2.90533  |
