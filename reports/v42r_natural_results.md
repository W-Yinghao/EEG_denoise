# V42R frozen natural development results

The models and outputs were frozen before the evaluator opened query EOG. Low-EOG retention is observation retention, not physiological preservation. Natural results did not tune training or checkpoint selection.

Natural POP validity: **False**. If false, MATCH minus POP is descriptive and is not interpreted as a general support effect.

| condition   | metric                        |   participant_mean |   participant_median |   bootstrap_low |   bootstrap_high |   participants |
|:------------|:------------------------------|-------------------:|---------------------:|----------------:|-----------------:|---------------:|
| MATCH       | heldout_eog_remaining_ratio   |           1.05386  |             1.02013  |        1.00908  |         1.10583  |             15 |
| MATCH       | artifact_attenuation_db       |          -0.057923 |             0.01297  |       -0.329206 |         0.194267 |             15 |
| MATCH       | low_eog_observation_retention |           0.90851  |             0.922939 |        0.876399 |         0.934585 |             15 |
| MATCH       | psd_distortion                |           0.116316 |             0.111432 |        0.091005 |         0.142931 |             15 |
| MATCH       | covariance_distortion         |           0.078549 |             0.057899 |        0.051305 |         0.115741 |             15 |
| MATCH       | output_input_rms              |           0.976058 |             0.983251 |        0.960746 |         0.987516 |             15 |
| POP         | heldout_eog_remaining_ratio   |           1.08203  |             1.02931  |        1.0229   |         1.14479  |             15 |
| POP         | artifact_attenuation_db       |          -0.133265 |            -0.077643 |       -0.42871  |         0.146455 |             15 |
| POP         | low_eog_observation_retention |           0.902997 |             0.922662 |        0.869814 |         0.931753 |             15 |
| POP         | psd_distortion                |           0.120569 |             0.11092  |        0.092025 |         0.15033  |             15 |
| POP         | covariance_distortion         |           0.083876 |             0.068206 |        0.054183 |         0.121801 |             15 |
| POP         | output_input_rms              |           0.97278  |             0.981675 |        0.956382 |         0.985867 |             15 |
| WRONG       | heldout_eog_remaining_ratio   |           1.19319  |             1.0135   |        1.04118  |         1.39556  |             15 |
| WRONG       | artifact_attenuation_db       |          -0.364022 |            -0.075566 |       -0.910632 |         0.052574 |             15 |
| WRONG       | low_eog_observation_retention |           0.82465  |             0.906389 |        0.694085 |         0.917622 |             15 |
| WRONG       | psd_distortion                |           0.141392 |             0.122246 |        0.098544 |         0.191433 |             15 |
| WRONG       | covariance_distortion         |           0.79843  |             0.087318 |        0.068162 |         1.94423  |             15 |
| WRONG       | output_input_rms              |           1.01961  |             0.993517 |        0.977537 |         1.07768  |             15 |
