# V40R natural development results

Low-EOG observation retention is not physiological preservation. Query EOG is unavailable to inference and opened only by the post-freeze evaluator.

| condition        | metric                        |   participant_mean |   participant_median |   bootstrap_low |   bootstrap_high |   participants |
|:-----------------|:------------------------------|-------------------:|---------------------:|----------------:|-----------------:|---------------:|
| ADAPTER_DISABLED | heldout_eog_remaining_ratio   |           4.01086  |             2.53635  |        2.49762  |         5.98539  |             15 |
| ADAPTER_DISABLED | artifact_attenuation_db       |          -5.3941   |            -4.90218  |       -7.12148  |        -3.72139  |             15 |
| ADAPTER_DISABLED | eeg_eog_coherence_reduction   |          -3.01085  |            -1.53635  |       -4.98539  |        -1.49762  |             15 |
| ADAPTER_DISABLED | low_eog_observation_retention |          -0.342096 |            -0.345902 |       -0.438538 |        -0.255292 |             15 |
| ADAPTER_DISABLED | psd_distortion                |           1.22724  |             1.01866  |        0.957369 |         1.54025  |             15 |
| ADAPTER_DISABLED | covariance_distortion         |           1.61459  |             1.55703  |        1.40443  |         1.842    |             15 |
| ADAPTER_DISABLED | output_input_rms              |           0.997619 |             0.993268 |        0.874953 |         1.1183   |             15 |
| MATCH            | heldout_eog_remaining_ratio   |           4.04658  |             2.53159  |        2.5012   |         6.0723   |             15 |
| MATCH            | artifact_attenuation_db       |          -5.4203   |            -4.90826  |       -7.17504  |        -3.72728  |             15 |
| MATCH            | eeg_eog_coherence_reduction   |          -3.04658  |            -1.53159  |       -5.0723   |        -1.5012   |             15 |
| MATCH            | low_eog_observation_retention |          -0.345741 |            -0.351523 |       -0.441356 |        -0.259352 |             15 |
| MATCH            | psd_distortion                |           1.21499  |             1.03895  |        0.954414 |         1.51583  |             15 |
| MATCH            | covariance_distortion         |           1.61554  |             1.55644  |        1.40451  |         1.84324  |             15 |
| MATCH            | output_input_rms              |           1.00387  |             0.993627 |        0.889306 |         1.12051  |             15 |
| POP              | heldout_eog_remaining_ratio   |           4.01086  |             2.53635  |        2.49762  |         5.98539  |             15 |
| POP              | artifact_attenuation_db       |          -5.3941   |            -4.90218  |       -7.12148  |        -3.72139  |             15 |
| POP              | eeg_eog_coherence_reduction   |          -3.01085  |            -1.53635  |       -4.98539  |        -1.49762  |             15 |
| POP              | low_eog_observation_retention |          -0.342096 |            -0.345902 |       -0.438538 |        -0.255292 |             15 |
| POP              | psd_distortion                |           1.22724  |             1.01866  |        0.957369 |         1.54025  |             15 |
| POP              | covariance_distortion         |           1.61459  |             1.55703  |        1.40443  |         1.842    |             15 |
| POP              | output_input_rms              |           0.997619 |             0.993268 |        0.874953 |         1.1183   |             15 |
| POP_MEAN         | heldout_eog_remaining_ratio   |           4.00764  |             2.52394  |        2.49676  |         5.97977  |             15 |
| POP_MEAN         | artifact_attenuation_db       |          -5.38776  |            -4.90151  |       -7.10852  |        -3.71651  |             15 |
| POP_MEAN         | eeg_eog_coherence_reduction   |          -3.00764  |            -1.52394  |       -4.97977  |        -1.49676  |             15 |
| POP_MEAN         | low_eog_observation_retention |          -0.340196 |            -0.340583 |       -0.435592 |        -0.253566 |             15 |
| POP_MEAN         | psd_distortion                |           1.22383  |             1.03338  |        0.950591 |         1.53902  |             15 |
| POP_MEAN         | covariance_distortion         |           1.61401  |             1.55232  |        1.40318  |         1.84253  |             15 |
| POP_MEAN         | output_input_rms              |           0.997829 |             0.992278 |        0.875198 |         1.11875  |             15 |
| SHUFFLED         | heldout_eog_remaining_ratio   |           4.08147  |             2.52956  |        2.58746  |         6.03009  |             15 |
| SHUFFLED         | artifact_attenuation_db       |          -5.60309  |            -4.90077  |       -7.20509  |        -4.11014  |             15 |
| SHUFFLED         | eeg_eog_coherence_reduction   |          -3.08147  |            -1.52956  |       -5.03009  |        -1.58746  |             15 |
| SHUFFLED         | low_eog_observation_retention |          -0.456471 |            -0.339872 |       -0.744179 |        -0.268289 |             15 |
| SHUFFLED         | psd_distortion                |           1.26099  |             1.04443  |        0.99442  |         1.56101  |             15 |
| SHUFFLED         | covariance_distortion         |           2.23667  |             1.57332  |        1.43231  |         3.6278   |             15 |
| SHUFFLED         | output_input_rms              |           1.08438  |             0.993361 |        0.897905 |         1.3197   |             15 |
| WRONG            | heldout_eog_remaining_ratio   |           4.29668  |             2.81028  |        2.6391   |         6.34213  |             15 |
| WRONG            | artifact_attenuation_db       |          -5.85615  |            -5.84028  |       -7.64799  |        -4.08528  |             15 |
| WRONG            | eeg_eog_coherence_reduction   |          -3.29668  |            -1.81028  |       -5.34213  |        -1.6391   |             15 |
| WRONG            | low_eog_observation_retention |          -0.660384 |            -0.384094 |       -1.12976  |        -0.321405 |             15 |
| WRONG            | psd_distortion                |           1.2956   |             1.05294  |        1.02594  |         1.59786  |             15 |
| WRONG            | covariance_distortion         |           6.36044  |             1.83364  |        1.6782   |        13.3552   |             15 |
| WRONG            | output_input_rms              |           1.2131   |             1.10257  |        0.960009 |         1.53902  |             15 |
