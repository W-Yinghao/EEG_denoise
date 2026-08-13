# V38P distribution fidelity and exposure

## Distribution

| method              |   conditional_energy_distance |   conditional_mmd_rbf |   conditional_covariance_discrepancy |   variance_retained |   within_query_diversity |   duplicate_rate |
|:--------------------|------------------------------:|----------------------:|-------------------------------------:|--------------------:|-------------------------:|-----------------:|
| Gaussian-Bridge     |                      0.213304 |              0.008467 |                             0.330533 |            0.736001 |                 41.3862  |         0        |
| OneStep-Bridge      |                      0.299691 |              0.012315 |                             0.464234 |            0.643356 |                  0       |         1        |
| SARD-Bridge         |                      0.504167 |              0.021693 |                             0.521702 |            0.520487 |                  1.58589 |         0        |
| Stratified-Resample |                      0.190002 |              0.007693 |                             0.31804  |            0.769369 |                 44.8665  |         0.000926 |

## Exposure

| method              |   exact_copy_rate |   near_copy_rate |   nearest_training_fiber_distance |   membership_attack_probability |
|:--------------------|------------------:|-----------------:|----------------------------------:|--------------------------------:|
| Gaussian-Bridge     |                 0 |         0        |                           6.24168 |                        0.001118 |
| SARD-Bridge         |                 0 |         0        |                           5.63003 |                        0.00463  |
| Stratified-Resample |                 0 |         0.416667 |                           2.30015 |                        0.494287 |
