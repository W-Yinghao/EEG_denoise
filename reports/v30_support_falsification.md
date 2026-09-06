# V30 support falsification

Correct, all-wrong, +1 s EOG lag, time-shuffled EOG, mean-context and exact population controls were evaluated without retraining and with fixed query/noise.

| method | condition | mean_risk |
|---|---|---|
| V25_SET_CALIB_DET | circular_lag_1s | 0.7505306076724081 |
| V25_SET_CALIB_DET | correct | 0.7515784628941521 |
| V25_SET_CALIB_DET | exact_pop | 0.761901031894592 |
| V25_SET_CALIB_DET | mean_context | 0.752861470455321 |
| V25_SET_CALIB_DET | registered_wrong | 0.7643708749823015 |
| V25_SET_CALIB_DET | time_shuffled | 0.7504034488964221 |
| V26_CALIB_SDEDIT | circular_lag_1s | 0.7532923820639823 |
| V26_CALIB_SDEDIT | correct | 0.7545025644384119 |
| V26_CALIB_SDEDIT | exact_pop | 0.761901031894592 |
| V26_CALIB_SDEDIT | mean_context | 0.7556412606491466 |
| V26_CALIB_SDEDIT | registered_wrong | 0.7686903993049378 |
| V26_CALIB_SDEDIT | time_shuffled | 0.7531405802572542 |
| V29_PA_SC_CDM | circular_lag_1s | 0.8166681807982966 |
| V29_PA_SC_CDM | correct | 0.8166679560556617 |
| V29_PA_SC_CDM | exact_pop | 0.8168807880274404 |
| V29_PA_SC_CDM | mean_context | 0.8166685139520539 |
| V29_PA_SC_CDM | registered_wrong | 0.8166689228485297 |
| V29_PA_SC_CDM | time_shuffled | 0.8166680503421555 |
| V29_PA_SC_DET | circular_lag_1s | 0.8156575488482809 |
| V29_PA_SC_DET | correct | 0.8156573736482877 |
| V29_PA_SC_DET | exact_pop | 0.8160266416020695 |
| V29_PA_SC_DET | mean_context | 0.8156579410483744 |
| V29_PA_SC_DET | registered_wrong | 0.8156584005938151 |
| V29_PA_SC_DET | time_shuffled | 0.8156574464503341 |
