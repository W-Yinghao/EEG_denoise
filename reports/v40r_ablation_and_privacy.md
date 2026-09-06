# V40R ablation and lightweight privacy

|   support_seconds |   rrmse_temporal |
|------------------:|-----------------:|
|                 0 |          1.19255 |
|                10 |          1.19272 |
|                30 |          1.28468 |

POP is an exact adapter bypass. POP_MEAN and ADAPTER_DISABLED isolate context and FiLM paths.

## Disjoint-half context linkage

|   fold |   top1_participant_classification |   verification_auroc |   context_state_bytes | state_stored   |
|-------:|----------------------------------:|---------------------:|----------------------:|:---------------|
|      0 |                          0.333333 |             0.760952 |                   512 | False          |
|      1 |                          0.266667 |             0.689206 |                   512 | False          |
|      2 |                          0.133333 |             0.659365 |                   512 | False          |
|      3 |                          0.266667 |             0.633968 |                   512 | False          |
|      4 |                          0.466667 |             0.702857 |                   512 | False          |

The support state is 128 float32 values (512 bytes), is not persistent, and should be deleted at session end. This is a linkage diagnostic, not an anonymity claim.
