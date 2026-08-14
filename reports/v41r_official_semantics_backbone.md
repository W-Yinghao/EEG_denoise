# V41R official-semantics backbone

The shared B×C -> [B*C,1,512] model retains the official dual branch, epsilon prediction, 500-step linear schedule, and conditional observation at every ancestral step. It has no 46-channel convolution.

```json
{
  "population_valid": false,
  "pop_temporal_rrmse": 0.6939456885370116,
  "raw_temporal_rrmse": 0.5501239840736768,
  "pop_snr_improvement": -8.097551776024806,
  "pop_output_input_rms_q99": 0.7789387549459934,
  "v40r_pop_temporal_rrmse": 1.192548
}
```
