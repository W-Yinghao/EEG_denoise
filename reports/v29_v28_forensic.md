# V29 V28 forensic

V28 absolute results were replayed from the committed method summary; `forensic_summary.csv` separates matched-architecture comparisons from historical strongest-denoiser positioning. The representative frozen replay reports observation residual and context/time/FiLM magnitudes in `v28_forensic_diagnostics.json`. V28's zero-context NULL was not ordinarily trained and is not a population route. V29 therefore uses an exact frozen-population architectural bypass.

```json
{
  "rms_y": 4.434592247009277,
  "rms_x_minus_y": 3.470427989959717,
  "rms_raw_network_output": 0.08656087517738342,
  "rms_scaled_network_output": 0.008656087331473827,
  "rms_prediction_minus_y": 0.008656087331473827,
  "support_context_norm": 11.35735034942627,
  "wrong_context_norm": 11.357535362243652,
  "time_embedding_norm": 3.3249125480651855,
  "context_time_norm": 12.100385665893555,
  "film_scale_norm": 21.98067855834961,
  "film_shift_norm": 7.6864166259765625,
  "match_wrong_feature_distance": 0.04834895581007004,
  "null_route_trained": false,
  "null_interpretation": "V28 zero context was not an ordinary trained population route"
}
```
