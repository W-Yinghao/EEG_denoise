# V44 Stage 0 — subtraction probe (CPU)

Query EOG is a declared runtime input in this deployment class. Operators use the V43-frozen EB gate unchanged. Paired panel, full-window temporal RRMSE vs clean, participant-first n=15; masked top-30% rows are a V19-comparability secondary.

Decision: **GO_to_S1** — G0-1 mean +0.312300 (margin +0.010), CI [+0.159823, +0.490611]; G0-2 pass **True** (mean +0.004770).

## Arm means

| arm       |   full_window_rrmse |   masked_top30_rrmse |
|:----------|--------------------:|---------------------:|
| RAW       |            0.607642 |           nan        |
| C0        |            0.747566 |             1.52136  |
| C_gated   |            0.435266 |             0.886283 |
| C_wrong   |            1.10414  |             2.18003  |
| C_wrong_g |            0.752336 |             1.51802  |
| C_query   |            0        |             0        |

## Endpoints

```json
{
  "G0-1": {
    "bootstrap_high": 0.4906107365120672,
    "bootstrap_low": 0.15982312551862257,
    "contrast": "RRMSE(y-C0*e) - RRMSE(y-C_gated*e)",
    "go": true,
    "margin": 0.01,
    "mean": 0.31230043556425624,
    "median": 0.2198287454475774,
    "participants": 15,
    "positive_count": 14
  },
  "G0-2": {
    "bootstrap_high": 0.04348329404081203,
    "bootstrap_low": -0.03329752069373752,
    "contrast": "RRMSE(y-C_wrong_g*e) - RRMSE(y-C0*e)",
    "margin": 0.01,
    "mean": 0.004769574988484958,
    "median": -0.0016216933889709878,
    "participants": 15,
    "pass": true,
    "positive_count": 7
  },
  "G0-3": {
    "oracle_row": {
      "mean_rrmse": 4.218923155353986e-08,
      "note": "degenerate on the paired panel: the Qgen operator reproduces the generative artifact exactly"
    },
    "ungated_wrong_harm": {
      "bootstrap_high": 0.6450811059338742,
      "bootstrap_low": 0.1612446582809501,
      "mean": 0.356571415505297,
      "median": 0.24978340234301227,
      "participants": 15,
      "positive_count": 14
    }
  }
}
```

## G0-4 natural windows (descriptive)

| arm     |   attenuation_db |   coherence_reduction |   low_eog_observation_retention |   output_input_rms |
|:--------|-----------------:|----------------------:|--------------------------------:|-------------------:|
| C0      |        -0.369905 |              0.013705 |                        0.775808 |           1.15968  |
| C_gated |         2.32214  |              0.181619 |                        0.858732 |           0.874715 |
