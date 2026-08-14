# V44 Stage 2 — ownership guard, re-baselining, online refinement

Inference-only on the frozen V44-S1 checkpoints; addendum rules frozen before submission. S1 verdicts unrevised.

Decision: OG-1 **False**, OG-2 **False**, OR-1 **False**.

## RB-1 honest re-baselining

```json
{
  "bad_anchor_worse_than_none_POP_minus_NO_A0": {
    "bootstrap_high": 0.2720311189518601,
    "bootstrap_low": -0.0511723184155542,
    "mean": 0.0773637761910019,
    "median": -0.01597895619488554,
    "participants": 15,
    "positive_count": 5
  },
  "condition_means": {
    "MATCH_gated": 0.43097083320090956,
    "NO_A0": 0.5737785470205482,
    "POP": 0.6511423232115501
  },
  "gain_vs_NO_A0": {
    "bootstrap_high": 0.18342107992032552,
    "bootstrap_low": 0.10612886040285954,
    "mean": 0.14280771381963858,
    "median": 0.13941557341604494,
    "participants": 15,
    "positive_count": 15
  },
  "gain_vs_POP": {
    "bootstrap_high": 0.40244942594839805,
    "bootstrap_low": 0.09292537851770855,
    "mean": 0.22017149001064049,
    "median": 0.10346880583286595,
    "participants": 15,
    "positive_count": 14
  }
}
```

## Ownership guard

```json
{
  "OG-1": {
    "bootstrap_high": 0.016055358866605165,
    "bootstrap_low": -0.008167545541196485,
    "contrast": "WRONG_gated_guard_minus_NO_A0",
    "detection_rate": 0.8833333333333333,
    "detection_required": 0.9,
    "margin": 0.005,
    "mean": 0.003919232635073253,
    "median": 0.004331900330726057,
    "participants": 15,
    "pass": false,
    "positive_count": 9
  },
  "OG-2": {
    "bootstrap_high": 0.12335430694255085,
    "bootstrap_low": 0.06368500420746286,
    "contrast": "MATCH_gated_guard_minus_MATCH_gated",
    "false_alarm_max": 0.1,
    "false_alarm_rate": 0.6333333333333333,
    "margin": 0.005,
    "mean": 0.09230606056096605,
    "median": 0.08079832393559627,
    "participants": 15,
    "pass": false,
    "positive_count": 14
  },
  "holm": {
    "alpha": 0.05,
    "p_adjusted": {
      "OG-1": 0.834,
      "OG-2": 1.0
    },
    "p_raw": {
      "OG-1": 0.417,
      "OG-2": 1.0
    }
  },
  "sensitivity": [
    {
      "detection": 0.9083333333333333,
      "false_alarm": 0.6916666666666667,
      "t_v": 10
    },
    {
      "detection": 0.8833333333333333,
      "false_alarm": 0.6333333333333333,
      "t_v": 30
    },
    {
      "detection": 0.8916666666666667,
      "false_alarm": 0.55,
      "t_v": 60
    }
  ]
}
```

## Online refinement

```json
{
  "OR-1": {
    "bootstrap_high": -0.12043487384754478,
    "bootstrap_low": -1.4926665089615927,
    "coldpop_end_mean": 1.0692687894077808,
    "contrast": "static_MATCH_gated_minus_warm_RLS_end",
    "gap_closure_fraction": -2.5417001163915156,
    "mean": -0.6205052494146608,
    "median": -0.1202372434005762,
    "oracle_gap_s1": 0.24413,
    "participants": 15,
    "pass": false,
    "positive_count": 1,
    "warm099_end_mean": 1.636684630268347
  }
}
```

### Calibration half-life (OR-2/OR-3)

|   t_seconds |   warm_mean_rrmse |   coldzero_mean_rrmse |   wrongwarm_mean_rrmse |   calibration_value_cold_minus_warm |
|------------:|------------------:|----------------------:|-----------------------:|------------------------------------:|
|          10 |          0.943731 |              0.979012 |               0.960958 |                            0.03528  |
|          30 |          0.654395 |              0.665359 |               0.662536 |                            0.010964 |
|          60 |          0.626134 |              0.65437  |               0.635792 |                            0.028236 |
|         120 |          0.60071  |              0.611173 |               0.614478 |                            0.010463 |
|         240 |          1.05148  |              1.06927  |               1.03706  |                            0.017793 |

## Natural panel

```json
{
  "attenuation_db": {
    "MATCH_guard_minus_MATCH": {
      "bootstrap_high": -1.0580903920720806,
      "bootstrap_low": -1.7724612739196426,
      "mean": -1.398329451905222,
      "median": -1.200974171714661,
      "participants": 15,
      "positive_count": 0
    },
    "NO_A0_ref_mean": 0.2766726779151144,
    "RLS_warm_end_minus_MATCH": {
      "bootstrap_high": -0.5576327997091974,
      "bootstrap_low": -1.2640727103188367,
      "mean": -0.9139623258079183,
      "median": -0.9104219523099171,
      "participants": 15,
      "positive_count": 1
    },
    "WRONG_guard_mean": 0.4522535259918046
  },
  "low_eog_observation_retention": {
    "MATCH_guard_minus_MATCH": {
      "bootstrap_high": 0.06444789343892927,
      "bootstrap_low": 0.026940136915398653,
      "mean": 0.044380044028362285,
      "median": 0.04137527918551309,
      "participants": 15,
      "positive_count": 13
    },
    "NO_A0_ref_mean": 0.9210058125484095,
    "RLS_warm_end_minus_MATCH": {
      "bootstrap_high": -0.037835257822303535,
      "bootstrap_low": -0.07818738387224906,
      "mean": -0.0569653441793049,
      "median": -0.04569542607979027,
      "participants": 15,
      "positive_count": 0
    },
    "WRONG_guard_mean": 0.8993524742769047
  }
}
```
