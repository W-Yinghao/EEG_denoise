# V43 Stage 1 — frozen-checkpoint floor probe

Preregistration: `reports/v43_preregistration.md` (frozen before submission). Frozen V42R checkpoints (job_941770), identical test banks, common noise across arms. The MATCH_EB120-POP gain reading is non-adjudicating in S1.

Decision: F1 **True**, F2 **True**, F3 **True**.

## Participant-first condition means (temporal RRMSE, n=15)

| condition          |   participant_mean_rrmse_temporal |
|:-------------------|----------------------------------:|
| RAW                |                          0.714933 |
| POP                |                          0.632308 |
| MATCH              |                          0.632335 |
| MATCH_EB120        |                          0.633036 |
| MATCH_RAW120       |                          0.632404 |
| MATCH_EB10         |                          0.632308 |
| WRONG_EB120        |                          0.634842 |
| MATCH_EB120_PERROW |                          0.634031 |

## Preregistered endpoints

| endpoint     | contrast                   |     mean |    median |   positive_count |   participants |   bootstrap_low |   bootstrap_high |   margin |   frozen_wrong_harm_mean |   pass |
|:-------------|:---------------------------|---------:|----------:|-----------------:|---------------:|----------------:|-----------------:|---------:|-------------------------:|-------:|
| F1           | WRONG_EB120_minus_POP      | 0.002534 | -0.000162 |                7 |             15 |       -0.000849 |         0.006986 |    0.01  |                 0.051481 |      1 |
| F1_reduction | frozen_harm_minus_new_harm | 0.048947 |  0.005301 |               13 |             15 |        0.003909 |         0.124469 |  nan     |               nan        |    nan |
| F2           | MATCH_EB10_minus_POP       | 0        |  0        |                0 |             15 |        0        |         0        |    0.01  |               nan        |      1 |
| F3           | MATCH_EB120_minus_POP      | 0.000728 | -0.000907 |                7 |             15 |       -0.003397 |         0.005028 |    0.005 |               nan        |      1 |

## Holm over {F1, F2, F3}

raw p: {"F1": 0.0008, "F2": 0.0, "F3": 0.0258}; Holm-adjusted: {"F2": 0.0, "F1": 0.0016, "F3": 0.0258}

## Anchors

```json
{
  "frozen_reference": {
    "match_minus_pop": 2.68e-05,
    "pop": 0.632308,
    "raw": 0.714933
  },
  "match_minus_pop_mean": 2.6784048289603866e-05,
  "per_cell_anchor_qc_max_abs_diff": 1.9073486328125e-06,
  "pop_mean": 0.632308163445244,
  "raw_mean": 0.7149332578243047
}
```

## Non-adjudicating gain reading and secondaries

```json
{
  "gain_reading": {
    "bootstrap_high": 0.0033969898084978876,
    "bootstrap_low": -0.005027599184677455,
    "contrast": "POP_minus_MATCH_EB120",
    "mean": -0.0007282333049564234,
    "median": 0.0009070347587112337,
    "note": "non-adjudicating in S1: checkpoint was trained on 30-s states",
    "participants": 15,
    "positive_count": 8
  },
  "secondaries": {
    "MATCH_EB120_PERROW_minus_POP": {
      "bootstrap_high": 0.005359178510786167,
      "bootstrap_low": -0.0017926268830099918,
      "mean": 0.0017224183001720425,
      "median": -0.0005899873140151612,
      "participants": 15,
      "positive_count": 7
    },
    "MATCH_RAW120_minus_POP": {
      "bootstrap_high": 0.005168847945218049,
      "bootstrap_low": -0.0052985565820775566,
      "mean": 9.633260209132763e-05,
      "median": -0.0007869410983403213,
      "participants": 15,
      "positive_count": 7
    }
  }
}
```
