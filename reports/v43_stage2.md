# V43 Stage 2 — floor-definitive round

Preregistration: V43-S2 addendum in `reports/v43_preregistration.md` (frozen before submission). Retrained gated model (duration-randomized EB conditioning), 15 cells; no gain endpoints anywhere; the S1.5 NO-GO stands.

Decision: D-F1 **True**, D-F2 **True**, D-F3 **True**, D-F4 **True**.

## Participant-first condition means (temporal RRMSE, n=15)

| condition          |   participant_mean_rrmse_temporal |
|:-------------------|----------------------------------:|
| RAW                |                          0.607642 |
| POP                |                          0.525978 |
| WRONG              |                          1.18027  |
| NO_TRANSFER_BRANCH |                          0.61778  |
| WRONG_EB120        |                          0.525755 |
| MATCH_EB10         |                          0.525978 |
| MATCH_EB30         |                          0.525978 |
| MATCH_EB60         |                          0.524084 |
| MATCH_EB120        |                          0.524547 |

## Definitive floor endpoints

| endpoint       | contrast                |      mean |   bootstrap_low |   bootstrap_high |   margin |   pass |     median |   positive_count |   participants |   one_sided_upper95 |
|:---------------|:------------------------|----------:|----------------:|-----------------:|---------:|-------:|-----------:|-----------------:|---------------:|--------------------:|
| D-F1           | WRONG_EB120_minus_POP   | -0.000223 |       -0.00342  |         0.00281  |    0.005 |      1 | nan        |              nan |            nan |          nan        |
| D-F1_reduction | WRONG_minus_WRONG_EB120 |  0.654513 |        0.271306 |         1.12777  |  nan     |    nan |   0.28607  |               15 |             15 |          nan        |
| D-F3           | MATCH_EB120_minus_POP   | -0.001432 |       -0.004113 |         0.00136  |    0.002 |      1 | nan        |              nan |            nan |            0.000952 |
| D-F4[10s]      | MATCH_EB10_minus_POP    |  0        |        0        |         0        |    0.002 |    nan |   0        |                0 |             15 |          nan        |
| D-F4[30s]      | MATCH_EB30_minus_POP    |  0        |        0        |         0        |    0.002 |    nan |   0        |                0 |             15 |          nan        |
| D-F4[60s]      | MATCH_EB60_minus_POP    | -0.001894 |       -0.004302 |         0.000771 |    0.002 |    nan |  -0.002689 |                4 |             15 |          nan        |
| D-F4[120s]     | MATCH_EB120_minus_POP   | -0.001432 |       -0.004113 |         0.00136  |    0.002 |    nan |  -0.003291 |                6 |             15 |          nan        |

D-F2 (construction check): {"construction_check": "hard gate routes <60-s support to the bit-identical POP state", "state_build_assert": true, "output_bypass_assert": true, "max_output_bypass_diff": 0.0, "pass": true}

Holm over {D-F1, D-F3, D-F4}: raw p {"D-F1": 0.0004, "D-F3": 0.0098, "D-F4": 0.0098}; adjusted {"D-F1": 0.0012000000000000001, "D-F3": 0.0196, "D-F4": 0.0196}

## Support-duration curve (descriptive)

|   support_seconds |   MATCH_EBd_mean_rrmse |   delta_vs_POP |
|------------------:|-----------------------:|---------------:|
|                10 |               0.525978 |       0        |
|                30 |               0.525978 |       0        |
|                60 |               0.524084 |      -0.001894 |
|               120 |               0.524547 |      -0.001432 |

## Branch necessity on the retrained model

```json
{
  "NO_TRANSFER_minus_MATCH_EB120": {
    "bootstrap_high": 0.1235537247294734,
    "bootstrap_low": 0.06546115380355935,
    "mean": 0.09323366824028197,
    "median": 0.07578132388395414,
    "participants": 15,
    "positive_count": 15
  },
  "NO_TRANSFER_minus_POP": {
    "bootstrap_high": 0.1236384024352547,
    "bootstrap_low": 0.06293354944079815,
    "mean": 0.09180202414160397,
    "median": 0.07614536003772326,
    "participants": 15,
    "positive_count": 14
  }
}
```

## Positioning (descriptive only, no superiority claims)

```json
{
  "DET_twin": {
    "DET_MATCH_EB120_mean": 0.5010101153480112,
    "DET_MATCH_EB120_minus_DET_POP": {
      "bootstrap_high": 0.001586368079582218,
      "bootstrap_low": -0.011558520440009185,
      "mean": -0.0032857389576141332,
      "median": 0.00012124328350182623,
      "participants": 15,
      "positive_count": 9
    },
    "DET_POP_mean": 0.5042958543056254,
    "DIFFUSION_POP_minus_DET_POP": {
      "bootstrap_high": 0.06356611133042558,
      "bootstrap_low": -0.01943969351247184,
      "mean": 0.02168238810958833,
      "median": 0.03069513648127517,
      "participants": 15,
      "positive_count": 12
    },
    "positioning_only": true
  },
  "LINEAR_EOG": {
    "LINEAR_EB120_mean": 0.4352657623093163,
    "LINEAR_EB120_minus_POP": {
      "bootstrap_high": -0.048199367016825846,
      "bootstrap_low": -0.13135399484403293,
      "mean": -0.09071248010589725,
      "median": -0.09929972571500705,
      "participants": 15,
      "positive_count": 2
    },
    "label": "requires query EOG at inference; not information-matched"
  }
}
```

## Retrained-state builder lambda distribution

```json
{
  "10": {
    "hard_gate_fraction": 1.0,
    "max": 0.0,
    "mean": 0.0,
    "min": 0.0
  },
  "120": {
    "hard_gate_fraction": 0.05161290322580645,
    "max": 0.9999289932609888,
    "mean": 0.8918303211472816,
    "min": 0.0
  },
  "30": {
    "hard_gate_fraction": 1.0,
    "max": 0.0,
    "mean": 0.0,
    "min": 0.0
  },
  "60": {
    "hard_gate_fraction": 0.053763440860215055,
    "max": 0.999893111360978,
    "mean": 0.8590192671518446,
    "min": 0.0
  }
}
```

## S2b — ceiling completion (descriptive; NO-GO final)

Pooled n=15 POP-ORACLE: mean -0.004830, CI [-0.056141, +0.065453], positive 4/15. Fold means: {"0": -0.007257403699137892, "1": 0.08645938058907632, "2": -0.09547318694725011, "3": -0.0021102624014019966, "4": -0.0057672094165657955}.

| participant   |   pop_minus_oracle |
|:--------------|-------------------:|
| sub-02        |          -0.006836 |
| sub-03        |          -0.018775 |
| sub-05        |           0.003838 |
| sub-06        |          -0.079512 |
| sub-07        |          -0.081998 |
| sub-09        |           0.420888 |
| sub-11        |          -0.065951 |
| sub-12        |          -0.186686 |
| sub-14        |          -0.033782 |
| sub-15        |           0.008123 |
| sub-17        |          -0.039028 |
| sub-18        |           0.024575 |
| sub-19        |          -0.009863 |
| sub-21        |          -0.003186 |
| sub-23        |          -0.004253 |

## S2c — lambda-privacy curve (descriptive; no privacy-safe claim)

| lambda_label   |   lambda_mean |   top1_accuracy |   top3_accuracy |   same_different_auroc |
|:---------------|--------------:|----------------:|----------------:|-----------------------:|
| 0.00           |      0        |        0.066667 |        0.2      |               0.5      |
| 0.25           |      0.25     |        0.413333 |        0.746667 |               0.860444 |
| 0.50           |      0.5      |        0.44     |        0.746667 |               0.86     |
| 0.75           |      0.75     |        0.44     |        0.76     |               0.859873 |
| 1.00           |      1        |        0.44     |        0.76     |               0.859556 |
| EB             |      0.955336 |        0.466667 |        0.746667 |               0.857841 |
