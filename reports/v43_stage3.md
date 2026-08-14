# V43 Stage 3 — natural-route repair

Preregistration: V43-S3 addendum (frozen before submission). Registered severity-mixture repair on the S2a recipe; frozen V42R natural gate criteria unchanged.

Decision: N-G1 **False**, N-G2 **False**, N-G3 **True**; S3b triggered: **False**.

## Gates

```json
{
  "N-G1": {
    "frozen_reference": {
      "attenuation_db": -0.133,
      "remaining": 1.082
    },
    "pass": false,
    "pop_attenuation_db_mean": -0.11681953565497127,
    "pop_output_input_rms_q99": 0.9996741196347607,
    "pop_remaining_mean": 1.0652394322028094
  },
  "N-G2": {
    "pass": false,
    "pop_paired_mean": 0.546106621830101,
    "reference_s2a": 0.526,
    "threshold": 0.536
  },
  "N-G3": {
    "match": {
      "bootstrap_high": 0.0036029069080905107,
      "bootstrap_low": -0.00371579056767587,
      "margin": 0.002,
      "mean": -2.8614954236325096e-05,
      "median": -0.0004346627574705053,
      "one_sided_upper95": 0.002991807791401381,
      "participants": 15,
      "positive_count": 7,
      "upper_margin": 0.005
    },
    "pass": true,
    "wrong_gated": {
      "bootstrap_high": 0.0017952922847143719,
      "bootstrap_low": -0.003349561523494026,
      "margin": 0.005,
      "mean": -0.0003997565916304439,
      "median": 0.001118616846118442,
      "participants": 15,
      "positive_count": 8,
      "reduction_vs_ungated": {
        "bootstrap_high": 0.3589155489208658,
        "bootstrap_low": 0.11983007400914936,
        "mean": 0.23274928933790054,
        "median": 0.13867724358169653,
        "participants": 15,
        "positive_count": 13
      }
    }
  },
  "holm": {
    "alpha": 0.05,
    "p_adjusted": {
      "N-G1": 0.973,
      "N-G2": 0.926,
      "N-G3": 0.38760000000000006
    },
    "p_raw": {
      "N-G1": 0.973,
      "N-G2": 0.463,
      "N-G3": 0.1292
    }
  }
}
```

## Natural utilities (MATCH_EB120 - POP, positive = better)

```json
null
```

## Cross-panel floor (S3c)

```json
{
  "bci2b": {
    "arm_means": {
      "C0": 0.5060596847436749,
      "C_gated": 0.4423356143116425,
      "C_wrong": 1.209706629547795,
      "C_wrong_gated": 1.1083505917996588,
      "GATED_10s": 0.5060596847436749,
      "GATED_30s": 0.5060596847436749,
      "GATED_60s": 0.4423356143116425
    },
    "duration_flatness": {
      "GATED_10s": {
        "bootstrap_high": 0.0,
        "bootstrap_low": 0.0,
        "margin": 0.002,
        "mean": 0.0,
        "median": 0.0,
        "participants": 9,
        "pass": true,
        "positive_count": 0
      },
      "GATED_30s": {
        "bootstrap_high": 0.0,
        "bootstrap_low": 0.0,
        "margin": 0.002,
        "mean": 0.0,
        "median": 0.0,
        "participants": 9,
        "pass": true,
        "positive_count": 0
      },
      "GATED_60s": {
        "bootstrap_high": 0.10925042448555611,
        "bootstrap_low": -0.31455100368262706,
        "margin": 0.002,
        "mean": -0.06372407043203239,
        "median": 0.0015179816234166255,
        "participants": 9,
        "pass": true,
        "positive_count": 5
      }
    },
    "gain_c0_minus_gated_descriptive": {
      "bootstrap_high": 0.31455100368262706,
      "bootstrap_low": -0.10925042448555616,
      "mean": 0.06372407043203239,
      "median": -0.0015179816234166255,
      "participants": 9,
      "positive_count": 3
    },
    "hard_gate_fraction": 0.1111111111111111,
    "units": 9,
    "wrong_gated_minus_c0": {
      "bootstrap_high": 1.5878012487596807,
      "bootstrap_low": 0.006355903403742934,
      "margin": 0.005,
      "mean": 0.602290907055984,
      "median": 0.13859475593325166,
      "participants": 9,
      "pass": false,
      "positive_count": 5
    },
    "wrong_ungated_minus_c0_descriptive": {
      "bootstrap_high": 1.8250667867189232,
      "bootstrap_low": 0.03569454067283865,
      "mean": 0.7036469448041203,
      "median": 0.1772284963589067,
      "participants": 9,
      "positive_count": 6
    }
  },
  "klados": {
    "arm_means": {
      "C0": 0.3885332549916096,
      "C_gated": 0.3885332549916096,
      "C_wrong": 0.49316222257014225,
      "C_wrong_gated": 0.3885332549916096,
      "GATED_10s": 0.3885332549916096
    },
    "duration_flatness": {
      "GATED_10s": {
        "bootstrap_high": 0.0,
        "bootstrap_low": 0.0,
        "margin": 0.002,
        "mean": 0.0,
        "median": 0.0,
        "participants": 54,
        "pass": true,
        "positive_count": 0
      }
    },
    "gain_c0_minus_gated_descriptive": {
      "bootstrap_high": 0.0,
      "bootstrap_low": 0.0,
      "mean": 0.0,
      "median": 0.0,
      "participants": 54,
      "positive_count": 0
    },
    "hard_gate_fraction": 1.0,
    "units": 54,
    "wrong_gated_minus_c0": {
      "bootstrap_high": 0.0,
      "bootstrap_low": 0.0,
      "margin": 0.005,
      "mean": 0.0,
      "median": 0.0,
      "participants": 54,
      "pass": true,
      "positive_count": 0
    },
    "wrong_ungated_minus_c0_descriptive": {
      "bootstrap_high": 0.18910562114279755,
      "bootstrap_low": 0.03361068893031222,
      "mean": 0.10462896757853268,
      "median": 0.06261392442774796,
      "participants": 54,
      "positive_count": 38
    }
  },
  "note": "gain rows descriptive; floor rules at S2 margins; gate frozen (no retuning)"
}
```

## Privacy onset grid (S3d)

|   lambda_label |   top1_accuracy |   same_different_auroc |
|---------------:|----------------:|-----------------------:|
|           0.05 |          0.4267 |                 0.8352 |
|           0.1  |          0.4533 |                 0.8671 |
|           0.15 |          0.4267 |                 0.8594 |
|           0.2  |          0.4133 |                 0.8604 |

## Secondary repair re-test (applied once, per the frozen addendum)

Support-only per-window Delta scaling (mean scale 0.46) improved but did not repair the
natural gate: POP remaining 1.0347 (needs < 1), attenuation −0.036 dB (needs > 0).
**Second failure: the natural route is CLOSED for the V43 arc; the flagship K2 rule
inherits this verdict.** S3b (SSVEP downstream) was never triggered.
