# V43 Stage 1.5 — oracle-trained ceiling probe (non-deployable)

Training conditioning is the query-fitted (Qgen, generative-truth) transfer signature; this route is an oracle diagnostic and is non-deployable by construction. Held-out test participants of folds 0 and 2, common noise, same episode banks as S1.

Decision: **NO-GO** (mean POP-ORACLE = -0.051365, CI [-0.107733, -0.009970]; GO requires mean >= +0.020 and CI-low > +0.005).

```json
{
  "bootstrap_high": -0.009969576567527838,
  "bootstrap_low": -0.10773292844532989,
  "cells": [
    {
      "fold": 0,
      "seed": 20261201
    },
    {
      "fold": 2,
      "seed": 20261201
    }
  ],
  "condition_means": {
    "ORACLE": 0.6872358274607299,
    "POP": 0.6358705321375359,
    "RAW": 0.6931990439964769
  },
  "contrast": "POP_minus_ORACLE",
  "go": false,
  "go_rule": {
    "ci_low_threshold": 0.005,
    "mean_margin": 0.02
  },
  "interpretation": "NO-GO: waveform-level gain claim dead on this panel; V43 proceeds floor-only",
  "mean": -0.051365295323194005,
  "median": -0.02627848892007023,
  "participants": 6,
  "per_participant": {
    "sub-02": -0.006835534702986479,
    "sub-03": -0.01877477380912751,
    "sub-05": 0.0038380974147003144,
    "sub-11": -0.0659512272104621,
    "sub-12": -0.1866861296002753,
    "sub-14": -0.03378220403101295
  },
  "positive_count": 1,
  "preregistration": "reports/v43_preregistration.md",
  "sealed_reads": 0,
  "stage": "S1.5_oracle_trained_ceiling_probe_nondeployable"
}
```
