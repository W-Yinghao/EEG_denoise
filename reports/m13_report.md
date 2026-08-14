# FLAGSHIP-M13 report

Preregistration: `reports/m13_preregistration.md` (frozen before submission).

## Stop point

The stage STOPPED at the preregistered P0 gate: the pooled-prior pilot fails PV-1 on
MobileBCI (posterior POP route worse than RAW: utility −0.053 [−0.090, −0.016]) and PV-2
on Klados (rms q99 0.812) and BCI2b (0.457) — amplitude collapse outside [0.90, 1.10].
P1 (pooled ×3, LODO ×4, ambient, per-dataset refs) and W3 were NOT run.

```json
{
  "bci2b": {
    "pv1_pass": true,
    "pv1_utility_raw_minus_route": {
      "bootstrap_high": 0.8202842163576296,
      "bootstrap_low": 0.08222017285537797,
      "mean": 0.401173764772517,
      "median": 0.2385480344557528,
      "n": 9,
      "positive_count": 7
    },
    "pv2_pass": false,
    "pv2_rms_q99": 0.45718637701121373,
    "raw_mean_rrmse": 1.21689731574964,
    "route_mean_rrmse": 0.815723550977123,
    "units": 9
  },
  "klados": {
    "pv1_pass": true,
    "pv1_utility_raw_minus_route": {
      "bootstrap_high": 0.23461031385349793,
      "bootstrap_low": 0.0982362327345035,
      "mean": 0.16356838634692336,
      "median": 0.12758062972598705,
      "n": 54,
      "positive_count": 37
    },
    "pv2_pass": false,
    "pv2_rms_q99": 0.8116670943733454,
    "raw_mean_rrmse": 0.7116395430562475,
    "route_mean_rrmse": 0.5480711567093243,
    "units": 54
  },
  "mobilebci": {
    "pv1_pass": false,
    "pv1_utility_raw_minus_route": {
      "bootstrap_high": -0.01603782893729642,
      "bootstrap_low": -0.08980578181706043,
      "mean": -0.05293275306198629,
      "median": -0.05338202727225749,
      "n": 15,
      "positive_count": 2
    },
    "pv2_pass": true,
    "pv2_rms_q99": 0.9812231842607985,
    "raw_mean_rrmse": 0.7149332631546206,
    "route_mean_rrmse": 0.7678660162166068,
    "units": 15
  }
}
```

## W1 — transport repair diagnosis

Math contracts (round-trip, rho=0 bit-identity, locality) pass on all panels. The
split-half abstention rule is adopted as the deployment rule (Klados
estimation-noise-dominated, 33/54 abstain; MobileBCI/BCI2b heterogeneity-dominated).
The rank-truncated whitening FAILED its own kappa target (whitener-spectrum cap does not
bound kappa of bar_root @ W @ L; MobileBCI kappa still 4.6e3) and degrades whitened arms
where it binds; it is reported descriptively and not deployed.

```json
{
  "bci2b": {
    "abstentions": 5,
    "between_median": 1.6842131471871717,
    "cells": 9,
    "diagnosis": "heterogeneity_dominated",
    "frame_angle_p50": 30.931760571573413,
    "frame_angle_within_15deg_fraction": 0.1111111111111111,
    "kappa_max": 1.2432380952885398,
    "kappa_median": 1.0509495127456958,
    "kappa_within_target_fraction": 1.0,
    "rho0_bit_identity": true,
    "rho_mean_nonabstained": 0.9669026109529864,
    "roundtrip_gate": true,
    "roundtrip_max": 1.63202784619898e-13,
    "split_half_median": 0.7873099136422771
  },
  "klados": {
    "abstentions": 33,
    "between_median": 0.5834179259992867,
    "cells": 54,
    "diagnosis": "estimation_noise_dominated",
    "frame_angle_p50": 10.605701449257866,
    "frame_angle_within_15deg_fraction": 0.7592592592592593,
    "kappa_max": 80.77340730966802,
    "kappa_median": 1.2850897235193064,
    "kappa_within_target_fraction": 1.0,
    "rho0_bit_identity": true,
    "rho_mean_nonabstained": 0.96834786018524,
    "roundtrip_gate": true,
    "roundtrip_max": 1.4497902549023674e-13,
    "split_half_median": 0.6650416650762243
  },
  "mobilebci": {
    "abstentions": 13,
    "between_median": 0.7924608703092724,
    "cells": 90,
    "diagnosis": "heterogeneity_dominated",
    "frame_angle_p50": 25.25568724378062,
    "frame_angle_within_15deg_fraction": 0.2,
    "kappa_max": 4613.405668144272,
    "kappa_median": 209.6983662200197,
    "kappa_within_target_fraction": 1.0,
    "rho0_bit_identity": true,
    "rho_mean_nonabstained": 0.9522218739165673,
    "roundtrip_gate": true,
    "roundtrip_max": 2.794403074102154e-13,
    "split_half_median": 0.3580699961934223
  }
}
```

## W4 — operator-posterior UQ (V44-S1 checkpoints)

UQ-1 coverage bands FAIL (0.271/0.440/0.516 vs [0.35,0.65]/[0.65,0.90]/[0.80,0.97]) but
dispersion is ~150x the V37T reference (0.0029) — materially dispersed, still
under-dispersed. UQ-2: the K-chain diffusion ensemble BEATS the DET-ensemble reference
on both CRPS (0.1529 vs 0.1548) and risk-coverage AUC (0.0920 vs 0.1129). UQ-3
conformalization saturates the registered scale grid at holdout coverage 0.765
(reported as conformalized, a downgrade).

```json
{
  "UQ-1": {
    "bands": {
      "0.5": [
        0.35,
        0.65
      ],
      "0.8": [
        0.65,
        0.9
      ],
      "0.9": [
        0.8,
        0.97
      ]
    },
    "pass": false,
    "v37t_reference": 0.0029
  },
  "UQ-2": {
    "crps_win": true,
    "risk_coverage_win": true
  },
  "UQ-3_conformalized": {
    "det_reference": {
      "holdout_coverage_80": 0.6517408288043479,
      "scale": 3.0
    },
    "diff": {
      "holdout_coverage_80": 0.7648458729619566,
      "scale": 3.0
    }
  },
  "det_reference": {
    "coverage_50": 0.21285984205163044,
    "coverage_80": 0.37179656834993957,
    "coverage_90": 0.43927149381038644,
    "crps": 0.1547910834032748,
    "participants": 15,
    "risk_coverage_auc": 0.11286150412569007
  },
  "diff": {
    "coverage_50": 0.2710544752038043,
    "coverage_80": 0.4404896022041063,
    "coverage_90": 0.516361785741244,
    "crps": 0.1528857012265091,
    "participants": 15,
    "risk_coverage_auc": 0.09195938797449925
  }
}
```

## Compute ledger (approximate)

W1 CPU ~1 h; P0 training 1 cell ~35 min A100 + eval; W4 15 inference cells ~8 GPU-h.
The planned ~500-650 GPU-h P1 campaign was not spent (stopped at P0).
