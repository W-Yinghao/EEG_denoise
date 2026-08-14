# FLAGSHIP-M13R report

Preregistration: M13R addendum in `reports/m13_preregistration.md` (frozen before submission).

## Repair decision — FAIL (pooled-prior axis CLOSED)

No panel passes both PV gates under R-A + R-B (bci2b: PV-2 q99 0.457; klados: PV-2 q99
0.812; mobilebci: PV-1 −0.053). The pilots reproduce P0's numbers because P0's readout
ALREADY confined corrections to span(U°) via the GLS likelihood step — R-A's complement
identity was therefore structurally a no-op, which sharpens the diagnosis: the amplitude
collapse lives INSIDE the rank-2 ocular coefficients (the prior over-estimates the
subtraction along U° sensor directions). This is a prior-training / contamination-model
mismatch, not a readout leak. Per the frozen rule: pooled-prior axis CLOSED as an honest
negative; no further repairs; the flagship descopes to
{matrix + V43/V44 legs + UQ + per-panel/analytic transport rows}. Reduced P1 and DIFF
rows were NOT run.

```json
{
  "consequence": "pooled-prior axis CLOSED as an honest negative with the two-mode diagnosis; flagship descopes to {matrix + V43/V44 legs + UQ + per-panel/analytic transport rows}",
  "panels_passing_both_gates": [],
  "pilots": {
    "bci2b": {
      "abstentions": 5,
      "panel": "bci2b",
      "pv1_pass": true,
      "pv1_utility_raw_minus_route": {
        "bootstrap_high": 0.8202056442002802,
        "bootstrap_low": 0.08212762319304892,
        "mean": 0.40108206677982006,
        "median": 0.23844560788699642,
        "n": 9,
        "positive_count": 7
      },
      "pv2_pass": false,
      "pv2_rms_q99": 0.45718339177186185,
      "raw_mean_rrmse": 1.21689731574964,
      "readout": "R-A canonical artifact-subspace (complement identity)",
      "route_mean_rrmse": 0.8158152489698199,
      "transport_config": "off",
      "units": 9
    },
    "klados": {
      "abstentions": 32,
      "panel": "klados",
      "pv1_pass": true,
      "pv1_utility_raw_minus_route": {
        "bootstrap_high": 0.2343378457261665,
        "bootstrap_low": 0.09796597856163587,
        "mean": 0.16329946751838503,
        "median": 0.1273125917874775,
        "n": 54,
        "positive_count": 37
      },
      "pv2_pass": false,
      "pv2_rms_q99": 0.8116836367652899,
      "raw_mean_rrmse": 0.7116395430562475,
      "readout": "R-A canonical artifact-subspace (complement identity)",
      "route_mean_rrmse": 0.5483400755378626,
      "transport_config": "full",
      "units": 54
    },
    "mobilebci": {
      "abstentions": 13,
      "panel": "mobilebci",
      "pv1_pass": false,
      "pv1_utility_raw_minus_route": {
        "bootstrap_high": -0.01615070135134821,
        "bootstrap_low": -0.08990301751474661,
        "mean": -0.05299297333038296,
        "median": -0.05345768266904105,
        "n": 15,
        "positive_count": 2
      },
      "pv2_pass": true,
      "pv2_rms_q99": 0.9810075447933859,
      "raw_mean_rrmse": 0.7149332631546206,
      "readout": "R-A canonical artifact-subspace (complement identity)",
      "route_mean_rrmse": 0.7679262364850036,
      "transport_config": "off",
      "units": 15
    }
  },
  "preregistration": "reports/m13_preregistration.md (M13R addendum)",
  "repair_pass": false,
  "sealed_reads": 0
}
```

## W3 transport factorial (analytic backbone; decoupled; PRIMARY)

TG-1 PASSES on both GO panels (Holm-adjusted p: klados 0.0052, bci2b 0.0058):

| panel | TG-1 gain (POP−MATCH) | CI | TG-2 wrong-gated | gauge | oracle headroom | abstain (ITT) |
|---|---|---|---|---|---|---|
| klados | +0.01596 | [+0.00378, +0.03038] | FAIL (+0.0169 [−0.0046,+0.0414]) | ok | +0.0396 | 33/54 |
| bci2b  | +0.01818 | [+0.00085, +0.03840] | PASS (mean ≤ +0.005) | ok | +0.0360 | 5/9 |

The transport channel's deployable, zero-training gain is confirmed under the frozen R-B
configs with intention-to-treat abstentions. The klados wrong-gated failure repeats the
known identity-blind-gate limitation (V43-S3c, V44-S1/S3) — ownership remains open on the
transport leg for M3-5 (the port noted in V44-S2 instructions).

```json
{
  "bci2b": {
    "TG-1": {
      "bootstrap_high": 0.03839887087372641,
      "bootstrap_low": 0.0008507899743195324,
      "mean": 0.01818210204045207,
      "median": 0.0,
      "n": 9,
      "p_raw": 0.0058,
      "pass": true,
      "positive_count": 4,
      "tost_equivalent_pm0.005": false
    },
    "TG-2": {
      "gauge_not_better": {
        "bootstrap_high": 0.03794266640079141,
        "bootstrap_low": -0.022348848652534208,
        "mean": 0.006532719615204975,
        "median": 0.0,
        "n": 9,
        "pass": true,
        "positive_count": 2
      },
      "oracle_headroom": {
        "bootstrap_high": 0.06478280132426534,
        "bootstrap_low": 0.010373513922852665,
        "mean": 0.036040859141766454,
        "median": 0.03957773859666658,
        "n": 9,
        "positive_count": 6
      },
      "wrong_gated_minus_pop": {
        "bootstrap_high": -0.0013050297596418625,
        "bootstrap_low": -0.024519674691533613,
        "margin": 0.005,
        "mean": -0.011256263717846037,
        "median": 0.0,
        "n": 9,
        "pass": true,
        "positive_count": 0
      }
    },
    "abstentions_itt": 5,
    "arm_means": {
      "GAUGE-NULL": 0.9448396254606638,
      "T-MATCH": 0.9331902430354168,
      "T-ORACLE": 0.9153314859341024,
      "T-POP": 0.9513723450758688,
      "T-WRONG": 0.9401160813580227
    },
    "transport_config": "off",
    "units": 9
  },
  "klados": {
    "TG-1": {
      "bootstrap_high": 0.030375115796620156,
      "bootstrap_low": 0.00377658502896934,
      "mean": 0.015959597376633213,
      "median": 0.0,
      "n": 54,
      "p_raw": 0.0026,
      "pass": true,
      "positive_count": 13,
      "tost_equivalent_pm0.005": false
    },
    "TG-2": {
      "gauge_not_better": {
        "bootstrap_high": -0.01613498199408653,
        "bootstrap_low": -0.049816089175845076,
        "mean": -0.03232080632675476,
        "median": 0.0,
        "n": 54,
        "pass": true,
        "positive_count": 3
      },
      "oracle_headroom": {
        "bootstrap_high": 0.06591241250207426,
        "bootstrap_low": 0.010632153238620795,
        "mean": 0.03959218083275198,
        "median": 0.030237664493996785,
        "n": 54,
        "positive_count": 36
      },
      "wrong_gated_minus_pop": {
        "bootstrap_high": 0.04143690653937528,
        "bootstrap_low": -0.004565592489456436,
        "margin": 0.005,
        "mean": 0.016877798135052066,
        "median": 0.0,
        "n": 54,
        "pass": false,
        "positive_count": 13
      }
    },
    "abstentions_itt": 33,
    "arm_means": {
      "GAUGE-NULL": 0.6645749664010006,
      "T-MATCH": 0.6162945626976125,
      "T-ORACLE": 0.592661979241494,
      "T-POP": 0.6322541600742458,
      "T-WRONG": 0.6491319582092978
    },
    "transport_config": "full",
    "units": 54
  }
}
```

## Compute ledger
3 pilot GPU cells (~25 min each) + W3 CPU (~40 min). Reduced P1 not spent.
