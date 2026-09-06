# V44 Stage 3 — drift-decomposed ownership rescoring

CPU-only re-routing of frozen S1 outputs; addendum frozen before rescoring. No further ownership attempts after this stage regardless of outcome.

SPATIAL: OG-1' **False**, OG-2' **False** (detection 0.200, false-alarm 0.078); GAIN (negative control): detection 0.200, false-alarm 0.078.

ROC AUC: spatial 0.7233, gain 0.7167, S2 Mahalanobis 0.7816.

```json
{
  "closure_statement": "ownership verification from operator features is CLOSED for the likelihood leg; complete two-family negative with the drift diagnosis",
  "families": {
    "gain": {
      "OG-1p": {
        "bootstrap_high": -0.0002471840359147742,
        "bootstrap_low": -0.023645489101585968,
        "margin": 0.005,
        "mean": -0.01111406105313412,
        "median": 0.0,
        "participants": 15,
        "pass": false,
        "positive_count": 1
      },
      "OG-2p": {
        "bootstrap_high": 0.06451572926841134,
        "bootstrap_low": 0.006207491646960583,
        "margin": 0.005,
        "mean": 0.03240515496209912,
        "median": 0.0,
        "participants": 15,
        "pass": false,
        "positive_count": 5
      },
      "detection_rate": 0.2,
      "false_alarm_rate": 0.07777777777777778,
      "p_raw": {
        "OG-1p": 0.0008,
        "OG-2p": 0.9814
      },
      "threshold": 1.836848839870554
    },
    "spatial": {
      "OG-1p": {
        "bootstrap_high": 0.0,
        "bootstrap_low": -0.015452550202543436,
        "margin": 0.005,
        "mean": -0.007066977543096681,
        "median": 0.0,
        "participants": 15,
        "pass": false,
        "positive_count": 0
      },
      "OG-2p": {
        "bootstrap_high": 0.04263025026432137,
        "bootstrap_low": 0.0016067634381922897,
        "margin": 0.005,
        "mean": 0.019288135159553764,
        "median": 0.0,
        "participants": 15,
        "pass": false,
        "positive_count": 4
      },
      "detection_rate": 0.2,
      "false_alarm_rate": 0.07777777777777778,
      "p_raw": {
        "OG-1p": 0.0,
        "OG-2p": 0.9276
      },
      "threshold": 1.3253798112348816
    }
  },
  "holm_spatial": {
    "alpha": 0.05,
    "p_adjusted": {
      "OG-1p": 0.0,
      "OG-2p": 0.9276
    },
    "p_raw": {
      "OG-1p": 0.0,
      "OG-2p": 0.9276
    }
  },
  "null_sizes": {
    "gain": 270,
    "spatial": 270
  },
  "ownership_closed": true,
  "prediction_check": {
    "gain_auc": 0.7166666666666667,
    "registered": "SPATIAL separates (AUC materially above S2); GAIN does not",
    "s2_auc": 0.7815624999999999,
    "spatial_auc": 0.7233333333333334
  },
  "preregistration": "reports/v44_preregistration.md (V44-S3 addendum)",
  "roc_auc": {
    "gain": 0.7166666666666667,
    "s2_mahalanobis": 0.7815624999999999,
    "spatial": 0.7233333333333334
  },
  "sealed_reads": 0,
  "stage": "V44_S3_drift_decomposed_ownership",
  "thresholds_R1": {
    "gain": 1.836848839870554,
    "spatial": 1.3253798112348816
  }
}
```
