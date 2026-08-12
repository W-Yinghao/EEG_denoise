# V28 final development diagnosis

```json
{
  "engineering": "valid",
  "clean_conditional_diffusion": "competitive",
  "support_mechanism": "paired_signal_weak",
  "natural_artifact": "promising",
  "natural_observation_retention": "concern",
  "task_valid_preservation": "unavailable",
  "next_route": "B. one small training refinement",
  "paired": {
    "support_vs_population": {
      "mean": -2.389306627137988e-05,
      "median": -0.00014763909566684053,
      "positive": 6,
      "participants": 15,
      "bootstrap_low": -0.001132494559982971,
      "bootstrap_high": 0.0010078679800447853
    },
    "match_vs_wrong": {
      "mean": 0.000596967529956464,
      "median": 0.0005821306347753574,
      "positive": 9,
      "participants": 15,
      "bootstrap_low": -4.084472472884033e-05,
      "bootstrap_high": 0.0012747057900757764
    },
    "cdm_vs_det": {
      "mean": -0.0019239284817363738,
      "median": -0.0010547648395843878,
      "positive": 5,
      "participants": 15,
      "bootstrap_low": -0.004507727685627251,
      "bootstrap_high": 0.0001374814661875575
    }
  },
  "natural": {
    "support_artifact": {
      "mean": 0.0009594061286489023,
      "median": 0.0013302582159870902,
      "positive": 10,
      "participants": 15,
      "bootstrap_low": -0.0021976808742911486,
      "bootstrap_high": 0.0038616978403488086
    },
    "support_observation_retention": {
      "mean": -0.0011669526460852768,
      "median": -0.0007932241644609261,
      "positive": 6,
      "participants": 15,
      "bootstrap_low": -0.0023000413141491165,
      "bootstrap_high": -9.792205747728221e-05
    },
    "support_psd": {
      "mean": -0.0005150357024653969,
      "median": -0.000989432402689741,
      "positive": 6,
      "participants": 15,
      "bootstrap_low": -0.0024300690253089125,
      "bootstrap_high": 0.001391805435723105
    }
  },
  "interpretation": "DET/CNN are competitive positioning controls, not diffusion survival gates; natural metrics are separated and no retention scalar is called physiological preservation.",
  "development_only": true,
  "query_EOG_inference_reads": 0,
  "query_operator_inference_reads": 0,
  "event_inference_reads": 0,
  "sealed_reads": 0
}
```
