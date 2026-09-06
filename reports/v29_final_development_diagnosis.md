# V29 final development diagnosis

```json
{
  "engineering": "valid",
  "absolute_denoising": "improved_over_v28",
  "support_mechanism": "clear_paired_signal",
  "capacity_control": "support_beats_pop_adapter",
  "diffusion_positioning": "competitive_with_det",
  "natural": "artifact_promising_retention_acceptable",
  "next_route": "A. freeze method and complete revision experiments",
  "development_only": true,
  "paired": {
    "cdm_match_pop_adapter": {
      "mean": 0.00018614593490614932,
      "median": 0.0001892729406015592,
      "positive": 15,
      "participants": 15,
      "bootstrap_low": 0.00016140274921902167,
      "bootstrap_high": 0.000208992024060453
    },
    "cdm_match_wrong": {
      "mean": 4.5516064675436375e-07,
      "median": 1.6706286298795447e-07,
      "positive": 11,
      "participants": 15,
      "bootstrap_low": 2.588705309873134e-08,
      "bootstrap_high": 9.814124330079212e-07
    },
    "det_match_pop_adapter": {
      "mean": 0.0002218754938109265,
      "median": 0.0002415456743144473,
      "positive": 15,
      "participants": 15,
      "bootstrap_low": 0.0001884388708179824,
      "bootstrap_high": 0.0002524387236723227
    },
    "cdm_det": {
      "mean": -0.0003409900920682607,
      "median": -0.001410303574230054,
      "positive": 4,
      "participants": 15,
      "bootstrap_low": -0.0015652653730362683,
      "bootstrap_high": 0.0013433844427723292
    }
  },
  "natural_effects": {
    "artifact": {
      "mean": 0.0005035998734063722,
      "median": 0.0005325091278349792,
      "positive": 14,
      "participants": 15,
      "bootstrap_low": 0.00039744612187578,
      "bootstrap_high": 0.0005946789156277039
    },
    "retention": {
      "mean": 8.127470132662425e-05,
      "median": 4.428345295748315e-05,
      "positive": 11,
      "participants": 15,
      "bootstrap_low": 3.517828976379108e-05,
      "bootstrap_high": 0.00013290323692257975
    },
    "psd": {
      "mean": -2.7501651446295088e-05,
      "median": -3.244998720472285e-05,
      "positive": 5,
      "participants": 15,
      "bootstrap_low": -9.304812423728476e-05,
      "bootstrap_high": 3.0911046304416575e-05
    }
  },
  "query_EOG_inference_reads": 0,
  "query_operator_inference_reads": 0,
  "event_inference_reads": 0,
  "sealed_reads": 0
}
```
