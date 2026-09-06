# V44 Stage 1 — EOG-guided diffusion

Query EOG is a declared runtime input in this deployment class. Any gain enters through the operator anchor; the claim is subject-aware SYSTEM gain, never score-network personalization and never diffusion superiority (C05). DET/LINEAR rows are reported wherever the diffusion arms are.

Decision: G1 **True**, G2-wrong-gated **False**, G2-shuffled **True**, G4 utilities adjudicable **True**.

## Paired arm means (temporal RRMSE, participant-first n=15)

| condition   |   participant_mean_rrmse_temporal |
|:------------|----------------------------------:|
| RAW         |                          0.607642 |
| POP         |                          0.651142 |
| MATCH_gated |                          0.430971 |
| WRONG       |                          1.08069  |
| WRONG_gated |                          0.689553 |
| SHUFFLED    |                          0.751127 |
| NO_A0       |                          0.573779 |
| ORACLE      |                          0.186842 |

## G1 per-participant forest data

| participant   |   pop_minus_match_gated |
|:--------------|------------------------:|
| sub-02        |               -0.000552 |
| sub-03        |                0.103469 |
| sub-05        |                0.096808 |
| sub-06        |                0.134012 |
| sub-07        |                0.05101  |
| sub-09        |                1.30983  |
| sub-11        |                0.059858 |
| sub-12        |                0.294555 |
| sub-14        |                0.204413 |
| sub-15        |                0.275358 |
| sub-17        |                0.497192 |
| sub-18        |                0.176994 |
| sub-19        |                0.012011 |
| sub-21        |                0.019412 |
| sub-23        |                0.068202 |

## Endpoints and controls

```json
{
  "G1": {
    "bootstrap_high": 0.40244942594839805,
    "bootstrap_low": 0.09292537851770855,
    "contrast": "RRMSE(POP) - RRMSE(MATCH_gated)",
    "mean": 0.22017149001064049,
    "median": 0.10346880583286595,
    "participants": 15,
    "pass": true,
    "positive_count": 14
  },
  "G2": {
    "no_a0_bridge_row_descriptive": {
      "bootstrap_high": 0.05117231841555418,
      "bootstrap_low": -0.2720311189518603,
      "mean": -0.0773637761910019,
      "median": 0.01597895619488554,
      "participants": 15,
      "positive_count": 10
    },
    "oracle_residual_gap": {
      "bootstrap_high": 0.5429381916664936,
      "bootstrap_low": 0.0635785231728692,
      "mean": 0.24412889857388412,
      "median": 0.05548123190237674,
      "participants": 15,
      "positive_count": 15
    },
    "shuffled": {
      "bootstrap_high": 0.3964593962172613,
      "bootstrap_low": 0.24924382960888275,
      "contrast": "SHUFFLED - MATCH_gated",
      "mean": 0.3201559226318851,
      "median": 0.26383137522740674,
      "participants": 15,
      "pass": true,
      "positive_count": 15
    },
    "ungated_wrong_harm_descriptive": {
      "bootstrap_high": 0.6124809817802149,
      "bootstrap_low": 0.28113890012267806,
      "mean": 0.4295530653633856,
      "median": 0.3789334527876538,
      "participants": 15,
      "positive_count": 14
    },
    "wrong_gated": {
      "bootstrap_high": 0.0697484814133855,
      "bootstrap_low": 0.008870303008558688,
      "contrast": "WRONG_gated - POP",
      "margin": 0.005,
      "mean": 0.03841100388114381,
      "median": 0.01636684765495977,
      "participants": 15,
      "pass": false,
      "positive_count": 9
    }
  },
  "holm": {
    "alpha": 0.05,
    "p_adjusted": {
      "G1": 0.0,
      "G2-shuffled": 0.0,
      "G2-wrong-gated": 0.9874
    },
    "p_raw": {
      "G1": 0.0,
      "G2-shuffled": 0.0,
      "G2-wrong-gated": 0.9874
    }
  }
}
```

The NO_A0 bridge row (a0 = 0) is the conditioning-class behavior inside the V44 system; its near-zero utility vs POP connects to the V43 null.

## G3 positioning (descriptive; competitive, no superiority claim)

```json
{
  "DET_MATCH_gated_mean": 0.4968236584651701,
  "DET_MATCH_gated_minus_DET_POP": {
    "bootstrap_high": -0.06030007319552169,
    "bootstrap_low": -0.38037680280630104,
    "mean": -0.1808152765314541,
    "median": -0.07901560780010186,
    "participants": 15,
    "positive_count": 1
  },
  "DET_POP_mean": 0.6776389349966242,
  "LINEAR_rows_from_S0": {
    "C0": 0.7475661814104312,
    "C_gated": 0.4352657458461749,
    "C_query": 4.218923155353986e-08,
    "C_wrong": 1.104137596915728,
    "C_wrong_g": 0.7523357563989161,
    "RAW": 0.6076416496460699
  },
  "wording": "competitive; no superiority claim in either direction (C05)"
}
```

## G4 natural windows

| condition   |   attenuation_db_mean |   retention_mean | meets_validity_bar   |
|:------------|----------------------:|-----------------:|:---------------------|
| POP         |              0.90909  |         0.772719 | True                 |
| MATCH_gated |              2.46324  |         0.843423 | True                 |
| WRONG       |             -1.52335  |         0.334269 | False                |
| WRONG_gated |              0.569913 |         0.78032  | True                 |
| SHUFFLED    |             -0.384088 |         0.365052 | False                |
| NO_A0       |              0.276673 |         0.921006 | True                 |
| ORACLE      |              2.23707  |         0.831969 | True                 |

```json
{
  "flags": {
    "MATCH_gated": {
      "attenuation_db_mean": 2.463239898163249,
      "meets_validity_bar": true,
      "retention_mean": 0.8434232344269811
    },
    "NO_A0": {
      "attenuation_db_mean": 0.2766726779151144,
      "meets_validity_bar": true,
      "retention_mean": 0.9210058125484095
    },
    "ORACLE": {
      "attenuation_db_mean": 2.237070937412354,
      "meets_validity_bar": true,
      "retention_mean": 0.8319687296301305
    },
    "POP": {
      "attenuation_db_mean": 0.9090899868623532,
      "meets_validity_bar": true,
      "retention_mean": 0.7727192170973683
    },
    "SHUFFLED": {
      "attenuation_db_mean": -0.3840880994861074,
      "meets_validity_bar": false,
      "retention_mean": 0.3650520448258559
    },
    "WRONG": {
      "attenuation_db_mean": -1.5233495821829603,
      "meets_validity_bar": false,
      "retention_mean": 0.3342692782313928
    },
    "WRONG_gated": {
      "attenuation_db_mean": 0.5699133096072282,
      "meets_validity_bar": true,
      "retention_mean": 0.780319581150184
    }
  },
  "natural_utilities_MATCH_gated_minus_POP": {
    "attenuation_db": {
      "bootstrap_high": 2.211197455462703,
      "bootstrap_low": 0.969346528461753,
      "mean": 1.5541499113008947,
      "median": 1.383979301410165,
      "participants": 15,
      "positive_count": 14
    },
    "coherence_reduction": {
      "bootstrap_high": 0.121120322161222,
      "bootstrap_low": 0.053502375860685625,
      "mean": 0.08492675326737022,
      "median": 0.059822000705536824,
      "participants": 15,
      "positive_count": 14
    },
    "low_eog_observation_retention": {
      "bootstrap_high": 0.10672279607812571,
      "bootstrap_low": 0.03821456758700018,
      "mean": 0.07070401732961266,
      "median": 0.05338824625597305,
      "participants": 15,
      "positive_count": 13
    }
  },
  "utilities_adjudicable": true,
  "validity_bar": {
    "attenuation_db": "> 0",
    "retention": ">= 0.75"
  }
}
```
