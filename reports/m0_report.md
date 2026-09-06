# FLAGSHIP-M0 — channel-ceiling matrix

K1 fired: **False**. U0-a geometry passed: **False**; U0-b coverage passed: **True**.

GO rule per cell: mean >= +0.020 and bootstrap CI-low > +0.005.

| channel      | panel     | stratum       |      mean |    ci_low |   ci_high | go    | note                                                               |
|:-------------|:----------|:--------------|----------:|----------:|----------:|:------|:-------------------------------------------------------------------|
| conditioning | mobilebci | all           |  0.006267 | -0.001607 |  0.01968  | False | V42R ORACLE-MATCH (banked, not re-measured)                        |
| transport    | mobilebci | all           | -0.1067   | -0.261644 |  0.082033 | False |                                                                    |
| likelihood   | mobilebci | all           |  0.763301 |  0.321095 |  1.44217  | True  | oracle operator on generated/near-exact pairs; see degeneracy note |
| transport    | mobilebci | high_severity |  0.303636 | -0.184432 |  0.856486 | False |                                                                    |
| likelihood   | mobilebci | high_severity |  1.76972  |  0.938213 |  2.76983  | True  | oracle operator on generated/near-exact pairs; see degeneracy note |
| transport    | mobilebci | high_eog      |  0.48397  | -0.174491 |  1.37244  | False |                                                                    |
| likelihood   | mobilebci | high_eog      |  1.67556  |  0.72697  |  2.90372  | True  | oracle operator on generated/near-exact pairs; see degeneracy note |
| transport    | klados    | all           |  0.039592 |  0.010632 |  0.065912 | True  |                                                                    |
| likelihood   | klados    | all           |  0.259916 |  0.192918 |  0.341251 | True  | oracle operator on generated/near-exact pairs; see degeneracy note |
| transport    | klados    | high_severity |  0.025633 | -0.005769 |  0.055238 | False |                                                                    |
| likelihood   | klados    | high_severity |  0.4586   |  0.315439 |  0.639368 | True  | oracle operator on generated/near-exact pairs; see degeneracy note |
| transport    | klados    | high_eog      |  0.031328 | -0.006224 |  0.066853 | False |                                                                    |
| likelihood   | klados    | high_eog      |  0.449468 |  0.331538 |  0.586306 | True  | oracle operator on generated/near-exact pairs; see degeneracy note |
| transport    | bci2b     | all           |  0.043394 |  0.016653 |  0.075024 | True  |                                                                    |
| likelihood   | bci2b     | all           |  0.595601 |  0.321111 |  0.940329 | True  | oracle operator on generated/near-exact pairs; see degeneracy note |
| transport    | bci2b     | high_severity |  0.109474 |  0.03629  |  0.197253 | True  |                                                                    |
| likelihood   | bci2b     | high_severity |  1.4065   |  0.976112 |  1.84496  | True  | oracle operator on generated/near-exact pairs; see degeneracy note |
| transport    | bci2b     | high_eog      |  0.079673 |  0.029661 |  0.141609 | True  |                                                                    |
| likelihood   | bci2b     | high_eog      |  1.31924  |  0.891532 |  1.78323  | True  | oracle operator on generated/near-exact pairs; see degeneracy note |
| weight_space | mobilebci | all           | -0.020444 | -0.036353 | -0.005621 | False | ORACLE-LoRA, generative-truth supervision, non-deployable          |

## GAUGE-NULL rows (must not GO)

```json
{
  "bci2b": {
    "bootstrap_high": 0.024117465655308368,
    "bootstrap_low": -0.034905829742845035,
    "mean": -0.00448264775343532,
    "median": 0.0,
    "n": 9,
    "positive_count": 2
  },
  "klados": {
    "bootstrap_high": 0.09145664887620816,
    "bootstrap_low": 0.03633350589388229,
    "mean": 0.06289013035021547,
    "median": 0.039559857768214135,
    "n": 54,
    "positive_count": 35
  },
  "mobilebci": {
    "bootstrap_high": 3.724554793676608,
    "bootstrap_low": 0.6077170236718256,
    "mean": 1.85339765211098,
    "median": 0.6955755247293454,
    "n": 15,
    "positive_count": 15
  }
}
```

## Transport deployable effect (T-MATCH - T-POP, descriptive)

```json
{
  "bci2b": {
    "bootstrap_high": 0.03757395082325267,
    "bootstrap_low": 0.0012831523411804824,
    "mean": 0.017587524365830293,
    "median": 0.0,
    "n": 9,
    "positive_count": 4
  },
  "klados": {
    "bootstrap_high": 0.05728228683983855,
    "bootstrap_low": 0.017234121306771617,
    "mean": 0.036923133137625165,
    "median": 0.009436537680094392,
    "n": 54,
    "positive_count": 29
  },
  "mobilebci": {
    "bootstrap_high": -0.19544841155105494,
    "bootstrap_low": -0.8012311603475123,
    "mean": -0.4383277127422912,
    "median": -0.24202446190332294,
    "n": 15,
    "positive_count": 1
  }
}
```

## Weight-space per-subject ceilings (non-deployable supervision)

```json
{
  "sub-02": -0.07153152045793831,
  "sub-03": -0.016421824577264488,
  "sub-05": -0.006662439904175699,
  "sub-06": -0.02351000625640154,
  "sub-07": -0.015127048798603937,
  "sub-09": -0.06967272015754133,
  "sub-11": -0.026969847036525607,
  "sub-12": 0.015261298802215606,
  "sub-14": -0.07120153371215565,
  "sub-15": 0.0016322021838277578,
  "sub-17": -0.04662768787238747,
  "sub-18": -0.009270469250623137,
  "sub-19": 0.003936330904252827,
  "sub-21": 0.015539004001766443,
  "sub-23": 0.013962638331577182
}
```
