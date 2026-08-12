# V27 Round A

Round A used only validation participants from folds 0 and 2 with seed
`20260828`.  The search was sequential: first `lambda_y`, then `lambda_a`,
then final-only versus stepwise.  Test participants were not used.

The frozen Round-B configuration is:

```text
lambda_y = 8
lambda_a = 1
mode = final-only
temporal confidence = training-fold q50/q90 with 100 ms smoothing
```

For CalibEnergySDEdit, the mean validation operating points were:

| Stage | lambda_a | lambda_y | mode | paired clean RRMSE | natural remaining ratio | natural preservation |
|---|---:|---:|---|---:|---:|---:|
| A | 1 | 0 | final-only | 1.501090 | 0.830581 | 0.844835 |
| A | 1 | 0.5 | final-only | 1.495471 | 0.839117 | 0.873749 |
| A | 1 | 2 | final-only | 1.495416 | 0.865772 | 0.917485 |
| A | 1 | 8 | final-only | 1.506649 | 0.915954 | 0.962242 |
| B | 0 | 8 | final-only | 1.515069 | 0.946259 | 0.975256 |
| B | 1 | 8 | final-only | 1.506649 | 0.915954 | 0.962242 |
| B | 4 | 8 | final-only | 1.497019 | 0.874676 | 0.934684 |
| C | 1 | 8 | final-only | 1.506649 | 0.915954 | 0.962242 |
| C | 1 | 8 | stepwise | 1.509757 | 0.933339 | 0.966003 |
| D | 1 | 8 | spatial-only | 1.509539 | 0.848391 | 0.884686 |

The selection prioritizes natural artifact--preservation validity.  It does
not require diffusion to beat the matched one-step model.  Final-only was
selected because stepwise produced only a small preservation increase while
worsening paired fidelity and natural artifact attenuation.  Therefore the
registered condition for optional energy-aware fine-tuning was not met.
