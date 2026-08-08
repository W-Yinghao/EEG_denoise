# BCI2b EOG residual diffusion V11.1

Development completion, same-session only. Participants 4–9 were unscored development holdouts, not sealed confirmation.

Decision: `SUBJECT_OPERATOR_SUPPORTED_BUT_DIFFUSION_INCREMENT_NOT_ESTABLISHED`. Global panel: K=8.

| method | RRMSE | paired spectral utility | MI-band distortion | preservation | covariance | EOG attenuation |
|---|---:|---:|---:|---:|---:|---:|
| RAW | 0.4885 | +0.0000 | 0.0000 | 1.0000 | 0.0000 | +0.0000 |
| LINEAR-POP | 0.1463 | +0.1560 | 0.0677 | 0.8054 | 0.0903 | +0.0892 |
| LINEAR-MATCH | 0.0876 | +0.1923 | 0.0672 | 0.8123 | 0.0909 | +0.0973 |
| LINEAR-WRONG | 0.1761 | +0.1652 | 0.0667 | 0.8075 | 0.0895 | +0.0900 |
| DET-POP | 0.1535 | +0.1450 | 0.0759 | 0.7884 | 0.1272 | +0.0985 |
| DET-MATCH | 0.1147 | +0.1713 | 0.0747 | 0.7961 | 0.1263 | +0.1041 |
| DET-WRONG | 0.1640 | +0.1522 | 0.0748 | 0.7877 | 0.1276 | +0.0985 |
| DIFF-POP | 0.1512 | +0.1427 | 0.0726 | 0.7904 | 0.1248 | +0.0992 |
| DIFF-MATCH | 0.1152 | +0.1663 | 0.0713 | 0.7980 | 0.1238 | +0.1047 |
| DIFF-WRONG | 0.1614 | +0.1501 | 0.0717 | 0.7897 | 0.1251 | +0.0996 |
| DIFF-TEMPORAL-SHUFFLED | 0.5797 | -0.0805 | 0.0416 | 0.8096 | 0.0673 | +0.0014 |

| effect | mean | median | positive |
|---|---:|---:|---:|
| U_D | -0.0005 | -0.0016 | 3/9 |
| U_L | -0.0276 | -0.0265 | 2/9 |
| U_P | +0.0360 | +0.0316 | 9/9 |
| U_S | +0.4646 | +0.4185 | 9/9 |
| U_W | +0.0462 | +0.0428 | 9/9 |

K=8 and K=32 are global panels; no participant-wise K selection occurred. LINEAR-MATCH remains a primary comparator. Cross-session participants 4–9 were not evaluated. Evidence is development, not confirmation.

K32 was evaluated only as the frozen secondary panel. It produced U_D mean/median `+0.0029/+0.0030` (7/9), but U_L remained `-0.0243/-0.0214` (2/9), and its DIFF-POP MI-kappa margin missed the frozen population safety requirement. It therefore did not replace K8 or establish diffusion increment.

Population denoising passed for K8 (DET-POP and DIFF-POP beat RAW for 9/9 participants). Subject operator utility also passed: U_P and U_W were positive for 9/9 participants with positive descriptive bootstrap intervals. Diffusion increment failed because K8 did not beat matched DET or LINEAR-MATCH; no extra seeds were submitted.

| panel | latency/window | posterior calls | peak GPU memory |
|---|---:|---:|---:|
| K8/DDIM25 | 0.0172 s | 200 | 153 MB |
| K32/DDIM25 | 0.0657 s | 800 | 153 MB |

## Slurm recovery

The first forensic attempts `930256` and `930275` exposed, respectively, NumPy's removed `trapz` alias and an incorrect assumption that V11 natural safety had a top-level CSV. `930278` completed all metrics but failed while serializing a NumPy boolean. The corrected forensic/contract chain `930282–930283` passed; failed attempts did not freeze a contract or open participants 4–9 outcomes. The attempted completed-job dependency submission before `930287` was rejected by Slurm; `930287–930288` were cancelled before aggregation, and the valid evaluator chain was `930289–930291`, followed by deterministic summary reruns `930351–930352` and `930365–930366`.
