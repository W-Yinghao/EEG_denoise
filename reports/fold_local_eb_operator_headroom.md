# Fold-local empirical-Bayes operator headroom

Every support, population, wrong, and evaluator-only generator transfer was re-fitted inside the same outer fold with identical EEG/EOG normalization and compatibility cell. Query outcomes were not used by the deployable lambda predictor.

| budget | effect | mean | 95% cluster CI | positive |
|---:|---|---:|---:|---:|
| 30s | oracle_relative_improvement | +0.1759 | [+0.1332, +0.2203] | 57/58 |
| 30s | deployable_relative_improvement | +0.0602 | [-0.0170, +0.1303] | 39/58 |
| 30s | match_relative_improvement | -0.0374 | [-0.1484, +0.0646] | 28/58 |
| 30s | match_vs_wrong_relative_improvement | +0.2161 | [+0.1014, +0.3064] | 52/58 |
| 60s | oracle_relative_improvement | +0.2200 | [+0.1796, +0.2644] | 57/58 |
| 60s | deployable_relative_improvement | +0.1167 | [+0.0702, +0.1600] | 47/58 |
| 60s | match_relative_improvement | +0.0826 | [+0.0005, +0.1630] | 39/58 |
| 60s | match_vs_wrong_relative_improvement | +0.3021 | [+0.2205, +0.3697] | 55/58 |
| 120s | oracle_relative_improvement | +0.2674 | [+0.2274, +0.3086] | 57/58 |
| 120s | deployable_relative_improvement | +0.1902 | [+0.1212, +0.2449] | 52/58 |
| 120s | match_relative_improvement | +0.1647 | [+0.0797, +0.2375] | 48/58 |
| 120s | match_vs_wrong_relative_improvement | +0.3606 | [+0.2950, +0.4152] | 57/58 |

This audit determines only whether the H/shrinkage branch has development headroom. It is not a hard gate for score-space LoRA.
