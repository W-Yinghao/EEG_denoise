# SGE-DYNTRANS-DIFF-v6

## Scope

Development exploration on 58 compatible SGEYESUB participant-stems (59 availability denominator; `study05/study05_p42` retained as blocked singleton). Early support is the first 30 seconds of block 1; query begins at block 2. A metadata-only audit over all 59 records found that the unused block-1 remainder provides an implicit guard of at least 106 seconds, exceeding the required 5 seconds. Grouped folds hold the target and its same-cell WRONG donor out of model fitting, normalization, and population transfer. The paired benchmark uses mutually disjoint generator, class-6 target, and class-1–5 EOG-source trials. Class 6 is a *low-artifact observed EEG target*, not physiological clean truth.

## Absolute results

| Method | RRMSE | correlation | delta-SNR | natural EOG reduction | class-6 preservation | PSD distortion | covariance distortion |
|---|---:|---:|---:|---:|---:|---:|---:|
| RAW | 0.8427 | 0.7808 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| DET-MATCH | 0.4655 | 0.8949 | 4.4516 | 0.0423 | 0.5854 | 0.2131 | 0.2479 |
| DET-POP | 0.4441 | 0.9033 | 4.9640 | 0.0413 | 0.6165 | 0.1931 | 0.1913 |
| DIFF-MATCH | 0.7657 | 0.7856 | -0.0282 | 0.0499 | 0.4031 | 1.2017 | 0.1876 |
| DIFF-POP | 0.7639 | 0.7873 | 0.0086 | 0.0503 | 0.3994 | 1.1937 | 0.1576 |
| DIFF-WRONG | 0.7611 | 0.7871 | 0.0255 | 0.0500 | 0.4069 | 1.1978 | 0.1796 |

## Primary effects

- U_D = -0.300187, 95% descriptive CI [-0.341968, -0.258371], positive 1/58.
- U_P = -0.001812, 95% descriptive CI [-0.007374, +0.003269], positive 26/58.
- U_W = -0.004595, 95% descriptive CI [-0.011486, +0.002220], positive 23/58.

Natural margins relative to DIFF-POP were preservation +0.003756, PSD -0.007965, and covariance -0.030017; covariance failed the frozen -0.02 margin.

## Decision

`current_transfer_conditioned_instance_no_go`. The one-seed gate failed, so seeds 20260807 and 20260808 were not submitted. This is a valid negative result for this dynamic-transfer-conditioned artifact-residual diffusion instance only. It is not a family-wide claim about diffusion or personalization.
