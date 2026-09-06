# BCI2a hierarchical Score-LoRA diffusion V10

Development exploration only; no confirmation or sealed outcomes were opened.

## Routing outcome

BCI2a did not meet the frozen subject-identifiability gate (`BCI2A_IDENTIFIABILITY_NOT_DETECTED`), so the hierarchical diffusion factorial was not run on BCI2a. The pre-specified BCI2b audit did meet identifiability and entered a three-subject real-data technical/efficacy ladder.

After correcting the physical-unit mismatch and retraining from regenerated microvolt-scale arrays, the BCI2b ladder still failed absolute validity: RAW RRMSE 0.4030, DET-MATCH 0.5543, DIFF-POP 0.6242, DIFF-MATCH 0.6233; preservation 0.4101, PSD distortion 0.1501, covariance distortion 0.5344.

The diagnostic contrasts below are reported for transparency but are not promoted to scientific effects because the population estimator failed absolute validity:

| protocol | U_D | U_P | U_W | U_S |
|---|---:|---:|---:|---:|
| same_session | -0.0684 | +0.0008 | +0.0001 | -0.0027 |
| cross_session | -0.0698 | +0.0009 | -0.0000 | -0.0026 |

Same-session and cross-session both show a negative diffusion-vs-deterministic contrast and negative temporal-shuffle specificity; the tiny MATCH−POP differences cannot establish subject utility under the failed absolute estimator.

The full 9-fold factorial and extra seeds were not authorized. This localizes the stopped route to the current population estimator/absolute-safety implementation before a fair adapter-success adjudication; it is not a hierarchical-adapter, diffusion, personalization, BCI2a, or BCI2b family-wide negative.

## Evidence boundaries

- BCI2a: complete 9/9 participant identifiability audit; no GPU denoising result.
- BCI2b: complete 9/9 participant identifiability audit; real-data diagnostic GPU ladder on 3/9 participants only.
- Paired targets are EOG-backed semi-simulation, not natural-clean ground truth.
- Natural evaluator fields were opened only after cleaner outputs were frozen.
