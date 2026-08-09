# BCI2b conditional clean-neural posterior diffusion V12

Development exploration; same-session and EOG-guided only.

Final decision: `CLEAN_POSTERIOR_HAS_NO_OUTER_TRAINING_HEADROOM`.

## Route

The zero-training score-component audit returned `CURRENT_ARTIFACT_RESIDUAL_SCORE_OBJECT_CLOSED`, so the artifact-residual score object was closed and V12 was authorized. The conditional clean posterior then failed its pre-heldout technical/outer-training gate; no participant 1–3 or 4–9 evaluator screen was run and no additional seeds were submitted.

## Technical evidence

t_start=0 exactly returned LINEAR: `True`. The sampler remained finite: `True`. Selected t_start: `50`. Fixed DET/DIFF RRMSE: `0.0695/0.2587`. Outer-training LINEAR/DIFF RRMSE: `0.3337/0.4998`.

This closes only the present observation-anchored clean-posterior implementation. It is not a family-wide diffusion or personalization conclusion.
