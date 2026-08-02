# `e_parallel` denominator audit

Date: 2026-08-02

This is a read-only semantic audit. It does not rename, rewrite, or reaggregate
historical metrics, checkpoints, frozen choices, or result summaries.

## Two existing meanings

| Location | Existing field | Formula | Meaning |
|---|---|---|---|
| `src/eeg_cgdr/evaluation/metrics.py` | legacy `e_parallel` | `||P(x_hat-x)||_F / ||P(y-x)||_F` | error relative to paired artifact energy |
| `src/eeg_cgdr/experiments/mechanism_runner.py` | repaired `e_parallel` | `||P(x_hat-x)||_F / ||Px||_F` | error relative to clean neural energy in the registered span |
| `src/eeg_cgdr/experiments/mechanism_runner.py` | `artifact_normalized_parallel_error` | `||P(x_hat-x)||_F / ||P(y-x)||_F` | explicitly named artifact-relative diagnostic |

The repaired mechanism runner already uses the paper-facing neural-energy
denominator. The conflict is the legacy generic metrics module retaining the
same bare name for the artifact-normalized quantity. Results from these two
pipelines must not be concatenated under one unqualified `e_parallel` column.

## Historical scope

- Pre-repair/full-fold outputs produced through `evaluation.metrics` retain the
  artifact-normalized meaning. They cannot be relabelled as neural-normalized.
- Repaired Klados mechanism-audit, duration-diagnostic, and B6 diagnostic
  outputs produced through `mechanism_runner` retain the neural-normalized
  primary meaning; their separate artifact-normalized column remains a
  sensitivity diagnostic.
- Existing frozen choices and reports remain unchanged. Cross-record medians,
  bootstrap intervals, and candidate rankings can differ between denominators
  even when a within-record comparison has the same direction.

## Additive side-by-side schema

Future diagnostic output may use
`subspace_parallel_error_side_by_side(...)` and the repaired runner aliases:

- `e_parallel_neural_normalized` = `||P(x_hat-x)||_F / ||Px||_F`;
- `e_parallel_artifact_normalized` =
  `||P(x_hat-x)||_F / ||P(y-x)||_F`;
- `parallel_error_norm`, `parallel_clean_norm`, and
  `parallel_artifact_norm` expose the three norms;
- denominator-valid flags distinguish an undefined ratio from a failed method;
- `parallel_error_denominators_v2_side_by_side` identifies the schema.

The generic legacy `subspace_error_metrics(...)` API is intentionally left
unchanged for reproducibility. The new helper is additive and does not feed an
old gate or overwrite an old result.

## Remaining limitation

The repaired mechanism runner currently treats a near-zero denominator as a
metric failure through its strict relative-norm helper, whereas the generic
metrics module returns `None`. The side-by-side norms make that distinction
auditable, but changing failure handling would alter the registered mechanism
pipeline and therefore requires a prospective protocol revision.
