# BCI2b EOG-guided residual diffusion V11

Development exploration; not confirmation. Both early support and later query use EEG+EOG, so the method is **EOG-guided**, not EEG-only.

## Final route

Decision: `CURRENT_EOG_ANCHORED_RESIDUAL_DIFFUSION_INSTANCE_NO_GO`.

## Evidence layers

- Operator/simulator bridge: `BASE_BRIDGE_VALID_J1_AUTHORIZED`.
- Technical estimator validity: `technical_validity_passed`.
- Three-participant population rescue: `CURRENT_EOG_ANCHORED_RESIDUAL_DIFFUSION_INSTANCE_NO_GO`.
- Full 9-participant subject and diffusion effects: `not_run`.

The three-participant gate is a technical route gate, not a final scientific estimate.

RAW RRMSE 0.4030; DET-POP 0.1556; DIFF-POP 0.1532. Preservation 0.8030, PSD distortion 0.3168, covariance distortion 0.1196, natural EOG attenuation +0.0956.

| protocol | U_D | U_P | U_W | U_S |
|---|---:|---:|---:|---:|
| same_session | -0.0007 | +0.0445 | +0.0560 | +0.4197 |
| cross_session | +0.0058 | -0.0322 | +0.0233 | +0.3448 |

Operator identity specificity, deployable MATCH-vs-POP subject utility, and DIFF-vs-DET incremental utility are not interchangeable and are reported separately.

V10 historical arrays were generated before inference, but its inference/model code never read evaluator fields. V11 physically separates inference and evaluator NPZ files. Paired results are real EEG/EOG-backed semi-simulation, not natural-clean ground truth.

The 3-person gate failure prevented the 9-person scientific experiment, so its diagnostic subject contrasts cannot establish or refute subject utility. Any negative result constrains only this EOG-anchored residual-diffusion instance; diffusion and personalization families remain untested.

Implementation provenance: the conditional 1-D diffusion is an adapted project implementation informed by common EEG diffusion recipes; it is not an exact official EEGDfus reproduction.

## Execution and recovery

- `930105` failed before computation because the new worktree was absent from `PYTHONPATH`; `930106` was cancelled, the launcher was corrected, and `930115–930117` completed J0.
- `930140` failed before its first training batch because the transfer helper lacked a batched `N×C×E` route. A batched scalar-equivalence test was added; repaired technical jobs `930146–930147` passed.
- `930143` exposed an over-specific source-inspection assertion; dependent `930144` became `DependencyNeverSatisfied`, was cancelled, and the corrected seven-test suite passed in `930146` and finally `930182`.
- `930152–930156` completed checkpoint replay, three-fold training/inference/evaluation, and the frozen gate. `930180–930182` regenerated final summaries and passed all targeted tests. The originally queued pre-commit clean-import job `930183` was intentionally cancelled and replaced after the scoped commit.

No scientific output from a failed pre-repair job entered the reported metrics.
