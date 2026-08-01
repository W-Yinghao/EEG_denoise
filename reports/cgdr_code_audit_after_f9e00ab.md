# CGDR paper-to-code audit after `f9e00ab`

Date: 2026-08-01 (Europe/Paris)

Scope: lightweight static review of the tracked CGDR implementation and the
extracted manuscript source after commit `f9e00ab`. No Python, data reader,
model, test, or experiment was run for this audit. Existing numerical files
were not reinterpreted as gate evidence.

## Bottom line

The repository now has a runnable research prototype for a single-channel
clean prior, explicit POP observation energy, P0 support calibration, guided
sampling, one Klados source-record comparison, and one Eye-BCI natural-record
comparison. It is not yet the method or confirmatory protocol described by the
paper. In particular, the existing result directories predate the repairs
listed below and are `exploratory_pre_repair_not_gate_evidence`. Formal G1 was
not run and remains blocked: **the single-source-record exploratory
effect-direction check failed; formal G1 was not executed.**

## Paper--implementation gaps

| Area | Paper contract | Implementation at `f9e00ab` | Consequence / required repair |
|---|---|---|---|
| Method name | `manuscript/main.tex` defines the method as `CSPD`. | Formal package and current protocol use `CGDR` and control `POP`. | Update the paper name and terminology before any result is inserted; do not maintain two scientific identities. |
| Clean prior architecture | A montage-compatible multichannel temporal prior, optionally conditioned on a registered acquisition key, trained with participant-grouped outer splits. | `CleanEEGDiffusionPrior` is trained on single-channel EEGdenoiseNet epochs and applies the same one-channel network independently to each EEG channel. It has no montage/acquisition condition and EEGdenoiseNet supplies no participant grouping. | This checkpoint is an engineering prior, not the paper's dataset-specific multichannel population prior. A participant-safe multichannel target/protocol is still needed. |
| Population observation model | Training contexts estimate a montage/reference-specific population state `Omega_0`, including a population projector or registered precision operator. | POP uses a fixed isotropic scalar precision times identity. No population projector or population observation model is fitted from outer-training contexts. | Population-only inference is observation-conditioned, but the claimed learned population observation state is absent. |
| P0 regressor restriction | A training-frozen retained regressor subspace `V_R` is selected before solving ridge. Unexcited directions are removed rather than manufactured by regularization. | P0 checks raw reference rank/condition but has no explicit retained-regressor-space construction. | Implement and freeze the retained reference rule on development participants before a confirmatory P0 run. |
| P0 ridge estimator | Solve `Y E^T (E E^T + lambda I)^{-1}` on the retained regressors. | `_fit_core` computes pseudoinverse OLS and then divides the complete transfer by `1 + lambda`. | The implemented estimator is not the displayed ridge solve. Repair and revalidate rank, conditioning, invariance, and all old P0 outputs. |
| Bootstrap stability | Median pairwise operator-norm distance between bootstrap projectors, plus effective block count. | Distances are normalized Frobenius distances from each bootstrap projector to the full-sample projector; the report uses median and 90th percentile. | Either implement the paper statistic or revise the paper and preregistered thresholds. Current eligibility values do not instantiate the manuscript definition. |
| Attenuation resolution | `a_tau` is sample- or frame-resolved and enters both registered population and context projector states. | External EOG is reduced to one scalar attenuation per 512-sample window. POP validates the source but remains isotropic and does not apply a population subspace attenuation. | The current approximation must be named explicitly; time-resolved attenuation and a defined population operator remain absent. |
| EEG-only attenuation | A frozen EEG-only detector `h_phi(y)` is evaluated as generalized Bayes and kept distinct from external-mask results. | Only external EOG/HEO-derived attenuation exists. | G4 EEG-only deployment evidence is not implemented or run. |
| Guidance transform | The paper specifies a timestep schedule `gamma_t`, residual-dimension and norm normalization, and trust-radius clipping `T_t`, with explicit ablations. | Sampling uses an exact autograd VJP of the plug-in energy with one fixed energy scale; there is no registered `gamma_t`, normalization, clipping, or related ablation. | The current sampler is a simpler guided process and cannot support the paper's stabilized-guidance claims. |
| Reliability gate | A training-only monotone calibrator `g_psi(z)` uses stability, conditioning, coverage, effective support, support--query shift, and attenuation quality. | Eligibility is a hard hand-written P0 accept/reject rule; accepted calibrations use a fixed `rho`. No learned reliability, benefit target, calibration curve, or risk--coverage rule exists. | Reliability and selective-deployment claims are not available. |
| POP short circuit | `rho=0` or rejection must reach population inference before context precision, residual, VJP, clipping, or context RNG consumption. | The lazy factory and shared-seed branch implement this property, and integration checks establish equality for the tested path. | This is an implemented engineering invariant, but it does not by itself validate drift detection or formal rollback performance. |
| Information-matched baseline | A trained one-pass amortized conditional estimator with matched data, inputs, conditioning, receptive field, and exposure. | `InformationMatchedOneStep` performs one prior denoising evaluation followed by a closed-form quadratic proximal update; it is not a trained amortized conditional estimator. | Rename it as a proximal one-step diagnostic or implement the paper's trained information-matched baseline before G3. |
| Strong task-matched baseline | A strong deterministic model trained with available paired corruption supervision and task/fidelity objectives. | Not implemented. | Diffusion necessity has not been tested. |
| Paired span error | Paper `e_parallel` divides by neural energy `||Pi X||`; overlap is `||Pi X||^2 / ||X||^2`. | `e_parallel` divides by oracle artifact energy `||Pi(Y-X)||`; `overlap_fraction` is projector-to-projector overlap, not clean neural overlap. | Existing G1-direction numbers answer a different question. Add separately named metrics and never map old columns into the paper table. |
| Frequency RRMSE | Relative error of Fourier magnitudes. | Implementation compares periodogram powers. | Rename the current metric or implement the manuscript amplitude definition. |
| Correlation aggregation | Channel correlations are Fisher transformed before context aggregation. | Raw channel correlations are averaged directly. | Repair before participant/context summaries. |
| Klados outer unit | Held-out participants with support/query blocks and enough independent units for paired inference. | The v4 release has no verified 54-record-to-27-participant map. The completed route holds out only source record `sim45`; one independent source cannot meet the gate minimum. | Retain it only as paired semi-simulation mechanism exploration. Formal G1 and participant-specific G2 are not run. |
| Population/wrong operator controls | Population operator from outer-training participants; wrong operator from a verified different participant/context. | Klados participant mapping is unresolved, so the population operator rolls back to POP and the wrong arm is only a different source record. | Klados G2 specificity is unavailable. Eye-BCI is the participant-safe route, but only one held-out participant has been evaluated. |
| Natural EEG protocol | Multiple held-out participants/sessions, cross-session and cross-paradigm tests, appropriate ERP/spectral/task preservation, fixed and retrained decoders. | Eye-BCI has one S01/Sess01/ME record, observation-relative preservation proxies, and no decoder or cross-session/paradigm evaluation. | Natural attenuation/preservation remains exploratory; G4/G5 are not run. |
| Classical/native baselines | Raw/standard preprocessing, EOG regression, ICA, ASR, Wiener, and native SGEYESUB with audited provenance. | The new CGDR fold lacks these registered baselines; native SGEYESUB is not integrated. | Required comparison coverage is incomplete. |
| Predictive uncertainty | Held-out coverage, width, proper interval score, risk--coverage, and explicit conditional semantics. | Multiple seeds are averaged, but no held-out predictive calibration or proper scoring analysis is implemented. | Samples cannot be described as calibrated posterior intervals. |
| Drift and rollback | Detect harmful calibration and real session/paradigm drift; report delay, false triggers, regret, fallback, and recovery. | Shared-seed POP equality exists, but no drift detector, recalibration controller, cross-session evaluation, or recovery experiment exists. | Formal G5 is not run. |
| Statistical gates | Participant-level paired effects, frozen practical margins/confidence/multiplicity, and all failures retained. | Gate thresholds remain `TBD-PREREG`; Klados has one source and Eye has one held-out participant. Historical summaries embedded a local direction check in the fold runner. | Formal G1--G5 remain NOT RUN/BLOCKED. The historical direction check is now isolated in the legacy exploratory audit runner instead of the training fold. |
| Reproducibility contract | Paper requires hashes for data, split, config, checkpoint, code, environment, and outputs before a value enters the manuscript. | The user explicitly selected `HARNESS_LEVEL=1`, which forbids those hashes for this private research iteration. | This is an intentional current workflow simplification, but paper tables cannot claim the manuscript's immutable provenance contract unless the paper is revised or final evidence is regenerated under it. |
| Deferred B1--B6 | Disabled until the required primary gates and a prospective single-family amendment. | Config stubs remain disabled and no backup comparison is present. | Aligned; keep disabled. |

## Evidence status after this audit

- `results/cgdr/klados_v4_source_fold_sim45/`: invalid inference semantics and
  exploratory pre-repair output only.
- `results/cgdr/klados_v4_source_fold_sim45_corrected/`: useful debugging
  output, but still exploratory pre-repair and single-source only.
- `results/cgdr/eye_bci_me_outer_fold_00/`: one-participant natural-record
  engineering result, exploratory pre-repair only.
- Formal G1: **NOT RUN / BLOCKED**. The single-source-record exploratory
  effect-direction check failed; formal G1 was not executed.
- Formal G2--G5: **NOT RUN / BLOCKED**.

The next valid use of old metrics is diagnosis through
`legacy_mechanism_audit.py` with
`legacy_mechanism_audit_klados_sim45.yaml`. That legacy stage is deliberately
analysis-only: it does not train a prior, fit an operator, sample a model, or
write a formal gate status. It is distinct from any new repaired mechanism
experiment dispatched through the general `mechanism-audit` CLI route.
