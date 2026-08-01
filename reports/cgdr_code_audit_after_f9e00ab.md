# CGDR paper-to-code audit after `f9e00ab`

Date: 2026-08-01 (Europe/Paris)

Scope: lightweight static review of the tracked CGDR implementation and the
extracted manuscript source after commit `f9e00ab`. No Python, data reader,
model, test, or experiment was run for this audit. Existing numerical files
were not reinterpreted as gate evidence.

## Bottom line at the `f9e00ab` snapshot

At the reviewed `f9e00ab` snapshot, the repository had a runnable research prototype for a single-channel
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

## Repair validation update

Commit `70b4057` freezes the repaired source-record mechanism audit at 10 s of
real calibration plus a 1 s guard.  The earlier 30 s setting is N/A for this
partition: the native records are 27.005--42.005 s, so it cannot leave a
non-overlapping query for most development and untouched records.  Slurm J0
job `919567` passed all 61 scheduled semantic tests and the real-record
validation over all 54 loaded records with targeted checks on sim31, sim45,
sim37 and sim54.  It verified disjoint support/query construction, training-
only normalization, zero normalized padding, the direct FP64 ridge reference,
rank-2 projector symmetry/idempotence, and terminal
`alpha_bar=4.0358297653756876e-05`.  These are implementation checks, not gate
evidence.  Formal G1 remains NOT RUN/BLOCKED.

## Post-repair J1--J6 addendum (2026-08-02)

The table above is intentionally retained as the static state at `f9e00ab`.
The repaired mechanism implementation subsequently reached code commit
`060141e`; the scheduler-wide `gpu-any` profile was added afterward and was
uncommitted during J5 execution. Its eventual repository state is reported by
the final Git status rather than inferred from the run.

The following previously identified implementation gaps are now repaired for
the Klados source-record mechanism route:

- P0 uses the FP64 ridge solve `Y E^T (E E^T + lambda I)^-1` without an
  explicit inverse, and thin SVD is applied to the fitted transfer `C_hat`.
- POP has a rank-2 population projector fitted jointly from all sim01--sim30
  training source records. Context uses its own projector, with the population
  and context precisions interpolated explicitly.
- External attenuation is frame-resolved; padding, guards and missing samples
  have zero valid-time weight before normalization and throughout the
  observation path.
- The Klados clean prior is a 19-channel model trained only on sim01--sim30,
  with a 1000-step linear schedule and terminal alpha-bar below `1e-4`.
- Guidance has a finite-difference-checked full VJP, raw trace fields,
  residual-dimension normalization and configurable prior-relative trust
  clipping. `energy_scale` no longer masquerades as a timestep schedule.
- POP and rho=0 share attenuation, initial state and random stream, while the
  rho=0 branch does not construct context-specific precision, residual or VJP.
- The sampler candidates are M0--M5 with M5 labeled
  `single_prior_eval_proximal`; it is not represented as a trained matched
  baseline.

Slurm J1 `919583` then passed real-record sampler integration on two independent
source records, including forward/loss/backward/optimizer update, checkpoint
save/reload/resume, POP/P0 branching, M0--M5, and a full-VJP finite-difference
relative error of `4.94e-6`. J2 `919584` trained the 19-channel prior for 200
epochs/3000 updates (470 training and 135 validation windows), selecting
validation loss `0.0657713`; the cross-channel dependency check passed. The
checkpoint is
`results/cgdr/klados_v4_repaired_mechanism_audit/checkpoints/best.pt`, and the
training trace is
`results/cgdr/klados_v4_repaired_mechanism_audit/training_history.csv`.

J3 evaluated all eight development records. J4 `919591` froze M2 final hard
Q-consistency with trust radius 0.1 in
`results/cgdr/klados_v4_repaired_mechanism_audit/development/frozen_choice.json`.
Consequently, the effective J5 configuration is the checked-in base
`resolved_config.yaml` plus that frozen choice; the null trust radius in the
base config is not the J5 runtime value. J5 completed all 16 evaluation source
records, and J6 `919608` produced the decision in
`results/cgdr/klados_v4_repaired_mechanism_audit/result_summary.json`. Full
task-to-job mapping is in `reports/slurm/cgdr_repair_job_ids.txt`.

The repaired result supports only a source-record mechanism statement under
M2: query-derived oracle context improved e_parallel and e_perp relative to
same-sampler POP in 16/16 records. It does not close the paper's formal gates.
Matching P0 was eligible on only 11/16 records and harmed the orthogonal
complement relative to population P0 (median Delta-e_perp `+0.1183`), so
operator specificity is not supported. Formal G1 remains
`NOT_RUN_BLOCKED`; G2--G5 remain unrun. The trained deterministic U-Net required
for diffusion-necessity claims is still absent, and M2 includes a deterministic
hard-consistency operation. Eye-BCI was therefore not submitted under the
frozen specificity rule.

The unique operator diagnosis is short-support instability in the presence of
a compatible population library. B6 `POP-SHRINK` is the only selected
diagnostic backup; B1--B5 remain closed. It cannot yet supply confirmatory
evidence because all current Klados development/evaluation records informed
selection or diagnosis. Details and exact source-record intervals are in
`reports/cgdr_mechanism_decision.md`.

The minimum disabled B6 interface is implemented in
`src/eeg_cgdr/operators/pop_shrink.py`. It performs a fixed-rank spectral
projection of the population/context convex combination, requires an exact
dataset/montage/reference/preprocessing/channel-order key and explicit
outer-training/support-only fit scopes, and reaches POP before reading context
at gamma=0 or after an ineligible calibration. The final Slurm validation J0c
`919634` passed 80 tests plus the existing real-record CGDR validator. This is B6 algebra and
leakage-contract validation, not a real-projector B6 experiment; the backup
config remains `enabled: false`.

J5 used a scheduler-selected heterogeneous GPU list. V100-32GB and A40 were
observed, but per-task allocation was not persisted and SlurmDBD was
unavailable. Runtime and memory values from that array are descriptive only
and cannot be pooled across records or compared against the V100-only J3 run.
