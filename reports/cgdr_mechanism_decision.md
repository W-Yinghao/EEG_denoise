# CGDR repaired mechanism decision

Date: 2026-08-02 (Europe/Paris)

## Decision

The original historical classifier returned `A_limited`. The absolute
baseline audit supersedes that interpretation with `B_geometry_only`; the
original `result_summary.json` is retained unchanged as history.

> Query-derived oracle geometry is useful under hard-Q consistency, but the
> diffusion-generated component is dominated by deterministic oracle
> orthogonal subtraction. Diffusion-specific value is not supported.

The corrected classifier cannot return A merely because an M0--M4 sampler
beats same-sampler POP. It additionally requires non-inferiority to
deterministic `Qy`, superiority to a trained task-matched multichannel U-Net,
and all frozen preservation conditions. Those requirements are not met.

Formal G1 is **`NOT_RUN_BLOCKED`**. These 54 Klados items have no verified
participant map, so source records, not participants, are the statistical
units. Independence among the source records is not established. The intervals
below are descriptive paired source-record bootstrap intervals after forming
the posterior-mean waveform across algorithmic seeds; they are not population
confidence intervals, and seeds are not independent statistical units.

The single-source-record exploratory effect-direction check failed; formal G1
was not executed.

## Absolute-baseline interpretation audit

- Oracle-projector M2 versus same-sampler POP: median
  `Delta e_parallel = -0.004969` across 16 source records.
- Oracle-projector M2 versus deterministic oracle `Qy`: median
  `Delta e_parallel = +1.145`, median `Delta RRMSE = +0.700`, and median
  `Delta correlation = -0.178` (method minus `Qy`).
- Deterministic `Qy` was better on all registered metrics in 16/16 source
  records.

Thus oracle geometry is useful, but the diffusion-generated component has no
demonstrated incremental value. The trained deterministic multichannel U-Net
required for a diffusion-specific comparison still does not exist; formal G3
was not run.

## Frozen development choice

Development used sim31--sim36, sim44 and sim45. It selected `M2`
(`final_hard_Q_consistency`) with trust radius 0.1. The effective evaluation
configuration is the base
`results/cgdr/klados_v4_repaired_mechanism_audit/resolved_config.yaml` plus the
frozen override in
`results/cgdr/klados_v4_repaired_mechanism_audit/development/frozen_choice.json`;
the `trust_radius_frozen: null` value in the base file denotes a candidate
before development selection, not the value used by J5.

On development, oracle-projector M2 versus same-sampler POP had median
Delta-e_parallel = -0.002689 (95% source-record bootstrap CI
[-0.005887, -0.001388]) and median Delta-e_perp = -0.033847 across 8/8 paired
records, with no failed records.

## Evaluation evidence

The evaluation source records were sim37--sim43 and sim46--sim54.

- Query-derived oracle projector plus M2 versus same-sampler POP: 16/16 paired,
  zero failures; median Delta-e_parallel = -0.004969 (95% CI
  [-0.005291, -0.002745]) and median Delta-e_perp = -0.040991 (95% CI
  [-0.048310, -0.035819]). The query clean target is used only to construct
  this mechanism upper bound; this arm is not deployable and is not an exact
  oracle posterior.
- Oracle orthogonal subtraction `Qy` versus corrupted identity: 16/16 paired;
  median Delta-e_parallel = -1.190589 (95% CI [-1.840268, -0.613063]) and
  essentially zero median e_perp change. This establishes useful geometry, not
  a diffusion advantage.
- Matching P0 specificity was **not supported**. Matching was eligible for only
  11/16 records (five bootstrap-stability fallbacks). Versus population P0 its
  median Delta-e_parallel was -0.007990 with CI
  [-0.034722, 0.011696], while median Delta-e_perp worsened by +0.118297,
  exceeding the 0.05 safety margin; safety preservation was 2/11. Only 4/16
  records were paired against the wrong-source control and 0/16 against the
  shuffled control, so neither comparison can support specificity.

The Klados duration diagnostic used development source records only. Of four
records with enough non-overlapping support/query time, eligibility rose from
2/4 at 5 s to 3/4 at 10 s and 4/4 at both 20 s and 30 s. Nevertheless,
matching-minus-population median `e_parallel` remained worse at every duration
(+1.523, +0.012, +0.021, +0.018 respectively). This does not reduce the
failure to a 10-second information shortage.

The Klados B6 gamma curve is exploratory and not confirmation: all 54 source
records have already participated in diagnosis and have no verified
participant mapping. On the eight development source records it selected
gamma 0.25 for M2, with only a small median `e_parallel` change relative to
gamma 0 and increasing orthogonal-complement error. Deterministic hard `Qy`
and soft proximal restoration were substantially better than M2 throughout
the same curve. B1--B5 remain closed, and no Klados B6 result is gate evidence.

Eye-BCI was not submitted: the preregistered condition requiring matching P0
specificity was not met. In addition, the old pre-repair S01 exploratory run
used S02--S26 for training and S27--S31 for validation. A new fold-wise protocol
could still be internally held out, but S07--S31 cannot honestly be called
historically untouched. G3 and diffusion necessity also remain untested: M2
contains hard deterministic Q-consistency, M0 was not evaluated on the final
records, and the required trained multichannel deterministic U-Net does not
exist yet.

## SGEYESUB deterministic operator-specificity result

The participant-stem experiment was moved to SGEYESUB without diffusion.
Development used study01/study03 (15 stems); the support-only objective froze
one global B6 gamma at **0.0**. Its mean support score worsened monotonically
from 0.0131 at gamma 0 to 0.0997 at gamma 1, and no query EOG, artifact labels,
trial labels, or outcomes were used for fitting or gamma selection.

Evaluation covered all 44 study02/study04/study05 stems. Forty-three had a
compatible same-cell population operator; the study05/study05_p42 singleton
remained explicitly `blocked_no_population`, with no cross-layout pooling.
Every method retained all 44 stems in its denominator, and gamma was not
reselected.

The automatic decision is
`personalization_failed_population_deterministic`, reason
`development_selected_gamma_zero`. B6-Qy is exactly the population-Qy arm at
gamma 0. Matching P0 did not establish a safe personalized advantage: its
mean held-out EOG-prediction remaining ratio was 0.609 versus 0.634 for
population-Qy, while non-artifact preservation was only 0.272 versus 0.289 and
covariance distortion was 0.910 versus 0.891. These are natural-EEG proxy
metrics, not clean recovery or RRMSE.

The source-faithful native SGEYESUB Python port was the strongest registered
baseline on several proxies (held-out EOG remaining ratio 0.198,
non-artifact preservation 0.799, covariance distortion 0.075), but it has not
been numerically cross-validated against MATLAB and is not called an exact
official reproduction. This comparison does not rescue B6/P0 specificity.

## Execution and evidence paths

- Current repair commits: `c77b36a`, `85a47fb`, `f06928a`, `ef0a724`.
- Current semantic/real-record J0: `919681` (162 tests plus validator passed).
- Repaired 19-channel prior checkpoint (not tracked by Git):
  `/home/infres/yinwang/denoiseNet/results/cgdr/klados_v4_padding_repair_development/checkpoints/best.pt`.
- Prior resume command: `scripts/slurm/submit.sh gpu-any cgdr mechanism-audit configs/cgdr/mechanism_audit_klados_padding_repair_development.yaml train-prior`.
- Klados duration result:
  `results/cgdr/klados_v4_development_diagnostics/calibration_duration_summary.json`.
- Klados exploratory B6 result:
  `results/cgdr/klados_b6_gamma_development/result_summary.json`.
- SGEYESUB development freeze:
  `results/cgdr/sgeyesub_operator_specificity/development/frozen_gamma.json`.
- SGEYESUB evaluation result:
  `results/cgdr/sgeyesub_operator_specificity/evaluation/result_summary.json`.
- Current Slurm jobs: `reports/slurm/cgdr_geometry_sgeyesub_job_ids.txt`.
- Code commit used by the repaired mechanism implementation: `060141e`.
  The later `gpu-any` dispatch edits were uncommitted during J5 execution; see
  the final Git status for their eventual repository state.
- Source config: `configs/cgdr/mechanism_audit_klados.yaml`.
- Split: `datasets/splits/klados_v4_mechanism_source_split.csv` (30 training,
  8 development, 16 evaluation source records; 10 s support and 1 s guard).
- Prior checkpoint: `results/cgdr/klados_v4_repaired_mechanism_audit/checkpoints/best.pt`.
- Population state: `results/cgdr/klados_v4_repaired_mechanism_audit/population_state.json`.
- Training trace: `results/cgdr/klados_v4_repaired_mechanism_audit/training_history.csv`.
- Development metrics: `results/cgdr/klados_v4_repaired_mechanism_audit/development/metrics.csv`.
- Evaluation metrics: `results/cgdr/klados_v4_repaired_mechanism_audit/untouched/metrics.csv`.
- Machine-readable decision: `results/cgdr/klados_v4_repaired_mechanism_audit/result_summary.json`.
- Slurm ledger: `reports/slurm/cgdr_repair_job_ids.txt` (J0--J6).
- Decision report: `reports/cgdr_mechanism_decision.md`.
- Slurm stdout/stderr pattern: `slurm_logs/cgdr/dn-cgdr-<job>.out` and
  `slurm_logs/cgdr/dn-cgdr-<job>.err`.

J5 used the scheduler-selected `gpu-any` partition list. V100-32GB and A40
allocations were observed while the array ran, but per-task allocation was not
persisted and SlurmDBD accounting was unavailable. Its latency and peak-memory
fields are therefore descriptive only: they must not be pooled across records
or compared with the V100-only development run. Any performance comparison
must be rerun on one fixed GPU type.

Current route: `stop_personalization_use_population_deterministic`. Eye-BCI,
formal G1 and G3 were not submitted. Diffusion may only be reconsidered after
a deterministic operator route is established and must then beat both `Qy`
and a task-matched multichannel deterministic U-Net.
