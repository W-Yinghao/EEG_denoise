# CGDR repaired mechanism decision

Date: 2026-08-02 (Europe/Paris)

## Decision

Conclusion **A, limited to the source-record mechanism audit**: under the
development-selected `M2` final hard-Q-consistency sampler, query-derived
oracle context geometry improved the paired semi-simulation metrics relative
to same-sampler POP in every one of the 16 evaluation source records. This is
evidence that corrected geometry can be useful in this enforced sampler. It is
not evidence that diffusion has unique value over a deterministic method.

Formal G1 is **`NOT_RUN_BLOCKED`**. These 54 Klados items have no verified
participant map, so source records, not participants, are the statistical
units. Independence among the source records is not established. The intervals
below are descriptive paired source-record bootstrap intervals after forming
the posterior-mean waveform across algorithmic seeds; they are not population
confidence intervals, and seeds are not independent statistical units.

The single-source-record exploratory effect-direction check failed; formal G1
was not executed.

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

The diagnosis points to short-support subspace instability: five individual
projectors failed the frozen bootstrap-stability gate, whereas the joint
sim01--sim30 population operator remained usable; reference excitation and rank
were not the failure. Therefore B6 `POP-SHRINK` is the only selected diagnostic
backup; B1--B5 are not opened. B6 remains disabled for confirmatory evidence until its
rule is frozen and a genuinely new evaluation set is available. All existing
Klados development and evaluation records have now informed either selection
or this diagnosis, so a B6 rerun on them would be post-selection exploratory.
The disabled mathematical interface and its fit-scope/compatibility guards are
implemented in `src/eeg_cgdr/operators/pop_shrink.py`; J0b `919631` passed its
initial tests, and final J0c `919634` passed all 80 current tests plus the
existing real-record CGDR validator. B6 has not been evaluated on real
projectors.

Eye-BCI was not submitted: the preregistered condition requiring matching P0
specificity was not met. In addition, the old pre-repair S01 exploratory run
used S02--S26 for training and S27--S31 for validation. A new fold-wise protocol
could still be internally held out, but S07--S31 cannot honestly be called
historically untouched. G3 and diffusion necessity also remain untested: M2
contains hard deterministic Q-consistency, M0 was not evaluated on the final
records, and the required trained multichannel deterministic U-Net does not
exist yet.

## Execution and evidence paths

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

Next valid scientific stage:
`single_diagnostic_operator_repair_before_Eye_BCI`. No Eye-BCI or G3 job is
running, because submitting either would cross a failed specificity condition.
