# Frozen diffusion incremental-value protocol

> **Superseded protocol:** the current frozen rule is
> `reports/cgdr_diffusion_incremental_decision_protocol.md`, which requires the
> complete M1, M4, operator-conditioned DDIM100, and EEGDfus matrices. The
> weaker completion language below is retained only as protocol history.

Frozen on 2026-08-02 before the new development and benchmark outputs were
read.  This protocol does not alter the retained Klados or SGEYESUB results.

## Klados exploratory source-record comparison

- Training source records: sim01--sim30.
- Development source records: sim31--sim36, sim44, sim45.
- Previously audited records sim37--sim43 and sim46--sim54 may only be used for
  explicitly labelled exploratory replay; they are not fresh confirmation.
- Frozen arms: M1 observation warm start, M2 final hard-Q, M4 per-step soft
  proximal, deterministic Qy, deterministic soft proximal, and the
  task-matched multichannel deterministic U-Net.
- Operator sources: population projector, matching P0, and query-derived
  oracle projector.  Oracle geometry and Qy are non-deployable diagnostics.
- Diffusion arms use five registered algorithmic seeds and 100 network calls
  per seed/window.  Seeds are averaged within a record and are not independent
  statistical units.  Algebraic arms use zero calls and U-Net uses one call;
  this compute difference is reported rather than hidden.
- The U-Net uses the same observed query, projector, framewise external-EOG
  attenuation, valid-time mask, training source records, windows, and paired
  corruption exposure. It additionally uses paired clean targets during
  training, unlike the clean-prior sampler. It is therefore a stronger,
  differently supervised exploratory comparator, not a same-supervision
  baseline and not formal G3 evidence. Population, matching-P0, and query-derived-oracle
  operator scopes train and select three independent checkpoints; no oracle
  cell can influence a deployable population or matching checkpoint.  Each
  scope trains for at least 3000 and at most 6000 updates and uses only its
  same-scope development cells for checkpoint selection.  The oracle-scope
  checkpoint remains explicitly non-deployable.
- The earlier shared-scope checkpoint from Slurm job 919785 is retained as
  `invalid_pre_operator_scope_isolation` and is ineligible for every scientific
  comparison.  Its downstream jobs 919786 and 919787 were cancelled before
  producing usable evidence.
- This source-record comparison is exploratory and cannot by itself establish
  formal G1/G3 or a diffusion-family conclusion.

The original scope-isolated v2 run contract remains preserved at
`configs/cgdr/klados_stage3_deterministic_comparison.yaml`. A prospective v3
amendment, frozen before any v3 outcome, additionally requires one common
matching-P0 eligibility set for all three scope-specific training/validation
bundles, separates eligible-only matching effects from the all-requested
fallback policy, retains method/seed failures, and makes terminal early-stop
resume idempotent.

Prospective implementation: protocol
`klados_stage3_deterministic_scope_isolated_v3` in
`configs/cgdr/klados_stage3_deterministic_comparison_v3.yaml`. Its results use
the separate `results/cgdr/klados_stage3_deterministic_scope_isolated_v3/`
root; v2 outputs are never overwritten.

## EEGDfus paired single-channel benchmark

- Official-native and strict source-epoch protocols are reported separately.
- Cells are protocol x EOG/EMG x conditional diffusion/matched deterministic.
- Each matched pair receives exactly the same prepared pairs, split, epochs,
  batches, optimizer schedule, and planned optimizer updates.
- Full runs are fixed at 4000 epochs, batch size 512, 500 diffusion steps, and
  11 mixtures.  The 1-epoch/8-step path is engineering smoke only.
- Official-native retains and discloses post-mixing train/validation source
  overlap.  Strict reanalysis splits clean and artifact source epochs 72/18/10
  before pairing or mixing.  EEGdenoiseNet has no participant IDs, so neither
  route makes a participant-independence claim.
- Outcomes are paired clean-component RRMSE/correlation/SNR metrics for the
  single-channel EOG/EMG stress test only; they do not test personalized EEG
  geometry or natural multi-channel deployment.

Frozen implementation: `configs/baselines/eegdfus_native_strict.yaml`.

## Decision boundary

The current label remains `current_M2_no_incremental_value`.  A broader tested-
protocol statement is considered only after at least M1 or M4, the matched
multichannel U-Net, and full EEGDfus conditional/matched cells complete.  The
SGEYESUB deterministic block-2 audit remains real-EEG context but cannot be
used as diffusion evidence because it contains no diffusion arm.  Formal G1
and G3 remain `NOT_RUN_BLOCKED`.
