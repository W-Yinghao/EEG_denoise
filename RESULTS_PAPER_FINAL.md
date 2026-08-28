# RESULTS_PAPER_FINAL — T1–T6 GPU inference + CPU reference rows

Branch `codex/paper-final-runs`. Frozen V44-S1 seed-20261201 checkpoints (T1 also uses
20261202/03 only where the dev temperature freeze consumed banked 3-seed arrays).
Participant-first aggregation, 5,000-resample participant bootstrap (seed 420)
throughout. Arrays in `paper_final_arrays/` (manifest at the end). Slurm jobs
962981–962987, ~2026-08-29.

One clarification recorded once: the 15 dev + 8 held-out participants are the
MobileBCI/Eye-BCI multimodal panel (46 EEG + bipolar VEOG/HEOG, 100 Hz); the sealed
EEGEyeNet-55 block is a separate asset and was not touched by any of these runs.

---

## S0 — Pipeline sanity check (ground rule 4): PASS

Re-ran the standard dev evaluation (15 fold-seed units, arms MATCH_gated and NO_A0,
frozen checkpoints, frozen noise seeds):

| quantity | this run | stored (V44-S1) | target |
|---|---|---|---|
| dev MATCH_gated | 0.430974 | 0.430971 | ≈ 0.4310 |
| dev NO_A0 (unguided) | 0.573775 | 0.573779 | ≈ 0.5738 |
| gain (matched − unguided) | +0.14280 [+0.1061, +0.1834], 15/15 | +0.14281 | — |

Reproduction to 5 decimals; gate passed, all new cells proceeded.

## T1 — Held-out predictive intervals (pending; highest priority)

Part A (done, committed before any held-out inference): ONE scalar temperature per
policy frozen from the development set, grid 0.50–6.00 step 0.05, smallest value
reaching 80% mean coverage over ALL 15 dev fold-seed cells jointly (the banked
IRIS-F4 K=32 arrays): **INFL 2.40, TEMP 3.00** (F4's leave-one-fold-out values were
2.30–2.55 / 2.90–3.20 — the joint scalars sit inside both ranges).
Part B/C (GPU job 962982 running): fold-99 sealed protocol, K=32 chains, 5-model-mean
per chain with shared noise and operator draw. Results appended when complete.

## T2 — Calibration-duration curve (pending; GPU job 962983)

## T3 — Ablation matrix completion (pending; GPU job 962984)

## T4 — Operator lifetime / staleness

CPU leg (complete). Closed-form ridge re-estimation on successive 120-s windows of
each dev record (per-window robust EOG scaling, fold EEG scaling), relative
Frobenius displacement from the 0–120-s calibration operator, participant-first:

| window start | 0 s | 120 s | 240 s | 360 s | 480 s |
|---|---|---|---|---|---|
| relative displacement | 0 | 0.861 | 0.854 | 1.029 | 1.428 |

The operator estimated two minutes later already differs by ~86% of the calibration
operator's norm, and the displacement grows monotonically from ~4 minutes on — the
calibration has a measurable lifetime.

Elapsed-time stratification of the STORED V44-S1 natural rows (window-position
metadata exists there; the stored paired rows carry none — the paired leg therefore
runs fresh episodes by record third, GPU job 962985):

| natural window position | ~300 s | ~406 s | ~513 s | ~619 s |
|---|---|---|---|---|
| attenuation gain, matched − unguided (dB) | +2.45 [1.80, 3.16] | +2.55 [1.83, 3.38] | +2.24 [1.59, 3.01] | **+1.51 [1.07, 1.95]** |
| retention delta (matched − unguided) | −0.096 | −0.094 | −0.088 | −0.033 |

The natural-window benefit of the calibrated guide decays by the ~10-minute mark
(attenuation gain 2.5 → 1.5 dB), consistent with the displacement curve: the
adaptation loop has a clock, and it points at recalibration.

## T5 — Natural-plane completion (complete; zero GPU)

Deviation note (one sentence): the mismatched and shuffled natural rows already
existed in storage — V44-S1 ran the full 7-arm natural panel and they were simply
never aggregated — so T5 is a CPU aggregation of stored evaluator outputs.

Five-condition natural table (participant-first means; mismatched reported both
gated — the paper's accepted-mismatched convention — and ungated):

| condition | EOG attenuation (dB) | low-EOG retention | coherence reduction |
|---|---|---|---|
| unguided (NO_A0) | 0.277 | **0.921** | 0.021 |
| population (POP) | 0.909 | 0.773 | 0.105 |
| matched (MATCH_gated) | **2.463** | 0.843 | **0.190** |
| mismatched, gated (WRONG_gated) | 0.570 | 0.780 | 0.084 |
| mismatched, ungated (WRONG) | −1.523 | 0.334 | 0.029 |
| shuffled EOG (SHUFFLED) | −0.384 | 0.365 | 0.083 |

Matched dominates population on both axes; a wrong person's operator without the
gate actively injects signal (negative attenuation at retention 0.33), and the gate
pulls the mismatched arm back to population-like behavior; destroying temporal
alignment is as bad as a wrong owner. Full stats with CIs in
`results/paper_final/t5_natural_plane.json`.

## T6 — Figure-data dump (partially complete)

1. **Waveform exemplar (dev)** — `paper_final_arrays/t6_waveform_exemplar_dev.npz`.
   Lowest-ID dev participant sub-02, ses-02 SSVEP, first non-zero-artifact episode
   with a clear ocular event (VEOG drive peak ≥ 2 robust units; episode index 1);
   fixed channels = the three most anterior of the montage: **Fp1, Fp2, AFz**.
   Arrays: EOG drive, contaminated, reference, linear regression, unguided, matched,
   plus the K=32 band (mean + 80% half-width, operator-posterior inflation at the
   dev-frozen temperature 2.40). Held-out exemplar arrays follow from T1.
2. **Scalp map** — `paper_final_arrays/t6_scalp_improvement.npz`. Per-channel
   improvement (per-channel RRMSE of unguided minus matched), participant-first;
   within-participant episode aggregation by MEDIAN (a handful of episodes have a
   near-flat clean window on single channels, which explodes the per-episode ratio —
   one-sentence deviation). Mean improvement 0.134; topography is ocular:
   Fp1 +0.290, Fp2 +0.267, AFz +0.224, F8/F4/F7 ≈ +0.19, smallest at ear/central
   electrodes (FC1 +0.014). 32 of the 46 channels have standard 10-20 positions;
   the 14 ear-electrode names are included for a custom layout.
3. **Width-locality scatter** — `paper_final_arrays/t6_width_locality.npz`
   (270 cell-points from the banked K=32 arrays; episode→cell mapping recovered by
   deterministic sampler reconstruction). Result is a null: the propagation-width
   share is nearly uniform (mean 0.30, max 0.52) and uncorrelated with the
   calibration within-variance (Pearson −0.017, Spearman +0.014, log-scale −0.021).
   The operator term contributes a roughly constant share of squared width rather
   than a cell-targeted one; its value shows up in CRPS/temperature (F4), not in
   per-cell localization. Reported descriptively; the planned scatter panel should
   either be dropped or shown as this (honest) flat cloud.
4. **Operating points / curves** — T5 array stored; T2/T3/T4 arrays appended when
   their jobs complete.

## CPU reference rows (partially complete; job 962987)

Environment note (one sentence): `asrpy` was pip-installed into the `icml` env for
the ASR row; ICA uses the installed mne 1.8.0 (fastica + EOG-correlation exclusion;
`mne-icalabel` is not installed, so ICLabel was not used).

**Calibrated eye-subspace subtraction (Kobler 2020 SGEYESUB style)** — complete.
Rank-2 subspace = column space of the same 120-s calibration ridge fit (raw,
unshrunk); correction projects it out; no runtime EOG. Same folds/episodes.

| endpoint | SGEYESUB-style | system (matched) | unguided |
|---|---|---|---|
| paired RRMSE | 0.827 | 0.431 | 0.574 |
| natural attenuation (dB) | 4.87 | 2.46 | 0.28 |
| natural low-EOG retention | 0.495 | 0.843 | 0.921 |
| natural coherence reduction | 0.213 | 0.190 | 0.021 |

The static projection attenuates the most and preserves the least — it removes the
ocular subspace everywhere, including where there is no ocular activity (retention
0.50 fails the 0.75 validity bar; paired error worse than raw 0.608). A faithful
context row for what a calibration-only linear subspace buys without a signal model.
(Implementation is "SGEYESUB-style" as specified — rank-2 projection from our own
calibration operator, not Kobler's full blink/vertical/horizontal decomposition.)

**ICA / ASR** — running; appended when complete.

---

## Arrays manifest (`paper_final_arrays/`)

| file | contents |
|---|---|
| `t5_natural_plane.npz` | conditions[6], attenuation_db, retention, coherence_reduction + CIs |
| `t6_waveform_exemplar_dev.npz` | eog_drive(2,512), contaminated/reference/linear_regression/unguided/matched(46,512), band_mean(46,512), band_halfwidth_80(46,512), eeg_names, fixed_channels, ids |
| `t6_scalp_improvement.npz` | improvement_noa0_minus_match(46), eeg_names(46) |
| `t6_width_locality.npz` | within_v(270), propagation_width_share(270), hard_gate, lam, cell |
| (pending) `t1_heldout_uq_summary.npz`, `t6_heldout_intervals_sub-01/04.npz`, `t2_duration_curve.npz`, `t3_ablation_matrix.npz`, `t4_staleness.npz`, `cpu_reference_rows.npz` | appended on job completion |
