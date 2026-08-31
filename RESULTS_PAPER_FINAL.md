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

## T1 — Held-out predictive intervals: 80/90% hold near nominal on unseen users

Part A (committed before any held-out inference): ONE scalar temperature per policy
frozen from the development set, grid 0.50–6.00 step 0.05, smallest value reaching
80% mean coverage over ALL 15 dev fold-seed cells jointly (banked IRIS-F4 K=32
arrays): **INFL 2.40, TEMP 3.00** (inside F4's leave-one-fold-out ranges 2.30–2.55 /
2.90–3.20).

Part B/C (M35 sealed episodes/protocol, K=32 chains, each chain = mean of the 5
seed-20261201 fold models under a shared noise and operator draw; temperatures
applied unchanged):

| policy | coverage 50/80/90% | Gaussian CRPS | risk–coverage |
|---|---|---|---|
| raw samples | 0.281 / 0.460 / 0.543 | 0.4016 | 0.0781 |
| temperature only (3.00) | 0.613 / 0.813 / 0.867 | 0.3824 | 0.0781 |
| **propagation + temperature (2.40)** | 0.610 / **0.810** / **0.864** | **0.3800** | 0.0786 |

The paper's question — do 80/90% hold near nominal on unseen users — answers yes:
0.810 / 0.864 under the dev-frozen temperature (dev: 0.802 / 0.853). The ordering
reproduces out of cohort: the operator-posterior width again beats temperature-only
on CRPS at a smaller temperature, and the 50% level stays conservative exactly as in
development. Per-participant 80% coverage: 6/8 in [0.77, 0.94]; sub-04 0.774,
sub-22 0.790; the one clear undercoverage is sub-01 at 0.456 (the same cohort's
weakest member in the point-estimate pass). Absolute CRPS is larger than dev
(0.380 vs 0.150) because held-out absolute errors are larger; cross-cohort CRPS is
not comparable and is reported per cohort. Arrays: per-window mean+width for
sub-01 and sub-04 in `paper_final_arrays/t6_heldout_intervals_sub-0{1,4}.npz`.

## T2 — Calibration-duration curve: 60 s already buys ~97% of the gain

Gain (matched − unguided; unguided = stored seed-20261201 NO_A0 rows, identical
episodes/noise), system reading = reliability rule active (reject → withhold guide),
rule-off = rejection floor bypassed (raw shrinkage curve):

| duration | 30 s | 60 s | 90 s | 120 s |
|---|---|---|---|---|
| system gain | −0.016 (gate fires 100%) | +0.146 | +0.148 | **+0.150** |
| rule-off gain | +0.124 | +0.136 | +0.140 | +0.143 |
| gate fired | 100% | 6.7% | 4.4% | 4.4% |

Two readings, one story: (a) the information is cheap — 60 s of calibration already
delivers 97% of the 120-s gain, and even a 30-s ridge estimate carries +0.124 if
forced through; (b) the floor at 60 s is the designed reliability boundary (< 60 s
the four sub-block statistics that drive shrinkage and the gate are undefined), and
at 30 s the system visibly refuses rather than degrades (−0.016 ≈ 0, the safe
unguided point). At 60–120 s the system reading sits slightly ABOVE rule-off at
every duration — the gate is not just safe, it nets a small positive by rejecting
the few unreliable cells. (Numbers are seed-20261201-only and therefore not
identical to the 3-seed dev headline +0.1428; the 120-s system point +0.150 is the
same contrast on the single-seed slice.) The 90-s point required relaxing the
frozen registry's duration whitelist {10,30,60,120} by replaying its constructor
verbatim (`pf_common.make_eb_registry`) — the only code deviation in this task.

## T3 — Ablation matrix: one cell lights up; shrinkage is load-bearing

All cells share the seed-20261201 episode bank and noise (the three campaign seeds
draw different episode banks, so absolute RRMSE differs by bank — e.g. MATCH_gated
is 0.633/0.438/0.222 across the three seeds, mean 0.431 — while contrasts are
bank-stable: MATCH−NO_A0 = +0.142/+0.152/+0.134). The matrix is therefore reported
as within-slice contrasts against the (matched, matched) cell (positive = that cell
is worse than matched-matched; participant bootstrap):

| cell (guide, features) | RRMSE (slice) | vs (matched, matched) | worse |
|---|---|---|---|
| **matched, matched** | 0.6334 | — | — |
| matched, population | 0.6387 | +0.0053 [−0.0043, +0.0183] | 7/15 |
| none, matched | 0.7753 | +0.1419 [+0.0795, +0.2067] | 13/15 |
| none, population | 0.7912 | +0.1577 [+0.0930, +0.2234] | 13/15 |
| population, population | 0.8196 | +0.1862 [+0.0949, +0.2975] | 14/15 |
| matched **unshrunk** (λ=1), matched | 0.9804 | +0.3470 [+0.0040, +1.0249] | 13/15 |

Readings: (1) with a matched guide, the calibration FEATURES add essentially
nothing (+0.005, CI spans 0) — the guide carries the payload; (2) without the
guide, matched features recover only ~0.016 of the ~0.15 gap; (3) the population
guide remains worse than no guide inside this slice too; (4) the unshrunk arm is
the table's cautionary cell: the raw 120-s ridge without empirical-Bayes shrinkage
is catastrophically unstable for a minority of participants (upper CI +1.02) —
shrinkage is what makes a 2-minute calibration deployable.

## T4 — GPU leg (record thirds): the paired gain does not decay; the natural gain does

Gain (matched − unguided) on fresh paired episodes restricted to record thirds
(same construction, recipients/donors/gains identical across thirds; the injection
operator is protocol-fixed at the 150–270-s Qgen fit):

| record third | early | mid | late |
|---|---|---|---|
| gain | +0.134 | +0.143 | +0.158 |

Interpretation (with the CPU leg above): in the paired protocol elapsed time enters
only through EOG/carrier content, and the guide keeps working on late-record
content — the paired gain is flat-to-rising. The real staleness signal lives in
the CPU leg: the operator itself drifts (relative displacement 0.86 → 1.43 by
480 s) and the real-record natural attenuation gain decays 2.5 → 1.5 dB by the
~10-minute mark. Together: the calibration's content validity persists, its
operator validity decays — recalibration is what the loop's clock should trigger.

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

## D-wave — downstream utility on the panel's native BCI tasks (dev cohort)

Design + interpretation grid frozen pre-compute (`DOWNSTREAM_UTILITY_WAVE_DESIGN.md`);
probe froze the stimulus frequencies (5.47/8.59/11.91 Hz — the dataset's 60/11, 60/7,
60/5 Hz monitor divisors), the occipito-parietal channel set (13), and verified
event-onset units (CCA accuracy 1.00 on the probe cell). Trials with onset < 120 s
excluded. Decoders are training-free (CCA) or shallow-deterministic (shrinkage-LDA,
within-participant 5-fold CV) — no deep classifier anywhere in the endpoint.

**D1 — SSVEP 3-class CCA accuracy** (RAW 0.773; chance 1/3; decoding under
ambulation, far from ceiling): every method sits within ±0.006 of RAW.
MATCH − RAW = −0.0014 [−0.0082, +0.0052] → the frozen grid's **no-harm cell**:
calibrated cleaning does not trade decoding for artifact removal. The
pre-registered high-contamination-tertile prediction did NOT materialize
(−0.0072 [−0.0216, +0.0071]); reported as-is. ICA is the only nominally positive
arm (+0.0058 [+0.0000, +0.0126], 7/15); ASR the only nominally negative
(−0.0061). Occipital SSVEP is simply robust to frontal ocular contamination —
an informative null the paper reports with the metric-utility-gap citations.

**D2 — ERP target/nontarget AUC** (RAW 0.7503): the one structured result.
The unguided pass is the only arm significantly BELOW raw
(NO_A0 − RAW = −0.0066 [−0.0107, −0.0030], 1/15 positive) — cleaning without the
guide costs task information. Every guided variant restores it: MATCH +0.0029
[−0.0102, +0.0145] (9/15), LINEAR +0.0071, POP +0.0099 [−0.0001, +0.0205] vs RAW;
the guide's increment over unguided is MATCH − NO_A0 = +0.0096 [−0.0027, +0.0196],
12/15. **ERP preservation** (correlation with the RAW low-contamination average):
MATCH 0.931 ≈ ICA 0.933 ≫ SGEYESUB 0.800, ASR 0.788 — the calibrated system cleans
without deforming the ERP, while the aggressive classical anchors visibly do.

Verdict per the frozen grid: D1 = no-harm; D2 = no-harm with a
mechanism-consistent guided-vs-unguided increment; nothing is over-claimed —
"the system does not trade decoding for cleaning, and where cleaning costs task
information (ERP), the calibrated guide is what pays it back."

**Held-out single pass (n=8, frozen protocol, run after the dev decision was
banked).** No-harm reproduces on both tasks: SSVEP MATCH − RAW = −0.0018
[−0.0118, +0.0072] (RAW 0.832); ERP AUC MATCH − RAW = −0.0016 [−0.0092, +0.0096]
(RAW 0.768); ERP preservation MATCH 0.966. The dev guided-vs-unguided structure
points the same way but is not adjudicable at n=8 (MATCH − NO_A0 = +0.0022
[−0.0031, +0.0082], 5/8; unguided − RAW = −0.0038, n.s.). The instructive held-out
result is the currency split among the anchors: ASR's ERP AUC is significantly
ABOVE raw (+0.0422 [+0.0106, +0.0753]) while its ERP preservation collapses to
0.637 and its retention to 0.62 — aggressive variance removal can help a linear
classifier while destroying the waveform — and SGEYESUB significantly harms SSVEP
decoding (−0.0100 [−0.0127, −0.0063], 0/8). Decodability, morphology preservation,
and retention are three different currencies; the calibrated system is the one
method that stays healthy in all three (AUC ≈ raw, preservation 0.966, retention
0.843), which is the deployment claim the paper makes. This pass was the sealed-8
cohort's authorized second/third contact (UQ per T1, downstream per the approved
D-wave design); point estimates from M35 remain final as banked.

## D-wave deep-decoder endpoint — EEGNet-8,2 on the banked waveforms

Adds decoder capacity to the D-wave question; no diffusion inference re-run.
Within-participant 5-fold CV, identical folds across arms, fresh init per arm.

- **ERP (AUC), dev n=15**: RAW 0.769; MATCH 0.765, LINEAR 0.766, POP 0.774 — all ≈ RAW.
  ICA 0.745, SGEYESUB 0.740 — the deep decoder resolves harm from the classical
  arms that shrinkage-LDA missed.
- **ERP heldout n=8**: RAW 0.752; MATCH 0.759, POP 0.763 — same picture, no harm.
- **SSVEP (3-class acc)**: 0.34–0.37 everywhere on both cohorts — EEGNet is
  underpowered for 3-class CCA-style SSVEP at these trial counts (CCA endpoint
  remains the informative one); no arm separates.
- Verdict: the no-harm conclusion is decoder-robust; deep capacity finds
  *classical*-arm harm, not diffusion-arm harm.

## Sealed-55 EEGEyeNet confirmation — both endpoints, preregistered

Opened 2026-08-30 against `reports/iris_prereg_sealed55.md` (amendments
SEALED55-1..3 logged before results); 55 subjects, 0 guard exclusions.

**Option A (S356 conditioning) — CONFIRMED.** Gain at n=259: **+0.0810**
[+0.0578, +0.1094], 53/55 positive; own−wrong **+0.1662**, 55/55; flat in n
(G3 pass). All five gates green. Dev reference (+0.0608) reproduced on a
second corpus.

**Option B (UQ transfer) — UQ_CONFIRMED_SECOND_CORPUS, with one honest miss.**
Frozen dev temperatures (INFL 2.45/TEMP 3.25) transported: coverage
0.830/0.891 at nominal 0.80/0.90 (tolerance ±0.05 → B1 pass, B2 pass).
**B3 FAILED**: temperature-only CRPS 0.1363 < inflation 0.1375 — the
"physics-shaped propagation" wording is NOT earned on EEGEyeNet; per-subject
80% coverage spans 0.218–0.998 (disclosed in full). Both verdicts reported per
the A2 disclosure rule.

## BCI-IV-2a (EOG panel) — the method family on a 4-class MI dataset

Two routes, gates report-only, official T→E split, EEGNet-8,2 decode, n=9.

- **Route 1 (linear operator only)**: RAW 0.593; LINEAR_MATCH 0.585
  (−0.0077 [−0.0274, +0.0081]), POP/WRONG likewise ≈ RAW.
- **Route 2 (full diffusion: 22-ch CalibSADDPMEOG, 3 folds, healthy training
  0.017–0.028 val)**: RAW 0.593; MATCH 0.582 (−0.0112 [−0.0428, +0.0224]),
  NO_A0 +0.0004, POP −0.0050 — every CI crosses zero.
- Verdict: on a third panel, third decoder, and both method routes, cleaning
  neither helps nor harms downstream decoding. Consistent with the ambulatory
  panels; strengthens the honest "denoising ≠ decodability" framing.

## Wave-5 — head-to-head with EEGDfus and DS-DDPM (repro + comparability)

**E1/E2 (EEGDfus grid, −5..5×11, single-channel EEGdenoiseNet, all 8 arms).**
Our CondDiff: EOG RRMSE_t 0.453 / CC 0.888; EMG 0.632 / 0.763. Beats Noisy,
SDEdit, FCNN, RNN_LSTM; **loses to SimpleCNN (0.321/0.934 EOG) and NovelCNN**.
Nothing on our bench approaches EEGDfus's published ~0.99 CC.

**EEGDfus on SSED (their code, their split bug quantified).** Training as
released succeeds. Released split (test overlaps train, their
`train_test_split(list(range(len(val_test_idx))))` bug): RRMSE_t 0.0846 /
CC **0.9805** — reproduces the flavor of their published 0.121/0.992.
Strict split (bug fixed): **0.373 / CC 0.889**. The leakage flatters their
SSED table; we report both.

**E3b/E4 (our 1-ch model on SSED).** Honest negative: trained-on-SSED CC 0.49,
zero-shot 0.48, 10%-finetune 0.48 — all below the identity baseline (CC 0.70;
resample bridge exonerated at CC 1.000). Full-generation sampling hurts
mostly-clean rows; the single-channel transplant is not where the method lives.

**E5 (PLV, Tables IV/V analogue).** Identity already ≈1.0 in alpha/beta/gamma;
contamination is low-frequency. EEGDfus repairs delta PLV 0.68→0.98 released /
0.87 strict; our transplant degrades delta to 0.53–0.58. Consistent with CC.

**DS-DDPM (collaborator repo, commit 12c339a).** Training reproduces
(100 epochs, loss 3.54→0.687, their loop mirrored verbatim). Two reproduction
findings, preserved not repaired:
1. **The release contains no real-data denoising path** — both samplers start
   from `torch.randn`; the paper-described inference was reconstructed from
   their own `q_sample`/`p_x0` primitives (t*=20, their apply_step default),
   disclosed.
2. **Table I is not reproducible from the release**: 10 matrices (raw / ICA /
   DS-DDPM-separated × SGD-paper / Adam / Adam+their-commented-l2 /
   Adam+zscore recipes) all land at chance (25.2–27.7% vs published M
   50.58/52.87), diagonals 26–40% vs published 74–92 — while our own
   EEGNet-8,2 reaches ~66% within-subject on the same trials. The bottleneck
   is their undocumented classifier protocol, not the data.
Table II analogue (PSD-signature correlation matrices, real-real vs
generated-real) queued; D4PM joint arm still training.

## Arrays manifest (`paper_final_arrays/`)

| file | contents |
|---|---|
| `t5_natural_plane.npz` | conditions[6], attenuation_db, retention, coherence_reduction + CIs |
| `t6_waveform_exemplar_dev.npz` | eog_drive(2,512), contaminated/reference/linear_regression/unguided/matched(46,512), band_mean(46,512), band_halfwidth_80(46,512), eeg_names, fixed_channels, ids |
| `t6_scalp_improvement.npz` | improvement_noa0_minus_match(46), eeg_names(46) |
| `t6_width_locality.npz` | within_v(270), propagation_width_share(270), hard_gate, lam, cell |
| `t1_heldout_uq_summary.npz` | held-out interval policies (coverage/CRPS/RC/per-participant) as JSON payload |
| `t6_heldout_intervals_sub-01.npz`, `t6_heldout_intervals_sub-04.npz` | per-window mean/sigma/var_op + contaminated/reference/eog_drive for two held-out participants |
| `t2_duration_curve.npz` | durations, system/rule-off gains + CIs, hard-gate fractions |
| `t3_ablation_matrix.npz` | matrix cells + contrasts (JSON payloads) |
| `t4_staleness.npz` | displacement curve + CIs, gain-by-third + CIs |
| `cpu_reference_rows.npz` | ICA/ASR/SGEYESUB paired + natural stats (JSON payload) |
| `dwave_dev.npz`, `dwave_heldout.npz` | downstream decision payloads (D1 accuracy/contrasts/tertiles, D2 AUC/contrasts, ERP preservation) |

All headline tables above are backed by JSONs under `results/paper_final/`; the
GPU spend of the whole campaign (S0, T1–T6, D-wave dev + held-out, recoveries) is
well under the 6 GPU-h budget.
