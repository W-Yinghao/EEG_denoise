# WAVE4-E2 — operational addendum for the Tier-S-authorized measurements

Committed BEFORE any E2 code runs. Base: `codex/wave4-optical` tip `97bebe4`. CPU only.
Authorization: E1R `tier_s = true` unlocks **M2**, **M4**, and the **segment-level variant
of M3**; `tier_e = false` keeps **M1** and **full M3** locked. Execution order, as
instructed: **M2 → M4 → M3**.

Part 1 of `reports/wave4_preregistration.md` already froze the M2/M3/M4 intent, the M3
scope declaration, the M4 statistic, the cross-panel caveat and the S27–S31 exclusion.
This addendum freezes only the operational detail those clauses left open. Nothing here
may move after this commit. The banked E1 verdict, the E1R verdicts and all WAVE3 numbers
are read-only.

## Substrate (shared by M2/M4/M3)

The **12 E1R-eligible recordings** (clock-valid ∧ Tobii validity ≥ 0.40): S01×2, S02×9,
S03×1. Clock reused verbatim; no drift fit recomputed. S27–S31 are excluded and counted
everywhere in E2 — vacuously here, since the eligible set is S01–S03.

**Optical regressor block**, resampled onto the 1000 Hz EEG clock through the frozen
affine map, z-scored per recording:

- `inval(t)` — both-eye invalidity indicator (the Tier-S-validated segment-level quantity)
- `gx(t)`, `gy(t)` — gaze position in **degrees of visual angle**, computed from
  `GazePointX/Y (ADCSmm)` and `mean(DistanceLeft, DistanceRight)` via
  `deg = atan2(mm, distance_mm)`, referenced to the recording median, sample-and-hold
  through invalid stretches
- `vel(t)` — gaze speed in °/s from the above
- `pupil(t)` — `mean(PupilLeft, PupilRight)`, median-filled

No per-event blink label enters any E2 measurement: Tier-E is locked, so every statistic
below is segment-level by construction.

**Program conventions reused verbatim**: `_ridge` with ratio **0.05**; 2-fold
cross-validation over disjoint contiguous halves; 1.0 s non-overlapping windows;
5000-draw bootstrap; Holm within each measurement family.

**FIR lag set (frozen)**: −100, −50, 0, +50, +100 ms at 1000 Hz.

## M2 — A4 reference-channel-error row (runs first)

Target block: the ocular-dominant frontal channels **FP1, FPZ, FP2, AF3, AF4, F7, F8**.

- **Artifact-rich window (frozen)**: a 1.0 s window whose optical invalidity fraction is
  **≥ 0.20**. Count reported per recording; recordings with < 20 such windows are reported
  and excluded from the M2 aggregate (counted, never repaired).
- **EOG-referenced regression**: target block on `HEO` with the frozen FIR lag set.
- **Optical-referenced regression**: target block on the optical regressor block with the
  same FIR lag set.
- Both fit by ridge (0.05) on artifact-rich windows under 2-fold CV; artifact estimates
  `â_EOG`, `â_OPT` are read on held-out folds.
- **Discrepancy statistics** per recording × channel:
  `D_rms = ‖â_EOG − â_OPT‖ / ‖â_EOG‖` (primary) and `D_corr = 1 − corr(â_EOG, â_OPT)`.

**Interpretation, frozen now**: `D_rms` is an **upper bound** on reference-channel error.
It contains EOG measurement noise and neural crosstalk (the quantity of interest) *and*
the optical reference's own limitations, which cannot be separated on this panel. It is
reported as a bound, never as a point estimate of EOG error.

## M4 — exogeneity (runs second)

Statistic already frozen in Part 2: during optical-confirmed fixation, regress `HEO` on
the posterior block (P7 P5 P3 P1 PZ P2 P4 P6 P8 PO7 PO5 PO3 POZ PO4 PO6 PO8 O1 OZ O2 CB1
CB2) with ridge ratio 0.05; report in-sample R² with a 2-fold cross-validated companion;
compare against OPERA's leakage R² **0.055**.

- **Optical-confirmed fixation mask (frozen)**: both eyes valid **and** gaze speed
  < **30 °/s** sustained for ≥ **100 ms**, excluding any sample inside a blink candidate.
  This is applied as a **segment mask built from directly measured quantities**, not as a
  κ-validated event label — M1 is locked, and this addendum claims no event-level typing.
- **Sensitivity**: the velocity threshold is swept over 20 / 30 / 50 °/s; the verdict is
  read at the frozen 30 °/s.
- Recordings yielding < 60 s of fixation-masked data are reported and excluded from the
  aggregate (counted).

## M3 — readout bound, segment-level variant (runs third)

**DIFF-class readout: NOT-MEASURABLE-THIS-PANEL**, per the Part-2 scope declaration — no
V44-class checkpoint applies to the 62-channel Neuroscan montage and porting checkpoints
across montages is prohibited. M3 therefore runs the LINEAR-vs-analytic version only.

Optical artifact reference: the frontal target block regressed on the optical regressor
block, **fit on disjoint segments** (2-fold contiguous halves), evaluated on matched 1.0 s
windows. Readout families are the banked WAVE3 T6 ladder applied to this regressor block:

- **LINEAR** = `indicator_linear` (ridge on the regressor block as-is)
- **ANALYTIC** = the minimum CV residual among `rank3_derivative`, `fir_lagged`,
  `amplitude_gain`, `kernel_ridge` — the same best-family rule WAVE3 T6 used

**Statistic**: relative gain `g = (residual_LIN − residual_ANALYTIC) / residual_LIN`,
matching the banked `relative_gain_vs_incumbent` convention.

**Frozen rule**: `g ≤ 0.03` ⇒ the readout ledger row is **BOUNDED** at 0.03.
`g > 0.03` ⇒ the row is **SIZED** at the measured `g`. The threshold is read on the
relative-gain scale, as in the banked ladder.

## Statistics and honest-power statement

Participant-first over the subjects present in the eligible set, then 5000-draw bootstrap;
Holm within each measurement family. **Only 3 subjects survive eligibility (S01, S02, S03)
and the design is heavily unbalanced (2 / 9 / 1 recordings)**, so the participant-level
bootstrap has almost no resolution. A recording-level bootstrap (n = 12) is reported
alongside it as a **declared secondary** companion, explicitly labelled as such; the
primary verdict is read participant-first regardless, and the low-power limitation is
stated in the report rather than worked around.

**Cross-panel caveat (frozen wording, applies to M2/M3/M4)**: these are measured on
Eye-BCI and enter the MobileBCI ledger as order-of-magnitude **BOUNDS** with an explicit
cross-panel label, unless a comparability check (montage/reference/task overlap) is
separately passed. No such check is performed in this wave.

Deliverables: `results/wave4_optical/{m2,m4,m3}/`, `reports/wave4_report.md`, ledger
addendum rows ADDED beside WAVE3 (never edits). Stop after M3.
