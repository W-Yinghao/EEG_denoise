# IRIS preregistration — P1R fallback repair + T/W3 drive-encoding repair

Committed BEFORE either rerun. Both follow the program's defect-repair discipline:
the banked P1 ladder verdict (DEAD_POINT_ESTIMATES) and the banked T FAIL stand
unedited; nothing below moves a frozen constant of an already-adjudicated gate; both
experiments here carry NEW, fresh-frozen estimands. CPU only.

## P1R — fallback repair and the vs-incumbent gate comparison (MobileBCI dev-15)

**Motivation (banked P1 diagnostics).** The deployed hard-gate fallback (POP
subtraction) loses to NO subtraction on exactly the cells the gate catches (1.8782 vs
1.6244 abstained; 0.7877 vs 0.5973 pooled); the incumbent violates the absolute
never-worse property on the same 6 cells the inflation arm did; inflation never
underperformed the incumbent anywhere (pooled 0.4246 vs 0.4340).

**Machinery.** Identical to P1 (same folds/seeds/episodes/drives/metrics; V44-S1
verbatim). New arms:

| Arm | Rule |
| --- | --- |
| BINARY_NOA0FB | incumbent binary gate; hard-gated cells fall back to NO subtraction (a0-path zeroed) instead of POP subtraction |
| INFLATION_NOA0FB | inflation (P1 construction, unchanged) on gate-open cells; hard-gated cells fall back to NO subtraction |
| BINARY, INFLATION, POP, NO_A0, ORACLE, WRONG_binary, WRONG_inflation | carried unchanged from P1 for paired comparison |

**Frozen gates.**
- **G-P1R-a (fallback repair, primary):** paired per-cell contrast
  RRMSE(BINARY) − RRMSE(BINARY_NOA0FB) over the abstained cells; PASS = mean > 0 with
  bootstrap CI-low > 0 (5000 draws, seed 420). This adjudicates the one-line
  deployment change on the incumbent itself, independent of inflation.
- **G-P1R-b (relative never-worse):** per-cell
  RRMSE(INFLATION_NOA0FB) ≤ RRMSE(BINARY_NOA0FB) + 0.005 for ALL cells; PASS = zero
  violations. (The vs-incumbent form P1b should have had.)
- **G-P1R-c (pooled non-inferiority + gain reading):** pooled
  RRMSE(BINARY_NOA0FB) − RRMSE(INFLATION_NOA0FB); reported with CI; PASS =
  CI-low > −0.005. A positive CI-low is additionally recorded as a gate-refinement
  gain, claimable only at "sized" strength (n = 15 dev participants).
- **G-P1R-d (harm, carried):** mean(RRMSE(WRONG_inflation) − RRMSE(POP)) ≤ +0.005 and
  mean(RRMSE(WRONG_binary_NOA0FB) − RRMSE(POP)) ≤ +0.005.

**Adoption rule (frozen).** IRIS's deployed gate for the F-stage fights =
- BINARY_NOA0FB if G-P1R-a passes and G-P1R-b/c fail;
- INFLATION_NOA0FB if G-P1R-a,b,c all pass;
- unchanged incumbent BINARY if G-P1R-a fails.
Whatever is adopted, the incumbent remains the floor sub-model and the abstention-row
RECLAMATION claim stays dropped (P1 verdict; not revisited).

## T/W3 repair — gaze-loss drive encoding (dots)

**Defect (verified, banked in M-D).** L-GAZE-X/Y and L-AREA snap to 0 during tracking
loss; loss concentrates in the artifact-rich evaluation windows (39.9% vs 12.4% on the
worst recording); the typed design is pathological exactly where the estimand reads.

**Repair (frozen).** Loss mask = (L-AREA ≤ 0 or non-finite). Gaze-x, gaze-y, and pupil
are linearly interpolated across loss gaps (edge gaps held at the nearest valid
value). Blink information enters through the blink event train ONLY. No other change:
the drive set, lag structures, ridge, thirds, rich-window definition (top 20% VEOG
energy), and both T gate constants (CI-low > 0 AND mean ≥ 0.05) are unchanged, as are
W3's referee bar (r ≥ 0.5), FIR lags, family ladder, and the 0.03 bound.

**Reruns.** T and W3 rerun under the repaired encoding; outputs written BESIDE the
banked versions (`t_typed_info_repaired.json`, `w3_readout_repaired.json`); W1 is also
recomputed under the same encoding as a declared companion (`w1_a4_repaired.json`) —
the banked W1 row remains the primary ledger entry unless the repair changes it by
more than its own CI width, in which case BOTH are reported and the discrepancy is
flagged, not resolved silently.

**Reading rule (frozen).** The typed-information verdict for the F2 fight is read on
the REPAIRED T gate. If it fails again, the family-finality extension to the rich
reference is adopted as a genuine negative (no further repairs without a new
preregistration naming a new defect).

## Amendment 1 (committed before the repaired rerun executes) — T2 increment estimand

**Disclosure.** Smoke-diagnosis of the repaired encoding on the worst recording
(EP39_DOTS2) shows the repair does not change the T outcome (delta −19.5 → −22.2), and
the verified mechanism is not a defect at all: in the artifact-rich (blink) windows the
STATIC [VEOG, HEOG] family is a near-autoregression of the artifact itself (its
residual approaches the noise floor; typed max|coef| 8.1 < static 15.1, prediction RMS
16.2 vs target 22.2 — no numerical blowup). The frozen T contrast therefore measures
whether typed drives can REPLACE the waveform reference — a question the IRIS design
never poses (the EOG columns are retained by construction; F9 leaves the family
question open only where the reference is genuinely RICHER, i.e. a superset). The
banked T FAIL and its preregistered repaired rerun stand as the decisive answer to the
REPLACEMENT reading. The thesis reading — does the typed set ADD information — is T2.

**T2 (fresh-frozen).** Same recordings, thirds, rich-window definition, repaired gaze
encoding, ridge, and lag structures. Designs: STATIC (as T); COMBINED = STATIC ∪ TYPED
(35 regressors); COMBINED_SHUFFLED = STATIC ∪ circularly-shifted TYPED (per-recording
uniform shift ≥ 5 s, rng seed 420) — the regressor-count null.
Estimands: Δ_inc = (resid_STATIC − resid_COMBINED)/resid_STATIC;
Δ_null = (resid_STATIC − resid_COMBINED_SHUFFLED)/resid_STATIC.
**Gates (frozen): participant-first CI-low(Δ_inc) > 0 AND mean(Δ_inc) ≥ 0.05 AND
CI-low(Δ_inc − Δ_null) > 0.** PASS → typed information is REAL as an increment → typed
family proceeds to F2 (as an addition to, never a replacement of, the EOG columns).
FAIL → the typed-family leg is dropped and the rich-reference information claim is
closed as a genuine negative — no further estimand changes.
