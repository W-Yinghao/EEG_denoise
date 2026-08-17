# IRIS K-stage preregistration — three cheap kills (CPU only)

Committed BEFORE any K-stage execution. Charter: `reports/iris_campaign_charter.md`
(commit 5cc8e05; sealed block frozen there). All inputs are banked artifacts or
published dev data; no sealed contact; no GPU. Decision JSONs →
`results/iris/k/{k1_abstention_taxonomy,k2_sigma_drift,k3_instrument}.json`.
RNG seed 20260817 wherever sampling appears; `bootstrap_draws` 5000; program ridge 0.05.

---

## K1 — Abstention-cause taxonomy (prices the gate-shrinkage bonus)

**Question.** Of the banked gate-shrinkage ledger row (WAVE3 T4: mean 0.13707 RRMSE on
the MobileBCI likelihood-leg span, 93 cells, abstained fraction 0.0645 at 1.9152/cell,
active cells 0.0144/cell), how much is attributable to causes that covariance inflation
could convert IN PRINCIPLE, and how much is structural?

**Inputs (read-only, banked).** `results/wave3/t4_gate_shrinkage.json` (denominators —
never edited); V43/V44 per-cell rows (`rgcc_v43/stage2/*/stage2_result.json`,
`rgcc_eog_v44/stage1/*/stage1_result.json`); gate state re-derived per cell from the
frozen registries (`TransferRegistry`/`EBTransferRegistry`, CPU re-derivation of gate
logic on stored data — no new estimation class, no new data contact).

**Cause classes (frozen).** Each of the 93 cells' shrinkage/abstention mass is assigned:
- **R (reliability-recoverable):** λ-shrinkage or abstention driven by estimation
  uncertainty with the reference present and support ≥ the hard-gate minimum —
  convertible by continuous inflation in principle.
- **H (hard-gate floor):** the 10-s hard gate fired — structural safety; NOT convertible.
- **N (no-reference / no-support):** reference or support absent below minimum —
  NOT convertible (population route regardless).
- **D (identity-hazard adjacency):** cells where the gate is what stands between the
  system and wrong-donor-class harm (wrong-context ownership per the banked cell
  metadata) — conversion BARRED (F3/F4).

**Estimand.** `f_conv = mass(R) / total row mass`, with the abstained and active
sub-masses reported separately (the row is abstention-concentrated; the split is the
finding). Cross-panel companion (descriptive only): the same taxonomy over the banked
Klados/BCI2b abstention events (W3 ITT cells, M35 U-1 100%-abstained anchor, V43-S3c).

**Validity check (frozen).** Reconstructed total must match the banked 0.13707 within
±10%; on mismatch, report the discrepancy and use the banked total as denominator.

**Decision rule (frozen in charter §7.3).** P1's reclamation bar =
`min(0.30, 0.75 × f_conv)`. K1 has no GO/NO-GO of its own; it prices P1.

---

## K2 — Σ_drift prior-predictive recheck under corrected accounting

**Question.** WAVE2's G0 failed with coverage 0.9267 (prior too wide) using
`Σ_drift = var_cells(C_query − C_support)` — a raw cross-cell variance that CONTAINS the
estimation-noise floors of both operator fits (the same accounting family as the P2/O4
units lesson). Does the drift prior pass its own coverage gate once debiased?

**Correction (frozen).** `Σ_drift_deb = clip(var_cells(C_query − C_support)
− V̂_est(support) − V̂_est(query), 0)` per coefficient, where the estimation floors are
the per-coefficient within-scatter estimates of the same class the shared layer already
uses (support: within/4 from the 4 sub-block scatter; query: same estimator class on the
Qgen sub-blocks — matching the banked debiased reading D_deb = D_raw − W/4 − W/4).
Everything else in the G0 instrument is reused UNCHANGED: same 5 folds, same
16 dev participants, band centre `eb120.transfer`, nominal 0.80, z = Φ⁻¹(0.9),
`sd = sqrt(post_var + Σ_drift_deb)`.

**Gate (frozen).** Corrected mean coverage ∈ **[0.70, 0.90]** → the drift term is
candidate-ON inside IRIS's inflation gate S. Coverage > 0.90 (still too wide) or < 0.70
(too narrow → overconfident gate, a safety failure) → drift term OFF: drift enters IRIS
only as a fallback-ladder trigger. Both the original and corrected coverages are
reported; the banked 0.9267 is never edited.

---

## K3 — EEGEyeNet instrument validity (true-VEOG correspondence gates)

**Scope.** DEV subjects only (28 antisaccade + 30 dots published under
`eegeyenet_min/`). The sealed block (charter §5) is untouched. This is the E1R
instrument, ported to a panel with a real vertical periocular axis and official ≤2 ms
EyeLink synchronization; the gates are the ones the Tobii panel failed.

**Frozen derivations (GSN-HydroCel-129).**
- VEOG_L = E25 − E127, VEOG_R = E8 − E126, VEOG = (VEOG_L + VEOG_R)/2;
  HEOG = E125 − E128 (right minus left outer canthus).
- Geometric verification precedes any correspondence test: the four periocular pairs
  must verify as most-anterior/superior-vs-inferior (VEOG) and left/right extreme
  (HEOG) in the recording's own chanlocs coordinates. A contradiction STOPS the run for
  a documented re-derivation from coordinates (defect-fix clause: the selection rule
  "standard EGI periocular derivation, geometry-verified" is what is frozen; a label
  correction moves no gate).
- Per-recording exclusion (ITT, counted): any recording whose interpolated/bad-channel
  list (automagic, where present) contains a frozen periocular electrode.
- Filters: VEOG band-pass 0.5–8 Hz for blink morphology; HEOG band-pass 0.5–20 Hz for
  saccade steps. Blink peak = local maximum > 3× MAD(VEOG) with width ≥ 50 ms; saccade
  step = |d(HEOG)/dt| > 3× MAD with the sign of the gaze displacement.

**Gates (frozen; antisaccade recordings are primary — the sealed fight runs there;
dots reported alongside).**
- **G-K3a forward blink correspondence:** fraction of EyeLink blink events with a VEOG
  blink peak inside [event start − 100 ms, event end + 100 ms]; PASS = median across
  recordings ≥ **0.70** (the bar the Tobii panel died on at 0.24).
- **G-K3b reverse correspondence:** fraction of VEOG-detected blinks inside an EyeLink
  blink-or-tracking-gap interval, elevation vs a 200-draw circular-shift null; PASS =
  pooled elevation bootstrap CI-low > 0 (5000 draws).
- **G-K3c saccade correspondence (gates the saccade-typed EOG drive only):** EyeLink
  saccades with horizontal amplitude ≥ 2°; HEOG step within ±50 ms of saccade onset;
  PASS = median match ≥ **0.70**. FAIL → the saccade-typed drive is carried by the
  continuous gaze channels only (dots, Class G) and never by EOG inversion.

**Verdict (frozen).** Instrument VALID = G-K3a AND G-K3b pass on antisaccade. VALID
unlocks the W-stage (true-VEOG A4 row, typed-label κ, event-level oracle, readout
re-bound) and the typed-family fight (F2). NOT VALID → the four WAVE4 dangling items
are re-closed honestly as instrument-limited on THIS panel too, and the F2 fight
proceeds reference-only (no typed claims). Per-recording table reported either way.

**Power statement.** 28 antisaccade recordings (one per subject), 177 dots recordings
(30 subjects). Blink counts from acquisition QC: 4,303 (antisaccade) + 11,108 (dots).
Medians over ≥28 recordings resolve the 0.70 bar far better than WAVE4's n=12.

---

## Execution

Order: K1, K2, K3 submitted concurrently (independent inputs); Slurm partition CPU,
single jobs (QOS submit headroom is occupied by the operator's GPU fleet). Each writes
its decision JSON; results committed before aggregates are discussed; digest appended
at milestone M-B with all three verdicts.
