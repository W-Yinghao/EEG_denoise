# WAVE3 Tier-0 preregistration (single merged document)

Frozen before ANY Wave-3 computation. Base: `codex/wave2-t1` tip 755ebe4.

## Part 1 — the frozen rulings (verbatim)

```text
(i) CORRECTED RESIDUAL SEMANTICS (program-wide, immediate): the oracle-residual is
  reported in ADDITIVE semantics everywhere, with the total-semantics reading given
  alongside once; the W3 conversion fractions restate as 0.29–0.50 band; no document
  cites the old 0.50–0.66 or mixed-semantics numbers again.

(ii) UNITS-vs-COMPOSITION DECISION TREE for O4/P2 (the 0.134 degeneracy fix):
  STEP 1 = B0 CODE READ FIRST: read eb_transfer_v43.py and the wave-2 shared-layer
  script; establish the literal λ estimator (the λ = τ²/(τ²+within/4) form is an
  INFERENCE — its sole support is the 0.4993 coincidence — label it
  inference-pending-B0 until the code says otherwise) and whether the gate's "within"
  is per-window variance or block-mean variance.
  STEP 2 = the T7 2×2 FACTORIAL {split granularity: window-random vs block-level} ×
  {stratification: none vs type-stratified}, both accounts' DISTINCT predictions frozen:
    units account:       W(block-level, unstratified) ≈ 0.134 AND
                         W(window, stratified) ≈ W(window, random) ≈ 1.09
    composition account: W(window, stratified) collapses toward 0.134
  The factorial, not any 0.134 landing, is the discriminator. If units wins: P2
  dissolves as a units lesson, O4 restates per-window (mundane candidate: 92 ridge
  parameters per ~3.7-s window), the parsimony narrative is dropped, and A1 survives
  only as the natural-data ceiling question. Composition-term closure check: W_random −
  W_strat reproduced within ×2 by mixture-fraction variance × T1's measured separation.

(iii) UNIFIED TYPE GATE (one gate for the one crux): natural-data T1 census gate =
  TSR ≥ 2 AND bootstrap CI-low > 1.3 AND ≥10/15 subjects AND classifier κ ≥ 0.8 AND
  mixture non-degeneracy (both types ≥15%). The ≥8/15 variant is labeled sensitivity
  only. κ < 0.8 → INCONCLUSIVE (not NO-GO).

(iv) INVALID-INSTRUMENT RULINGS (from the hostile audit, binding):
  T1-pre (banked per-window residual typing) is DESCRIPTIVE PLUMBING ONLY, |e|²-
  normalized, NO kill authority — on single-injection semi-sim pairs type-independence
  holds by construction and spurious type-structure appears via |e|-scaling.
  The S0b query-fitted typed-vs-pooled RRMSE probe on banked pairs is DELETED for the
  same reason. The single RRMSE-unit oracle instrument is Panel-T (below), built once,
  jointly owned, attribution-only.

(v) CLOSURE LEDGER (binding rulings): type-indexed STATIC operators are LEGAL
  (indicator-expansion identity: type-indexed 46×2 ≡ static 46×4 on [e·1_blink;
  e·1_sacc] — same legality class as the confirmed +0.143 channel). RED LINES:
  ψ (the type classifier) thresholds frozen GLOBALLY in this prereg, never tuned
  per-subject or on query data; starved/abstained types degrade bit-identically to the
  incumbent; no output of any diagnostic oracle (T2, Panel-T) converts into a deployed
  within-query estimator this wave; a T7/Pack-B explanation of the DT-Gibbs G0 death
  does NOT resurrect the semi-blind family (new registered design required); ONCE
  reopens nothing (U-1 was a factorial verdict on naive stacking; the naive stack is
  retained as comparator); M-B does not bar ocular type-resolution.

(vi) DISJOINT LEDGER-ROW DEFINITIONS (F6): the residual ledger rows {delivered,
  bookkeeping, gate-shrinkage, estimation-noise, readout, family, fluctuation, drift,
  unattributed-remainder} are defined as a disjoint decomposition with stated
  covariances (B2 and B4 both derive from λ̂ — partition, do not double-count;
  B3 overlaps A1's ceiling term — assign by the frozen rule).

(vii) ONCE noise-corrected decision rule (F5): the Stage-0 primary statistic is the
  deficit decomposition (≥60% of the joint-vs-best deficit along the shared subtraction
  span WITH excess removal ⇒ bookkeeping conviction; orthogonal-variance-dominated ⇒
  noise conviction) — no "impossibility" rhetoric; compute a CI on the banked −0.0072
  before asserting over-subtraction.

(viii) A4 REFERENCE-CHANNEL ERROR is added to the attribution tree (EOG measurement
  noise + neural crosstalk caps every e-regressing arm) with a ledger row; its clean
  instrument (Eye-BCI optical-vs-EOG) is priced but deferred.

(ix) Pack-A hard cap 25 GPU-h; A2' sandbox runs ONLY if A1' inconclusive AND ≥1 density
  fingerprint fired at A0. TROCA S2 (Eye-BCI) deferred out of wave 3; S3 retraining
  unfunded. Statistics everywhere: participant-first n=15, 5000-draw bootstrap,
  sign-flip, Holm within families; C05/C08 wording; PV-2/q99 guards on any new arm.
```

## Part 2 — registered operational definitions (frozen here, before computation)

**B0 code read (STEP 1 of the decision tree).** Deliverables: (a) the literal λ
estimator expression and the literal `within` estimator, quoted from
`src/eeg_scad/data/eb_transfer_v43.py`; (b) the literal wave-2 shared-layer estimator
from `src/eeg_chart/run_wave2.py`; (c) the DEPLOYED per-cell (τ², within, λ̂) triples
read from the banked V43 manifests `results/rgcc_v43/state/fold_*/eb_state_manifest_s2.csv`
(descriptive read of banked artifacts); (d) an explicit reconciliation of any estimator
difference between (a) and (b), decomposed term by term. The λ-form label
"inference-pending-B0" is lifted only by (a).

**T7 2×2 factorial (STEP 2).** For every dev cell the artifact operator is refit on the
120-s support under four estimators; W is the mean squared deviation of the four
sub-estimates around the cell's full-prefix fit (the deployed within-form), reported per
cell and pooled:
- *block-level*: 4 contiguous 30-s blocks (the deployed split).
- *window-random*: the support is cut into 512-sample windows; windows are assigned to 4
  groups uniformly at random (seed 20269301), each group refit.
- *unstratified*: all windows/blocks used.
- *type-stratified*: windows are labeled by ψ (below) and each of the 4 groups is drawn
  to hold a FIXED type mixture equal to the cell's global mixture (stratified sampling
  removes mixture-fraction fluctuation between groups).
Composition-term closure check as written in (ii).

**ψ, the frozen type classifier (global thresholds, never tuned).** Input: the VEOG
column of the V41R bipolar latent, 100 Hz, 1.0-s windows (100 samples), robust-z scaled
by the support median/MAD. Per window compute: `peak` = max |z|; `step` =
|mean(z[last 30]) − mean(z[first 30])| / max(peak, 1e-9); `width` = fraction of samples
with |z| ≥ 0.5·peak. Labels: **blink** iff peak ≥ 3.0 AND step ≤ 0.40 AND width ≤ 0.35;
**move** (vertical EM/saccade) iff peak ≥ 2.0 AND step ≥ 0.60; **unclassified**
otherwise. Unclassified windows are excluded from typed fits and counted.

**κ instrument (no labels exist).** κ = Cohen's κ between ψ and an INDEPENDENT
unsupervised reference instrument on the same windows: a 2-component Gaussian mixture
fitted per subject on a DISJOINT feature set (log-power in 0.5–3 Hz and 3–8 Hz bands of
the VEOG derivative, plus zero-crossing rate), with components mapped to
{blink, move} by which has the higher 0.5–3 Hz power share. κ ≥ 0.8 required; κ < 0.8 →
INCONCLUSIVE per (iii).

**TSR.** Per subject: `sep²` = mean squared difference between the blink-typed and
move-typed 46×2 operator fits; `W_type` = mean of the two within-type 2-fold refit
scatters. TSR = sep²/W_type. Gate per (iii); bootstrap over subjects.

**T2 nested variance decomposition.** Operator-entry variance partitioned as
between-subject / between-type-within-subject / within-type (residual), by nested
ANOVA-style moment matching on the typed fits.

**T6 nested family ladder (natural queries, CPU).** Families in order:
(1) indicator-linear = the incumbent static 46×2; (2) rank-3 = 46×3 with a third
regressor from the VEOG derivative; (3) FIR-lagged = 46×2 with lags {−2,−1,0,1,2};
(4) amplitude-gain = 46×2 with a per-window scalar gain fit on the support-side
statistic; (5) kernel ridge = RBF ridge on the 2-D drive (support-fit, applied causally).
Metrics: 2-fold CV residual variance on the SUPPORT side (no query fitting), plus
natural EOG-band attenuation and low-EOG retention guards (retention ≥ 0.75 required for
any family to be quotable). Adjudication: the ladder is a family-misspecification
adjudicator only; no deployment claim.

**T4 exact gate-shrinkage recompute.** Closed-form: for each cell the shrinkage loss is
‖(1−λ)(C_own − C_pop)‖² in operator units, converted to the RRMSE scale by the measured
local slope; compare shrink-to-pop (deployed) against shrink-to-zero (NO_A0-consistent).

**T3 oracle-by-readout.** The semi-sim comparison is STRUCTURALLY DEGENERATE (the
query-fitted operator reproduces the injected artifact exactly, so the LINEAR oracle
residual is 0 by construction) and is reported as non-adjudicating. The adjudicating
version runs on NATURAL windows with the attenuation-based residual: for the DIFF
readout and the LINEAR readout separately, residual = attenuation(oracle operator) −
attenuation(gated operator) in dB; |residual_DIFF − residual_LIN| ≤ 0.03 (in the RRMSE-
equivalent scale via the measured local slope) exonerates the sampler.

**ONCE Stage 0/1/2.** Stage 0: CI on the banked bci2b joint−best (−0.0072) by bootstrap
over records; deficit decomposition = project the joint correction onto the span shared
by the two single-leg corrections; ≥60% of the deficit along that span, WITH the excess
(over-subtraction) removed, ⇒ bookkeeping conviction; otherwise noise conviction.
Stage 1: the projection identity is derived in-house and MUST retrodict both the banked
additivity 0.596 and joint−best −0.0072 (within ±0.05 and ±0.005 respectively) before
Stage 2 runs. Stage 2: orthogonalized composite = anchor correction projected onto the
complement of the transport correction's span, applied on banked windows; hard gates as
written in WP-C.

**Pack-A A0 entry gate.** U-ratio = ‖U°ᵀ x̂‖ / ‖U°ᵀ x_true‖ on dev paired episodes with
the banked M13R P0 checkpoint; gate = mean < 0.95 with bootstrap CI excluding 1. STOP on
fail (finding 7 softens to a single-run observation). A0 CPU fingerprints:
(a) masking-frequency correlation = corr(selection statistic, U° energy) over candidate
windows; (b) amplitude ladder = U° energy quantiles of selected vs unselected windows;
(c) prevalence tables of ψ types in the selected vs unselected sets.
A1' guidance-weight sweep: the analytic likelihood step's weight δ swept over
{1e-8, 1e-4, 1e-2, 1e-1, 1} on the frozen P0 checkpoint; C4/guidance confirmed iff the
U-ratio deficit is monotone in δ and vanishes at the small-δ end.

**Disjoint ledger rows (F6 partition rule).** Total = NO_A0-referenced oracle span per
panel. Rows, in this fixed order, each computed on the residual left by the previous:
1 `delivered` (measured deployed gain), 2 `bookkeeping` (ONCE deficit attributable to
double-subtraction), 3 `gate-shrinkage` (T4, the λ̂-derived shrinkage loss —
**partition rule**: the λ̂-derived quantity is assigned ENTIRELY to row 3; row 4 takes
only the residual estimation noise not already in row 3), 4 `estimation-noise`
(support-fit sampling error net of row 3), 5 `readout` (T3), 6 `family` (T6 ladder gain;
**overlap rule**: where T6's family gain overlaps A1's ceiling term, it is assigned to
row 6 and A1's ceiling is reported net of it), 7 `fluctuation` (per-window realization
variability), 8 `drift` (support→query, Σ_drift-derived), 9 `unattributed-remainder`
(closes the identity by construction). A4 reference-channel error is reported as an
annotation on rows 4–7 (priced, instrument deferred), not a separate subtracted row.

**Caps and stops.** Pack-A ≤ 25 GPU-h; Panel-T ≤ 10; TROCA S1 15–30; wave total ≤ 100
with re-prioritization on overrun. Stop points: after DP1, after DP2, after ledger
assembly. No sealed contact anywhere in Wave-3.
