# IRIS W/T preregistration — dots instrument stage + typed-information kill

Committed BEFORE any W/T execution. Addendum to the K3 verdict (cc98a3d): the
instrument is VALID on dots (177/177; forward 0.982, reverse +0.345, saccade 0.990)
and NOT VALID on antisaccade — so the four dangling instrument items and the
typed-information thesis bet run on DOTS dev. Sealed (antisaccade) untouched; nothing
here enters any sealed-fight asset (charter §5 cross-paradigm rule). CPU only.
Decision JSONs → `results/iris/w/{w1_a4,w2_kappa,w3_readout,t_typed_info}.json`.

Shared frozen conventions: ridge 0.05; bootstrap 5000 draws seed 420; participant-first
aggregation primary (n = 30), recording-level secondary (n = 177, declared
non-independent); ITT — excluded recordings counted with reasons; VEOG/HEOG derivations
and filters exactly as K3 froze them; disjoint-thirds protocol: each recording is cut
into contiguous thirds — FIT = thirds {1,3}, EVAL = third {2}; every oracle/template/
family is fit on FIT and scored on EVAL only.

## W1 — true-VEOG A4 row (replaces the axis-limited WAVE4-M2 bound)

Mirror of WAVE4-M2 with a real vertical reference. Per recording: EOG-referenced
artifact estimate  = ridge regression (0.05) of the frontal block on [VEOG, HEOG] with
lags {−100,−50,0,+50,+100} ms, fit on FIT, applied on EVAL; optical-referenced estimate
= same regression form on [gaze-x, gaze-y, pupil] (L-GAZE-X/Y, L-AREA). Frontal block
(frozen): the 8 most-anterior non-periocular EEG channels by the recording's own
chanlocs X (deterministic). Estimands: D_rms = RMS(est_EOG − est_optical)/RMS(est_EOG)
and D_corr = 1 − corr(est_EOG, est_optical), on EVAL, per recording → participant-first
mean with CI. No pass/fail gate — this is a ledger row: it ENTERS beside (never over)
the banked WAVE4 bound (1.7673 axis-limited), labelled same-axis this time.

## W2 — typed-label κ (the WAVE3-O1 instrument, retried where a referee exists)

EOG-morphology classifier (frozen rule, closed-form): events = union of EyeLink blinks
and saccades on EVAL thirds with EyeLink type hidden; features per event window
[onset−100 ms, end+100 ms]: VEOG peak prominence / HEOG step magnitude ratio ρ and
VEOG half-width; classify BLINK if ρ > 1 and half-width ≥ 50 ms else SACCADE.
Estimand: Cohen's κ vs the EyeLink labels, pooled and participant-first.
**Gate (frozen): κ ≥ 0.60 → typed labels VALID on this panel** (the WAVE3 instrument
died at κ = −0.25); else the type question stays open and is reported so.

## W3 — event-level oracle + readout re-bound (the T3-blocker removal)

Oracle construction (frozen principles): the optical event-level artifact reference is
operator-independent by construction — drive trains are built ONLY from EyeLink events
and gaze/pupil channels (blink train with per-event duration; signed saccade train with
amplitude; gaze-position terms), convolved with per-type templates estimated on FIT
thirds by ridge (0.05) against the frontal block; evaluated on EVAL thirds.
Cross-referee (frozen): the fitted reference must correlate with measured VEOG on EVAL
at r ≥ 0.5, else the recording is excluded-and-counted.
Readout ladder on the oracle target (the WAVE3-T6 families, verbatim):
indicator_linear, rank3_derivative, fir_lagged, amplitude_gain, kernel_ridge —
`relative_gain_vs_incumbent` of the best family vs indicator_linear on EVAL.
**Estimand: participant-first mean relative gain with CI against the 0.03 readout
bound** (WAVE4-M3 sized 0.0620 with CI straddling 0.03 at n=3; here n=30).
Verdict rule (frozen, same as M3): SIZED if point ≥ 0.03, BOUNDED otherwise; the CI
position is reported and governs the wording either way.

## T — typed-information kill (the "rich reference is NEW information" thesis bet)

Per recording, two reference families predict the frontal block on EVAL (fit on FIT,
identical ridge and lag structure to W1):
- STATIC (incumbent class): [VEOG, HEOG] static linear with the W1 lag set — the
  2-channel EOG family F9 declared near-final;
- TYPED: the event-typed drive set {blink train ⊗ template, signed saccade train,
  gaze-x, gaze-y, pupil} with the same lag structure.
Estimand: per-recording relative EVAL-residual reduction
Δ = (resid_STATIC − resid_TYPED)/resid_STATIC on the frontal block, artifact-rich
windows only (top 20% VEOG-energy 1-s windows, the ARTIFACT_RICH=0.20 program
convention); participant-first mean with CI.
**Gate (frozen): CI-low(Δ) > 0 AND mean(Δ) ≥ 0.05** → typed information is REAL →
the typed family proceeds to the F2 fight. CI-low ≤ 0 → F9's family-finality extends
to the rich reference on this panel; the typed-family leg of F2 is DROPPED (fight
becomes reference-only) — a first-class negative.
The 0.05 floor guards against declaring victory on a statistically-real but
practically-empty increment; both numbers report regardless.

Power: 30 participants × up to 6 recordings; the K3 per-recording table showed blinks
and saccades abundant in every recording. No GPU anywhere in this file.
