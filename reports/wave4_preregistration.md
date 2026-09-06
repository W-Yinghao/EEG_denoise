# WAVE4 preregistration — the optical instrument

Frozen before ANY Synapse call. Base: `codex/wave3` tip ec9ff37. CPU only; zero sealed
contact; WAVE3 numbers are read-only (rows are ADDED, never edited).

## Part 1 — frozen rulings (verbatim)

```text
E0 DOWNLOAD DECISION RULE (frozen; no blind download):
  Query the Synapse manifest ONLY (PAT auth, no file content): enumerate Tobii files +
  the E-Prime logs (needed for cross-stream alignment; they are small — include them),
  per-subject coverage, bytes, formats, sampling rates.
  PROCEED to download iff ALL of:
    (a) Tobii covers ≥ 10 subjects that also have local EEG;
    (b) total download ≤ 300 GiB;
    (c) current data-root free space ≥ 2× the download size.
  Else STOP and report the manifest verbatim (the operator decides).
  Download = resumable, sharded, published atomically (the 919-series pattern);
  E-Prime logs always downloaded (tiny) if present.

E1 ALIGNMENT GATES (per subject; 3 pilot subjects first, then all covered):
  clock model = linear drift fit on shared events (E-Prime triggers primary;
  blink-onset cross-matching fallback);
  gate: ≥ 80% of Tobii-detected blinks match a VEOG deflection within |lag| ≤ 50 ms
  after the drift fit, drift-fit residual ≤ 20 ms RMS.
  Subjects failing the gate are EXCLUDED-AND-COUNTED (ITT discipline), never repaired
  ad hoc.

E2 MEASUREMENTS (all preregistered, all CPU, all on aligned subjects):
  M1 TYPE LABELS: optical blink/saccade/fixation labeling (Tobii validity flags +
     gaze-velocity thresholds, thresholds frozen here). κ vs hand labels on a
     ≥200-event subsample; instrument valid iff κ ≥ 0.8. If valid: rerun the WAVE3 T1
     census on Eye-BCI natural EEG with the optical ψ under the SAME unified gate
     (TSR ≥ 2, CI-low > 1.3, ≥2/3 of covered subjects, mixture non-degeneracy ≥15%).
     Verdict adjudicates A1 (typed-operator separation) — measurement only; any
     Panel-T/TROCA consequence is a separate operator decision.
  M2 A4 ROW: on artifact-rich aligned windows, compare EOG-referenced vs
     optical-referenced artifact regressions; the discrepancy (with CI) bounds
     reference-channel error (EOG noise + crosstalk). Ledger row added with the
     cross-panel caveat below.
  M3 READOUT BOUND (valid T3): construct the optical artifact reference (EEG regressed
     on optical gaze/blink regressors, fit on disjoint segments); evaluate banked-style
     LINEAR vs DIFF-class residuals against this operator-independent reference on
     matched windows; |residual_DIFF − residual_LIN| ≤ 0.03 bounds the readout row;
     larger values SIZE it. (If no V44-class checkpoint applies to the Eye-BCI montage,
     run the LINEAR-vs-analytic-readout version and label the diffusion readout bound
     as not-measurable-this-panel — do not port checkpoints across montages.)
  M4 EXOGENEITY: during optical-confirmed fixation (no saccade/blink), the residual
     EOG-channel variance that correlates with posterior-EEG activity = the neural
     crosstalk estimate; report as the direct A0-exogeneity measurement (compare
     OPERA's leakage R² 0.055).

CROSS-PANEL CAVEAT (frozen wording): M2/M3/M4 are measured on Eye-BCI and enter the
MobileBCI ledger as order-of-magnitude BOUNDS with an explicit cross-panel label,
unless a comparability check (montage/reference/task overlap) is separately passed.

EXPOSURE NOTE: the old exposure audit flagged Eye-BCI S27–S31 as
validation-role-declared; keep those subjects out of all E2 measurements (counted,
excluded) so the panel stays claim-clean.

PROHIBITIONS: no GPU; no deployment arms; no edits to WAVE3 ledger numbers (rows are
ADDED); no sealed contact (MobileBCI-8 spent; BrainID Day-200 / PhysioMotion-10 /
SHU untouched); no checkpoint porting across montages; no writing-round text.
```

## Part 2 — registered operational definitions (frozen here, before any Synapse call)

**Local panel facts established before this commit** (read-only inspection of already-local
data, no Synapse call): Eye-BCI Neuroscan CSVs at
`/projects/EEG-foundation-model/eye_bci/syn64005218-neuroscan`, subjects S01–S31, 315
files, 196.6 GB, 1000 Hz, columns = 62 scalp channels + `HEO`, `Trig`, `Cues`,
`PhanFrame`, `PhanTime`, `RelTime`, `RecordingTimestamp`, `LocalTimeStamp`, `Blinks`
(10 files in S03/Sess01 and S06/Sess01 lack the timestamp pair; P3004L101 and ME181 lack
the phantom fields). Data-root free space at commit time: **875 GiB**, so rule (c)
admits a download up to 437 GiB and rule (b)'s 300 GiB is the binding constraint.

**E0 mechanics.** `synapseclient` is absent from every environment; the manifest query
uses the raw Synapse REST API with the existing PAT in `~/.synapseConfig` (the 919-series
pattern). The manifest query reads entity metadata ONLY — no file content, no download —
and is recorded verbatim before the rule is applied. Download, if it fires, writes to
`/projects/EEG-foundation-model/eye_bci/` under a staging directory and is published
atomically by rename.

**VEOG surrogate (E1).** This panel has `HEO` (horizontal EOG) but no vertical EOG
channel. The registered vertical ocular reference is
`VEOG_surrogate = mean(FP1, FPZ, FP2)` band-passed 0.5–8 Hz (4th-order Butterworth,
zero-phase) — the standard frontal blink proxy. Every E1/E2 statement that names "VEOG"
means this surrogate on this panel.

**Clock model (E1).** Primary: if both streams carry `RecordingTimestamp` on a shared
clock, the linear drift model is fit on those timestamps and the gate is applied
unchanged. Secondary: E-Prime trigger events. Fallback: blink-onset cross-matching. The
path actually used is reported per subject.

**M1 optical labeling thresholds (frozen, I-VT family).** Blink = both-eye validity
invalid for a contiguous run of 50–500 ms flanked by ≥50 ms of valid samples. Saccade =
gaze velocity ≥ 30 °/s for ≥ 20 ms with both eyes valid. Fixation = velocity < 30 °/s for
≥ 100 ms with both eyes valid. Events shorter than the minimum durations are unlabeled
and counted.

**M1 κ reference — registered deviation.** No human labeler exists in this pipeline, so
"hand labels" are replaced by an INDEPENDENT-INSTRUMENT reference on the same ≥200-event
subsample: a dispersion-based I-DT classifier (dispersion threshold 1.0°, window 100 ms)
plus explicit pupil-diameter dropout detection, i.e. a different algorithm family on the
same raw gaze stream. κ is therefore instrument-vs-instrument agreement, exactly as in the
WAVE3 κ protocol. This is a declared deviation from the verbatim wording, frozen here; the
κ ≥ 0.8 bar is unchanged.

**M3 scope declaration.** No V44-class checkpoint applies to the 62-channel Neuroscan
montage, and porting checkpoints across montages is prohibited. M3 therefore runs the
LINEAR-vs-analytic-readout version, and the diffusion readout bound is reported as
NOT-MEASURABLE-THIS-PANEL.

**M4 statistic.** During optical-confirmed fixation, regress `HEO` on the posterior-EEG
block (P7 P5 P3 P1 PZ P2 P4 P6 P8 PO7 PO5 PO3 POZ PO4 PO6 PO8 O1 OZ O2 CB1 CB2) with the
same ridge convention as the program (ratio 0.05); the reported crosstalk statistic is the
in-sample R² with a 2-fold cross-validated companion, compared against OPERA's 0.055.

**Statistics.** Participant-first over aligned, non-excluded subjects; 5000-draw bootstrap;
Holm within each measurement family. S27–S31 excluded and counted everywhere in E2.
