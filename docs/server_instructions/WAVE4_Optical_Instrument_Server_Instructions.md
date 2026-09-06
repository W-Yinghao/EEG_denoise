# WAVE4 — The Optical Instrument
### Server execution instructions (Claude Code) — Eye-BCI Tobii activation: one instrument, four customers

WAVE3 closed the residual ledger to 0.9% unattributed. Four dangling items remain, and all
four point at one missing instrument — an OPERATOR-INDEPENDENT artifact reference:

```text
customer 1: the READOUT ledger row (currently unbounded — T3 non-adjudicating because
            both available oracles are degenerate/tautological)
customer 2: the A4 reference-channel-error row (EOG measurement noise + neural
            crosstalk — the exogeneity assumption under every e-regressing arm)
customer 3: valid artifact-type labels (the EOG-morphology classifier failed at
            κ = −0.25; O1's type question is open for lack of an instrument)
customer 4: a non-degenerate oracle for any future family/readout adjudication
```

The instrument: the Eye-BCI Tobii eye-tracking modality — registered on Synapse
(syn64005218, PAT authentication previously worked), NEVER downloaded (only the 315
Neuroscan EEG CSVs are local). This wave is measurement-only: it ADDS ledger rows and
bounds; it never edits a WAVE3 number, never deploys anything, never touches sealed
assets, and does not by itself resurrect any gated route (TROCA/Panel-T eligibility is a
separate operator decision after M1's verdict).

---

# 1. Workspace

```text
base    : codex/wave3 tip ec9ff37
branch  : codex/wave4-optical
compute : CPU only (network + disk are the real costs; no GPU anywhere in this wave)
data    : existing Eye-BCI EEG at the path recorded in datasets/registry/eye_bci.json;
          data root /projects/EEG-foundation-model (CHECK current free space first)
```

# 2. Preregistration (`reports/wave4_preregistration.md`, commit BEFORE any Synapse call)

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

# 3. Execution order and stop points

```text
STOP 0 : commit prereg → E0 manifest query → report manifest + decision-rule verdict.
         If PROCEED fires, continue without waiting; else stop for the operator.
E0b    : sharded resumable download → atomic publish → registry update
         (datasets/registry/eye_bci_tobii.json: paths, bytes, coverage, job IDs).
STOP 1 : E1 pilot (3 subjects) alignment verdict → if ≥2/3 pass, extend to all covered
         subjects; report the alignment table (matched-blink rates, drift residuals,
         exclusions).
E2     : M1 → M2 → M3 → M4 (M1's κ gate first — it is also the cheapest kill:
         κ < 0.8 on the optical labels would leave O1 permanently inconclusive and
         downgrade M2/M3 to blink-only variants).
STOP 2 : final report — the four measurements verbatim, the updated ledger addendum
         (rows added, WAVE3 numbers untouched), and the A1 verdict. Commit, push, stop.
```

Deliverables: `reports/wave4_{preregistration, e0_manifest, e1_alignment, report}.md`,
`results/wave4_optical/{manifest, alignment, m1..m4}/`, registry entry, one short
ledger section.

# 4. Kickoff prompt

```text
Read WAVE4_Optical_Instrument_Server_Instructions.md in full and execute it. Create
branch codex/wave4-optical from codex/wave3 tip ec9ff37. Commit the preregistration
FIRST (it freezes the download decision rule, alignment gates, all four measurement
protocols, the cross-panel caveat, and the S27–S31 exclusion). Run E0 as a
manifest-only Synapse query and apply the frozen download rule — no blind download; if
PROCEED fires, download sharded/resumable/atomic and continue through E1 alignment
(pilot 3 subjects, then all covered) and the E2 measurements M1–M4 in order; write the
three stop-point reports; commit, push, stop at each declared stop point. CPU only;
Slurm for anything heavy; zero sealed contact; WAVE3 numbers are read-only.
```
