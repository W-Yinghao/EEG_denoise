# IRIS milestone M-B — K-stage verdicts (decision JSONs verbatim)

Branch `codex/iris`. CPU only; 0 GPU-h spent. Raw results banked pre-analysis at
dbbfba4. Preregistrations: charter 5cc8e05, K-stage 162fe2e — every threshold below was
frozen before its data existed.

## K1 — abstention taxonomy: `results/iris/k/k1_abstention_taxonomy.json`

```json
"reconstruction": {"total_mean": 0.1370699264499657, "discrepancy_vs_banked": 0.0,
                   "n_cells": 93, "validity_pass": true}
"taxonomy": {"R_hard_within_outlier": {"cells": 6,  "mass_share": 0.9014},
             "R_active_shrinkage":    {"cells": 87, "mass_share": 0.0986},
             "H_support_floor":  {"cells": 0}, "N_no_reference": {"cells": 0},
             "D_identity_hazard": {"cells": 0}}
"f_conv": 1.0000, "p1_reclamation_bar": 0.30
```

The banked gate-shrinkage row reproduced exactly; every abstention is a within-outlier
hard-gate event at full 120-s support (all six on SSVEP). Nothing structural, nothing
identity-hazardous. The row IRIS's inflation gate attacks is 100% reliability-class —
and the frozen formula therefore sets the P1 bar at the full 0.30.

Prereg description slip, disclosed: the K1 text called the support-floor class "the
10-s hard gate" (echoing the digest's duration-cell language); the deployed constant is
`HARD_GATE_MIN_SECONDS = 60`. The class was defined by cause; no threshold moved; the
class is empty either way.

## K2 — Σ_drift recheck: `results/iris/k/k2_sigma_drift.json`

```json
"banked_g0": {"coverage_mean": 0.9266983695652173, "pass": false}
"raw_sigma_reproduction": {"max_abs_diff_vs_banked_npz": 0.0,
                           "raw_coverage_recomputed": 0.9266983695652173}
"correction": {"removed_variance_fraction": 0.3253,
               "coefficients_zeroed_fraction": 0.4565, "n_cells": 450}
"corrected": {"coverage_mean": 0.6506}
"gate": {"band": [0.70, 0.90], "pass": false,
         "verdict": "drift term OFF — fallback-ladder trigger only"}
```

Debiasing flips the failure mode: too-wide (0.9267) → too-narrow (0.6506). The
query−support displacement is not independent-Gaussian drift once estimation noise is
subtracted. The IRIS inflation gate ships WITHOUT a Σ_drift term; drift stays a
fallback trigger. (Consistent with WAVE2-P1: removing drift converts only +0.039.)

## K3 — EEGEyeNet instrument: `results/iris/k/k3_instrument.json`

```json
"verdict": {"instrument_valid": false,
            "basis": "G-K3a AND G-K3b on antisaccade (sealed-fight panel)",
            "saccade_typed_eog_drive": true}
"antisaccade": {"n_recordings": 28, "n_excluded_bad_periocular": 26, "n_usable": 2,
  "G_K3a_forward": {"median": 0.1627, "bar": 0.70, "pass": false},
  "G_K3b_reverse": {"pooled_elevation": 0.01826, "ci_low": 0.01698, "pass": true},
  "G_K3c_saccade": {"median": 0.9990, "pass": true}}
"dots": {"n_recordings": 177, "n_usable": 177,
  "G_K3a_forward": {"median": 0.9821, "bar": 0.70, "pass": true},
  "G_K3b_reverse": {"pooled_elevation": 0.3449, "ci_low": 0.3191, "pass": true},
  "G_K3c_saccade": {"median": 0.9904, "pass": true}}
```

Three findings, ranked by consequence:

1. **The antisaccade "minimal" release largely lacks a RECORDED periocular axis**:
   automagic interpolated ≥1 frozen periocular electrode in 26/28 recordings (E25
   23/28, E8 20/28, E127 20/28, E126 17/28; canthi mostly intact: E125 3/28, E128
   9/28). The dev rate (~93%) is the best available estimate for the sealed block.
2. **Synchronization is perfect; the blink layer is not**: on the 2 intact recordings,
   saccade↔HEOG match 1.000/0.998 with sign consistency 0.98/0.99 — but forward blink
   match only 0.176/0.149 (34/47 task-suppressed blinks against 310/319 VEOG peaks;
   reverse elevation +0.017, 8× its null but tiny in absolute terms). The failure is a
   property of antisaccade blink physiology/annotation, not of the clock.
3. **Dots is the instrument-grade panel**: 177/177 usable, forward 0.982, reverse
   +0.345, saccade 0.990 — the exact gates the Tobii panel failed at 0.24 pass with
   huge margins, on the panel that also carries continuous gaze.

## Sealed status

55/55 subjects fetched to quarantine (14.64 GiB, 0 failures, 0 bad headers), tree at
mode 000. Zero analysis contact. `results/iris/sealed/sealed_fetch_report.json`.

## Consequences now frozen into the campaign

- **P1 (inflation-gate pilot)** proceeds at the FULL bar: reclamation ≥ 0.30 on
  abstained cells, per-cell never-worse-than-NO_A0 (ε 0.005), wrong-donor harm
  ≤ +0.005. No drift term (K2). Prereg: `reports/iris_prereg_p1.md`.
- **W-stage moves to dots** by preregistered addendum (`reports/iris_prereg_w.md`):
  true-VEOG A4 row, typed-label κ, event-level oracle, readout re-bound — all four run
  on the panel where the instrument is valid. The typed-family fight (F2) runs on dots.
- **Sealed-fight design constraint recorded**: on antisaccade, the saccade-typed
  EyeLink drive is validated (0.999); the blink axis has no clean VEOG referee in ~93%
  of recordings. The sealed fight's evaluation instruments must not assume one.
- Budget: 0 GPU-h spent of 400 (`results/iris/budget.json`).
