# IRIS milestones M-G (F1/F4) + campaign close-out

Branch `codex/iris`. All preregistrations committed before execution (5cc8e05,
162fe2e, 4543b37, 4a4d2ea, ea72515, f217fde, F4); all raw results banked
pre-analysis. Total compute: **~0.3 GPU-h** of the 400-h soft cap (everything else
CPU). Job monitoring per operator instruction: squeue only.

## F1 — `results/iris/f1/f1_decision.json` (D1)

```json
"F1_primary": {"contrast": "MATCH_gated - MATCH_NOA0FB, hard-gated cells",
  "paired_mean": 0.061387578646341874, "ci_low": -0.025764490167299967,
  "ci_high": 0.19190456966559077, "n": 4, "verdict": "TIE"}
"F1_anchor": {"mean": 0.14703706115380755, "ci_low": 0.11209677690473957,
  "ci_high": 0.18713747360949737, "positive_count": 15, "preserved": true,
  "banked_incumbent_dev": 0.1428}
"incumbent_anchor_this_run": {"mean": 0.14280149830608732}
"F1_natural_hard_cells": {"low_eog_observation_retention":
  {"MATCH_NOA0FB": 0.9837, "MATCH_gated": 0.9169}}
```

**D1: TIE.** The incumbent survives as IRIS's point system on MobileBCI; the adopted
fallback is linear-class-proven (+0.2537 CI-low +0.0070), retention-improving
(0.984 vs 0.917 where the gate fires), and non-inferior in the diffusion class. The
banked headline (+0.1428) reproduced to four decimals with the bit-identity guard
never firing.

## F4 — `results/iris/f4/f4_decision.json` (D4)

```json
"reproduction_guard": {"gap": 0.00175, "tol": 0.01, "pass": true}
"W-INFL-TEMP": {"coverage": {"0.5": 0.6219, "0.8": 0.8024, "0.9": 0.8530},
  "crps_gaussian": 0.1503, "crps_ratio_vs_sharp": 0.99,
  "risk_coverage_auc": 0.0504, "temperatures": {2.3–2.55},
  "verdicts": all gates PASS, "uq_head": true}
"W-TEMP":      {"0.8": 0.8015, "0.9": 0.8514, "crps": 0.1524 (1.00x), PASS}
"physics_informed_wording_permitted": true
"wording": "operator-posterior width calibrates the bands"
```

**D4: PASSED, stretch demolished.** The mission bar was calibrated 80/90 at < 3.0×
CRPS (prior best 3.1×, drift-widened). The adopted UQ head — operator-posterior
inflation + leave-one-fold-out scalar temperature — calibrates at **0.99×**:
calibration is free in CRPS under this family, because the sharp bands were
underdispersed. The physics wording is earned per the frozen rule (beats pure
temperature 0.1503 < 0.1524 with smaller temperatures 2.3–2.55 vs 2.9–3.2). This is
where the covariance-inflation mechanism, retired twice from point estimates, ends
up paying — exactly the division of labor F8 predicted for diffusion.

## The IRIS synthesis (what the method IS, post-campaign)

- **Points**: the incumbent (exact sub-model, floor) + the NO_A0 fallback on
  hard-gated cells. No inflation, no drift term, no typed drives, no BEM.
- **UQ**: K-chain operator-posterior sampling, inflation-informed width, scalar
  temperature; calibrated 80/90 at ~1× CRPS; RC-AUC not degraded.
- **Validity**: fail-closed abstention, now also spatial (per-channel), with the
  measured exogeneity hazard (E3) as the referee it protects against.
- **Instruments delivered** (dots): true-axis A4 row 0.8955 [0.8417, 0.9524];
  readout re-bound 0.0812 [0.0575, 0.1044] (first CI clear of 0.03); typed-label κ
  0.34 (honest fail); the T/T2 information decomposition (+7% additive, 0 replacement).

## Sealed (D5) — awaiting operator decision

Frozen, quarantined (chmod 000), zero contact, 55 subjects. My recommendation stands:
**do not open** — no current claim clears the bar for a one-shot spend (typed arm
dead by F2-b; incumbent class exogeneity-limited on this corpus by F2-a). The block
loses nothing by staying frozen for paper time. Opening requires your sign-off; if
you elect it, the opening protocol gets its own preregistration first.

## Honest limitations register

- F1's primary reads on n=4 cells (the phenomenon is that small); the TIE is the
  honest verdict, not a power failure — the abstention row was 6/93 cells to begin with.
- F4's CRPS is Gaussian closed-form (declared); the 0.99× ratio is internally
  consistent (same functional both sides); the cross-metric DET-AUC comparison is
  labeled approximate; the 50% band over-covers (0.62).
- F2's E3 hard gate was measured on linear-class arms only; the diffusion class was
  never fielded on EEGEyeNet (no per-panel prior was trained — "showcase never").
- The dots↔antisaccade subject-ID unmappability caveat stands wherever EEGEyeNet
  assets are combined.
