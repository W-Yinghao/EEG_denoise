# WAVE2-T1 report — the cheap-kill battery

Preregistration: `reports/wave2_preregistration.md`; sealed ledger: `docs/sealed_spend_ledger.md`
(committed first; no sealed byte read anywhere in this wave).

## Shared layer (per-panel, both D readings per the language rule)

| panel | tau2 | W | D_raw/tau2 | D_deb/tau2 |
|---|---|---|---|---|
| mobilebci (fold-mean) | 0.2707 | 1.0858 | 4.00 | 1.99 |
| klados | 0.0219 | 0.0243 | 0.25 | - |
| bci2b | 0.0158 | 0.0042 | 0.24 | - |

## Verdict table

| unit | verdict |
|---|---|
| P2 ledger | **SOFT-FAIL** (lambda_pred 0.50 vs banked 0.89; Sigma_drift downgraded to empirical; rho-hat prediction language stripped from sealed plans) |
| P1 same-block | **mixed attribution** (fraction 0.620 -> 0.660 with drift removed; delta +0.039 [-0.151, +0.200]; neither drift nor prior dominates; both Tier-2 prizes deflate) |
| P7 dose-response | **superlinearity FALSIFIED** (harm ~linear: 0.037/0.100/0.171/0.213; manifold-displacement prediction corrected) |
| drift-widened W4 | **partial repair** (coverage 0.715/0.890/0.923 vs 0.271/0.440/0.516 — 80/90% bands now inside; 50% over-covers; CRPS cost 0.153 -> 0.480) |
| MOKA | M-A GO (0.109) but **M-B NO-GO** (own-vs-pop motion gain -0.001 ERP / -0.016 SSVEP) — family closed at zero GPU |
| OPERA | A0 PASS (leakage 0.055 <= 0.15) but **A1 NO-GO** (no double dissociation: censored U-ratio 1.005, no over-subtraction; ambient over-retains 1.343) — the in-vitro M13R censoring diagnosis did not reproduce; family closed with the diagnostic note |
| DT-Gibbs | **G0 FAIL** (prior-predictive coverage 0.927 > 0.90 — drift prior over-wide, consistent with P2); G1 gated off; semi-blind family closed at its calibration gate |
| THRESH | T0: d_eff 2.67 of 92, harm survives on paper -> T1a full mode; **T1a: no size-dependent transition** (all sizes cross at n=8) — threshold law rejected on this family |

## Portfolio outcome

Zero of four families survive their cheap kills (~30 GPU-h spent of the 250-400 budget).
The flagship is nonetheless strengthened as designed: three mechanism corrections banked
(ledger inconsistency, linear-not-superlinear misalignment harm, drift-widened UQ band
repair with its sharpness cost), and the sealed ledger is intact (no byte read; G2's
rider slot moot). Tier-2 launches: nothing qualifies; the operator's call is writing.
