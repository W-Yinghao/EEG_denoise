# IRIS F4 preregistration — UQ adjudication (D4): calibrated bands at bounded CRPS cost

Committed BEFORE execution. The banked W4 state (read-only): K-chain beats the
DET-ensemble on CRPS (0.1529 vs 0.1548) and RC-AUC (0.0920 vs 0.1129) but bands fail
(0.271/0.440/0.516 vs 0.5/0.8/0.9); the drift-widened variant calibrates 80/90 at
3.1× CRPS. Target frozen by the mission brief: calibrated 80/90 at < 3.0× CRPS
(stretch 1.5× reported). The inflation mechanism's surviving role (P1/P1R) is width,
adjudicated here.

## Machinery (frozen)

Regenerate the M13-W4 K-chain ensembles BIT-IDENTICALLY (the frozen chain seeds
910000 + fold·1000 + seed%100 + chain·17 and DDIM seeds base+31·(chain+1); K = the
banked k_chains; same episode banks). Per fold×seed store per-sample |error|,
chain σ, and the operator-anchor variance projection Var_op,i(t) = Σ_j V_ij ψ_j(t)²
(U0-b posterior V; first-order anchor→output sensitivity 1, declared approximation)
to `/projects/EEG-foundation-model/derived/denoiseNet/iris_f4/` (npz, float16);
JSON summaries in-repo. DET reference numbers are consumed from the banked W4
decision verbatim (never regenerated, never edited).

## Width policies (frozen)

| Policy | Predictive σ' | Temperature |
| --- | --- | --- |
| W-SHARP | σ_chain | none (reproduction guard: must reproduce the banked coverage/CRPS at tolerance 0.01) |
| W-TEMP | s·σ_chain | scalar s* per evaluation fold, chosen by leave-one-fold-out: the smallest s on a [0.5, 6.0] grid (step 0.05) with calibration 80% coverage ≥ 0.80 |
| W-INFL | sqrt(σ_chain² + Var_op) | none — tests whether physics-informed width alone calibrates |
| W-INFL-TEMP | s·sqrt(σ_chain² + Var_op) | leave-one-fold-out scalar as W-TEMP |

Coverage: exact per-sample Gaussian bands (z·σ'). CRPS: Gaussian closed form from
(error, σ') per sample (declared; the s=1 empirical-vs-Gaussian gap is reported once
as context). All holdout evaluation is on the left-out fold's episodes only, pooled.

## Gates (frozen)

- **G-F4-cal**: holdout 80% AND 90% coverage within ±0.05 of nominal.
- **G-F4-cost**: holdout CRPS ≤ 3.0× the W-SHARP holdout CRPS (stretch 1.5× reported).
- **G-F4-rank**: holdout risk-coverage AUC ≤ the banked DET reference 0.1129.
- **Adjudication**: IRIS's UQ head = the passing policy with the lowest CRPS. The
  physics-informed wording ("operator-posterior width calibrates the bands") is
  permitted ONLY if a W-INFL* policy passes AND beats W-TEMP on holdout CRPS;
  otherwise the honest wording is "calibration, not physics, fixes the bands."
- 50% coverage, conformal-wrapper companion, and per-participant tables reported
  regardless; no gate reads on them.

Budget: ≤ 20 GPU-h (single job, per-fold-seed resume). Decision JSON →
`results/iris/f4/f4_decision.json`.
