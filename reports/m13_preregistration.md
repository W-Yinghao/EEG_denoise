# FLAGSHIP-M13 preregistration

```text
FLAGSHIP-M13 preregistration — frozen before first submission.

W1 TRANSPORT REPAIR RULES (engineering, no outcome-tuning):
  whitening = rank-truncated + Ledoit-Wolf: keep the top-q eigendirections of the
  shrunk canonical covariance with q chosen by a FROZEN rule (captured variance 0.99,
  capped so kappa(T_s) <= 100); below-cap directions pass through unwhitened.
  Frame-concentration bar RE-DIAGNOSED, not lowered: report split-half frame agreement
  per subject; subjects whose disagreement exceeds the cohort between-subject spread are
  ABSTAIN (rho := 0), counted, never dropped. The 15-degree/80% bar stays as the
  aspiration; the deployment rule is the abstention rule.
  After repair: re-run the U1-a analytic probes on all three panels (descriptive;
  the M0 GO map is not revised — a newly-stable MobileBCI transport row is reported
  as post-repair evidence, labeled as such).

W2 PRIOR VALIDITY GATES (per evaluation panel, participant/record-first):
  PV-1: canonical-space posterior POP route beats RAW on the panel's paired metric
        (temporal RRMSE), CI-low > 0.
  PV-2: output/input RMS q99 in [0.90, 1.10] (no amplitude collapse/inflation).
  PV-3: pooled prior >= per-dataset prior on >= 2 of 3 paired panels (non-inferiority
        margin 0.005) — the pooling-does-not-hurt gate.
  LODO gate: prior trained with dataset D held out, evaluated on D: must still pass
        PV-1/PV-2 on D (transfer-of-prior claim; this is the headline axis).
  Training-data rule: paired panels contribute clean carriers; truth-free corpora
  (SGEYESUB, SHU, OpenBMI, PhysioMotion) contribute LOW-ARTIFACT-WINDOW selections
  under the frozen repo criterion (primary); an ambient-loss arm is a preregistered
  secondary on one seed only. EEGdenoiseNet contributes single-channel segments lifted
  through a 1-channel montage mask. Eye-BCI deferred (not in this stage).

W3 TRANSPORT GAIN ENDPOINTS (GO panels only: Klados, BCI2b; MobileBCI descriptive
  post-repair):
  TG-1 (primary, per panel): T-MATCH − T-POP with the TRAINED prior, mean > 0 with
        CI-low > 0. TOST band ±0.005 preregistered as the equivalence alternative.
  TG-2 controls: T-WRONG gated ≈ T-POP (within +0.005); GAUGE-NULL not better than
        T-POP; T-ORACLE reported as residual headroom.
  TG-3 positioning (descriptive, C05): DIFF vs DET1 / DET-ITER (50-step unrolled
        deterministic, no injected noise) / LINEAR twins on identical backbones and
        common noise. No superiority wording in either direction.

W4 UQ ENDPOINTS (on V44-S1 checkpoints, EOG-guided class, MobileBCI):
  UQ-1: operator-posterior K-chain sampling (each chain draws C ~ EB posterior from
        U0-b; K = 8 primary / 32 secondary, weighted per the registered particle scheme).
        Empirical 50/80/90% interval coverage must land in [0.35,0.65]/[0.65,0.90]/
        [0.80,0.97] respectively (vs V37T's 0.0029 — the bar is "materially dispersed
        and honest", not perfection).
  UQ-2: CRPS and risk-coverage AUC adjudicated against the 3-seed DET-ensemble
        reference (the bar that beat diffusion before). Report win/lose honestly.
  UQ-3: conformal recalibration preregistered as a DOWNGRADE (reported as
        "conformalized", never as native posterior calibration).

LIKELIHOOD-CEILING WORDING RULE: the U1-b oracle ceilings (+0.26–1.77) are degenerate
  (oracle operator near-exact on generated pairs) and may not ground any claim; the
  likelihood channel's claimable numbers come from V44's deployable arms only.

Statistics: 5000-draw bootstrap, Holm within each family {TG-1×2 panels}, {UQ-1..2}.
No threshold changes after this commit. Sealed cohorts untouched.
```

Registered implementation notes (fixed at commit time, before any submission):
- OpenBMI is not present on this cluster's corpus root; the truth-free pool is
  SGEYESUB + SHU (sessions 1-3) + PhysioMotion (development subjects). Recorded as a
  data-availability deviation, not a rule change.
- The W4 "registered particle scheme" is fixed here as EQUAL-WEIGHT chains (uniform
  mixture of the K chain outputs; intervals and CRPS from the pooled empirical ensemble).
- The V44 DET-ensemble reference has 2 seeds per fold (the V44-S1 DET twin set);
  reported as a 2-member ensemble deviation from the 3-seed wording.

# FLAGSHIP-M13R addendum

```text
FLAGSHIP-M13R addendum — frozen before first submission.

HONESTY RULE (immediate, unconditional): P0's Klados/BCI2b PV-1 utilities (+0.164, +0.401)
  are amplitude-shrinkage-confounded (q99 0.812/0.457) and may not be cited as denoising
  gains anywhere. PV-2 is reported as having performed its designed function.

R-A CANONICAL ARTIFACT-SUBSPACE READOUT (the amplitude fix, P3-precedented):
  the posterior updates ONLY the ocular canonical coordinates: correction c ∈ span(U°)
  (rank 3); every other canonical coordinate is pinned to the observation (hard data
  consistency). Sensor readout x̂ = y − T(ρ)^+ (U° û). Complement identity holds by
  construction: outside the sensor image of span(U°), x̂ = y bit-exactly. Expected
  q99 ≈ 1 structurally. The diffusion prior's role: the clean-EEG prior constrains the
  rank-3 coefficient trajectory via the posterior step (the P3 design in canonical
  coordinates). Training loss unchanged; only the sampler/readout change.

R-B PER-PANEL TRANSPORT CONFIG (frozen from the W1 diagnosis, not tunable):
  heterogeneity_dominated panels (MobileBCI, BCI2b): covariance alignment OFF (G = I);
  transport = ocular Procrustes ∘ montage lift only, with the split-half abstention rule.
  estimation_noise_dominated panel (Klados): G active for non-abstained subjects only.
  Rank-truncated whitening stays undeployed (failed its own target).

REPAIR GATES: PV-1 and PV-2 per panel, margins unchanged. PASS = both gates on ≥ 2 of 3
  panels including at least one cross-montage panel. FAIL → the pooled-prior axis is
  CLOSED as an honest negative with the two-mode diagnosis; no further repairs; the
  flagship descopes to {matrix + V43/V44 legs + UQ + per-panel/analytic transport rows}.

W3 TRANSPORT FACTORIAL (decoupled; runs regardless of the repair outcome):
  primary backbone = the validated analytic canonical cleaner; panels = Klados, BCI2b
  (GO panels); arms per the M13 prereg (T-POP / T-MATCH(ρ̂) / T-WRONG(gated) / T-ORACLE /
  GAUGE-NULL); TG-1 primary MATCH−POP with CI-low > 0, TOST ±0.005 alternative;
  TG-2 controls; LINEAR/DET rows as the backbone itself is deterministic — the DIFF row
  is added ONLY if the repair passes (repaired prior, same arms, common noise).
  Transport configs per R-B; abstentions counted in the denominator (ITT).

If repair passes: run a reduced P1 before any DIFF claims — pooled 3 seeds + LODO ×2
  (hold out Klados; hold out BCI2b). PV-3 and the LODO gate as originally frozen.
No threshold changes after this commit. Sealed cohorts untouched. W4 is banked; no
further UQ work this round.
```

Registered implementation note (fixed at commit time): the canonical ocular frame in this
build has r = 2 (VEOG/HEOG; blink not separable in the V19 bipolar convention, as
registered in M0) — "rank 3" reads as rank r = 2 here.
