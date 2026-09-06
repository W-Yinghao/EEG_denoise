# FLAGSHIP-M13 — Canonical prior campaign + the two live gain legs
### Server execution instructions (Claude Code) — Months 1–3, keyed off the M0 ceiling matrix

Matrix verdict consumed: K1 not fired. Gain thesis alive in TRANSPORT (Klados, BCI2b —
all-strata GO on BCI2b, deployable zero-training effects already +0.037/+0.018) and
LIKELIHOOD (MobileBCI EOG-at-query; deployable +0.312 from V44-S0; the +0.26–1.77 oracle
ceilings carry the degeneracy caveat and ground no claim). Weight-space and conditioning
are closed (three independent kills of score-network personalization). MobileBCI transport
is unstable (κ up to 4.6e3; frame concentration failed) — engineering repair, not theory
failure (all mathematical contracts passed).

This stage: (W1) repair the transport estimation engineering; (W2) train the flagship's
headline object — ONE montage-invariant canonical population prior across the corpus, with
leave-one-dataset-out; (W3) the transport gain factorial with the trained prior on the GO
panels; (W4) the operator-posterior UQ layer on the V44-S1 checkpoints when they land.

---

# 1. Workspace and Git

```text
base    : codex/flagship-m0, tip b808c10
branch  : codex/flagship-m13   (same worktree denoiseNet_flagship_m0 is fine)
```

Consume read-only: M0 results, V44-S0/S1 results (rgcc-eog-v44), V43-S2/S3 results
(rgcc-v43). Harness unchanged: Slurm only (CPU/cpu-high; H100/A100/L40S/A40/V100;
A100/H100 `--time=24:00:00`), no verification ceremonies, sealed cohorts never read.

---

# 2. Preregistration (`reports/m13_preregistration.md`, commit BEFORE submissions)

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

---

# 3. W1 — transport estimation repair (CPU, run first)

Implement the frozen whitening rule in `src/eeg_chart/transport.py` (new code path;
Stage-0 contracts and the ρ=0 bit-identity test must still pass — extend the unit tests).
Re-run the geometry battery + the analytic ceiling probes on all three panels.
Output `results/flagship_m13/w1_repair/{geometry.json, u1a_rerun.json}` + a short section
in `reports/m13_report.md` (split-half diagnosis: estimation noise vs heterogeneity).

# 4. W2 — canonical prior campaign (the big one)

**P0 pilot** (auto-proceed on pass): pooled prior on the three paired panels only
(MobileBCI + Klados + BCI2b clean carriers), 1 seed, the K=121 canonical U-Net from the
design (≈28 M params, x0-parameterized observation-centred zero-init residual — V42R's
decisive fix, reused verbatim), montage-mask augmentation. Evaluate PV-1/PV-2 per panel
with the analytic likelihood step. If any PV gate fails: STOP and report (base-validity
first, the V40R/V41R lesson).

**P1 full campaign** (on P0 pass):

```text
pooled prior      : full corpus per the training-data rule, 3 seeds
LODO runs         : hold out {MobileBCI, Klados, BCI2b, SGEYESUB} one at a time, 1 seed each
ambient arm       : 1 seed, secondary
per-dataset refs  : 3 single-panel priors, 1 seed each (for PV-3)
```

Slurm: `m13_prior.sbatch`, A100/H100 `--time=24:00:00` per cell (checkpoint-resume across
submissions if a cell needs more than 24 h; V100 fallback 36 h). Budget expectation
~500–650 GPU-h total. Evaluate all PV/LODO gates; write
`results/flagship_m13/w2_prior/{validity.json, lodo.json}`.

# 5. W3 — transport gain factorial (after P1, inference-only, cheap)

On frozen pooled-prior checkpoints, GO panels first:

```text
arms      : T-POP, T-MATCH(ρ̂), T-WRONG(gated), T-ORACLE, GAUGE-NULL
backbones : DIFF (50-step DDIM, K=8), DET1, DET-ITER, LINEAR  (identical conditioning)
noise     : common initial noise across all arms; participant/record-first stats
```

Decision JSON with TG-1/TG-2/TG-3 per panel; MobileBCI post-repair row descriptive.
`results/flagship_m13/w3_transport/decision.json`.

# 6. W4 — operator-posterior UQ layer (conditional on V44-S1 completion)

When `results/rgcc_eog_v44/stage1/decision.json` exists: load the V44-S1 EOG-guided
checkpoints (read-only), implement K-chain operator-posterior sampling
(`src/eeg_chart/posterior_sampling.py`: per chain draw C ~ N(C_gated, Σ_post) from the
U0-b posterior, compute a0 = C·e, run the registered DDIM trajectory; weighted combination
per the registered scheme), evaluate UQ-1..3 on the MobileBCI paired panel + the natural
panel if V44's G4 validity flag passed. `results/flagship_m13/w4_uq/decision.json`.
If V44-S1's G1 gain failed, W4 still runs (UQ is orthogonal to the gain sign) but the
report says so.

---

# 7. Deliverables and stop points

```text
reports/m13_preregistration.md, m13_report.md
src/eeg_chart/{transport repair, prior training, posterior_sampling}.py + tests
results/flagship_m13/{w1_repair, w2_prior, w3_transport, w4_uq}
one short ledger section
```

STOP points: (a) if P0 pilot fails any PV gate — stop, report; (b) otherwise stop after
W3 decision + W4 (or after W3 if V44-S1 has not landed, noting W4 pending). Report
verbatim: PV/LODO table, TG table per panel, UQ table vs DET ensemble, the W1 diagnosis,
and updated compute ledger. Months 3–5 (unification factorial + modules) follow from the
operator's read.

Prohibitions: no sealed reads; no natural-route tuning (V43-S3 owns that; consume its
verdict); no revision of the M0 GO map; no claim wording from degenerate likelihood
ceilings; no threshold changes post-commit.

---

# 8. Kickoff prompt

```text
Read FLAGSHIP_M13_Prior_And_GainLegs_Server_Instructions.md in full and execute it.
Create branch codex/flagship-m13 from codex/flagship-m0 tip b808c10. Commit the
preregistration BEFORE any submission. Run W1 (transport repair + geometry battery +
U1-a reruns, CPU) first; then the W2 P0 pilot (pooled prior, 3 paired panels, 1 seed) —
stop and report if any prior-validity gate fails, otherwise auto-proceed to the P1 full
campaign (pooled 3 seeds + LODO ×4 + ambient arm + per-dataset references); then the W3
transport factorial on Klados and BCI2b with the trained prior; run W4 (operator-posterior
UQ on V44-S1 checkpoints) when V44-S1's decision JSON exists. Aggregate, write all
decision JSONs and the report, commit, and stop. Slurm only; A100/H100 at
--time=24:00:00 with checkpoint-resume; no verification ceremonies; no sealed reads.
```
