# IRIS MISSION BRIEF
### For the server-side Claude Code (Fable 5) — full autonomy within stated constraints

You are being handed a research mission, not a script. You decide the staging, the gate
thresholds, the experiment designs, and the order of battle. This brief gives you the
goal, the assets, the established facts that bind any design, the non-negotiable
disciplines, and the boundaries of your autonomy. Everything else is yours.

Companion documents (the operator copies these into `docs/` alongside this brief —
read all three in full before planning):

```text
docs/arc_results_digest.md        — every established fact and number (the ground truth)
docs/IRIS_method_design.md        — the method you are building (design + judge warnings)
docs/paper_design_cleanslate.md   — the paper frame the results will eventually serve
```

---

## 1. THE MISSION

Build, validate, and honestly adjudicate **IRIS** — the operator-marginalized,
typed-reference, structured-prior method for subject-aware diffusion EEG denoising
specified in `docs/IRIS_method_design.md` — to the point where every claim the paper
frame needs is either measured with a confidence interval or honestly closed as a
preregistered negative.

**Definition of done:**

```text
D1. The mechanism-pure benchmark fight is adjudicated: IRIS vs the incumbent
    (MATCH_gated, dev +0.143 [15/15], sealed +0.1537 [7/8]) on MobileBCI, EOG-only,
    identical reference and prior — win, tie, or loss, with CIs, under a
    preregistration you froze before running.
D2. If EEGEyeNet has landed: the dual-reading fight (same-reference AND
    native-reference, both preregistered) is adjudicated, and the four dangling
    instrument items (true-VEOG A4 row, typed labels, event-level oracle, readout
    bound) are serviced or honestly re-closed.
D3. The covariance-inflation gate's properties are measured: abstention-cliff
    reclamation, wrong-donor harm under the harm certificate, fallback-ladder behavior,
    never-worse-than-NO_A0.
D4. UQ is adjudicated against the DET-ensemble bar (CRPS/RC-AUC) with the calibrated-
    bands-at-reasonable-CRPS target (current best: calibrated 80/90 at 3× CRPS —
    beat the 3×).
D5. A sealed EEGEyeNet block (~40–56 subjects) is FROZEN before any query contact and
    spent at most ONCE, only with operator sign-off, under the C-1 protocol
    (sha256-frozen outputs, logged reads, report regardless of outcome).
D6. The results digest is extended with every verdict — positive and negative alike —
    in the same verbatim-numbers style it already uses. No manuscript writing.
```

The eventual product is ONE paper (`docs/paper_design_cleanslate.md`); your job ends at
experimental completion and documentation. `taas_submission/**` stays untouched.

## 2. WHAT YOU HAVE

```text
- The repo: 70+ branches of finished work. Frozen V44-S1 checkpoints (the incumbent),
  the V42R/V43 machinery (EB builder, gates, common-noise arms, paired panels), the
  M0 operator/covariance atlas, the E1R alignment instrument (clock layer validated at
  5 ms; tiered correspondence gates), the wave-2 shared drift layer, all decision JSONs.
- Panels: MobileBCI (the benchmark panel; 15 dev + 8 sealed ALREADY SPENT), Klados v4,
  BCI2b, SGEYESUB; EEGEyeNet incoming by manual operator download (356 subjects,
  129-ch EGI with periocular electrodes → true VEOG; EyeLink typed events; minimally-
  preprocessed = artifacts intact). If it has not landed, structure the campaign so
  nothing blocks on it.
- Compute: Slurm (CPU/cpu-high; H100/A100/L40S/A40/V100; A100/H100 cap 24 h/job,
  checkpoint-resume for longer). Single-user project: no verification ceremonies
  (no sha256 binding rituals, no clean-checkout replays), commit and push per milestone.
- Sealed ledger (spend-once assets, all requiring operator sign-off to open):
  BrainID Day-200, PhysioMotion-10, SHU Day-4/5 — reserved for paper-time; the new
  EEGEyeNet sealed block you will freeze. MobileBCI-8 is spent and closed.
```

## 3. WHAT IS TRUE (binding on any design you produce)

The full record is `docs/arc_results_digest.md`. The facts that must never be violated:

```text
F1  Score-network personalization is dead three independent ways (+0.0063 ceiling /
    oracle-trained −0.051, ceiling below floor / LoRA −0.020), and scale does not
    rescue it (n=8 crossing). The network never receives subject-identity inputs.
F2  What converts: per-window artifact realization via runtime references
    (ceiling +0.312, deployable +0.143/+0.1537) and operator geometry on sparse
    montages (+0.016/+0.018 of +0.04–0.11 ceilings).
F3  Reliability gates cannot verify identity (4 replications); ownership from operator
    features is drift-defeated (3 attempts, closed). Safety = graceful degradation.
F4  A bad population anchor is worse than none (0.651 vs 0.574). Misalignment harm is
    LINEAR (slope 1.390 RRMSE per unit RMS operator displacement).
F5  Causal online RLS loses to dedicated calibration at every horizon; learned
    components have lost to closed forms four times in this record. Closed forms first.
F6  The two physics channels are partially redundant (additivity 0.596; the deficit is
    variance accumulation, not double-counting) — fuse in one likelihood, never stack.
F7  Clean-window pooled priors over-subtract along ocular directions (cause unknown;
    U-ratio paradox 1.40); canonical cross-montage priors failed validity (amplitude
    collapse; κ≈4600 whitening instability). No covariance whitening anywhere.
F8  Diffusion earns its keep in UQ, not points (DET twins competitive-to-ahead; K-chain
    beat the DET ensemble on CRPS/RC-AUC). DET twins carry all point claims.
F9  Ocular-only scope (motion operators carry no subject-specificity). The static
    linear family is near-final ON THE 2-CHANNEL EOG REFERENCE; the family question is
    open only where the reference is genuinely richer.
F10 Privacy is a step function; only the population route is zero-linkage; abstention
    is the privacy mechanism. The gaze stream is itself biometric — a new privacy
    surface you must probe before any privacy language.
```

Judge warnings carried into the design (details in the method doc): the template-BEM
physics is a gated prior-mean hypothesis, not a load-bearing core (orbital anatomy is
where template physics is worst; the lid is a sliding shunt, not a dipole); EEGEyeNet
comparisons must preregister BOTH readings (same-reference and native-reference) or the
win is a reference claim, not a mechanism claim; the exogeneity hazard is subtracting
saccade-locked brain activity (λ waves) — retention gates must be hard; oracle
instruments on operator-injected panels are tautological (fit oracles on disjoint
segments, cross-referee with held-out reference channels).

## 4. NON-NEGOTIABLE DISCIPLINES (values, not scripts)

```text
1. Preregister before you run: for every experiment, commit the estimands, gates, and
   thresholds (YOUR choice of thresholds — but frozen before execution) first. No
   threshold moves after its data exists.
2. Cheap kills first: adjudicate thesis-level bets with the least expensive decisive
   instrument available before spending integration compute. (The method doc's Step-0
   trio — abstention-cause taxonomy, corrected-Σ_drift prior-predictive recheck,
   EEGEyeNet instrument port — are strong candidates, but the choice of what to run
   first is yours.)
3. Honest negatives are first-class: every failed gate is documented with its numbers
   in the digest style; defective runs are banked unedited beside their corrections.
4. Strongest-baseline honesty: subject-aware gains are measured against the strongest
   subject-free route (the NO_A0 lesson); reference-rich gains carry both readings;
   point claims say "competitive", never diffusion-superiority (C05); the retention
   metric is named "low-EOG observation retention" (C08); Dev/Sealed labels never mix.
5. Sealed discipline: freeze the EEGEyeNet sealed split BEFORE any query contact with
   that data; no sealed asset opens without operator sign-off; one opening, one pass,
   report regardless.
6. Budget: soft cap 400 GPU-h for the whole campaign. Crossing it — or any single
   planned spend over 150 GPU-h — is a check-in with the operator, never a silent run.
7. Report at your own milestones (major verdicts, kills, direction changes) with
   verbatim decision JSONs; commit and push at each. Between milestones you run free.
```

## 5. YOUR AUTONOMY

You may: design the staging and choose all gate thresholds; alter or extend the IRIS
architecture where evidence argues for it (document the argument against F1–F10 and the
judge warnings before deviating); invent sub-experiments this brief never mentions;
re-order or drop components the evidence kills (every component death must leave a
functioning method — the incumbent is IRIS's exact sub-model and is your floor);
allocate compute freely under the budget rule; decide what is worth measuring beyond
the definition of done if it serves the paper frame.

You may not: violate F1–F10 or the closures recorded in the digest; touch sealed assets
without sign-off; write manuscript text; delete or rewrite banked results; exceed the
budget rule silently; convert diagnostic oracles into deployed estimators; run
verification ceremonies (they are banned, not required).

If EEGEyeNet has not landed when you start: run the MobileBCI mechanism-pure campaign
and every legacy-panel leg first; leave the EEGEyeNet legs as frozen preregistrations
that execute when the data appears.

## 6. KICKOFF PROMPT (operator pastes this)

```text
Read docs/IRIS_MISSION_BRIEF.md, docs/arc_results_digest.md,
docs/IRIS_method_design.md, and docs/paper_design_cleanslate.md in full. You have the
mission, the facts, the assets, and the autonomy boundaries. Plan your own campaign:
write your preregistrations, choose your gates, run your kills cheapest-first, build
and adjudicate IRIS against the incumbent per the definition of done, extend the
digest with every verdict, and report at your own milestones with verbatim decision
JSONs. Branch from the current tip (new branch, your naming). Slurm only; budget rules
and sealed discipline per the brief; no manuscript writing. Begin.
```
