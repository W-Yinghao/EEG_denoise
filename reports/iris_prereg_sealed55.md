# PREREGISTRATION — EEGEyeNet sealed-55 confirmation opening

**Status: FROZEN, NOT EXECUTED. The sealed tree is untouched (chmod 000, zero contact).**
Opening requires operator sign-off on §1 (the endpoint question) plus §7 (the opening record).
Written 2026-08-30 on branch `codex/paper-final-runs`. Committed before any sealed byte is read.

---

## 1. The endpoint question the operator must settle first

The binding earmark is Operator Ruling 1 (2026-08-18), verbatim from
`docs/EEG_denoise_arc_results_digest.md`:

> **Ruling 1 — sealed block**: agreed, NOT opened; chmod 000 and zero contact stand.
> **EARMARK added**: at paper time the block's best use is likely the UQ-calibration sealed
> confirmation (band coverage + CRPS on unseen subjects — the UQ claim is relatively immune to
> the exogeneity limitation and is now the headline chapter) ± instrument-row stability.
> Any opening: paper-time, separate preregistration, operator sign-off.

**Disclosure (found while preparing this document, before any sealed contact).** The endpoint the
earmark names is not available with existing assets:

- The UQ head (`scripts/iris/f4_uq.py`, K=32 operator-sampled trajectories, W-INFL-TEMP width
  policy) is bound to the V44 MobileBCI machinery. **No diffusion prior was ever trained on
  EEGEyeNet** ("showcase never", milestone-G limitations).
- The only frozen models that exist on EEGEyeNet are the ten S356 compact **deterministic**
  denoisers (`iris_s356/model_n{30,60,120,200,259}_{COND,BLIND}.pt`). A single deterministic
  network has no predictive distribution; the five pool-size checkpoints are not a valid
  posterior ensemble (different training pools, not seeds).
- A literal band-coverage/CRPS confirmation on EEGEyeNet therefore requires training a new
  diffusion prior on that corpus — new training, which contradicts both the paper-final
  zero-training rule and Ruling 2's program closure.

**Two further facts that bear on the choice, and that post-date the earmark:**

1. The earmark was written on 2026-08-18, when the UQ chapter was development-only. **T1 has since
   closed that gap** (2026-08-29): one scalar temperature frozen on all 15 development cells, applied
   unchanged to the 8 held-out MobileBCI participants, coverage held at 0.810/0.864 with the
   operator-posterior width again beating temperature-only on CRPS out of cohort. The UQ claim now
   *has* a held-out confirmation on the panel where the machinery actually lives.
2. The claim that is now confirmation-hungry is the other one. Paper §5.5 (calibrated subject
   information generalizes across interfaces — the section that reconciles our design with the
   DS-DDPM / SADDPM subject-conditioning literature) rests on **14 guarded development subjects**
   from S356. Sealed-55 would confirm it on ~50 never-touched subjects with roughly a third of the
   interval width.

**Options.**

- **Option A — S356-style conditioning confirmation (this document's protocol).** Machinery
  complete, ~0.2–0.5 GPU-h, zero training, single pass. Confirms paper §5.5 on unseen subjects.
  Deviates from the earmark's letter; needs the operator to re-scope the earmark.
- **Option B — literal UQ confirmation.** Requires training a diffusion prior on EEGEyeNet
  (≈30+ GPU-h) and porting the F4 width policy to that corpus, i.e. reopening the closed program.
  Marginal value reduced by T1 having already confirmed the UQ claim on MobileBCI.
- **Option C — do not open.** Keep the block frozen; the paper says a sealed 55-participant block
  is reserved for confirmatory evaluation (current v3 wording). Costs nothing, gains nothing.

**Recommendation: Option A**, on the grounds that the earmark's purpose (spend the one-shot block
on whichever claim most needs unseen-subject confirmation at paper time) is now better served by the
conditioning claim than by the UQ claim, and that Option B cannot be executed under the standing
rulings. **This is a re-scope of Ruling 1 and is not mine to make: the protocol below is frozen but
will not run until the operator picks A, B, or C.**

## 2. Question and estimand (Option A)

Does the S356 finding — that calibration information delivered through a learned 32-d representation
gives a subject-specific gain that is flat in training-population size — reproduce on 55 subjects
that took no part in any modelling or analysis decision?

Estimand: per-subject $\text{gain} = \rrmse_{\text{zero-embedding}} - \rrmse_{\text{own-embedding}}$
on query episodes, evaluated with the frozen n=259 COND model; and the specificity contrast
$\rrmse_{\text{wrong}} - \rrmse_{\text{own}}$. Participant-first, 5,000-draw participant bootstrap
(seed 420), as in every prior campaign.

## 3. Frozen protocol

**Assets (read-only, never modified).** `iris_s356/model_n259_COND.pt` (primary),
`model_n30_COND.pt` (flatness companion), `model_n259_BLIND.pt` (sanity twin). Weights frozen;
no fine-tuning of any kind.

**Cohort.** All 55 non-empty sealed subjects (Drive listing positions 31–90; the 5 upstream-empty
folders AH0/AH6/AJ8/AJ9/AH9 are recorded and counted, never replaced). ITT: every subject is
reported, including preparation failures, with a reason code.

**Preparation and episodes** — byte-identical semantics to `scripts/iris/s356_probe.py`, which the
confirmation driver imports rather than reimplements: resample 500→100 Hz; 46 channels = 8
most-anterior non-periocular E-channels by chanlocs X plus 38 evenly spaced remaining non-periocular
channels; latent = [VEOG, HEOG] from the six periocular channels E8/E25/E125/E126/E127/E128;
`peri_interp` status carried from automagic metadata; SUPPORT = first half of the record, QUERY =
second half; per-subject operator $C_s$ = ridge-0.05 regression of the 46-channel block on the
robust-centred latent, fit on SUPPORT only; clean windows drawn from the lowest-30% EOG-energy pool,
drives from the top-30% pool circularly shifted; $y = x + C_s e$; 64 episodes per half;
`rng seed = 20260818 + crc32(subject) % 100000`, support drawn before query on one stream.

**Evaluation.** Per subject, a 32-d embedding is fitted on that subject's **own SUPPORT** episodes
(Adam, lr 1e-2, 200 steps, model frozen) and evaluated on the disjoint QUERY episodes. This is the
declared evaluator-only ceiling of the S356 protocol — a per-subject calibration act on frozen
weights, **not** asset tuning, and therefore not a violation of the charter's "no IRIS or incumbent
asset may be tuned after sealed contact". Arms per subject: `own`, `zero`, and `wrong` (mean RRMSE
under every other guarded sealed subject's embedding). BLIND is scored once as a sanity twin
(COND-with-zero-embedding should track BLIND).

**Episode-validity guard** (frozen at S356 amendment b1dcc2f, unchanged): a subject is
excluded-and-counted if its support or query injection RMS ratio falls outside $[0.1, 20]$. This
caught the AA4 defect (ratio 167 against 0.33–4.02 for all others) in development.

## 4. Gates and pre-committed interpretation grid

$\varepsilon = 0.02$ (the S356 epsilon convention). All intervals are 95% participant bootstrap.

| # | Quantity | Rule |
|---|---|---|
| G1 (primary) | own-gain at n=259, guarded subjects | CI-low $> \varepsilon$ |
| G2 (co-primary) | own $-$ wrong at n=259 | CI-low $> 0$ |
| G3 (secondary) | trend, gain(n=259) $-$ gain(n=30) | CI contains 0 $\Rightarrow$ flat |
| QC1 | guard exclusions | $\le 11$ of 55 (20%); more $\Rightarrow$ instrument-limited |
| QC2 | preparation failures | reported per subject with reason code |
| QC3 | BLIND sanity | COND-zero within 0.05 RRMSE of BLIND |

| Outcome | Verdict | What the paper may say |
|---|---|---|
| G1 ✓ and G2 ✓ | **CONFIRMED** | §5.5 upgraded: the calibrated-conditioning gain and its subject specificity replicate on 55 unseen subjects (with G3 ✓, "and remain flat in training-population size") |
| G1 ✓, G2 ✗ | **PROTOCOL-GENERIC** | a gain exists but is not owner-specific on this cohort; §5.5 must be narrowed to development evidence and the discrepancy reported |
| G1 ✗, G2 ✓ | **SPECIFIC BUT SMALL** | specificity replicates, magnitude does not clear $\varepsilon$; report both and keep §5.5 as development evidence |
| G1 ✗ and G2 ✗ | **NOT CONFIRMED** | §5.5 is reported as a development-only finding that did not replicate; stated plainly in the paper |
| QC1 fails | **INSTRUMENT-LIMITED** | adjudication withheld; exclusion statistics reported |

The verdict is reported **regardless of outcome**. No outcome licenses re-running, re-scoping, or
re-tuning anything: this is a single pass.

## 5. What is explicitly NOT claimed

Nothing here touches the paper's main MobileBCI results (dev +0.1428, held-out +0.1537, the UQ
tables) — those are final as banked and are not re-litigated. A CONFIRMED verdict upgrades §5.5
only. EEGEyeNet remains a supporting corpus; per the campaign charter it is never a showcase panel.

## 6. Single-pass discipline (charter §5, verbatim)

> at most once, only with operator sign-off, single inference pass, outputs digest-frozen before
> evaluation, reported regardless of outcome (C-1 protocol). No IRIS or incumbent asset may be
> tuned after sealed contact.

Execution order, each step committed before the next begins: (1) this prereg, committed; (2)
dev-class probe of the driver on `eegeyenet_ext` subjects — **no sealed contact**, validates the
pipeline end to end; (3) operator sign-off recorded; (4) open, prepare, run inference, **bank raw
rows and sha256-freeze outputs before any aggregation**; (5) aggregate and adjudicate against §4;
(6) two-commit convention — results-only commit first, interpretation second.

## 7. Opening mechanics and required record

`scripts/iris/freeze_sealed.py` has no unseal mode by design. Opening is a deliberate act:
`chmod 0o755` on `/projects/EEG-foundation-model/eegeyenet/eegeyenet_sealed` (user-owned; no
administrator needed), performed by a new `open` mode that **refuses to run unless** it can write an
opening record containing: this prereg's commit SHA, a UTC timestamp, the operator sign-off
reference, the chosen option (A/B/C), and the pre-opening `sha256` of `sealed_freeze.json`. The
record is written to `results/iris/sealed/sealed_opening_record.json` and committed before
preparation starts. Re-sealing after the run is mandatory.

## 8. Budget

GPU ≈ 0.2–0.5 h (inference only, 55 subjects × 3 arms × up to 2 models). CPU ≈ 1–2 h (preparation of
14.6 GiB). Disk ≈ 2 GB under a fresh derived root `iris_sealed_confirm/` — the frozen `iris_s356/`
tree is never written to. Program total stands at ~0.7 GPU-h of 400; the charter staged ≤10 GPU-h
for a sealed opening.

## 9. Power

S356 implies a per-subject gain SD ≈ 0.04. With ~50 guarded sealed subjects the bootstrap CI
half-width is ≈ 0.011, decisive against $\varepsilon = 0.02$ in both directions — the confirmation
is adequately powered to fail if the effect is not there.

---

# AMENDMENT SEALED55-1 — operator elects BOTH endpoints (2026-08-30)

**Committed before any sealed contact and before any Option-B training begins.**
Operator decision: *"AB 都做一做吧，最后看结果决定，现在 GPU 是空闲的"* — run both endpoints,
decide at the end which the paper features.

## A1. One opening, two endpoints

The block still opens **exactly once**. Options A and B are scored in the **same single inference
pass** over the same prepared sealed episodes; they are independent (A uses the frozen S356
deterministic checkpoints; B uses a new diffusion prior trained only on dev-class data), so neither
contaminates the other. Two openings would violate charter §5 and are not permitted.

## A2. Mandatory disclosure rule (the honesty clause)

Both endpoints are preregistered here with their own gates **before** opening, and **both verdicts
are reported in `RESULTS_PAPER_FINAL.md` regardless of outcome**. The operator may choose which
result the paper *features*; choosing not to feature one is an editorial decision and is permitted.
**Suppressing one because it came out worse is not.** If only one endpoint is featured in the
manuscript, the other is still stated in the paper's evaluation section or appendix, with its
verdict. Selection of the featured endpoint happens after both verdicts are banked and is recorded
as such.

## A3. Option B protocol — UQ-calibration confirmation on EEGEyeNet

**Why it is now executable.** S356 episodes are 46 channels x 512 samples at 100 Hz with a 2-row
[VEOG, HEOG] latent and a 46x2 per-subject ridge operator — the same tensor geometry and operator
shape as the MobileBCI V44 class. The V44 architecture (`CalibSADDPMEOG`, x0-parameterized around
EOG regression) and the F4 UQ head therefore port to this corpus without redesign. What is new is
one training run on EEGEyeNet dev-class data; nothing about the MobileBCI assets changes.

**Training (dev-class only, zero sealed contact).** One `CalibSADDPMEOG` prior is trained on
EEGEyeNet antisaccade **dev-class** episodes (listing positions 1-30 and the 247 ext subjects at
positions 91-370; the sealed positions 31-90 are excluded by construction and the code refuses any
root but the dev-class ones during training). Recipe frozen from V44-S1: guide
$a_0 = C_s e$ with $C_s$ the SUPPORT-fitted ridge operator, x0-prediction
$\hat{x}_0 = (y - a_0) + \Delta_{\text{pop}} + \Delta_{\text{cal}}$, 1,000 linear diffusion levels
($\beta_1 = 10^{-4}$, $\beta_{1000} = 0.02$), 80,000 AdamW updates, batch 8, lr $10^{-4}$, weight
decay $10^{-4}$, gradient clip 1.0, EMA 0.999, guide-dropout 0.30, feature-dropout 0.20, Smooth L1.
Seed 20260830. Validation on held-out dev-class subjects every 2,000 updates; EMA weights selected
by validation RRMSE. **Zero fine-tuning after sealed contact.**

**Operator posterior and width policy.** Per cell, the entrywise EB posterior variance is
$\sigma^2_{A,cr} = (1/\tau^2_{cr} + B/v_{cr})^{-1}$ with $\tau^2$ the across-dev-class-subject
variance of SUPPORT-fitted operators and $v$ the variance of $B{=}4$ SUPPORT sub-block refits —
the V44-S2 `_posterior_variance` construction, recomputed on this corpus. $K = 32$ chains jointly
sample the DDIM initial noise and an entrywise operator draw; predictive width
$w = s\sqrt{w_{\text{chain}}^2 + w_A^2}$ with $w_A^2 = \sum_r \sigma^2_{A,cr} e_r(t)^2$.

**Temperature freeze (before opening).** One scalar per policy is chosen on **dev-class evaluation
subjects** by the frozen grid rule — smallest $s$ in `arange(0.50, 6.00, 0.05)` reaching 80% mean
coverage — and committed to `results/iris/sealed_confirm/uq_temperature.json` **before** the block
opens. Applied unchanged to the sealed cohort, exactly as T1 did on MobileBCI.

**Endpoints.** Empirical coverage at nominal 50/80/90%, Gaussian CRPS, risk-coverage area, and
per-subject 80% coverage spread, for three policies: raw samples, temperature-only, and
operator-posterior inflation + temperature.

## A4. Option B gates and pre-committed interpretation grid

| # | Quantity | Rule |
|---|---|---|
| B1 (primary) | 80% and 90% coverage, adopted policy (inflation + temperature) | both within 0.05 of nominal |
| B2 (co-primary) | raw-sample coverage at 80% | $< 0.70$ (the samples must be genuinely underdispersed, i.e. the calibration is doing work) |
| B3 (secondary) | CRPS ordering | inflation + temperature $\le$ temperature-only |
| B4 (descriptive) | per-subject 80% coverage | spread reported; no gate |
| BQC | prepared sealed subjects usable for UQ | $\ge 40$ of 55; else instrument-limited |

| Outcome | Verdict | What the paper may say |
|---|---|---|
| B1 ✓, B2 ✓ | **UQ CONFIRMED (second corpus)** | the interval mechanism transfers to an independent corpus and cohort under a temperature frozen in advance |
| B1 ✓, B2 ✗ | **TRIVIALLY COVERED** | coverage holds but the raw samples were already wide enough — calibration unnecessary here; reported plainly, no transfer claim |
| B1 ✗ (undercovers) | **UQ DOES NOT TRANSFER** | stated plainly; the MobileBCI T1 result stands on its own panel and the corpus-transfer claim is withdrawn |
| B1 ✗ (overcovers) | **CONSERVATIVE** | intervals valid but loose on this corpus; report coverage and width |
| BQC fails | **INSTRUMENT-LIMITED** | adjudication withheld; preparation statistics reported |

B3 failing does not overturn B1: it would only mean the physically motivated width earns nothing on
this corpus, which is reported as such (the "physics wording" is earned per corpus, per the F4 rule).

## A5. Revised execution order

Each step committed before the next begins; steps 1-5 involve **zero sealed contact**.

1. This amendment, committed. ✅
2. Option-B training on dev-class episodes; validation curve banked.
3. Option-B temperature frozen on dev-class evaluation subjects; committed.
4. Dev-class probe of the **dual-endpoint** driver (A and B scored end to end on ext subjects).
5. Operator sign-off recorded.
6. **Single opening** → prepare sealed episodes → one inference pass scoring **both** endpoints →
   raw rows banked and sha256-frozen before any aggregation.
7. Aggregate and adjudicate **both** grids (§4 for A, §A4 for B); two-commit convention.
8. Reseal.

## A6. Revised budget

Option A ≈ 0.2-0.5 GPU-h (inference only). Option B ≈ 8-14 GPU-h (one 80k-update training run on
dev-class, plus K=32 inference on the sealed cohort). Total well inside the charter's ≤10 GPU-h
staging for the opening plus a one-off training allowance; program total remains far below the
400 GPU-h campaign budget. Option B's training is new compute on an already-closed program and is
authorized by this amendment alone — it changes no banked result and touches no MobileBCI asset.

---

# AMENDMENT SEALED55-2 — Option-B episode defect and repair (2026-08-30)

**Committed before the repaired training re-runs. Zero sealed contact throughout; Option A is
unaffected (it reads the S356 episodes through `s356_probe`, a separate derived tree).**

## Defect

The first Option-B training run (job 963709, cancelled at update 6,000) showed the training loss
falling while the guided validation RRMSE rose monotonically: 0.716 (2k) → 1.048 (4k) → 1.966 (6k).
Diagnosis, verified in the source: `sealed_uq.episodes` injected the artifact with the
SUPPORT-fitted operator and `sealed_uq.train` built the guide from the *same* operator, so

$$y - a_0 = (x + C_{\text{sup}} e) - C_{\text{sup}} e = x$$

exactly. Because `CalibSADDPMEOG` is parameterized as
$\hat{x}_0 = (y - a_0) + \Delta_{\text{pop}} + \Delta_{\text{cal}}$ with zero-initialized residual
heads, the guided mode was **perfect at initialization and could only be degraded by training**;
the 30% guide-dropout batches supplied the only real gradient signal, and that learning perturbed
the guided mode away from its exact solution. The rising validation curve is that perturbation.
This is an instrument defect in the episode construction, not a scientific outcome.

## Repair (V44-faithful)

The paper's own protocol never guides with the operator that generated the artifact: V44 injects
with a generative operator fitted on a disjoint region (`query_transfer`, samples 15000:27000,
never a model input) and guides with the calibration operator fitted on the 120-s prefix. The
Option-B episodes now mirror that structure exactly:

- $C_{\text{gen}}$ — ridge-0.05 operator fitted on the **QUERY half**, used to inject the artifact
  in both halves' episodes; never a model input, never conditioned on;
- $C_{\text{sup}}$ — ridge-0.05 operator fitted on the **SUPPORT half**, used to build the guide
  $a_0 = C_{\text{sup}} e$ and the 46x53 signature, exactly as before;
- the residual the network must learn is therefore
  $(C_{\text{gen}} - C_{\text{sup}}) e$ — precisely the within-subject operator drift the
  operator-posterior width policy is meant to cover, which makes the UQ endpoint well posed rather
  than degenerate.

The EB posterior variance construction, the K=32 chain machinery, the temperature grid rule, and
every gate in §A4 are **unchanged**. The draw order of the clean/drive windows is unchanged, so the
clean windows $x$ are identical to the banked S356 episodes; the contaminated $y$ now differs by
construction, and the SEALED55-1 sentence "so x/y are unchanged" is corrected here to "so the clean
windows x are unchanged".

## Discipline

No result has been aggregated and no sealed byte has been read. The cancelled run's logs are kept
unedited as the defect record. This amendment is committed before the repaired training starts,
following the S356-1 precedent (bank the defect, freeze the repair, then re-run).
