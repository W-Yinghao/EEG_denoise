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
