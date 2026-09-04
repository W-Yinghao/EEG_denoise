# WAVE-6 pre-registration (FROZEN) — subject-calibrated ocular propagation

Frozen 2026-09-05, before any wave-6 compute. Source plan: `added_exp.md`
(experiments 1-5). This document fixes questions, endpoints, contrasts,
aggregation, QC gates and the interpretation grid BEFORE the numbers exist.
Amendments append below with a timestamp; nothing above is edited after freeze.

## 0. What already exists (survey, 2026-09-05)

| plan item | already banked | what is missing |
|---|---|---|
| E1 propagation structure | `t4_staleness.py cpu()` re-fits a ridge operator on every non-overlapping 120-s window of every dev cell, but stores ONLY a scalar displacement-vs-first-window (`results/paper_final/t4_staleness.json`) | the operators themselves, the four comparison classes (repeat / within-record-time / across-condition / across-participant), gain-vs-direction decomposition, shrinkage and reliability diagnostics, V/H topographies |
| E2 donor transfer | ONE mismatched donor per episode (`WRONG`, `WRONG_gated` arms in V44-S1 `stage1_result.json`), plus `POP` and `MATCH_gated` | the full donor sweep, and any relation between donor distance and transfer cost |
| E3 event selectivity | natural windows exist with `attenuation_db` / retention metrics; no event typing | V/H event labels, direction-resolved donor classes, the interaction, `D_active` |
| E4 recalibration | displacement curve + natural gain stratified by elapsed position (`t4_staleness.json`) | old-vs-new calibration compared ON THE SAME evaluation segment |
| E5 calibration content | duration curve only (`t2_duration.json`) | equal-duration calibration sets differing in V/H activity composition |

Nothing in wave-6 retrains a network. All GPU work is inference with the frozen
V44-S1 checkpoints (`fold_{0..4}_seed_{20261201,20261202,20261203}/best.pt`, EMA).

## 1. Common setting

- Corpus: MobileBCI dev cohort, 15 participants x 3 sessions x 2 tasks, 100 Hz.
- Operator: `EBTransferRegistry(data, fold, registry30, 120).operator(p, s, t, "EB")`,
  a 46x2 map from the V41R bipolar-EOG latent to fold-scaled EEG. The EB variant
  (`A = A_pop + lambda * (A_raw - A_pop)`) is the primary object because it is what
  the deployed system uses; `variant="RAW"` (`A_raw`) is analysed alongside so that
  similarity created by shrinkage toward a shared population mean can be told apart
  from similarity of the underlying fits.
- Guide: `A @ drive`, where `drive` is the recorded bipolar EOG of the window being
  restored. Swapping a donor swaps `A` ONLY; the EOG always comes from the window
  under restoration.
- **Intervention isolation (plan convention 3).** In every wave-6 arm the static
  calibration signature is held at the recipient's POPULATION signature
  (`registry30.signature(p, s, t, "POP")`). Consequently the own-calibration
  reference of this wave, `OWN`, is a NEW arm and is NOT numerically the published
  `MATCH_gated` (which carries `sig_gated`). Wave-6 contrasts are internal to
  wave-6; the published arms are not restated as wave-6 results.
- Donor eligibility: same `(session, task)` cell, donor drawn from the fold's
  TRAINING participants (never a test participant), operator estimated from
  calibration data only. An all-participant sweep, if run, is reported separately
  and labelled exploratory.
- Inference unit for statistics is the PARTICIPANT: episodes/windows are averaged
  within participant first, then across participants. Per-participant values are
  always retained and reported, negative cases included.

## 2. Experiment 1 - repeatable structure in the propagation relation (CPU)

Bank, for every dev cell and every non-overlapping 120-s window, the ridge
operator (the same estimator `t4_staleness.cpu()` already uses), plus per-cell
`lambda`, calibration reliability, and EOG activity summaries.

**Distances.** Primary is probe-based, secondary is the raw matrix distance:

- Probe bank `E*`: bipolar-EOG latent segments taken from TRAINING participants'
  calibration windows only, fixed per `(session, task)` before any distance is
  computed. `D_probe(i,j) = ||(A_i - A_j) E*||_F / mean_k ||A_k E*||_F`.
- `D_mat(i,j) = ||A_i - A_j||_F / mean_k ||A_k||_F`.
- Decomposition: gain `g_i = ||A_i||_F`; direction `Ahat_i = A_i / g_i`;
  `D_dir(i,j) = ||Ahat_i - Ahat_j||_F`; `D_gain(i,j) = |log(g_i/g_j)|`.

**Comparison classes** (the four rows of the plan's table):
R1 same participant, same cell, adjacent non-overlapping windows (the
short-term repeatability reference); R2 same participant, same cell, first vs
later windows; R3 same participant, different `(session, task)`; R4 different
participants, same `(session, task)`.

**Primary endpoint.** Per participant, the median of R4 minus the median of R1
on `D_probe`. Positive means cross-participant differences exceed what
short-term re-estimation noise explains. Reported per participant plus the
cohort distribution.

**Secondary.** The same statistic on `D_dir` and on `D_gain` separately; the
R3-vs-R4 comparison (does changing condition move the operator as far as
changing person); all four classes computed on `A_raw` as well as `A_EB`;
correlation of every distance with `lambda` and with calibration reliability.

## 3. Experiment 2 - does calibration transfer along propagation similarity? (GPU)

For each recipient cell, restore the SAME episodes/windows with the guide built
from: `OWN` (recipient's own EB operator), every eligible training donor
(`DONOR_<id>`), and `POP`. Identical inputs, identical model weights, identical
sampling seed across arms.

**Primary endpoint.** `dR(i<-j) = R(i<-j) - R(i<-i)` with `R = rrmse_temporal`
on the paired episodes, episodes averaged within recipient first. The primary
contrast is the WITHIN-RECIPIENT Spearman correlation between `D_probe(i,j)`
(computed in E1, from calibration data only) and `dR(i<-j)`; reported as the
per-recipient rho distribution and its participant-level mean.

**Secondary.** Near/mid/far donor tertiles cut on `D_probe` BEFORE any
restoration result is inspected, compared as `dR` per recipient; the same
analysis with `D_dir` and `D_gain` substituted, to test whether an apparent
distance effect is only overall gain matching; donor calibration reliability as
a competing explanation entered alongside distance; the natural-recording
endpoints (`attenuation_db`, `low_eog_observation_retention`,
`coherence_reduction`) under the same donor sweep.

## 4. Experiment 3 - is the effect selective to the current eye movement? (CPU on E2 output)

**Event labels.** For every natural window, `v` and `h` are the RMS of the
vertical and horizontal bipolar EOG. `ratio = v / (v + h)`. Cut points (the
upper and lower tercile of `ratio`, and the low-activity cut on total EOG RMS)
are fixed on TRAINING participants and then applied unchanged; windows are
labelled V-dominant, H-dominant, mixed, or low-activity.

**Donor classes.** For each recipient-donor pair the difference operator
`A_i - A_j` is split into its vertical and horizontal columns; the pair is
V-different or H-different according to which column carries the larger share.
This uses calibration data only.

**Primary endpoint.** The interaction: mean `dR` for (V-different donors,
V-dominant events) + (H-different, H-dominant) minus (V-different, H-dominant) +
(H-different, V-dominant), computed within recipient, aggregated across
recipients. Events are matched on total EOG energy before the contrast, and the
achieved match is reported.

**Secondary.** Explanatory comparison of `D_probe` (static distance), EOG energy,
and `D_active(i,j,q) = ||(A_i - A_j) e_q||_F` for predicting `dR` at the
window level, with energy entered first so that direction information must add
explanatory power on held-out recipients to count.

**Guard against circularity (plan section 5).** The paired data are constructed
by injecting `A_gen e`, so an operator-difference effect there is partly
structural. The paired result is therefore reported as a controlled-restoration
check only; the load-bearing evidence is the natural-recording arm, where the
event-related spatial pattern is estimated from FUTURE recorded EEG and compared
against the prediction from the independent earlier calibration.

## 5. Experiment 4 - when does drift actually invalidate an old calibration? (GPU)

On each evaluation window, three guides on identical input: `A_0` (the 0-120 s
calibration), `A_t` (the 120-s window immediately preceding the evaluation
window, never overlapping it), and `POP`.

**Primary endpoint.** `R(A_t) - R(A_0)` per participant per elapsed-time
stratum. Related, per stratum, to (a) `||A_t - A_0||` normalised as in E1,
(b) `||(A_t - A_0) e_q||_F` on that window's own EOG, and (c) the R1
short-term repeatability band from E1 - so that "the matrix moved" can be
separated from "the matrix moved in a direction the current activity uses".

## 6. Experiment 5 - content vs duration (deferred)

Per the plan's own ordering, E5 is designed only after E1-E3 report. Not frozen
here.

## 7. QC gates (probe must pass before any fleet)

The probe runs ONE (fold, seed) unit with a reduced donor set and asserts, in
code, exiting non-zero on failure:

- **P1 determinism** - re-running the OWN arm with the same seed reproduces the
  banked prediction bit-identically.
- **P2 the manipulation acts** - at least one donor arm differs from OWN by more
  than 1e-9 in `rrmse_temporal`; an all-identical sweep is a wiring failure, not
  a null result.
- **P3 real-id addressing** - every row's `donor` field equals the intended donor
  id, `OWN` rows carry the recipient's own id, and no donor is a test
  participant of that fold.
- **P4 operator sanity** - all operators are 46x2 and finite; probe distances are
  finite, zero on the diagonal, and not all equal.
- **P5 magnitude sanity** - the OWN arm's participant-mean `rrmse_temporal` lies
  within [0.5x, 2x] of the published `MATCH_gated` mean for that unit. It is NOT
  expected to be equal (the signature differs by design, section 1).

Coverage guard for the fleet: a unit counts as done only when its result file
contains the full expected row count (arms x episodes), counted by unique
`(participant, session, task, start/episode, arm)` keys - never by file
non-emptiness.

## 8. Pre-committed interpretation grid

**E1.** (a) R4 median exceeds R1 median for most participants, and R2 stays near
R1 -> a participant-related structure persists within a recording. (b) R3
approaches R4 -> calibration is specific to "participant x recording condition";
the individual-level wording must narrow accordingly. (c) Distances track
lambda, reliability, or EOG activity more than participant identity -> the
estimation problem must be fixed before any individuality claim. (d) All classes
overlap -> the value of calibration is not broad individuality; look for it in a
few directions or in unusual recordings.

**E2.** (a) Within-recipient rho > 0 for most recipients, surviving the gain-only
and reliability substitutions -> transfer follows propagation compatibility, and
"own calibration" is a special case of "compatible calibration". (b) All donors
are bad and distance explains nothing -> the current distance misses what
matters, OR operator estimates are too noisy; identity is NOT thereby proven to
act on its own. (c) Reliability explains most of it -> short-calibration estimate
quality dominates similarity. (d) The effect disappears once gain is matched ->
the individual difference that matters is overall propagation strength, not
spatial structure. (e) Nearest donors match OWN with wide per-participant
intervals -> under-determined; report the uncertainty, do not claim equivalence.

**E3.** (a) The predicted interaction appears at matched EOG energy AND the
natural event-related spatial pattern matches the independent prediction ->
support for context-dependent action of propagation differences. (b) Only an
energy effect remains -> report intensity dependence, drop the directional
claim. (c) Paired-data interaction without a natural-data counterpart -> the
effect may belong to the additive injection construction. (d) Natural spatial
prediction holds but restoration does not change accordingly -> the relation is
predictable but the restoration model does not exploit it.

**E4.** (a) Large `||A_t - A_0||`, small activated difference, unchanged result ->
some drift is irrelevant to the current activity. (b) Large activated difference
and a reliable recent estimate improve the result -> applicability genuinely
decays. (c) Drift is within the R1 repeatability band -> not attributable to real
drift. (d) The recent calibration is worse and reliability explains it ->
chasing the newest estimate adds variance.

## 9. Reporting rules

Neutral `*_RESULTS.md` with per-participant tables, the QC block and file
checksums; surprises disclosed before conclusions; results committed separately
from any interpretation; no manuscript text written from this wave (the paper is
owner-written).

---

## AMENDMENT W6-1 (2026-09-05, before any endpoint was computed)

Disclosure: at the time of writing, the E2 and E4 probes had run and their QC
blocks were inspected (gate values only — no endpoint, no donor ranking, no
restoration aggregate has been viewed). The E2 fleet was cancelled 1.5 minutes
after launch to make this amendment before compute was spent.

**W6-1a — the missing donor condition.** The plan's donor table
(`added_exp.md`, "本人另一记录或条件的校准，如有") requires a condition that the
first freeze omitted: the recipient's OWN calibration taken from a DIFFERENT
`(session, task)` cell. Without it the sweep cannot separate "same identity"
from "compatible propagation relation", which is the central question of E2.
Added as the arm family `OWN_OTHER_k`, k = 0..4, indexing that participant's
other cells in sorted order (missing indices produce no rows). Its rows carry
the full donor cell id `participant|session|task`, and gate P3 is extended: an
`OWN_OTHER_k` row must name a real cell of the SAME participant that is not the
recipient's own cell.

New pre-committed contrast (E2): the per-recipient comparison of
`dR(OWN_OTHER)` against `dR(DONOR)` at matched `D_probe`. If a
propagation-near stranger beats the recipient's own less compatible recording,
the applicable scope of calibration is set by the propagation relation and not
by identity; if own-other always wins regardless of distance, identity carries
information the current distance does not capture.

**W6-1b — E5 is un-deferred and frozen now.** Design: inside each cell's
pre-evaluation region the EOG is cut into 2-s blocks; blocks are ranked by
`v / (v + h)`; three EQUAL-DURATION calibration sets are assembled — V-heavy
(top-ranked blocks), H-heavy (bottom-ranked), and balanced (stratified across
the ranking) — with total EOG energy equalised as far as the blocks allow and
the achieved match reported. Three independent draws per composition guard
against a lucky segment. Each set is fitted with the SAME estimator and the same
shrinkage rule, then used to restore the SAME later natural windows, which are
labelled V-dominant / H-dominant / mixed / low by the E3 rule.
Primary endpoint: the composition x event-type interaction, i.e. mean
`rrmse`-equivalent natural endpoint for (V-heavy calibration, V-dominant events)
+ (H-heavy, H-dominant) minus the crossed cells, within recipient. Secondary:
whether the balanced composition is the most even across event types, and
whether composition effects are explained instead by total energy or estimate
reliability. Concatenating blocks is legitimate here because only a static
linear map is fitted — no filtering or spectral estimate crosses a seam, and no
metric is computed on the concatenated signal.

**W6-1c — analysis scripts are frozen as part of the design.** The endpoints of
E1-E5 are produced by committed analyzers (`x*_analyze.py`), not by ad-hoc
inspection; each writes a neutral table plus its QC block.

No other part of the freeze is changed.

---

## DEFECT W6-D1 (2026-09-05, disclosed on discovery)

The first E5 run completed all 15 units and its analyzer produced a primary
interaction of -0.546 (CI [-1.084, -0.046], 3/14 participants positive) - the
OPPOSITE sign to the plan's prediction. Before that number is interpreted, the
composition diagnostics were read, and they show the run does not test what the
design says:

| draw | V-heavy mean ratio | H-heavy mean ratio | separation |
|---|---|---|---|
| 0 | 0.815 | 0.447 | +0.368 |
| 1 | 0.659 | 0.643 | +0.015 |
| 2 | 0.651 | 0.650 | +0.001 |

Cause: `_compose` varied the three draws by ROTATING the whole ratio-ranked
block list, which moved draws 1 and 2 into the middle of the ranking and
collapsed every composition onto the cell average. Two thirds of the E5 rows are
therefore composition-blind, and the pooled interaction is dominated by them.
The probe gate did not catch it because it only required that SOME arm pair
differed, which draw 0 satisfied.

Actions, in this order: the defective units and results are preserved as
`e5_units_DEFECT_W6D1/`, `e5_results_DEFECT_W6D1.json`,
`e5_probe_DEFECT_W6D1.json` and are NOT deleted; `_compose` now varies draws
strictly WITHIN each composition's own region of the ranking; gate P4 now
requires EVERY draw to separate V-heavy from H-heavy by more than 0.05, so a
single separated draw can no longer hide two collapsed ones. E5 is re-run from
scratch. The defective interaction value is void and is not reported as a
result.

No other experiment is affected: E1-E4 do not use `_compose`.
