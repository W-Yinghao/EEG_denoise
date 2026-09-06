# WAVE6_RESULTS — subject-calibrated ocular propagation (neutral report)

Frozen design: `reports/prereg_wave6_propagation_FROZEN.md` (+ amendment W6-1,
defect W6-D1). Corpus: MobileBCI dev, 15 participants x 3 sessions x 2 tasks,
100 Hz. No network was retrained; all GPU work is inference with the frozen
V44-S1 EMA checkpoints. Every wave-6 arm holds the static calibration
signature at POP, so `OWN` here is a wave-6 arm and is NOT the published
`MATCH_gated`. Statistics are participant-first throughout.

This document reports numbers and the pre-committed reading of each. It does
not choose the manuscript framing.

## Coverage / QC

| experiment | units | rows | unique keys | incomplete |
|---|---|---|---|---|
| E1 structure | 90 cells | 2358 distances | - | 0 |
| E2 donor sweep | 15 | 23040 | 23040 | 0 |
| E4 recalibration | 15 | 4320 | 4320 | 0 |
| E5 content | 15 | 10800 | 10800 | 0 |

Rows equal unique keys in every unit (no duplicate or partial unit). All
probe gates passed before their fleet launched; E5 was re-run after defect
W6-D1 and its re-run separates every draw (0.393 / 0.294 / 0.295).

## E1 — repeatable structure in the propagation relation

- PRIMARY `probe R4-R1` (EB): -0.0001 (med +0.3076, CI [-0.6299, +0.3688], 12/14 positive) — **crosses zero**
- direction axis `R4-R1` (EB): +0.1516 (med +0.1900, CI [+0.0113, +0.2588], 13/14 positive)
- gain axis `R4-R1` (EB): +0.2492 (med +0.2695, CI [+0.1620, +0.3432], 13/14 positive)
- `R4-R3` probe (EB): -0.2051 (med -0.1239, CI [-0.3011, -0.1210], 2/14 positive)
- `R4-R3` gain (EB): -0.1707 (med -0.1926, CI [-0.2859, -0.0534], 5/14 positive)
- `R4-R3` direction (EB): -0.0148 (med -0.0086, CI [-0.0888, +0.0593], 6/14 positive)
- `R2-R1` probe: +0.0003 (med +0.0353, CI [-0.0858, +0.0591], 10/15 positive)

Class medians (probe distance): R1_adjacent_repeat 0.352, R2_within_record_time 0.388, R3_across_condition_EB 0.820, R3_across_condition_RAW 0.886, R4_across_participant_EB 0.666, R4_across_participant_RAW 0.775

Shrinkage confound: Pearson(lambda, cross-participant probe distance) = -0.314 over 84 cells.

Reading against the frozen grid: the composite primary is inconclusive. The
scale-free direction axis does exceed short-term repeatability (13/14, CI
excludes zero). The eye-catching ordering R3 > R4 (same person in another
condition sits farther than another person in the same condition) lives
almost entirely on the gain axis, which the per-cell EOG latent
normalisation can itself produce; the direction axis shows no R3-vs-R4
difference. Within-record time (R2) is not beyond repeatability (R1).

## E2 — transfer along propagation similarity

- PRIMARY within-recipient Spearman(D_probe, dR): +0.6000 (med +0.5667, CI [+0.5278, +0.6756], 15/15 positive)
- substitution rho(direction): +0.4433 (med +0.4500, CI [+0.3322, +0.5500], 14/15 positive)
- substitution rho(gain): +0.3356 (med +0.4000, CI [+0.1600, +0.5022], 11/15 positive)

| donor tertile by calibration distance | dR vs OWN (rrmse) |
|---|---|
| near | +0.0583 (med +0.0620, CI [+0.0337, +0.0833], 14/15 positive) |
| mid | +0.1338 (med +0.1050, CI [+0.0641, +0.2180], 14/15 positive) |
| far | +0.1507 (med +0.1303, CI [+0.0997, +0.2041], 14/15 positive) |

- OWN_OTHER (own calibration, other cell) - OWN: +0.2846 (med +0.2126, CI [+0.1803, +0.4217], 15/15 positive)
- near stranger - OWN: +0.0583 (med +0.0620, CI [+0.0337, +0.0833], 14/15 positive)
- **near stranger - OWN_OTHER: -0.2263 (med -0.1451, CI [-0.3636, -0.1252], 0/15 positive)**

Arm means (paired rrmse): OWN 0.4352, POP 0.6512, OWN_OTHER_0..4 0.4792, 0.7055, 0.6120, 0.6980, 1.1043.

Guard G-B (confound decomposition of the OWN_OTHER penalty):

| axis | own-other | near stranger | difference |
|---|---|---|---|
| gain_log | 0.591 | 0.260 | +0.3314 (med +0.3470, CI [+0.2525, +0.4069], 15/15 positive) |
| direction | 0.535 | 0.456 | +0.0791 (med +0.0855, CI [+0.0303, +0.1250], 11/15 positive) |

Reading: the primary is positive in every recipient and the tertile ordering
is monotone; the direction substitution stays positive, so the effect is not
only overall gain matching. The own-other comparison is the plan's most
discriminating cell and is negative for all 15 participants, but G-B shows
the own-other distance exceeds the near-stranger distance about four times
more on the gain axis (+0.331, 15/15) than on the direction axis (+0.079,
11/15). The gain part is what the per-cell EOG normalisation can manufacture,
so this cell supports 'compatibility, not identity, sets the applicable
range' only with that scale caveat stated; a smaller but reliable direction
component is also present.

## E3 — event selectivity

- layer 1, energy-matched direction x event interaction: -0.7019 (med -0.8626, CI [-1.6998, +0.3105], 4/13 positive) (n=13) — **crosses zero**
- variance explained: energy only 0.0045, energy + direction 0.0048
- layer 2 vertical: own cosine median +0.9605; own minus stranger +0.0513 (med +0.0510, CI [+0.0271, +0.0812], 13/15 positive)
- layer 2 horizontal: own cosine median +0.8408; own minus stranger +0.1046 (med +0.1185, CI [+0.0601, +0.1463], 14/15 positive)

Reading: the load-bearing natural prediction succeeds — an earlier
calibration predicts the participant's own later natural coupling better
than any stranger's does, on recorded EEG the calibration never saw. The
directional interaction in restoration does not appear, and direction adds
essentially nothing over EOG energy (0.0045 -> 0.0048). This is the frozen
grid's case (d): the relation is predictable, but the current restoration
model does not turn that predictability into event-selective differences.

## E4 — old versus new calibration on the same window

- PRIMARY recent minus initial (attenuation_db): -0.4861 (med -0.5813, CI [-0.9601, -0.0179], 2/15 positive)
- POP minus initial: -1.2190 (med -0.8761, CI [-1.9482, -0.5559], 2/15 positive)
- Spearman(operator displacement, effect) -0.1633; Spearman(activated displacement, effect) -0.1894

| activated-displacement stratum | windows | median displacement | recent - initial |
|---|---|---|---|
| low_activated | 360 | 0.231 | -0.1583 (med -0.1576, CI [-0.2792, -0.0392], 2/14 positive) |
| mid_activated | 360 | 0.463 | -0.3481 (med -0.2562, CI [-0.6210, -0.1197], 3/15 positive) |
| high_activated | 360 | 1.068 | -1.2632 (med -1.5085, CI [-2.2776, -0.2170], 2/14 positive) |

Reading: re-estimating on the most recent 120 s is WORSE than keeping the
initial calibration, and the deficit grows with how much of the change the
current activity activates. Consistent with E1's R2-R1 result (within-record
movement does not exceed short-term repeatability), i.e. there is little
real drift to chase. Confound that cannot be excluded from this design: the
initial 120 s is the designated calibration segment, whereas the 'recent'
120 s is arbitrary natural recording whose EOG activity may be poorer, so
estimate reliability is an alternative explanation to 'drift does not
matter'.

## E5 — calibration content (after defect W6-D1 was fixed)

- PRIMARY composition x event interaction: -0.9014 (med -0.2672, CI [-2.1386, +0.2100], 7/14 positive) — **crosses zero**
- evenness |V - H| by composition (smaller = more even across event types):
  - BALANCED: +1.5749 (med +0.9571, CI [+0.9805, +2.2432], 14/14 positive)
  - VHEAVY: +2.0091 (med +1.8812, CI [+1.4183, +2.6425], 14/14 positive)
  - HHEAVY: +1.9109 (med +1.9407, CI [+1.4818, +2.2860], 14/14 positive)

| calibration composition | V-dominant events | H-dominant events |
|---|---|---|
| VHEAVY | 1.255 | 1.848 |
| HHEAVY | 1.745 | 1.450 |
| BALANCED | 2.303 | 2.492 |
| OWN_EB | 2.949 | 3.116 |

(attenuation_db; OWN_EB is the deployed 120-s calibration, shown for scale.)

Guard G-A (the plan's identifiability check):

| composition | achieved v/(v+h) | design condition number (median / max) | operator |V col| | operator |H col| |
|---|---|---|---|---|
| VHEAVY | 0.796 | 1.92 / 5.2 | 2.034 | 1.032 |
| BALANCED | 0.592 | 1.61 / 3.3 | 1.564 | 1.179 |
| HHEAVY | 0.469 | 1.47 / 3.2 | 1.309 | 1.414 |

Reading: the directional prediction is not supported — the interaction
crosses zero. The condition numbers are small (median 1.5-1.9, max 5.2), so
the reversed pattern is NOT the unidentifiable-second-column artefact the
plan warns about. A post-hoc observation, flagged as post-hoc: a
composition inflates the operator column of the direction that dominated it
(V-heavy gives |V col| 2.03 against 1.31 for H-heavy), which would
over-subtract during events of that same direction; the balanced
composition is both the most even across event types and the best overall,
but its interval overlaps the others.

## Artifact checksums (sha256, first 16)

| file | sha256 |
|---|---|
| `wave6/e1_structure.json` | `794f728e2cd51e1b` |
| `wave6/e2_probe.json` | `943b093855913d6d` |
| `wave6/e2_results.json` | `197a61df767eb500` |
| `wave6/e3_event_selectivity.json` | `2efbc22d75137b89` |
| `wave6/e4_probe.json` | `8edf530eb4562679` |
| `wave6/e4_results.json` | `ea4f528191a81198` |
| `wave6/e5_probe.json` | `4e2a8aff046650f2` |
| `wave6/e5_probe_DEFECT_W6D1.json` | `cd8919adc116d6bc` |
| `wave6/e5_results.json` | `688a69c90f983a57` |
| `wave6/e5_results_DEFECT_W6D1.json` | `6d2a57805266a280` |
| `wave6/guards.json` | `dfd3b5e7df8e79da` |
| `wave6/e1_operators.npz` | `bd39ab203a88f6d1` |

Defective E5 evidence is preserved, not deleted: `e5_units_DEFECT_W6D1/`,
`e5_results_DEFECT_W6D1.json`, `e5_probe_DEFECT_W6D1.json`.
