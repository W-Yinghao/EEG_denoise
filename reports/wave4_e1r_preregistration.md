# WAVE4-E1R — preregistration addendum (the ONE registered repair of the optical-correspondence instrument)

Committed BEFORE any E1R code runs. Base: branch `codex/wave4-optical`, tip `ebddc76`
(the STOP-1 commit). CPU only.

**Standing invariants.** The banked E1 verdict (`results/wave4_optical/alignment/e1_pilot.json`,
reports at `ebddc76`) is READ-ONLY — E1R is a new instrument row beside it, never an edit
to it. All WAVE3 numbers are read-only. No gate constant in this addendum may move after
this commit. This is the only repair round: if the repaired instrument fails its weakest
tier, the optical instrument is CLOSED on this panel.

## Frozen scope — the clock layer is untouched

E1R rebuilds the **correspondence layer only**. The clock layer passed and is reused
verbatim from the banked artifact: per-recording `slope`, `intercept_ms`, the
`clock_gate_passed` flag, the 26-recording clock-valid set, and the nine exclusions
(4 stream discontinuities, 5 no-sync). No drift fit is recomputed. E1R runs on exactly
the **26 clock-valid recordings** of the banked pilot (S01–S03); E1 was never extended
past the pilot and E1R does not extend it either.

The VEOG surrogate is unchanged from the E1 preregistration: `mean(FP1, FPZ, FP2)`,
0.5–8 Hz 4th-order Butterworth zero-phase, robust-z peaks at height 4.0 with 200 ms
minimum separation. The validity cutoff stays at the E1-declared `≤1` (both eyes);
"invalid sample" means NOT(both eyes ≤ 1).

## REPAIR 1 — physiological lag window (frozen from blink kinematics, not from banked lags)

The eye tracker loses the pupil at lid-closure **onset**; the VEOG surrogate peaks near
**maximal closure**. Human blink kinematics place lid-closure duration at roughly 50–100 ms,
so the expected tracking-loss → VEOG-peak lag is **~+30 to +100 ms**.

**Frozen window: lag ∈ [−20, +120] ms**, where `lag := t_VEOG_peak − t_tracking_loss_onset`,
both expressed in the EEG clock after the frozen affine map. The −20 ms lower edge admits
detector jitter in the opposite sign; the +120 ms upper edge admits slow closures.

The banked E1 matched-lag median of 30–55 ms is **consistent with** this window and is
cited as corroboration only. It is **not** the source of the window: the window is derived
from closure kinematics and would be identical had the banked lags never been observed.
The retired E1 window (±50 ms, centred on zero) was mis-centred — it assumed simultaneity
between two events that are physiologically offset.

## REPAIR 2 — fragmentation-robust blink segmentation

At ~59% tracker invalidity a single blink is shattered into several short invalid runs,
and the E1 clause "flanked by ≥50 ms of valid samples" then rejects **real** blinks
(it demands clean data exactly where this tracker has none). That clause was correct for
a clean tracker and wrong at this invalidity; it is replaced, not merely loosened:

1. **Merge**: invalid runs separated by ≤ **40 ms** of valid samples are merged into one
   candidate.
2. **Duration**: a blink candidate is a merged run of duration ∈ **[50, 500] ms**.
3. **No flanking requirement.**
4. **Auxiliary, never required**: pupil-diameter collapse (`PupilLeft`/`PupilRight` going
   non-finite or dropping ≥50% below the recording median across the candidate) is
   computed and reported per recording as corroborating evidence. It enters **no** gate.

## REPAIR 3 — reverse VEOG-anchored instrument (co-primary, segmentation-free)

Measures correspondence **without segmenting Tobii blinks at all**, so it cannot be
starved by fragmentation.

For each VEOG-detected blink at EEG time `d`, the tracking-loss window implied by
REPAIR 1 is `W(d) = [d − 120, d + 20]` ms (EEG clock), mapped to the Tobii clock by the
frozen affine inverse. Let `f(d)` = fraction of Tobii samples inside `W(d)` that are
invalid; recordings contribute only windows containing ≥1 Tobii sample.

- **observed** = mean over blinks of `f(d)`.
- **null** = the same statistic under **≥200 circular shifts** of the VEOG blink train
  within the recording span (shift amounts spread uniformly, excluding near-zero shifts).
- **elevation** = observed − mean(null).
- **CI** = 2000-draw bootstrap resampling blinks with replacement, recomputing
  (observed_boot − mean(null)); report the 95% percentile interval.
- **p** = one-sided fraction of null draws ≥ observed.

A recording is *reverse-positive* iff its elevation CI-low > 0.

## REPAIR 4 — eligibility gating

Recording-level Tobii validity fraction ≥ **0.40** (frozen). The full sweep **0.30–0.60**
(step 0.05) is reported; the verdict is read at the frozen 0.40. Excluded recordings are
counted with their reason, never repaired. Clock exclusions are unchanged and take
precedence: only clock-valid recordings are eligible at all.

## Tiered E2 gates (gates sized per customer)

The single E1 gate was sized to the most demanding customer (event-level type labels) and
so killed customers needing far less (segment-level bounds). Retired and replaced:

**TIER-S — segment level.** Unlocks **M2**, **M4**, and the segment-level variant of **M3**.
Passes iff BOTH:
- reverse-instrument elevation CI-low > 0 in **≥60%** of eligible recordings, AND
- **pooled** elevation CI-low > 0, where pooled = mean per-recording elevation with a
  2000-draw bootstrap **over eligible recordings**.

**TIER-E — event level.** Unlocks **M1** and full **M3**. Passes iff BOTH:
- forward blink match rate (REPAIR 2 segmentation, REPAIR 1 window) **median ≥ 0.70**
  across eligible recordings, AND
- per-recording circular-shift null **p < 0.01** in **≥60%** of eligible recordings
  (forward null = ≥200 circular shifts of the Tobii blink-candidate train).

The retired gate (≥80% matched at ±50 ms) is recorded as retired with the rationale above.

**CLOSE RULE.** Tier-S fail ⇒ the optical instrument is **CLOSED on this panel**; all four
WAVE3 customers (readout row, A4 reference-channel error, type labels, non-degenerate
oracle) are documented **instrument-limited-unserved**; no further repairs, no re-download,
no new panels this wave.

## Statistics and reporting

Per-recording first, then aggregation. Sweeps are reported; verdicts are read at frozen
values only. Deliverables: `results/wave4_optical/e1r/correspondence_table.csv`,
`results/wave4_optical/e1r/decision.json`
(`{tier_s, tier_e, eligible_n, close_fired}`), `reports/wave4_e1r_report.md`, one ledger
line. Execution STOPS after the tier verdicts: E2 measurements are *authorized* by the
tiers but wait for the operator read.
