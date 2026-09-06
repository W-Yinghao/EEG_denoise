# WAVE4 — STOP 1: E1 alignment verdict (3-subject pilot)

**Verdict: the E1 gate FAILS on 0/3 pilot subjects. The frozen extension rule (≥2/3 pass
→ extend to all covered subjects) does not fire. The wave stops here for the operator.**

Pilot: S01, S02, S03 — 35 recordings. Artifact: `results/wave4_optical/alignment/e1_pilot.json`
(run 2, commit `a4ea551`). Run 1 is banked separately at commit `e361c56`; see the
disclosure below.

---

## 1. Instrument disclosure — run 1 was defective, run 2 is the verdict

Run 1 failed all 35 recordings. That result is committed unmodified. On inspection the
run-1 implementation deviated from the frozen protocol in three ways, all fixed in run 2
**without changing any gate constant**:

1. **The preregistration's PRIMARY clock path was never implemented.** The prereg says
   "if both streams carry `RecordingTimestamp` on a shared clock, the linear drift model
   is fit on those timestamps". They do: the Neuroscan export writes the *Tobii*
   recording-clock value onto each trial-marker row (40–50 exact pairs per recording,
   `NA` elsewhere). Run 1 went straight to event matching and never read the column.
2. **The secondary matcher could manufacture a fit.** It scored candidates by hit count
   into a dense EEG marker train (164–2104 marker runs vs 42 Tobii pulses); with a ±100 ms
   window over a train that dense, almost any lag "matches". Replaced by pairwise-difference
   vote concentration with a circular-shift null control, which requires many events to
   agree on one offset.
3. **`optical_blinks()` did not implement the frozen flanking clause.** The prereg
   requires an invalid run "flanked by ≥50 ms of valid samples"; the code tested a
   one-sample gap, inflating blink counts by ~100× (2171 vs 1 on S01/Sess01/ME011).

These fixes make the instrument strictly *better* and it still fails, which strengthens
rather than weakens the negative result. No threshold was touched.

**One parameter the preregistration left unfrozen**: it froze "both-eye validity invalid"
without naming the Tobii validity-code cutoff. Declared choice: `≤1`. Full 0–3 sensitivity
is reported in §4 and the verdict is invariant across it.

## 2. Clock model — PASSES, at ~5 ms

| Path | Recordings | Drift residual | Clock sub-gate (≤20 ms) |
| --- | --- | --- | --- |
| `shared_recording_timestamp` | 30 | median **5.00 ms**, max 5.49 ms on passers | 26 pass |
| — discontinuity failures | 4 | 1846 / 2367 / 4049 / 13633 ms | 4 fail |
| `none` (no clock available) | 5 | — | 5 fail |

The four large-residual recordings (ME011, SSVEP011, P3004L023, ME033) are **stream
discontinuities, not noise**: their largest single-offset cluster holds only 0.11–0.33 of
the marker pairs, i.e. the pairs split across several offsets. On ME011 the EEG time base
is perfectly monotonic at 1.000 ms steps while the embedded Tobii stamps drift 11.6 s
across the recording. The frozen least-squares fit is what gates; the collinear-fraction
statistic is diagnostic only and changes no decision.

The five `none` recordings are all of S03/Sess01, which lacks the timestamp pair *and*
carries **zero** `EventMarkerOn` pulses (`EventMarkerValue` is constant 0 — the sync marker
was never driven), against 41 pulses in S03/Sess02. No clock path exists for them; the
secondary path correctly declined rather than inventing one.

## 3. Blink correspondence — FAILS, and this is what kills E1

Restricted to the 26 recordings whose clock is valid, so clock error cannot be blamed:

| Statistic | Value |
| --- | --- |
| Recordings reaching match rate ≥ 0.80 | **0 / 26** |
| Pooled match rate | **0.305** (46 of 151 blinks) |
| Per-recording match rate | median 0.23, max 0.75 |
| Optical blinks detected | 151 total, median 4 per recording, 6 recordings with 0 |

Two independent causes, both properties of this panel rather than of the code:

- **The optical blink train is starved.** Tobii tracking validity runs 33–42% in this
  panel; ~59% of samples are code 4 ("eye not found"). Invalid runs are therefore dense
  and short, and the frozen "≥50 ms valid flanking" clause — which exists precisely to
  separate blinks from tracking dropout — rejects nearly all of them. On ME011: 5522
  invalid runs, 2171 in the 50–500 ms band, **1** with clean flanking. Against recordings
  spanning 381–766 s (E1 diagnostic), the rule recovers on the order of one blink per
  minute where spontaneous blinking is 10–20/min.
- **The survivors sit on the tolerance boundary.** Median |lag| on matched blinks is
  ~30–55 ms against a frozen ±50 ms window. This is the expected physiological offset
  between tracking loss (eyelid begins to close) and the VEOG-surrogate peak (mid-closure).
  **No offset correction was applied** — doing so after seeing the outcome would be
  outcome-tuning. An offset-corrected variant would require a new preregistration and is
  an operator decision.

## 4. Sensitivity to the unfrozen validity cutoff

Pooled over the 26 clock-valid recordings:

| Validity cut | Blinks | Matched | Pooled rate | Per-rec median | Recordings ≥ 0.80 |
| --- | --- | --- | --- | --- | --- |
| ≤0 | 151 | 46 | 0.305 | 0.23 | 0 |
| **≤1 (declared)** | 151 | 46 | **0.305** | 0.23 | **0** |
| ≤2 | 134 | 34 | 0.254 | 0.17 | 0 |
| ≤3 | 167 | 48 | 0.287 | 0.17 | 2 |

The verdict is invariant: no cutoff brings the panel near the 0.80 bar (at ≤3, 2 of 26
recordings clear it while the pooled rate still falls).

## 5. Subject-level outcome (ITT, excluded-and-counted)

| Subject | Recordings | Gate passes | Clock-valid | Blinks | Median match rate | Aligned |
| --- | --- | --- | --- | --- | --- | --- |
| S01 | 5 | 0 | 3 | 35 | 0.00 | **No** |
| S02 | 15 | 0 | 14 | 120 | 0.06 | **No** |
| S03 | 15 | 0 | 9 | 96 | 0.00 | **No** |

`aligned_subjects: []` — `excluded_and_counted: [S01, S02, S03]`. No subject was repaired
ad hoc.

## 6. Consequence

The frozen extension rule requires ≥2/3 pilot subjects to pass; 0/3 did, so E1 was **not**
extended to the remaining covered subjects. Every E2 measurement (M1–M4) is defined "on
aligned subjects" and there are none, so **M1–M4 do not run** and no WAVE3 ledger row is
added or altered. WAVE3 numbers remain untouched, as required.

What the wave did establish, and what is worth banking: **the cross-stream clock is
solid** — an exact shared timestamp yields ~5 ms alignment on 26 of 35 recordings — while
**the optical blink instrument is not validated on this panel**, because eye-tracker data
quality (~59% sample loss) starves the frozen blink definition. The failure is located in
the optical instrument, not in the alignment machinery.

Open to the operator (each would need a new preregistration; none taken here):
a blink definition robust to fragmented tracking; a lag tolerance that admits the
physiological tracking-loss-to-VEOG-peak offset; or restricting the panel to
high-validity recordings before gating.
