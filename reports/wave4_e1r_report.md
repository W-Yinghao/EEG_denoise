# WAVE4-E1R — repaired optical-correspondence instrument: tier verdicts

**TIER-S: PASS. TIER-E: FAIL. Close rule does NOT fire.**

Preregistration addendum: `reports/wave4_e1r_preregistration.md`, commit `d7f42f7`
(frozen before any repair code ran). Artifacts:
`results/wave4_optical/e1r/{correspondence_table.csv, decision.json}`.
Clock layer reused verbatim from the banked E1 artifact — no drift fit recomputed, no
banked E1 number edited, no WAVE3 number touched. 26 clock-valid recordings measured,
CPU only (Slurm job 943487).

```json
{"tier_s": true, "tier_e": false, "eligible_n": 12, "close_fired": false}
```

## Verdicts at the frozen values

| Tier | Requirement (frozen) | Observed | Verdict |
| --- | --- | --- | --- |
| **S** | reverse elevation CI-low > 0 in ≥60% of eligible | **12/12 = 100%** | PASS |
| **S** | pooled elevation CI-low > 0 | pooled **0.2203**, CI-low **0.1778** | PASS |
| **E** | forward match median ≥ 0.70 | **0.2409** | **FAIL** |
| **E** | circular-shift p < 0.01 in ≥60% of eligible | 10/12 = 83.3% | pass |

Tier-E fails on the median clause alone; its significance clause passes. Both Tier-S
clauses pass with margin.

## What the two instruments say, and why they disagree

The co-primary design was built precisely to separate these, and it did:

- **Reverse (VEOG-anchored, segmentation-free): strongly positive.** Every eligible
  recording shows Tobii invalidity elevated inside the physiological window around real
  VEOG blinks, elevation 0.09–0.35 over the circular-shift null. Real blinks *do* drive
  tracker invalidity.
- **Forward (Tobii-anchored, segmented): real but low-yield.** Only ~24% of blink
  candidates carry a VEOG peak in the window — yet 10/12 recordings sit at the permutation
  floor (p = 0.005), so the association is unambiguously real, just sparse.

The asymmetry is a **precision** problem, not a correspondence failure: most 50–500 ms
invalid runs are not blinks (micro-dropouts, partial occlusion, gaze-away), while blinks
reliably produce invalidity. Critically, this is **not** an artifact of poor tracking — in
the three best-tracked recordings (validity 0.94/0.92/0.90) forward match is still only
0.27/0.35/0.29. Better tracking does not rescue the forward direction, which is why the
event-level tier is genuinely out of reach on this panel rather than merely underpowered.

The auxiliary pupil-collapse evidence (never gating) agrees: collapse fractions are
0.00–0.28 across eligible recordings, i.e. most candidates show no pupil signature either.

## Per-recording correspondence table (26 clock-valid recordings)

Eligible (Tobii validity ≥ 0.40, frozen):

| Recording | Validity | nVEOG | nCand | Forward | p | Rev. elevation | Rev. CI-low | Pupil |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S02/Sess03/SSVEP023 | 0.942 | 192 | 33 | 0.273 | 0.005 | 0.1613 | 0.0931 | 0.00 |
| S02/Sess02/P3005L022 | 0.915 | 312 | 200 | 0.350 | 0.005 | 0.2318 | 0.1882 | 0.07 |
| S02/Sess02/MI022 | 0.896 | 181 | 160 | 0.287 | 0.005 | 0.3532 | 0.2859 | 0.03 |
| S02/Sess02/P3004L022 | 0.818 | 273 | 276 | 0.264 | 0.005 | 0.2569 | 0.2078 | 0.01 |
| S02/Sess03/ME023 | 0.805 | 165 | 49 | 0.449 | 0.005 | 0.2088 | 0.1276 | 0.00 |
| S02/Sess02/SSVEP022 | 0.787 | 228 | 159 | 0.157 | 0.005 | 0.1703 | 0.1036 | 0.04 |
| S02/Sess02/ME022 | 0.757 | 179 | 225 | 0.111 | 0.015 | 0.3430 | 0.2768 | 0.02 |
| S02/Sess03/MI023 | 0.735 | 123 | 49 | 0.265 | 0.005 | 0.1321 | 0.0308 | 0.00 |
| S02/Sess03/P3005L023 | 0.720 | 279 | 267 | 0.195 | 0.005 | 0.0905 | 0.0426 | 0.01 |
| S03/Sess02/SSVEP032 | 0.517 | 97 | 214 | 0.023 | 0.692 | 0.3009 | 0.2480 | 0.07 |
| S01/Sess01/P3005L011 | 0.417 | 283 | 209 | 0.187 | 0.005 | 0.1975 | 0.1740 | 0.09 |
| S01/Sess01/P3004L011 | 0.412 | 266 | 161 | 0.217 | 0.005 | 0.1975 | 0.1782 | 0.12 |

Excluded and counted (validity < 0.40, never repaired): S02/Sess01/SSVEP021 (0.393),
S01/Sess01/MI011 (0.326), S02/Sess01/ME021 (0.266), S02/Sess01/P3004L021 (0.256),
S03/Sess02/MI032 (0.241), S03/Sess03/P3004L033 (0.190), S03/Sess02/ME032 (0.093),
S03/Sess02/P3005L032 (0.086), S03/Sess03/P3005L033 (0.055), S03/Sess03/SSVEP033 (0.054),
S03/Sess02/P3004L032 (0.050), S03/Sess03/MI033 (0.045), S02/Sess01/P3005L021 (0.045),
S02/Sess01/MI021 (0.024). Full rows for all 26 are in `correspondence_table.csv`.

**Panel note.** Tracking quality is strongly heterogeneous — S02/Sess02–03 run 0.72–0.94
validity while S03/Sess02–03 and S02/Sess01 run 0.02–0.52. The E1 report's "33–42%"
characterisation came from the four diagnostic recordings and understates the good half
of the panel; the eligibility gate is what makes this tractable.

## Eligibility sweep (verdict read at the frozen 0.40)

| Threshold | Eligible | Rev. positive | Pooled CI-low | Tier-S | Forward median | p-frac | Tier-E |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.30 | 14 | 0.93 | 0.1601 | PASS | 0.2061 | 0.71 | FAIL |
| 0.35 | 13 | 0.92 | 0.1563 | PASS | 0.2174 | 0.77 | FAIL |
| **0.40** | **12** | **1.00** | **0.1778** | **PASS** | **0.2409** | **0.83** | **FAIL** |
| 0.45 | 10 | 1.00 | 0.1733 | PASS | 0.2649 | 0.80 | FAIL |
| 0.50 | 10 | 1.00 | 0.1733 | PASS | 0.2649 | 0.80 | FAIL |
| 0.55 | 9 | 1.00 | 0.1632 | PASS | 0.2653 | 0.89 | FAIL |
| 0.60 | 9 | 1.00 | 0.1632 | PASS | 0.2653 | 0.89 | FAIL |

Both verdicts are invariant across the entire sweep: Tier-S passes everywhere
(pooled CI-low 0.156–0.178), Tier-E fails everywhere (median 0.206–0.265, never within
0.43 of the 0.70 bar). Neither verdict depends on the frozen threshold choice.

## Authorization state (E2 NOT run — stopped for the operator)

| Measurement | Tier required | Status |
| --- | --- | --- |
| **M2** — A4 reference-channel-error row | S | **AUTHORIZED** |
| **M4** — exogeneity vs OPERA 0.055 | S | **AUTHORIZED** |
| **M3** — readout bound, *segment-level variant* | S | **AUTHORIZED** |
| **M3** — full (event-level) | E | LOCKED |
| **M1** — optical type labels, κ gate, WAVE3 T1 re-census | E | LOCKED |

The close rule did not fire, so the optical instrument stays **open on this panel** at
segment level. Of the four WAVE3 customers: the A4 reference-channel-error row and the
segment-level readout bound are now servable; **customer 3 (valid artifact-type labels)
and customer 4 (a non-degenerate event-level oracle) remain instrument-limited-unserved**,
and with them O1's type question and the A1 typed-operator-separation verdict stay open.

Per the execution order, E2 measurements are authorized by these verdicts but wait for the
operator read. Nothing further has been run.

## Ledger line

`WAVE4-E1R optical correspondence (Eye-BCI, 26 clock-valid recordings, 12 eligible at
validity ≥ 0.40): reverse VEOG-anchored elevation 0.2203 [CI-low 0.1778], 12/12 positive
→ TIER-S PASS, segment-level optical reference established; forward event match median
0.2409 (p<0.01 in 10/12) → TIER-E FAIL, event-level typing unavailable on this panel.
Added beside the banked E1 row; no E1 or WAVE3 number edited.`
