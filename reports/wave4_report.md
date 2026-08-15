# WAVE4 — STOP 2: final report (optical instrument, four customers)

Branch `codex/wave4-optical`. CPU only throughout. Zero sealed contact. **No WAVE3 number
was edited; every row below is ADDED.** All E2 measurements carry the frozen cross-panel
caveat: measured on Eye-BCI, entering the MobileBCI ledger as order-of-magnitude **BOUNDS**
with an explicit cross-panel label — no comparability check was performed in this wave.

## Wave trajectory

| Stage | Outcome |
| --- | --- |
| **E0** manifest + download rule | **PROCEED** (31 covered subjects, 15.67 GiB, 874 GiB free) |
| **E1** alignment, frozen gate | **FAIL 0/3 pilot subjects** — banked, never edited |
| **E1R** one registered repair | **TIER-S PASS, TIER-E FAIL**, close rule did not fire |
| **E2** M2 → M4 → M3 (Tier-S authorized) | measured; **M1 and full M3 remain locked** |

The clock layer passed at ~5 ms and was reused verbatim from E1 into E1R and E2; no drift
fit was ever recomputed. E2 ran on the 12 E1R-eligible recordings (S01×2, S02×9, S03×1).

## M1 — optical type labels — **NOT RUN**

Locked by `tier_e = false`. The event-level instrument does not exist on this panel:
forward blink match reached median 0.2409 against a 0.70 bar, and crucially was **not
rescued by good tracking** (0.27–0.35 at validity ≥ 0.90). No κ was computed, no WAVE3 T1
re-census was run, and no optical ψ was substituted anywhere.

## M2 — A4 reference-channel-error row (12/12 recordings included)

| Statistic | Participant-first (n=3) | Recording-level, declared secondary (n=12) |
| --- | --- | --- |
| `D_rms` | **1.7673** [1.6025, 2.0136] | 1.6506 [1.3745, 1.9114] |
| `D_corr` | **0.9000** [0.8004, 1.0435] | 0.8827 [0.7352, 1.0249] |

Per-subject `D_rms`: S01 1.6856, S02 1.6025, S03 2.0136.

`D_rms > 1` means the discrepancy between the EOG-referenced and optical-referenced
artifact estimates **exceeds the magnitude of the EOG-referenced estimate itself**, and
`D_corr ≈ 0.90` means the two estimates correlate at only ≈ 0.10.

**This number must not be read as "EOG references are ~177% wrong."** It is
panel-specific and the dominant cause is identifiable: **this panel's only ocular channel
is `HEO`, which is horizontal**, while the frontal target block (FP1 FPZ FP2 AF3 AF4 F7 F8)
is dominated by *vertical* blink artifact. The measurement is therefore dominated by
reference-*axis* mismatch rather than by generic EOG measurement noise or crosstalk. As
frozen, `D_rms` is an **upper bound** that also absorbs the optical reference's own
limitations, which this panel cannot separate.

**Ledger reading**: the A4 row is entered as an upper bound, explicitly labelled
horizontal-reference-limited and cross-panel. It does not tighten A4 for a panel carrying
a proper VEOG channel.

## M4 — exogeneity (11/12 included; P3004L011 excluded at 53 s < 60 s)

Reported after the declared velocity-window repair (addendum 2). The **banked first run
returned 0/12** with a 1-sample velocity estimator whose median during valid gaze was
24.5–27.0 °/s — a noise floor sitting at the very 30 °/s threshold it was testing. That
result is preserved unedited at `m4/m4_exogeneity.json`; the repair moved no frozen
constant (30 °/s, 100 ms, 60 s, ridge 0.05, OPERA 0.055 all unchanged) and only declared
the previously unfrozen estimation window at the vendor-standard 20 ms.

| Statistic | Participant-first (n=3) | Recording-level secondary (n=11) |
| --- | --- | --- |
| **R² in-sample** (frozen primary) | **0.6198** [0.4904, 0.8509] | 0.5257 [0.3919, 0.6517] |
| **R² 2-fold CV** (frozen companion) | **−0.7596** [−2.7916, 0.4164] | −2.2374 [−5.4431, 0.0241] |

Velocity sweep: 20 °/s → n=9, R²_in 0.5207, R²_cv −4.2447; **30 °/s (frozen)** → n=11,
R²_in 0.6198, R²_cv −0.7596; 50 °/s → n=12, R²_in 0.4706, R²_cv −0.9269.
Vendor I-VT companion mask (n=12, bypasses our estimator entirely): R²_in 0.4725
[0.4323, 0.4946], R²_cv −0.5646 — the same picture, so the finding is not an artifact of
the mask choice.

**Verdict: NOT STABLY IDENTIFIABLE — not comparable to OPERA's 0.055.** The in-sample R²
is an order of magnitude above OPERA, but the frozen cross-validated companion is
**negative**, meaning the fitted HEO↔posterior relationship fails to transfer between
contiguous halves of the *same* recording. A negative CV R² refutes the stability that a
crosstalk estimate requires; the in-sample value is consistent with shared low-frequency
drift and common-reference structure rather than a stable neural-crosstalk pathway.
Reporting 0.62 as a crosstalk estimate — "11× OPERA" — would be wrong, and this wave does
not do so. A0-exogeneity is **not** measured on this panel.

## M3 — readout bound, segment-level variant (12/12 included)

**DIFF-class readout: NOT-MEASURABLE-THIS-PANEL** — no V44-class checkpoint applies to the
62-channel Neuroscan montage and checkpoint porting is prohibited. LINEAR-vs-analytic only.

| Statistic | Participant-first (n=3) | Recording-level secondary (n=12) |
| --- | --- | --- |
| relative gain `g` | **0.0620** [0.0021, 0.1374] | 0.0466 [0.0239, 0.0730] |

Per-subject: S01 0.0021, S02 0.0464, S03 0.1374. Best analytic family: `amplitude_gain`
8/12, `fir_lagged` 2, `kernel_ridge` 1, `rank3_derivative` 1.

**Verdict: SIZED at 0.0620** — the frozen rule reads on the point estimate and
0.0620 > 0.03, so the readout row is sized rather than bounded.

**Stated plainly: neither interval excludes the bounded regime.** The participant-first CI
runs [0.0021, 0.1374] and the recording-level CI [0.0239, 0.0730]; both straddle 0.03. The
verdict follows the frozen rule, but the evidence separating "sized at ~0.06" from
"bounded at 0.03" is weak at n=3 subjects. This is the honest state of the readout row,
not a demonstration that richer readouts buy 6%.

What M3 *does* settle: unlike WAVE3 T3, this instrument is **non-degenerate**. The optical
reference is not the generative source of the EEG artifact, so LINEAR enjoys no tautological
advantage, and a finite, non-trivial gain is measurable at all. T3's "both oracles
degenerate" blocker is removed at segment level.

## Ledger addendum — rows ADDED beside WAVE3

```text
WAVE4-M2  A4 reference-channel-error (Eye-BCI, 12 rec / 3 subj):
          D_rms 1.7673 [1.6025, 2.0136], D_corr 0.9000 [0.8004, 1.0435].
          UPPER BOUND, horizontal-reference-limited (panel has HEO only, no VEOG),
          cross-panel. Does not tighten A4 for VEOG-bearing panels.
WAVE4-M4  A0 exogeneity (Eye-BCI, 11 rec / 3 subj): R2_in 0.6198 [0.4904, 0.8509],
          R2_cv -0.7596 [-2.7916, 0.4164]. NOT STABLY IDENTIFIABLE; not comparable to
          OPERA 0.055. No crosstalk estimate entered.
WAVE4-M3  Readout row, segment-level (Eye-BCI, 12 rec / 3 subj): relative gain
          0.0620 [0.0021, 0.1374] -> SIZED at 0.0620; CI does not exclude 0.03.
          DIFF-class NOT-MEASURABLE-THIS-PANEL. Instrument non-degenerate (T3 blocker
          removed at segment level).
WAVE4-M1  NOT RUN (TIER-E locked). Customers 3 and 4 remain
          instrument-limited-unserved.
```

## A1 verdict

**A1 (typed-operator separation) is UNADJUDICATED.** Its adjudication was defined to run
through M1's optical type labels under the unified WAVE3 gate; M1 is locked by the Tier-E
failure, so no A1 evidence was produced in either direction. O1's type question likewise
stays open. This wave neither resurrects nor closes any gated route: TROCA/Panel-T
eligibility remains a separate operator decision, and nothing here supports one.

## The four customers, final state

| Customer | State after WAVE4 |
| --- | --- |
| 1 — readout ledger row | **SERVED at segment level** (sized 0.0620, CI straddles the 0.03 bound) |
| 2 — A4 reference-channel error | **SERVED as an upper bound**, horizontal-reference-limited |
| 3 — valid artifact-type labels | **UNSERVED** — instrument-limited (Tier-E fail) |
| 4 — non-degenerate event-level oracle | **UNSERVED** — instrument-limited; segment-level oracle *is* non-degenerate |

Two of four customers are served, both with explicit bounds and caveats. Two remain
unserved for want of an event-level instrument that this panel's eye-tracker data quality
cannot support.

## Honest-power statement

Only **3 subjects** (S01, S02, S03) survive clock validity and eligibility, distributed
2/9/1 across recordings. Participant-first bootstrap CIs on n=3 have very little
resolution; recording-level intervals are reported alongside as a declared secondary and
are not independent (S02 contributes 9 of 12). Every verdict above should be read as
provisional in that light. E1 was never extended past the 3-subject pilot because the
frozen extension rule did not fire.

## Provenance

Preregistrations: `wave4_preregistration.md` (`e20eeb4`), `wave4_e1r_preregistration.md`
(`d7f42f7`), `wave4_e2_preregistration.md` (`0567bbf`),
`wave4_e2_addendum2_m4_velocity.md` (`0b05876`) — each committed before the code it
governs. Stop-point reports: `wave4_e0_manifest.md`, `wave4_e1_alignment.md`,
`wave4_e1r_report.md`, this file. Artifacts under `results/wave4_optical/{manifest,
alignment, e1r, m2, m3, m4}/`. Two instrument defects were disclosed and corrected in
flight (E1 run-1, M4 run-1); in both cases the superseded result is preserved unedited and
the fix moved no frozen constant.
