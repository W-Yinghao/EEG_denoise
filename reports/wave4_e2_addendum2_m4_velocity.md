# WAVE4-E2 addendum 2 — M4 velocity-estimation window (declared before rerun)

Committed BEFORE the M4 rerun. Base: `codex/wave4-optical`. CPU only.

## What happened

M4 as first run returned **0 of 12 recordings included** at the frozen 30 °/s threshold:
no recording reached the 60 s minimum of fixation-masked data. That result is banked
(`results/wave4_optical/m4/m4_exogeneity.json`, commit before this addendum) and is
**not edited**. The repaired run is written beside it, never over it.

## Why this is an estimator defect, not a data property

Neither `wave4_preregistration.md` nor `wave4_e2_preregistration.md` froze the **window
over which gaze velocity is estimated**. The implementation therefore used a 1-sample
finite difference at the 300 Hz Tobii rate (3.3 ms). That estimator cannot resolve the
frozen threshold, and the argument does not depend on the M4 outcome:

| Estimator | Median velocity during valid gaze | Fraction < 30 °/s |
| --- | --- | --- |
| 1-sample (3.3 ms) finite difference | **24.5–27.0 °/s** | 0.56–0.62 |
| 20 ms window (vendor I-VT standard) | 8.2–9.0 °/s | 0.91–0.93 |

The 1-sample estimator's **noise floor sits at the threshold it is meant to test** — its
median during ordinary valid gaze is already ~25 °/s, so a "< 30 °/s sustained ≥ 100 ms"
criterion can almost never fire regardless of what the eye is doing. Independently, the
vendor's own I-VT classifier (`GazeEventType`, exported with the data and computed with
proper filtering) labels **84–87% of samples as Fixation** in the same recordings.
Fixation is abundant; the estimator could not see it.

## Declared repair (no frozen constant moves)

- **Velocity-estimation window: 20 ms** (vendor I-VT standard), i.e. displacement measured
  across ±10 ms rather than one sample.
- **Unchanged**: the 30 °/s threshold, the ≥100 ms sustained requirement, the 60 s
  minimum, the sweep 20/30/50 °/s, the ridge convention, the posterior block, and the
  OPERA 0.055 comparison. Nothing that was frozen moves.
- **Independent companion (declared, reported alongside)**: the same M4 statistic computed
  on the vendor's `GazeEventType == "Fixation"` mask, which bypasses my velocity estimator
  entirely. It is reported as a companion, not as the primary.

Primary verdict is read on the frozen 30 °/s with the 20 ms window. Both the banked
1-sample result and the vendor-mask companion are reported in full, so the reader can see
every version rather than only the one that worked.
