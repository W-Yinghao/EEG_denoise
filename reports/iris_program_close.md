# Program close-out — S356 verdict and the final state (operator ruling 2 executed)

The funded final experiment is complete; the experimental program is CLOSED.
Branch codex/iris; preregs ac0762b + amendment b1dcc2f before execution; both the
banked ITT pass and the amended pass are in results/iris/s356/ unedited.

## S356 decision JSONs verbatim

Banked ITT (never edited): verdict INCONCLUSIVE; gain(259) -1.4276
[-4.4154, +0.0760] — one objectively defective EVAL subject (AA4, injection ratio
167.2 vs 0.33-4.02 all others) dominates the mean.

Amended (guard [0.1,20] excludes AA4 only; WRONG-embedding control added):

```json
"verdict": "SCOPED_C1_COUNTEREXAMPLE_FLAT_SUBJECT_SPECIFIC"
"gain_by_n_guarded": {"30": 0.0832 [0.0379,0.1177], "60": 0.0805 [0.0535,0.1072],
  "120": 0.0655 [0.0440,0.0861], "200": 0.0585 [0.0381,0.0798],
  "259": 0.0608 [0.0406,0.0835], positive 13-14/14 throughout}
"trend_max_minus_min": -0.0224 [-0.0491, +0.0158]   (FLAT)
"subject_specificity(n=259)": {"own": +0.0608, "wrong": -0.1015,
  "own_minus_wrong": +0.1623 [+0.1314, +0.1946]}
```

## Reading (the paper-relevant sentence)

The scale objection to C1 is dead a second time — on real data, across n=30..259 the
conditioning gain is FLAT. What S356 adds is a decomposition, not a reversal: the
conditioning channel carries nothing UNCALIBRATED (the three banked kills stand),
but a support-calibrated embedding carries subject-specific artifact-coupling
information (+0.0608, wrong-embedding control decisive at +0.1623, harm law
reproduced in embedding space). C1's wording costs one word: "universal" ->
"uncalibrated"; the information account (subject information lives in the artifact
coupling) is the invariant and is strengthened. Caveats frozen in the amendment
travel with every citation: injected episodes, single panel, fresh compact class,
oracle ceiling (no deployment claim).

## Program totals

- Compute: **~0.7 GPU-h** of the 400-h cap (F1 0.1, F4 0.2, S356 0.4; all else CPU).
- Preregistrations: 8 documents + 4 amendments, every threshold frozen pre-data.
- Verdicts banked: K1-K3, P1, P1R, W1-W3, T, T2, F1, F2-a/b, F2-R, F4, S356
  (ITT + amended) — positives and negatives alike in the digest.
- Sealed: EEGEyeNet-55 frozen and earmarked (paper-time UQ confirmation);
  BrainID / PhysioMotion / SHU untouched.
- Defect log: every defective run banked unedited beside its correction
  (P1 episode machinery clean; T gaze encoding; F2-R gate scoping; S356 AA4).

The next act is the paper. No manuscript text was written here (per mission rules);
docs/EEG_denoise_paper_design_cleanslate.md + the digest are the writing-round
inputs, with the C1 re-scoping and the F4 UQ headline as the two changes the
cleanslate design must absorb.
