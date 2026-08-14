# V43 preregistration

```text
V43 preregistration — frozen before first Slurm submission.

GATE RULE (the method):
  lambda = clip(tau2 / (tau2 + within/4), 0, 1)          [V19 closed form, scalar, primary]
  tau2   = mean squared deviation of fold-TRAIN owners' 120-s transfers around the
           population transfer of the (session, task) cell
  within = mean squared deviation of the four 30-s sub-block transfers around the
           full 120-s transfer of the support owner
  h_gated = h_pop + lambda * (h_120 - h_pop)
  HARD GATE: effective support < 60 s OR within above the frozen threshold
             (95th percentile of fold-train within values) -> lambda := 0 exactly.
  lambda = 0 MUST produce a signature array bit-identical to the frozen POP state
  (quality features clamped to the fold-train 30-s range; registry30 continuous
  center/scale reused for normalization so POP comparability is exact).
  Secondaries (reported, never primary): per-row lambda; raw un-shrunk 120-s state.

S1 (frozen-checkpoint floor probe) ADJUDICATES ONLY:
  F1 wrong-donor harm elimination:
     mean[RRMSE(WRONG_EB120) - RRMSE(POP)] <= +0.010
     AND reduction vs frozen WRONG harm has bootstrap CI-low > 0.
  F2 short-support harm elimination:
     mean[RRMSE(MATCH_EB10) - RRMSE(POP)] <= +0.010   (frozen 10-s spike is +0.0822).
  F3 provisional non-inferiority (frozen checkpoint, OOD-caveated):
     mean[RRMSE(MATCH_EB120) - RRMSE(POP)] <= +0.005.
  The MATCH_EB120 - POP gain reading is NON-ADJUDICATING in S1 (the checkpoint was
  trained on 30-s states; a positive or null here neither opens nor closes the gain claim).
  Definitive non-inferiority margin for the retrained model (S2, if run): delta = 0.002.

S1.5 (oracle-trained ceiling probe) GO RULE for the gain leg:
  train with query-fitted (generative-truth) conditioning; evaluate ORACLE vs POP
  on held-out participants of the same cells.
  GO  iff mean[RRMSE(POP) - RRMSE(ORACLE)] >= +0.020 AND bootstrap CI-low > +0.005.
  NO-GO -> the waveform-level gain claim is declared dead on this panel; V43 proceeds
  floor-only; S2 trains the gated model for the floor endpoints only.

STATISTICS: participant-first (n=15) mean, median, positive count, 5000-draw bootstrap;
Holm over the family {F1, F2, F3}. No other corrections, no post-hoc endpoint additions.
```
