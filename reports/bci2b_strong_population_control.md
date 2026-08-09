# BCI2b strong population control

This is a development-only robustness experiment. Each LOSO backbone excludes only the recipient and uses the other eight participants. The primary duration is frozen at 120 s; 30 s and 60 s are robustness analyses. Eligible protocol units are 26/27, 26/27, and 26/27, respectively, while all 9 participants remain represented. The blocked short-support unit stays in the availability denominator. FULL_AVAILABLE is not reported because support availability is not identical for every unit.

| support | U_P8 mean | median | wins | exact two-sided p | U_W8 mean | median | wins | exact two-sided p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 s | +0.00069 | -0.00035 | 4/9 | 0.914062 | +0.02748 | +0.02797 | 9/9 | 0.003906 |
| 60 s | +0.00173 | +0.00231 | 5/9 | 0.710938 | +0.02587 | +0.02653 | 9/9 | 0.003906 |
| 120 s | +0.00194 | +0.00202 | 6/9 | 0.671875 | +0.01879 | +0.02124 | 8/9 | 0.007812 |

At 120 s, U_P8 is +0.00194 (6/9) and U_W8 is +0.01879 (8/9). All compatible WRONG8 donors were scored separately and averaged within recipient; they are training-seen donor context sensitivity. The frozen two-heldout cyclic donor remains a separate unseen-WRONG sensitivity and is never pooled with WRONG8. Seeds and protocol units are first aggregated within participant; scientific n=9. Participant-first primary RRMSE is RAW 0.49133, LINEAR-MATCH 0.20041, DET-MATCH 0.17156, DIFF-POP8 0.15418, and DIFF-MATCH8 0.15224. DIFF-MATCH8 natural means are EOG attenuation +0.10145, preservation 0.78163, MI-band distortion 0.07806, covariance 0.13720, MI kappa 0.34829, and ERD preservation 0.93516. Participant bootstrap intervals are descriptive. Full participant and seed values are in the accompanying CSV/JSON files.

## Execution notes

Training completed 27/27 checkpoints at the frozen 8,000-update budget. Slurm QOS submit/GPU limits delayed inference submissions but did not alter the protocol. One 15.6-second support unit was blocked from duration-labelled effects rather than silently truncated.
