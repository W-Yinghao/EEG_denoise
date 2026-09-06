# V33P full-sampler model selection

## Selection contract

For each outer fold and seed, Stage A used three training participants and a
separate three-participant validation group. Validation privacy attackers were
retrained on sanitized Session-T outputs and tested on Session E. No outer-test
participant or outcome entered epoch selection.

SANDiff retained two checkpoint rules from the same training trajectory:

- `single_timestep`: minimum deterministic t=500 reconstruction/task objective;
- `full_10_step`: maximum validation privacy–utility balance after deployed
  K=1, ten-step sampling at strong replacement.

The simple balance was the mean of fixed/retrained task BA, minus 0.25 times
adaptive subject BA, minus a small penalty for verification AUROC above chance.
It is a selection score, not a scientific endpoint.

## Selected epochs

| fold | seed | EEGNet | SANDiff single-t | SANDiff full sampler | one-step full output |
|---:|---:|---:|---:|---:|---:|
| 0 | 20260920 | 58 | 9 | 20 | 20 |
| 0 | 20260921 | 73 | 18 | 50 | 80 |
| 1 | 20260920 | 44 | 4 | 80 | 80 |
| 1 | 20260921 | 68 | 13 | 10 | 30 |
| 2 | 20260920 | 11 | 7 | 20 | 10 |
| 2 | 20260921 | 48 | 16 | 70 | 30 |

The SANDiff rules selected different epochs in 6/6 cells. Stage B then refit
from scratch on all six non-test Session-T participants and captured the two
frozen epochs from the same trajectory.

## Outer-test ablation

| checkpoint | fixed BA | retrained BA | adaptive subject BA | verification BA | verification AUROC |
|---|---:|---:|---:|---:|---:|
| single-timestep | 0.352238 | 0.351273 | 0.662616 | 0.559799 | 0.578079 |
| full sampler | 0.352816 | 0.353588 | 0.661265 | 0.543789 | 0.575094 |

Full-sampler selection changed fixed BA by +0.000579 and adaptive privacy
utility by +0.001350. Participant bootstrap intervals crossed zero for both;
the improvement is directionally aligned but small and heterogeneous. The
single-timestep result remains committed as an ablation.
