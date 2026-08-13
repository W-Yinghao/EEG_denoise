# V35P head-aware attacks

Attackers were retrained separately for every release using outer-test Session T as
gallery and Session E as query. Both linear and adaptive MLP attacks received H, Z,
[H,Z], and [H,U] explicitly.

## Adaptive MLP balanced accuracy

| Method | A_H | A_Z | A_HZ | A_HU | Primary max(H,Z,HZ) |
|---|---:|---:|---:|---:|---:|
| RAW | 0.579090 | 0.743441 | 0.739390 | 0.736690 | 0.743441 |
| HEAD_ONLY | 0.579090 | 0.557485 | 0.557677 | 0.479167 | 0.579090 |
| LEACE | 0.579090 | 0.669367 | 0.683063 | 0.688272 | 0.683063 |
| Fiber-OneStep | 0.579090 | 0.559799 | 0.557099 | 0.559028 | 0.579090 |
| Fiber-Stratified-Resample | 0.579090 | 0.498457 | 0.503665 | 0.479938 | 0.579090 |
| Fiber-SANDiff | 0.579090 | 0.521412 | 0.535301 | 0.511960 | 0.579090 |

Once A_H is included, the three exact strong channels have the same registered
primary adaptive leakage. Lower A_Z values for stochastic channels are finite-model
geometry effects, not privacy below the head-visible boundary.

## Conditional fiber closure

`CE(A_H)-CE(A_HU)` was negative for every exact channel:

| Method | Linear | Adaptive MLP |
|---|---:|---:|
| Fiber-OneStep | -0.006491 | -0.030675 |
| Fiber-Stratified-Resample | -0.234742 | -0.251974 |
| Fiber-SANDiff | -0.082069 | -0.103122 |

Thus no registered finite attacker gained subject prediction from U' after H was
provided explicitly. Negative values indicate worse finite generalization with the
additional coordinates, not negative information.
