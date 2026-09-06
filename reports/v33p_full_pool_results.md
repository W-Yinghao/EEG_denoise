# V33P six-subject full-pool results

## Absolute frontier

All values use the same three outer test groups. Seeds are averaged within
participant before participant-first inference.

| method | fixed BA | retrained BA | adaptive subject BA | verification AUROC | ECE | worst-participant BA |
|---|---:|---:|---:|---:|---:|---:|
| RAW | 0.358603 | 0.354552 | 0.741127 | 0.630805 | 0.296559 | 0.261574 |
| LEACE | 0.355517 | 0.349730 | 0.658372 | 0.577914 | 0.298801 | 0.277778 |
| DANN | 0.346836 | 0.345679 | 0.664545 | 0.602607 | 0.499576 | 0.277778 |
| one-step strong | 0.353781 | 0.354938 | 0.657986 | 0.583221 | 0.378814 | 0.292245 |
| SANDiff strong | 0.352816 | 0.353588 | 0.661265 | 0.575094 | 0.456326 | 0.289352 |

## Strong SANDiff versus RAW

- Fixed-head BA: −0.005787; participant bootstrap 95% interval
  [−0.023341, +0.014275], 3/9 positive.
- Adaptive privacy utility: +0.079861; interval [+0.051505, +0.110340],
  9/9 positive.
- Verification-AUROC reduction: +0.053964; interval
  [+0.028387, +0.079703], 9/9 positive.
- Retrained-head BA changed by −0.000965 at the aggregate level.

Thus full-pool SANDiff gives a reproducible privacy reduction with a small,
heterogeneous fixed-head utility cost. It is a positive privacy–utility trade,
not a simultaneous improvement in every outcome. Calibration is worse than
RAW, while worst-participant task BA is higher.

## SANDiff versus LEACE and one-step

Versus LEACE, SANDiff fixed-head BA was −0.002701 (interval −0.016975 to
+0.011767; 4/9 positive), but retrained-head BA was +0.003858. Adaptive subject
BA was 0.002894 higher for SANDiff, whereas verification AUROC was 0.002820
lower. The generative replacement comparison with linear erasure is mixed.

Versus strong one-step, SANDiff fixed-head BA was −0.000965 (interval
−0.009645 to +0.007909; 4/9), adaptive privacy utility was −0.003279
(−0.016204 to +0.009066; 4/9), and verification-AUROC reduction was +0.007268
(−0.001163 to +0.019651; 6/9). The two mechanisms are practically equivalent
at current participant resolution.

## V32P versus V33P

Six-subject refitting raised RAW fixed/retrained BA by +0.038387/+0.022762,
but also raised RAW adaptive subject BA by +0.072531. Strong SANDiff
fixed/retrained BA increased by +0.027778/+0.023727; its absolute adaptive
subject BA increased by +0.007137 and verification AUROC by +0.001619.

The larger training pool therefore improved task generalization and greatly
increased SANDiff's *relative* privacy gain over RAW. It did not reduce the
absolute residual linkage below V32P. Full-pool LEACE rank was five in every
fold/seed, versus rank two in the V32P three-subject training pool.

## Cost and seed variability

| method | batch | median latency | p95 | peak measured GPU memory | parameters |
|---|---:|---:|---:|---:|---:|
| one-step | 1 | 0.201 ms | 0.230 ms | 24.2 MB | 133,248 |
| one-step | 64 | 0.204 ms | 0.236 ms | 24.4 MB | 133,248 |
| SANDiff | 1 | 9.239 ms | 9.351 ms | 24.2 MB | 508,384 |
| SANDiff | 64 | 9.701 ms | 9.906 ms | 24.6 MB | 508,384 |

SANDiff fixed BA was 0.360725/0.344907 across the two seeds, and adaptive
subject BA was 0.658565/0.663966. EEGNet seed variability is material and is
reported separately; seeds are never treated as biological samples.
