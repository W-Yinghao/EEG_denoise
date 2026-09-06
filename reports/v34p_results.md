# V34P Fiber-SANDiff results

## Protocol

V34P reused the V33P three-way outer folds, two seeds, BCI-IV-2a Session T/E
protocol, and frozen EEGNet checkpoints. Stage A selected Fiber-OneStep full-output
and Fiber-SANDiff deployed ten-step checkpoints using three train and three
participant-disjoint validation subjects. Stage B refit the fiber models on all six
non-test Session-T participants for the frozen epochs. Attackers were retrained on
sanitized outer-test Session T and evaluated on Session E.

## Strong operating point

| Method | Fixed BA | Retrained BA | Adaptive subject BA | Verification AUROC |
|---|---:|---:|---:|---:|
| RAW | 0.358603 | 0.354552 | 0.741127 | 0.630805 |
| HEAD_ONLY | 0.358603 | 0.356481 | 0.562114 | 0.599027 |
| LEACE | 0.355517 | 0.349730 | 0.658372 | 0.577914 |
| Fiber-OneStep | 0.358603 | 0.359182 | 0.574460 | 0.609631 |
| Fiber-SANDiff | 0.358603 | 0.350116 | 0.534529 | 0.579262 |
| V33P strong one-step | 0.353781 | 0.354938 | 0.657986 | 0.583221 |
| V33P strong SANDiff | 0.352816 | 0.353588 | 0.661265 | 0.575094 |

Relative to RAW, strong Fiber-SANDiff reduced adaptive subject recall by `0.206597`
with 9/9 participants positive and a participant-bootstrap 95% interval of
`[0.161458, 0.246335]`. Verification AUROC reduction was `0.049157` participant-first
with 9/9 positive and interval `[0.028873, 0.072217]`. Fixed-head BA changed by
exactly zero.

## Diffusion versus conditional mean

Relative to Fiber-OneStep, Fiber-SANDiff reduced adaptive leakage by `0.039931`
(`6/9`, 95% interval `[0.013696, 0.065201]`) and verification AUROC by `0.030370`.
Its retrained-head BA was `0.009066` lower. The stochastic model was materially
closer to the observed fiber distribution:

| Strong method | Covariance discrepancy | Energy distance | MMD | Variance retained |
|---|---:|---:|---:|---:|
| Fiber-OneStep | 0.941062 | 4.291420 | 0.156030 | 0.179976 |
| Fiber-SANDiff | 0.912793 | 1.671643 | 0.060592 | 0.784008 |

The one-step sanitizer exhibits conditional-mean variance collapse. Fiber-SANDiff
retains substantially more pooled fiber variance and improves all three registered
distribution discrepancies, while not winning retrained utility.

## H-only diagnostic

HEAD_ONLY adaptive BA was `0.562114` and verification AUROC `0.599027`, showing
substantial subject linkage already in task-head-visible coordinates. Fiber-SANDiff
adaptive BA was `0.027585` below the H-only point, with a participant interval that
crosses zero. Because irrelevant stochastic dimensions can alter finite attacker
behavior, H-only is treated as an empirical diagnostic rather than a strict numerical
lower bound or a conditional-mutual-information estimate.

## Cost

Fiber-OneStep used 99,709 parameters and approximately `0.257 ms` median latency.
Fiber-SANDiff used 474,077 parameters; ten-step K=1 latency was `8.680 ms` at batch 1
and `9.098 ms` at batch 64. Recorded peak allocation was about 22.5 MiB.
