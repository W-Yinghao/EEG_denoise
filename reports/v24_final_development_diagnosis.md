# V24 final development diagnosis

## Coordinate and provenance

Coordinate verdict: `V23_COORDINATE_MISMATCH_CONFIRMED`. The prepared EOG is
physical microvolt data, not dimensionless support-standardized EOG. Correct raw and
canonical constructions agreed to a maximum relative error of 2.571e-16 across 90
cells and 9,000 windows, whereas the committed V23 route had median relative error
1.173398 (range 0.096218–10.897984). V23 artifacts remain immutable and are
superseded as scientific evidence; V24 is a new corrected development analysis, not
an exact recovery of V23.

## Engineering

Classification: `valid`.

- Population-anchor, TemporalEOGNet, and residual-diffusion single-batch loss
  reductions were 98.18%, 98.45%, and 99.09%.
- POP structural identity was exact (`max difference = 0`). MATCH/POP/WRONG swaps
  changed only the registered deviation decoder.
- All 15 development recipients were evaluated over five fixed folds and three
  seeds. Test query EOG/operator/event inference reads and sealed reads were zero.
- Checkpoint/resume state includes optimizer, scheduler, EMA, AMP scaler, and all
  registered RNG streams.
- A100 latency on 100 windows was 0.077 ms/window for the population anchor,
  0.136 ms/window for DET, and 1.721 ms/window for SCAD-K1 (27 NFE).

## Population anchor and EOG latent

Population anchor paired participant-first RRMSE: `0.674453`.
Natural DET latent correlation: `0.610227`.

Population-anchor classification is
`not_interpretable_vs_v23_due_coordinate_supersession`: a numerical improvement over
the invalid-coordinate V23 result is not treated as a valid cross-version contrast.
EOG-latent classification is `moderate_predictability`. Paired latent correlation was
0.332335 for DET and 0.265467 for SCAD; natural latent correlation was 0.610227 for
DET and 0.410562 for SCAD. Thus the temporal target is learnable, but diffusion
degrades rather than refines it.

## Subject correction and diffusion

Subject correction: `context_harmful`.
Diffusion: `deterministic_better`.
Natural trade-off: `artifact_reduction_insufficient`.

Next route: **C. Replace fixed operator with raw support-set encoder**.

DET MATCH−POP `-0.041906`; DET MATCH−WRONG `+0.003479`; SCAD MATCH−POP `-0.202303`; SCAD MATCH−WRONG `+0.021900`; SCAD−DET `-0.160397`.

The fixed support deviation retains weak donor-specific ordering against WRONG, but
does not improve over the exact shared population anchor. Residual diffusion is worse
than DET for every participant. K8 was therefore not run, and no energy bridge,
routing, rollback, or operator portfolio was introduced.

## Natural trade-off and route

The population anchor retained preservation 0.828301 with remaining ratio 0.989638.
DET MATCH reduced preservation to 0.782816 without improving remaining ratio. SCAD
MATCH further degraded both axes (remaining ratio 1.363523, preservation 0.596115).

The evidence therefore selects **C. Replace fixed operator with raw support-set
encoder**. This is a future-development recommendation, not authorization executed
in this round. The direct evidence is that EOG temporal prediction is moderately
successful while applying the fixed support deviation is harmful versus POP; making
the temporal network larger or adding diffusion does not address that bottleneck.

All evidence is development-only. Sealed reads were zero; no manuscript was modified.
