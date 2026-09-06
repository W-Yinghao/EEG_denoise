# WAVE3 report — residual ledger, type probe, anomaly resolutions

Preregistration: `reports/wave3_preregistration.md` (Tier-0, frozen before any
computation, commit c1e25b9). Sealed contact: **none**. Compute: CPU battery plus two
small GPU units (A0, T3), far inside the 25 h Pack-A and 100 h wave caps.

## The guaranteed deliverable — the disjoint residual ledger

Span = NO_A0-referenced ADDITIVE oracle span on the MobileBCI likelihood leg =
delivered + oracle residual = **0.38694** RRMSE.

| row | RRMSE units | share of span | instrument |
|---|---|---|---|
| delivered | 0.14281 | 36.9% | T0 / V44-S1 banked |
| bookkeeping | 0.00000 | 0.0% | ONCE Stage 0 |
| gate-shrinkage | 0.13707 | 35.4% | T4 (slope-converted) |
| estimation-noise | 0.03438 | 8.9% | T4 slope x lambda x sqrt(median within/4) |
| readout | 0.00000 | 0.0% | T3 |
| family | 0.00160 | 0.4% | T6 ladder |
| drift | 0.06760 | 17.5% | drift_row (median per-cell RMS) x T4 slope |
| fluctuation | 0.00000 | 0.0% | P7 dose-response (linear) |
| unattributed-remainder | 0.00347 | 0.9% | closure |

Nine independently measured first-order instruments close the identity to
**0.9%** unattributed remainder. Partition rules held as
frozen (F6): the entire lambda-derived quantity sits in gate-shrinkage, estimation-noise
takes only the surviving lambda-weighted term, T6's family gain is assigned to `family`
with A1's ceiling reported net of it. A4 (reference-channel error) annotates rows 4-7;
its clean instrument is priced and deferred.

The single largest attributed term is **gate-shrinkage 0.1371**,
and its sub-split is the wave's most deployment-relevant number: **active cells
0.0144 (median
0.00281) versus
1.915 per abstained cell at
6.5% abstention**. The EB gate costs
almost nothing where it engages; the cost is the fail-closed fallback. Shrink-to-zero
(0.4084) is far worse than shrink-to-pop — a free
deployment note.

## Corrected residual semantics (rule i, adopted program-wide)

| panel | delivered | oracle residual (additive) | R* | 95% CI | delta_conv |
|---|---|---|---|---|---|
| mobilebci_likelihood | 0.14281 | 0.24413 | 0.3691 | [0.176, 0.724] | +0.5228 |
| klados_transport | 0.01596 | 0.02363 | 0.4031 | [0.089, 1.381] | -0.0265 |
| bci2b_transport | 0.01818 | 0.01786 | 0.5045 | [0.045, 1.246] | -0.0748 |

Restated band **0.369-0.504**,
inside the frozen 0.29-0.50. Conversion fractions are ratios of means (the per-unit
mean-of-ratios is unstable: it returns -0.06 on Klados) with the mean-of-ratios retained
as a declared sensitivity.

## O4/P2 — units, decisively (B0 + T7)

B0 lifted the "inference-pending" label: the estimator is literally
`clip(tau2/(tau2 + within/4), 0, 1)`, `within` is **block-mean variance**, and `tau2`
measures 120-s deviates against a **30-s** population centroid. Per-cell recompute matches
the banked manifests to **7.1e-15**.

The decisive quantity: `within` has mean 1.0509
but median 0.0031
(341x;
15/465 cells above 1.0). Hence
lambda(pooled means) = 0.5959,
lambda(pooled medians) = 0.9819,
mean-of-per-cell lambda = 0.8918. **"0.134" is
exactly the `within` that pooled-mean bookkeeping needs to reproduce the banked lambda-hat
— a bookkeeping artifact, not a measured physical quantity.**

T7's frozen LEVELS are reproduced by neither account at any arm (all four 2x2 arms sit at
median W ~0.003). The frozen DISCRIMINATOR is the contrast, and it is unambiguous:
stratification 6.7%, granularity
13.6x, aggregation 316x.
Composition closure: W_random - W_strat = 0.000187,
consistent with the ~zero composition term T1's null separation implies. **Frozen
consequence applied**: P2 dissolves as a units lesson; O4 restates per-window; the
parsimony narrative is dropped; A1 survives only as the natural-data ceiling question.

B1: transform reproducible for **100%** of active cells, slope 1.000 (CI [1.000, 1.000]).
B1.5: symbolic identity holds; registered estimator note — `within` is (k-1)/k = 0.75x the
block variance, so `within/4` slightly UNDER-states the standard error.

## O2 — ONCE: branch B with a corrected mechanism

Stage 1 retrodicts the banked values exactly (bci2b additivity 0.596
vs banked 0.596; deficit
-0.00723 vs -0.00723).
But the **increment** cross-correlation is ~0 (0.031 /
0.000), and the orthogonalized composite is
WORSE than the best single leg (-0.00788 /
-0.00326); P2 non-inferiority fails, superiority
not claimable. Stage 0's "bookkeeping conviction" on bci2b is therefore **contradicted**:
its shared-span statistic measures overlap of the TOTAL subtractions (necessarily large),
while the decisive increment cross-term is ~0.03. The joint deficit is **variance
accumulation, not double-counting**, and orthogonalization cannot repair what is not
overlapping. Branch B stands: the two channels tap one shared ocular budget.

## O3 — Pack-A closed at its entry gate

A0 U-ratio = **1.4023** [1.0422, 2.0187], n=15: the CI
excludes 1 but ABOVE it. The banked P0 prior **inflates** ocular-coefficient energy rather
than removing it, so the M13R U0-censoring mechanism does not reproduce in vivo. Per the
frozen gate Pack-A stops, A1' is not run, and **finding 7 softens to a single-run
observation**. The CPU fingerprints did fire (masking-frequency correlation
0.784; selected windows at
0.805x median U0 energy): **the censoring is real in
the data selection; its predicted prior-side consequence is not.**

## O1 — the type probe is INCONCLUSIVE

TSR 0.517 (CI-low 0.378),
0/16 subjects at TSR>=2, mixture
non-degenerate (blink 0.28, move
0.16) — but **kappa = -0.245**, far below
0.8. Per frozen rule (iii) this is INCONCLUSIVE, not NO-GO: psi does not agree with the
independent GMM instrument even at chance, so the low TSR carries no kill authority.
Panel-T and TROCA S1 are therefore **not authorized**, and A1 is NOT closed as
family-final this wave.

T6 answers the family question anyway: FIR-lagged
+0.0066
[+0.0041,
+0.0091] is statistically
clear and numerically trivial; richer families overfit.

| family | relative CV gain vs incumbent | 95% CI |
|---|---|---|
| amplitude_gain | -0.1627 | [-0.2533, -0.1042] |
| fir_lagged | +0.0066 | [+0.0041, +0.0091] |
| indicator_linear | +0.0000 | [+0.0000, +0.0000] |
| kernel_ridge | -0.2941 | [-0.3759, -0.2209] |
| rank3_derivative | +0.0021 | [+0.0013, +0.0029] |

## T3 — non-adjudicating (instrument failure, reported)

Both available oracle instruments are degenerate: on the paired panel the query-fitted
operator reproduces the injected artifact exactly, and on natural windows the metric's
proxy teacher IS `C_query.e`, so the LINEAR oracle arm reproduces it exactly
(150.6 dB). The prereg
anticipated this for the paired panel; it extends to the natural panel. **T3 neither
exonerates nor convicts the sampler**; a valid test needs an operator-independent artifact
reference (A4 Eye-BCI optical, priced and deferred). The readout ledger row is entered as
0 and flagged UNBOUNDED-BY-MEASUREMENT.

## Decision Point 2

TROCA S1: **not authorized** (Panel-T not built, T1 INCONCLUSIVE). A2': **not authorized**
(A1' never ran, so the "A1' inconclusive" precondition is unmet even though >=1 density
fingerprint fired). Pack-A tail closed. No Tier-2 stage qualifies.
