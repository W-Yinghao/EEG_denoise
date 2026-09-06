# v19 Null-Floor Provenance Audit

Final audit verdict: `INVALID_NO_EXACT_RECOVERY`

Terminal label: `NULL_FLOOR_PROTOCOL_OR_PROVENANCE_INVALID_ROUTE_CLOSED`

The reported implementation is numerically reproducible, but the statistical-null provenance needed to validate the routing floor is not uniquely preregistered. No recovery or O1 was run.

## Exact numeric replay

- O0-A floor: reported 0.2293; replayed 0.229309363911899.
- O0-B floor: reported 0.1745; replayed 0.174469145999967.
- Maximum original-vs-independent implementation difference: 0 (tolerance 1.0e-12).
- N_P=0.167627509034797; N_W=0.163276731715251; H_P=0.720564444385842; H_W=0.662091832839082.

## What generated 0.2293

For each held-out participant, the code selected the other 15 participants' participant-first `POP − TIME_SHIFT` natural-risk effects, clipped each at zero, and applied `numpy.quantile(..., 0.95)` with the library-default linear interpolation. It then took the maximum of 0.010, 5% of that held-out participant's POP normalized risk, and this fold-local q95. The published 0.22930936391189927 is the mean of the resulting 16 participant-specific floors.

There is no null replicate ID, permutation vector, RNG schedule, group-null statistic, or joint P/W max-stat operation. The time-shift rows are participant-level falsification/stress cells. They are not exchangeable realizations of a group scientific null.

## Unit and scale checks

- Observed and stress-cell effects both use participant-first normalized risk; ERP/SSVEP are equal-weighted and sub-24 is policy fallback zero.
- WRONG rows are averaged within donor before method/unit/task/participant reduction.
- The 5% component is converted to contrast units as `0.05 × participant POP risk`; it is not used as a bare dimensionless 0.05.
- Fold floors exclude the held-out participant and use the other 15 development participants.

## Why exact recovery is not authorized

- no permutation/null-replicate indices or floor RNG schedule
- no preregistered group null statistic
- no preregistered panel-local/global P/W max-stat axis
- percentile interpolation method absent from preregistration
- EOG time-shift falsification cells are not exchangeable null replicates
- multiple reasonable corrections exist and none is uniquely preregistered

Because several scientifically different max-stat/null constructions are reasonable and none is uniquely frozen, choosing one now would be a new analysis rather than an exact recovery.

## Governance

Original v19 files and decision remain unchanged. No raw EEG, marker, event, annotation, sealed outcome, GPU job, model training, O1, or manuscript operation occurred. Mobile sealed-8, PhysioMotion sealed-10, SHU Day-4/5, and PhysioTrait Day-200 remain unopened.
