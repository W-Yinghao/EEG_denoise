# PhysioMotion J1R retrieval fairness audit

Decision: `DEPLOYABLE_SUBJECT_INCREMENT_HEADROOM_PRESENT`.

This is a CPU-only development fairness audit. It trains no deterministic or diffusion model and never opens the ten sealed participants.

## Data and mask audit

Official-layout annotation mapping was 100.0000% before and 100.0000% after strip/case normalization. Normalization repaired 0 names; unresolved names: none.

Empty reconstructed masks: 0. Query state/run units with multiple baseline annotation rows, where the old rowwise cap could repeat sampling: 67.

Run-01 support participant/state units with multiple baseline annotation rows: 12. In the old builder, per-row candidate concatenation followed by `[:16]` could favor earlier rows. The fairness cache instead uses fixed-seed uniform sampling over the complete baseline for both support and query.

## Participant-first effects

### Observable selector

- H_P_eq: mean +0.03704, median +0.02400, 13/17 positive, one-sided exact p=0.014450, descriptive 95% bootstrap [+0.00757, +0.06889].
- H_W_eq: mean +0.03670, median +0.02367, 13/17 positive, one-sided exact p=0.014954, descriptive 95% bootstrap [+0.00727, +0.06874].
- H_HYB: mean +0.02153, median +0.01519, 13/17 positive, one-sided exact p=0.012833, descriptive 95% bootstrap [+0.00486, +0.03973].
- H_HYB_W: mean +0.03277, median +0.02356, 14/17 positive, one-sided exact p=0.001480, descriptive 95% bootstrap [+0.01434, +0.05262].
- H_MATCH_LARGE: mean +0.00009, median -0.00382, 8/17 positive, one-sided exact p=0.498375, descriptive 95% bootstrap [-0.03435, +0.04022].
- H_HYB_LARGE: mean +0.00019, median +0.00203, 9/17 positive, one-sided exact p=0.494545, descriptive 95% bootstrap [-0.02150, +0.02553].

### Oracle selector

- H_P_eq: mean +0.01817, median +0.02670, 12/17 positive, one-sided exact p=0.049683, descriptive 95% bootstrap [-0.00240, +0.03755].
- H_W_eq: mean +0.03895, median +0.04183, 12/17 positive, one-sided exact p=0.006340, descriptive 95% bootstrap [+0.01194, +0.06640].
- H_HYB: mean +0.01589, median +0.01382, 13/17 positive, one-sided exact p=0.003632, descriptive 95% bootstrap [+0.00601, +0.02627].
- H_HYB_W: mean +0.02045, median +0.01791, 13/17 positive, one-sided exact p=0.001335, descriptive 95% bootstrap [+0.00931, +0.03225].
- H_MATCH_LARGE: mean -0.10687, median -0.10060, 0/17 positive, one-sided exact p=1.000000, descriptive 95% bootstrap [-0.12815, -0.08648].
- H_HYB_LARGE: mean -0.06821, median -0.06095, 0/17 positive, one-sided exact p=1.000000, descriptive 95% bootstrap [-0.08360, -0.05477].

## Descriptive strata and oracle gap

- same-day H_P_eq: mean +0.06392, median +0.03510, 4/5 positive.
- multi-date H_P_eq: mean +0.02585, median +0.02282, 9/12 positive.
- Oracle-minus-observable H_P_eq effect gap: mean -0.01887 across 17 participants.
- Oracle-minus-observable H_W_eq effect gap: mean +0.00225 across 17 participants.
- Oracle-minus-observable H_HYB effect gap: mean -0.00564 across 17 participants.
- Oracle-minus-observable H_HYB_W effect gap: mean -0.01231 across 17 participants.

## Interpretation

Equal-budget observable MATCH exceeds both POP and donor-averaged WRONG, so the fixed clean support bank contains subject information under the corrected sampling contract. The equal-total-budget hybrid effects also pass the frozen directional gates, and observable HYBRID-MATCH is aggregate-point-estimate non-inferior to POP-LARGE in both mean and median (9/17 participant effects are positive and the descriptive interval crosses zero). This yields `DEPLOYABLE_SUBJECT_INCREMENT_HEADROOM_PRESENT`, which authorizes only a future single hybrid masked-diffusion screen; no model is trained here.

The oracle equal-budget contrasts are also positive, while oracle POP-LARGE is substantially stronger than the smaller MATCH/hybrid pools. Thus large population coverage remains an important ceiling and the result is not evidence that subject retrieval universally dominates a sufficiently large population bank.

## Boundary

All 64 deterministic subsampling repeats are averaged inside each evaluation unit before state/run, equal-family, and participant aggregation. They are not scientific replicates. POP fallback units appear only in the 20-person policy estimand; mechanism effects use the evaluable estimand. Same-day and multi-date strata remain descriptive.

No model training is authorized in this round regardless of the routing label.
