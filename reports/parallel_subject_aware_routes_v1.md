# Parallel subject-aware route screen v1

This report covers only the independent worktree
`codex/parallel-subject-explore`.  It does not use the FIR R2 implementation or
cache and must not be combined with the R2 repair result as one execution.

## Execution and scope

- Screening seed: `20260811`.
- Coverage: 16/16 Klados source records and 58/58 compatible SGEYESUB stems
  for each of P1--P6; 104/104 dependent route/fold metric files.
- Aggregation: P100 jobs `924576` and `924577` on `node46`.
- Scientific role: complete-real-data, one-seed route screening.  No top route
  has been promoted to a three-seed or confirmatory result.

All utilities below use the frozen convention that positive is better.  The
Klados utility is negative RRMSE difference; the SGE utility is EOG coherence
reduction difference.

| Route | Klados DIFF-DET | SGE DIFF-DET | Klados MATCH-POP | SGE MATCH-POP | Klados MATCH-3WRONG | SGE MATCH-3WRONG | Screen interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| P3 activity gate | +0.01562 | +0.01840 | -0.08851 | -0.03895 | +0.04671 | +0.02119 | strongest continuous signal; still below population |
| P4 support adapter | +0.00709 | +0.01517 | -0.09705 | -0.04218 | +0.03562 | +0.01229 | second candidate; still below population |
| P1 full-C residual | +0.00404 | +0.01443 | -0.10010 | -0.04292 | +0.03333 | +0.01196 | weak mechanism signal only |
| P2 full-C FiLM | +0.00187 | +0.00959 | -0.10227 | -0.04776 | +0.03294 | +0.01237 | weak mechanism signal only |
| P5 posterior guidance | -0.06466 | -0.00211 | -0.16880 | -0.05946 | -0.06401 | +0.00518 | no signal |
| P6 anchored SDEdit | -0.06997 | -0.00472 | -0.17410 | -0.06207 | -0.06649 | +0.00460 | no signal |

P3 has the best absolute DIFF-MATCH values among these routes: Klados mean
RRMSE `0.52908` versus DET-MATCH `0.54470` and route DIFF-POP `0.44057`; on
SGE, EOG reduction is `0.28314` versus DET-MATCH `0.26473` and route DIFF-POP
`0.32208`.  Its preservation (`0.77255`) is close to route POP (`0.77275`),
but artifact reduction is lower.  It is therefore not valid to call the route
safer or subject-aware merely from preservation.

P4 is the second continuous screen candidate.  Its Klados RRMSE is `0.53762`
and SGE EOG reduction is `0.27990`; both remain worse than the route population
endpoint.  P5 and P6 are worse than their matched deterministic estimator on
both datasets and are not candidates for expansion.

## Decision boundary

No route demonstrates MATCH > POP, so this screen does not establish a
subject-specific advantage.  P3 and P4 are retained only as the two least-bad,
mechanistically informative candidates for a possible three-seed follow-up;
that follow-up requires the separate support/operator audit to justify the
carrier and reliability definition.  The screen is neither a diffusion-family
negative result nor a subject-awareness-family negative result.

Compact outputs:

- `results/cgdr/parallel_subject_aware_routes_v1/screen_aggregation/method_summary.csv`
- `results/cgdr/parallel_subject_aware_routes_v1/screen_aggregation/effect_summary.csv`
- `results/cgdr/parallel_subject_aware_routes_v1/screen_aggregation/paired_effects.csv`
- `results/cgdr/parallel_subject_aware_routes_v1/screen_aggregation/result_summary.json`
