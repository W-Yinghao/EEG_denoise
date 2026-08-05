# Parallel subject-aware route screen v1

This work is isolated on `codex/parallel-subject-explore`. It does not alter
the completed wide-exploration-v2 report or its scientific decision on
`master`.

## Scope

The screen uses one fixed canonical `standardized_artifact_latent` target.
Changing matching, population, or wrong same-cell operators changes only the
conditioning/reconstruction intervention, never the target. Query-time EOG,
eye tracking, artifact labels, outcomes, and participant identifiers are not
model inputs.

The independently scheduled routes are:

- P0: three-seed canonical population latent diffusion backbone.
- P1: full-C population-residual reconstruction with fixed `g=1` and exact
  `g=0` population fallback.
- P2: full-C support summaries injected by FiLM into every major U-Net
  residual block.
- P3: query-EEG-only artifact-activity gate.
- P4: support-only low-rank adapter fitted after P0.
- P5: bounded per-step full-C posterior guidance.
- P6: observation-anchored SDEdit initialized from EEG-space coordinates.

P1--P6 use one screening seed but the complete frozen Klados and SGEYESUB
development units. They are route screens, not confirmation or a final route
selection. Every compatible unit includes RAW, POP, DET-MATCH, DIFF-POP,
DIFF-MATCH, a true K=1 DIFF-MATCH sample, and three same-cell DIFF-WRONG
controls. A cell with fewer than three legitimate wrong donors is retained as
blocked coverage rather than filled with duplicate donors.

The FIR job only caches fixed lag/ridge support cross-fit candidates. It does
not choose a lag and does not launch FIR diffusion.

## Status

The job ledger is maintained in
`reports/slurm/parallel_subject_aware_routes_v1_job_ids.txt`. Scientific
metrics and route-level status will be added after all independent screens and
the compact aggregation complete. No parallel result is merged into or pushed
to `master` during this stage.
