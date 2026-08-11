# V23 final development diagnosis

V23 repairs the frozen V22 supervision conflict, uses an online continuous-EOG counterfactual generator, and makes the population/deviation operator basis the explicit artifact decoder. Results are development-only.

## Projection geometry

Mean projection errors: POP 0.4901, MATCH 0.2961, WRONG 0.4109, query oracle 0.2226.

## Participant-first effects (positive utility is better)

- OF_DET_MATCH_POP_SWAP: +0.057644
- OF_DET_MATCH_WRONG_SWAP: +0.008791
- OF_DET_SUBJECT: -0.092447
- OF_SCAD_MATCH_POP_SWAP: +0.167918
- OF_SCAD_MATCH_WRONG_SWAP: +0.045371
- OF_SCAD_SUBJECT: -0.092598
- DIFF_K1_vs_DET1: -0.003304

## Natural trade-off

Subject-vs-population attenuation utility was -0.421162; preservation utility -0.227200.

## V22 repair

V22-FIX MATCH utility against POP was +0.011448 and against WRONG was +0.003176. The frozen forensic audit established conflicting ordinary supervision, a 1,200-update budget, first-64 validation, fixed t=500 diffusion validation, last-weight checkpointing, hard-masked fixed pairs, and zero-artifact SNR contamination. Frozen V22 outputs were not changed.

## Uncertainty

- OF_DET_MATCH_POP_SWAP: mean +0.057644, median +0.107710, 13/15 positive, participant-bootstrap 95% CI [-0.053143, +0.141810]
- OF_DET_MATCH_WRONG_SWAP: mean +0.008791, median +0.032398, 12/15 positive, participant-bootstrap 95% CI [-0.095080, +0.090496]
- OF_DET_SUBJECT: mean -0.092447, median -0.049521, 3/15 positive, participant-bootstrap 95% CI [-0.185838, -0.017678]
- OF_SCAD_MATCH_POP_SWAP: mean +0.167918, median +0.167602, 14/15 positive, participant-bootstrap 95% CI [+0.054061, +0.266374]
- OF_SCAD_MATCH_WRONG_SWAP: mean +0.045371, median +0.038495, 11/15 positive, participant-bootstrap 95% CI [-0.050063, +0.129244]
- OF_SCAD_SUBJECT: mean -0.092598, median -0.096414, 3/15 positive, participant-bootstrap 95% CI [-0.170150, -0.024085]
- DIFF_K1_vs_DET1: mean -0.003304, median -0.011943, 6/15 positive, participant-bootstrap 95% CI [-0.034198, +0.029283]

## Classification

- Engineering: `valid`
- V22 repair: `objective_repair_helped`
- Operator-factorized context: `weak_or_heterogeneous_signal`
- Diffusion incremental value: `deterministic_better`
- Natural trade-off: `artifact_reduction_insufficient`
- Next route: `B. improve temporal model`

K8/DET8 is not used to rescue a poor K1 result. EEGDfus remains the frozen V22 `local_results_reasonable_but_nonidentical` reference; D4PM remains `blocked_incomplete_release`. No energy bridge, sealed data, manuscript change, or confirmation claim is included.
