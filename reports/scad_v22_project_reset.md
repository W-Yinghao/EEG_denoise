# SCAD V22 project reset

This development-only round implements an observation-anchored artifact diffusion model: the network estimates ocular artifact and returns clean EEG as `Y - A_hat`. A query-disjoint support operator is canonicalized and injected through one FiLM mechanism. MATCH, POP, WRONG and null contexts share a checkpoint. No reliability routing, analytic inversion, energy bridge, old SADDPM evidence, or sealed data was used.

## Engineering and baselines

The canonical DET and SCAD models have 886,094 and 956,430 trainable parameters (SCAD +7.9%). Both passed finite-gradient, fixed-batch overfit, observation-anchor, context-response, trajectory, fixed-noise, checkpoint and interrupted optimizer/EMA/RNG resume checks. EEGDfus official-native and D4PM official-native claims are blocked by their unlicensed/incompletely documented releases; the clean-room public architecture reimplementation gave RRMSE 0.497–0.534 and correlation 0.845–0.867 across −5 to +5 dB.

## Paired development evidence

Participant-first mean temporal RRMSE was 1.7795 for DET-MATCH and 1.8571 for SCAD-MATCH. SCAD MATCH−POP utility was -0.00133 (median -0.00159, 5/15 positive), and MATCH−WRONG -0.00047 (median +0.00148, 8/15). The K1 diffusion-minus-DET1 utility was -0.07761 with 0/15 positive; thus DET was better for every participant.

Removing context ranking improved mean RRMSE from 1.8571 to 1.8466. The v-parameterized pilot was worse at 2.3162. DDIM10/25/50 MATCH RRMSE was 1.8495/1.8571/1.8459; iterative step effects were small and non-monotonic.

## Natural development evidence

For held-out EOG remaining ratio (lower is better), SCAD-MATCH was 0.9061, compared with DET-MATCH 0.8396. SCAD natural MATCH−POP utility was +0.00333 (9/15), and MATCH−WRONG +0.00021 (10/15). MATCH preservation was 0.8424; relative to SCAD-POP its preservation utility was -0.00212, PSD utility -0.00094, and covariance utility -0.00172.

## Development diagnosis

- Engineering validity: `valid`
- Baseline reproduction: EEGDfus `local_results_reasonable_but_nonidentical`; D4PM `blocked_incomplete_release`
- Subject-context evidence: `weak_or_heterogeneous_signal`
- Diffusion incremental value: `deterministic_better`
- Natural EEG trade-off: `preservation_concern`
- Recommended next step: `B. improve context representation`

Latency was 0.40 ms/window, NFE=1, 135 MiB for DET-MATCH and 7.88 ms/window, NFE=25, 43 MiB for SCAD-MATCH. K1 is the primary diffusion comparison against DET1. K8 and DET8 were not run, so no ensemble- or compute-matched K8 diffusion claim is made. RAW and STANDARD are numerically identical observation references in the already-standardized V19-derived harness; they are not presented as two independent methods. This is development/model-building evidence, not confirmation.

Energy-bridge refinement is not recommended next: subject context was not consistently load-bearing and SCAD did not complement or beat DET, so fewer than two registered preconditions were met. The concrete next development action is to improve the support-context representation while retaining the current simple artifact target and strong DET comparator; if that does not establish a consistent context effect, the priority should switch to the artifact backbone/target rather than posterior guidance.
