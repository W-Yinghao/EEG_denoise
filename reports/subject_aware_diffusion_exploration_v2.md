# Subject-aware diffusion wide exploration v2

This is a development wide screen, not confirmation or a frozen submission protocol.

## Corrected current-result audit

The old uncertainty verdict is invalid under its original definition. Comparable final-EEG replay gives diffusion normalized risk–coverage AUC 0.5436 versus deterministic 0.4625; uncertainty is secondary and provides no current advantage. True end-to-end K=1 and DET-POP are reported in the re-audit directory.

## Data suitability

SGE full-C support cross-fit beat population in 32/58 stems and the mean of three same-cell wrong donors in 38/58. Median errors were {'FIR': 0.7198559903688613, 'full_C': 0.8505213364404112, 'population': 0.8484409427573466, 'state_specific': 0.7766012295123007}. Post-output-freeze support-to-query evidence was {'matching_beats_mean_three_wrong_count': 51, 'matching_beats_population_count': 56, 'median_operator_distance': 0.7706272522534281}; legacy-rho diagnostics were {'pearson_with_negative_split_half_distance': nan, 'saturation_fraction_at_one': 0.0}.

## Complete carrier ranking

| Rank | Carrier | Score | DIFF−DET | Deployment MATCH−POP | Deployment MATCH−WRONG | Equal-attenuation preservation |
|---:|---|---:|---:|---:|---:|---:|
| 1 | R2_FIR_residual | 1.07667 | 1.18527 | -0.07478 | -0.00763 | 0.00000 |
| 2 | R0_projector | 0.05694 | 0.05922 | -0.00515 | 0.00286 | nan |
| 3 | R3_state_gated_residual | -0.08096 | 0.02318 | -0.07484 | -0.01315 | 0.00000 |
| 4 | R1_full_C_residual | -0.08155 | 0.05022 | -0.11770 | -0.00234 | 0.00000 |

Conditioning variants were ranked only after complete 26-fold coverage: [{'conditioning': 'structured', 'diffusion_value': 1.1852700007951835, 'equal_attenuation_preservation': 0.0, 'equal_attenuation_units': 58, 'preservation': -0.10982957688765982, 'route': 'R2_FIR_residual', 'score': 1.1028619848979386, 'specificity': -0.007628541672721256, 'subject_value': -0.07477947422452347}, {'conditioning': 'support_FiLM', 'diffusion_value': 0.9799863426826372, 'equal_attenuation_preservation': 0.0, 'equal_attenuation_units': 58, 'preservation': -0.07681916823490031, 'route': 'R2_FIR_residual', 'score': 0.8909073761816232, 'specificity': -0.0058624176525004235, 'subject_value': -0.08321654884851362}, {'conditioning': 'structured', 'diffusion_value': 0.02318236934115503, 'equal_attenuation_preservation': 0.0, 'equal_attenuation_units': 58, 'preservation': -0.03574464871244909, 'route': 'R3_state_gated_residual', 'score': -0.06480720502083942, 'specificity': -0.013148701713228933, 'subject_value': -0.07484087264876552}, {'conditioning': 'support_FiLM', 'diffusion_value': 0.027486890073048498, 'equal_attenuation_preservation': 0.0, 'equal_attenuation_units': 58, 'preservation': -0.016425245941983357, 'route': 'R3_state_gated_residual', 'score': -0.07022499712062275, 'specificity': -0.013474708797440338, 'subject_value': -0.0842371783962309}]

## Final two routes

| Route | Deployment MATCH−POP | Deployment MATCH−3 WRONG | Mechanism g=1 MATCH−POP | Mechanism g=1 MATCH−3 WRONG | DIFF−DET | Equal-attenuation preservation |
|---|---:|---:|---:|---:|---:|---:|
| R2_FIR_residual|structured | -0.07780 [-0.0961573889680479, -0.05976115879462844] | -0.00647 [-0.02183236101709162, 0.008876072629557556] | -0.12642 [-0.14909115666781456, -0.10472171522105386] | -0.01066 [-0.0315638532341278, 0.010868589836829595] | 1.09425 [0.528537330491792, 1.7656017933107142] | 0.00000 [0.0, 0.0] |
| R2_FIR_residual|support_FiLM | -0.08179 [-0.10161546230991353, -0.06282981576848617] | -0.00612 [-0.023231671205667635, 0.01103197507211929] | -0.13133 [-0.15354611642655527, -0.10956982100428517] | -0.00757 [-0.029571415772829384, 0.015077071920104497] | 0.55483 [-0.23898564833072986, 1.2059634976994205] | 0.00000 [0.0, 0.0] |

The ranking separates diffusion incremental value, carrier value, support-only reliability, and equal-attenuation natural-EEG trade-offs. The corrected uncertainty remains secondary. Klados remains 16-source-record mechanism evidence; SGE is 58/59-stem development evidence. R0 is a frozen historical baseline and not a full-C representation. No Eye-BCI confirmation outcome, EEGEyeNet retry, TAAS edit, or family-wide claim was made.
