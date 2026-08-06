# MobileBCI protocol repair and direct diffusion interaction screen (v5)

This is development exploration, not confirmation. The eight sealed participants were not opened.

## Data and protocol validity

The repaired experiment is explicitly **processed-EEG+IMU**; source EOG is disabled. Event onsets use the official 100 Hz sample-index interpretation. Of 160 protocol units, 150 were eligible, 0 were blocked for no later block, and 10 were missing. ERP/SSVEP eligible counts were 78/72.

## Direct one-seed factorial

| Protocol | H_D DIFF−DET [95% CI] | H_S MATCH−NULL [95% CI] | MATCH−WRONG [95% CI] | MATCH−SHUFFLED [95% CI] | Interaction [95% CI] | Safety | Extra seeds |
|---|---:|---:|---:|---:|---:|---|---|
| S0_STATIC_XSESSION | -0.00351 [-0.00954, +0.00228] | -0.00323 [-0.00513, -0.00133] | -0.00043 [-0.00167, +0.00071] | +0.00053 [-0.00031, +0.00143] | -0.00344 [-0.00688, -0.00059] | fail | not authorized |
| S1_MOTION_WITHIN_SESSION | -0.00360 [-0.00950, +0.00187] | -0.00259 [-0.00415, -0.00117] | +0.00015 [-0.00102, +0.00125] | -0.00006 [-0.00107, +0.00099] | -0.00331 [-0.00666, -0.00031] | pass | not authorized |
| S2_MOTION_XSPEED | -0.00416 [-0.01361, +0.00428] | -0.00300 [-0.00542, -0.00079] | -0.00026 [-0.00251, +0.00221] | -0.00045 [-0.00138, +0.00041] | -0.00466 [-0.01066, +0.00123] | pass | not authorized |

### Deterministic support probe

| Protocol | DET MATCH−NULL [95% CI] | DET MATCH−WRONG [95% CI] | DET MATCH−SHUFFLED [95% CI] |
|---|---:|---:|---:|
| S0_STATIC_XSESSION | +0.00021 [-0.00233, +0.00274] | +0.00058 [-0.00173, +0.00322] | +0.00007 [-0.00224, +0.00280] |
| S1_MOTION_WITHIN_SESSION | +0.00072 [-0.00249, +0.00406] | +0.00214 [-0.00112, +0.00519] | +0.00029 [-0.00140, +0.00212] |
| S2_MOTION_XSPEED | +0.00166 [-0.00383, +0.00678] | +0.00438 [+0.00006, +0.00842] | +0.00034 [-0.00221, +0.00323] |

Units are participants, not windows. DIFF-NULL and DIFF-POP are the same learned context-dropout population arm in this formulation and are not counted as independent replications. The nonzero-gamma Pareto sweep found no strict population domination, but the raw operating point of strong POP had materially higher motion-coherence reduction than DIFF-MATCH in all protocols.

## Method operating points

| Protocol | Method | Motion-coherence reduction | Low-motion preservation | PSD distortion | Covariance distortion | Observation change |
|---|---|---:|---:|---:|---:|---:|
| S0_STATIC_XSESSION | DET-MATCH | +0.01675 | +0.79064 | +0.25388 | +0.20664 | +0.21431 |
| S0_STATIC_XSESSION | DET-NULL | +0.01654 | +0.76509 | +0.28504 | +0.22815 | +0.24073 |
| S0_STATIC_XSESSION | DIFF-MATCH | +0.01324 | +0.77215 | +0.22495 | +0.18613 | +0.22564 |
| S0_STATIC_XSESSION | DIFF-NULL | +0.01647 | +0.74476 | +0.26814 | +0.22110 | +0.25395 |
| S0_STATIC_XSESSION | DIFF-SHUFFLED | +0.01271 | +0.78075 | +0.22376 | +0.18437 | +0.21806 |
| S0_STATIC_XSESSION | POP | +0.03210 | +0.73096 | +0.31270 | +0.27071 | +0.27209 |
| S0_STATIC_XSESSION | RAW | +0.00000 | +1.00000 | +0.00000 | +0.00000 | +0.00000 |
| S1_MOTION_WITHIN_SESSION | DET-MATCH | +0.01779 | +0.78781 | +0.26203 | +0.20928 | +0.21760 |
| S1_MOTION_WITHIN_SESSION | DET-NULL | +0.01708 | +0.76706 | +0.28081 | +0.22842 | +0.23945 |
| S1_MOTION_WITHIN_SESSION | DIFF-MATCH | +0.01420 | +0.77363 | +0.23381 | +0.19673 | +0.22631 |
| S1_MOTION_WITHIN_SESSION | DIFF-NULL | +0.01679 | +0.74604 | +0.26611 | +0.22120 | +0.25365 |
| S1_MOTION_WITHIN_SESSION | DIFF-SHUFFLED | +0.01425 | +0.76766 | +0.24027 | +0.20169 | +0.23222 |
| S1_MOTION_WITHIN_SESSION | POP | +0.03308 | +0.73398 | +0.30242 | +0.26806 | +0.27083 |
| S1_MOTION_WITHIN_SESSION | RAW | +0.00000 | +1.00000 | +0.00000 | +0.00000 | +0.00000 |
| S2_MOTION_XSPEED | DET-MATCH | +0.01950 | +0.78818 | +0.25812 | +0.21181 | +0.22862 |
| S2_MOTION_XSPEED | DET-NULL | +0.01783 | +0.75811 | +0.28818 | +0.23807 | +0.26035 |
| S2_MOTION_XSPEED | DIFF-MATCH | +0.01534 | +0.78885 | +0.24529 | +0.21224 | +0.21765 |
| S2_MOTION_XSPEED | DIFF-NULL | +0.01834 | +0.75412 | +0.28780 | +0.24573 | +0.25290 |
| S2_MOTION_XSPEED | DIFF-SHUFFLED | +0.01579 | +0.78262 | +0.25368 | +0.21856 | +0.22426 |
| S2_MOTION_XSPEED | POP | +0.03642 | +0.72741 | +0.31367 | +0.27543 | +0.28575 |
| S2_MOTION_XSPEED | RAW | +0.00000 | +1.00000 | +0.00000 | +0.00000 | +0.00000 |

## P-C bounded-candidate selector diagnostic

Status: `completed_bounded_candidate_selector_diagnostic`. Infeasible units abstain to POP and remain in the full denominator. M0 uses the seven frozen features with training-unit achieved-coverage calibration; M1 adds output disagreement. This diagnostic is independent of the Mobile factorial.

| Dataset | Route | Success/denominator | Coverage | AUROC | AUPRC | Matching safe-rate | Wrong-support safe-rate |
|---|---|---:|---:|---:|---:|---:|---:|
| klados | M0 | 16/16 | +0.44186 | +0.78983 | +0.77534 | +0.67252 | +0.64159 |
| klados | M1 | 16/16 | +0.45436 | +0.77674 | +0.77279 | +0.62981 | +0.62582 |
| sgeyesub | M0 | 58/59 | +0.00000 | N/A | N/A | +0.00000 | +0.00000 |
| sgeyesub | M1 | 58/59 | +0.00000 | N/A | N/A | +0.00000 | +0.00000 |

## Scientific boundary

The current v5 formulation did not meet the pre-specified route for extra optimization seeds. This closes only this implementation, not diffusion or personalization families.

Factorial run code began from commit `d2bfe9a`; aggregation/report code HEAD was `c75e327a05f9f197ca8e126e2e3b07a0175ad22b`. Confirmation eligibility is false.
