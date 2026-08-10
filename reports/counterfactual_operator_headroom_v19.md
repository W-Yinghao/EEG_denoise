# Counterfactual Operator-Swap Headroom v19

This is development evidence for participant-session calibration and paired operator-swap semi-simulation only. It is not a natural EEG counterfactual, stable brain physiology result, or validated subject-aware denoiser.

## Routing outcome

- O0: `FAIL`
- Route: `SUPPORT_TO_QUERY_OPERATOR_TRANSFER_NO_GO`
- O1: `NOT_RUN`
- Evaluable: 15/16; policy denominator: 16.
- All source waveform/marker reads were allowlist-checked before open; sealed reads: 0.

## Participant-first effects

| Effect | Mean | Median | Positive | Exact one-sided p | Descriptive 95% bootstrap | Frozen mean floor |
|---|---:|---:|---:|---:|---:|---:|
| N_P | 0.167628 | 0.138442 | 15/16 | 0.000031 | [0.100838, 0.255047] | 0.229309 |
| N_W | 0.163277 | 0.123513 | 15/16 | 0.000031 | [0.100804, 0.252500] | 0.229309 |
| H_P | 0.720564 | 0.699026 | 15/16 | 0.000031 | [0.548056, 0.889160] | 0.174469 |
| H_W | 0.662092 | 0.696399 | 15/16 | 0.000031 | [0.499456, 0.813033] | 0.174469 |

## Gate audit

Failed criteria (2): N_P_mean_above_floor, N_W_mean_above_floor

O0-A uses evaluator-only later EOG solely to score natural transfer. O0-B constructs `y=x+C_query e` with operator recipient, carrier donor, and EOG donor distinct and keeps `x/e/y/mask` identical across arms. `C_query` is absent from inference packages.

The strong population is the participant-equal mean over all 15 nonrecipient development participants. WRONG donors are scored separately then averaged within recipient.

## Boundaries

Mobile sealed participants, PhysioMotion sealed participants, SHU Day-4/5, and PhysioTrait Day-200 remained unopened. No DET, diffusion, identity model, or other neural network was trained.
