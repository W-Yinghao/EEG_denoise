# CGDR diffusion incremental-value decision v2 protocol

This decision protocol was frozen before the SGEYESUB natural-EEG evaluation
outcomes were available. It adds the missing natural-EEG matched comparison to
the v1 local evidence without rewriting or relabelling any historical result.

## Evidence boundaries

- `current_M2_no_incremental_value` remains a separate historical result. It
  concerns the unconditional clean prior, 100-step deterministic DDIM and
  frozen final-hard-Q M2 instance under the Klados source-record protocol.
- The SGEYESUB input is a prospective, release-internal block1-to-block2 natural
  EEG comparison between operator-conditioned conditional DDIM100 and a
  task-matched multichannel deterministic U-Net. It has no clean target and
  therefore cannot support clean-waveform recovery claims.
- EEGDfus contributes its independently frozen official/strict source-epoch
  benchmark local status.
- Klados M1, M4 and operator-conditioned DDIM100 are exploratory development
  source-record comparisons. They can complete the all-negative truth-table
  branch, but they cannot turn a natural-SGE/EEGDfus positive result into a
  formal G1 or G3 claim.
- Formal G1 and G3 remain `NOT_RUN_BLOCKED` for every v2 outcome.

## Required natural-SGE structure

The evaluator rejects partial or ambiguous aggregates. The input must retain
all 44 available participant stems, report 43 compatible performance stems and
the single preblocked `study05/study05_p42`, and complete the 15 frozen
evaluation folds. Each of the two learned arms must have exactly 6,000
successful optimizer updates in every fold. The aggregate must also assert the
matched input/data contract, prove that query EOG/classes/labels opened only
after all outputs froze, and explicitly deny clean-target and clean-waveform
recovery semantics.

The natural-SGE local decision is not recomputed or tuned here. v2 accepts only
the status produced from the already-frozen
`configs/cgdr/sgeyesub_diffusion_incremental.yaml` prospective threshold
section. Any other threshold source, post-evaluation threshold change, partial
coverage, invalid protocol ID or fabricated local status is rejected.

## Frozen truth table

| Natural SGE local status | EEGDfus local status | Klados M1/M4/operator-conditioned status | v2 conclusion |
|---|---|---|---|
| pass | meets frozen stability | any valid local result | `conditional_diffusion_supported` |
| no detectable | no detectable | all three no detectable | `diffusion_no_detectable_incremental_value_under_tested_protocols` |
| any other complete combination | any | any | `inconclusive` |
| missing, partial or invalid input | — | — | fail-closed `inconclusive` |

The positive conclusion is limited to the tested SGE natural-EEG and EEGDfus
protocols. The negative conclusion is limited to the tested datasets, tasks,
splits, objectives and configurations. Neither conclusion licenses the phrases
“diffusion is useless”, “diffusion sampler has no value”, “EEG diffusion is
disproved”, or “personalization failed”.

