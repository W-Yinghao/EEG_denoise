# V10 scientific-status correction

This addendum preserves every V10 result and changes no historical JSON or CSV.

- BCI2a: `CURRENT_BAND_TRANSFER_IDENTIFIABILITY_NOT_ESTABLISHED`. This does not imply that BCI2a is unsuitable for personalization.
- BCI2b: `OPERATOR_IDENTITY_SPECIFICITY_PRESENT_BUT_BASE_DENOISING_PIPELINE_INVALID`. The historical MATCH-WRONG contrast is operator-identity specificity, not MATCH-POP subject utility.
- Hierarchical Score-LoRA: `TECHNICALLY_IMPLEMENTED_BUT_SCIENTIFICALLY_NOT_ADJUDICATED`.
- V10 WRONG used known-training/session-mismatched donors and must not be described as unseen-WRONG.
- V10 SHUFFLED was a batch-level pair permutation, not a temporal shuffle.
- Evaluator arrays were generated before inference, but inference/model code never read evaluator fields. The stronger historical wording that evaluator fields were opened only after outputs were frozen was inaccurate.

V11 is an independent EOG-guided development experiment. Both support and later-query inference use EEG+EOG; it is not an EEG-only method.
