# V41R project plan

V41R tests a narrow increment: a two-regressor, query-disjoint EOG-to-EEG transfer signature
conditions a shared channel-wise EEGDfus model. V40R remains frozen and is not repaired.

## Frozen contracts

- Base: `ade827ebc587f4edf8c4eede11a5d4472116338f`.
- Official EEGDfus: `a19a652b3b6346188ae77067e1daf8b90cad005f`, unchanged checkout.
- Development participants and five folds are inherited from V24/V31.
- Support is a chronological, non-overlapping prefix; normalization reads only that prefix.
- Query-generating transfer is fitted independently on Qgen and never enters the support estimate.
- Inference reads query EEG and a precomputed transfer signature, never query EOG or clean target.
- Sealed data, V20–V40R, A-track, and `taas_submission/**` remain read-only.

## Provenance correction

The active 15-participant counterfactual panel is the audited V19/V24 participant-session
semi-simulation resource derived from the 46-EEG/four-eye-electrode acquisition. Historical Klados
v4 cannot establish participant identity and is not silently relabeled as participant-held-out.
V41R therefore preserves the established participant panel while naming its source contract exactly.

## Execution

1. Audit bipolar VEOG/HEOG construction and transfer coordinates.
2. Validate an official-shape shared single-channel population backbone on 512-sample windows.
3. Jointly train population and zero-initialized transfer FiLM with 20% population-context dropout.
4. Evaluate POP/MATCH/WRONG/SHUFFLED/ORACLE with common random numbers.
5. Run 0/10/30 s support, minimal ablations, lightweight linkage, and frozen natural evaluation only
   after paired POP validity is known.
6. Aggregate participant-first, update ledger, test a clean archive, and push only the V41R branch.

No scientific threshold automatically stops training. Engineering hard stops are leakage,
coordinate/provenance failure, nonfinite or scale collapse, checkpoint mismatch, random-stream
mismatch, and invalid participant aggregation.
