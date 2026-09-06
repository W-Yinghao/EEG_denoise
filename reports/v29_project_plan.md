# V29 project plan

V29 freezes the V28 population clean models and the V25 DeepSets raw-support encoder. It trains only zero-initialized residual adapters. MATCH receives ordinary clean supervision; WRONG and exact POP bypass appear only in counterfactual intervention/ranking. All CDM context swaps share the query, Gaussian noise, and DDIM10 schedule. Natural held-out inference reads query EEG and query-disjoint support only; evaluator auxiliaries open after output digest freeze. This is development work, not confirmation.

The authoritative ledger was synchronized byte-for-byte from the user-provided v1.9 handoff before implementation. It will be updated to v2.0 only after V29 results are complete.
