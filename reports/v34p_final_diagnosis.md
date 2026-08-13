# V34P final diagnosis

## Decision

```text
A. Fiber-SANDiff positive method
```

This positioning is supported by four distinct facts:

1. The frozen fixed task function is preserved exactly: zero prediction mismatches
   and zero balanced-accuracy change across all 42 audited outputs.
2. Strong Fiber-SANDiff reduces adaptive subject leakage versus RAW for all nine
   participants and reduces cross-session verification leakage for all nine.
3. It improves adaptive and verification privacy over Fiber-OneStep while retaining
   78.4% rather than 18.0% of fiber variance and improving all registered conditional
   distribution discrepancies.
4. It removes the fixed-head utility cost observed for V33P SANDiff.

The result does not establish across-the-board diffusion superiority. Fiber-OneStep
has `+0.009066` higher retrained-head BA and is about 34 times faster. Fiber-SANDiff
also retains substantial absolute subject leakage (`0.534529` adaptive BA), much of
which is consistent with information transmitted by the task-head-visible component.

The allowed next step is a single larger-participant-cohort replication of this
frozen exact-fiber method. No further method family should be added on BCI-IV-2a.

Claims remain development-only. V34P does not establish anonymity, mutual-information
removal, causal identification of nuisance, cross-dataset generalization, or clinical
fitness.
