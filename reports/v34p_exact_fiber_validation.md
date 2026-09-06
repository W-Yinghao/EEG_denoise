# V34P exact fiber validation

All six fold/seed cells used the frozen V33P EEGNet heads. Every centered head had
rank 3 and a 125-dimensional fiber. Across cells, the largest `CN` residual was
`5.07e-16`; orthonormality and row/null cross-products were also at machine precision.

The output audit covers HEAD_ONLY plus weak, medium, and strong Fiber-OneStep and
Fiber-SANDiff outputs: 42 method/cell rows in total.

```text
maximum centered-logit error:       2.164e-7
maximum softmax probability error:  7.703e-8
prediction mismatches:              0
maximum fixed-head BA difference:   0
maximum fixed-head ECE difference:  5.210e-10
```

Thus exact fixed-function preservation passed as an engineering identity. Small
nonzero logit/probability residuals arise only from float32 representation materialization;
they never changed a decision.
