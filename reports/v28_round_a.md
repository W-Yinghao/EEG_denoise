# V28 Round A

Round A trained PopCleanDET, SupportCleanDET, PopCleanCDM, paired-only
SupportCleanCDM, and natural-consistency SupportCleanCDM on validation folds 0
and 2 with seed 20260901. All ten cells completed with finite outputs and
registered best/last checkpoints; updates ranged from 7,250 to 30,000.

The first evaluation (job 938670) and selection (job 938680) are superseded
because the inference loader routed both support-CDM labels to the paired-only
checkpoint. The checkpoints were not affected. Recovery job 938681 replayed
the same validation samples, diffusion noise, and step counts after separating
the two checkpoint routes; selection job 938685 then froze the corrected
result.

The selected Round-B configuration is:

```text
training stream: paired 70% + natural consistency 30%
lambda_low: 0.05
lambda_Q: 0.05
support encoder: frozen V25 DeepSets
DDIM steps: 10
K: 1
```

Across the two validation folds, the selected SupportCleanCDM had mean clean
RRMSE 0.763879. MATCH-minus-WRONG utility was +0.000337, while
MATCH-minus-PopCleanCDM utility was -0.000516. The context evidence is therefore
weak at Round A, not a positive scientific finding. The model remains stable
and the protocol calls for a complete participant-first Round B rather than a
threshold-based early stop.

The selected natural corrected remaining ratio was 1.000525 and low-EOG
observation change was 0.006958 on validation data. These are separate
artifact/retention outcomes; neither is described as physiological
preservation. ERP, SSVEP, and ERD/ERS remain unavailable.
