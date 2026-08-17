# IRIS milestone M-F(part 1) — F2 verdicts (decision JSONs verbatim)

Branch codex/iris. Prereg ea72515 pre-execution; raw banked pre-analysis. Slurm 947298
(CPU). F1 (947295, GPU) still queued behind the operator's fleet. 0 GPU-h spent.

## F2-a — results/iris/f2/f2a_decision.json

```json
"arm_levels": {
  "INCUMBENT_NATIVE":  {"attenuation_db": 7.803 [6.760, 8.846],
                        "retention": 0.517, "e3": 0.404, "e3_pass": false},
  "INCUMBENT_SAMEREF": {"attenuation_db": 8.030 [7.001, 9.061],
                        "retention": 0.464, "e3": 0.376, "e3_pass": false},
  "IRIS_TYPED":        {"attenuation_db": 7.969 [6.923, 9.008],
                        "retention": 0.439, "e3": 0.354, "e3_pass": false}}
"readings": {
  "native_reference":  {"verdict": "INVALID(E3/E2)",
    "attenuation_delta_db": 0.1667 [-0.0527, 0.3707], "retention_delta": -0.078},
  "same_reference":    {"verdict": "INVALID(E3/E2)",
    "attenuation_delta_db": -0.0606 [-0.2062, 0.0553], "retention_delta": -0.0251}}
```

Adjudication: no valid contest exists in the unrestricted linear class — every arm,
incumbent included, fails the lambda-wave exogeneity bar (removes 60-65% of
post-saccadic posterior norm). Had the arms been valid, both readings are TIEs on
attenuation. A spatially-gated repair (per-channel abstention) is preregistered
(addendum below) before any rerun.

## F2-b — results/iris/f2/f2b_decision.json

```json
"delta_inc":  {"mean": -0.0252, "bootstrap_low": -0.0400,
               "bootstrap_high": -0.0133, "n": 28, "positive_count": 5}
"delta_null": {"mean": 0.0073, "bootstrap_low": 0.0029, "bootstrap_high": 0.0133}
"intact_companion": {"n": 2, "delta_inc": [-0.0097, -0.0013]}
"gate": {"pass": false, "verdict": "typed leg dead on the sealed panel class; any
          sealed opening proposal shrinks to incumbent-class confirmation"}
```

## Sealed recommendation (operator decision, no action taken)

With the typed arm dead on the sealed panel class (F2-b) and the incumbent class
failing exogeneity on this corpus's richer panel (F2-a), I see no claim on EEGEyeNet
currently worth the one-shot sealed spend. My recommendation: the sealed block stays
FROZEN and unopened; it remains available for paper-time confirmation if a later
valid claim emerges (e.g. the spatially-gated class below). Opening requires your
sign-off in any case; I will not propose it again unless a gated-class result
qualifies.
