# IRIS milestone M-E — P1R + T2 verdicts (decision JSONs verbatim)

Branch `codex/iris`. Prereg `reports/iris_prereg_p1r_trepair.md` (4543b37 body,
amendment 4a4d2ea) committed before execution; raw banked pre-analysis. Slurm job
947242, partition CPU. Still **0 GPU-h spent** of 400.

## P1R — `results/iris/p1r/p1r_decision.json`

```json
"G_P1R_a_fallback_repair": {"paired_mean": 0.2537441915974991,
  "ci_low": 0.007016523646379835, "ci_high": 0.5004718595486184, "n": 4,
  "binary_mean": 1.8781846012054177, "binary_noa0fb_mean": 1.6244404096079188,
  "pass": true}
"G_P1R_b_relative_never_worse": {"violations": 7, "pass": false}
"G_P1R_c_pooled": {"mean": 0.001771597763114603,
  "ci_low": -0.0002842500224749395, "ci_high": 0.004009711214277391,
  "pass": true, "gain_sized": false}
"G_P1R_d_harm": {"wrong_inflation_minus_pop": -0.062311027751067,
  "wrong_binary_noa0fb_minus_pop": -0.19145261449339424, "pass": true}
"adopted_gate_for_fights": "BINARY_NOA0FB"
"pooled_means": {"BINARY": 0.43396946508269535, "BINARY_NOA0FB": 0.4226919454561398,
  "INFLATION": 0.42458695860822515, "INFLATION_NOA0FB": 0.4209203476930253,
  "NO_A0": 0.5972533284650511, "POP": 0.7877236257267984}
```

**Adopted**: the incumbent binary gate with the NO_A0 fallback (hard-gated cells stop
subtracting instead of subtracting the population operator). The inflation gate is
retired from point estimates on two independent readings (P1 ladder; P1R-b with 7
gate-open violations) and lives on only as a UQ-width mechanism (F4). The
abstention-row RECLAMATION claim stays dropped; what the row yielded instead is a
deployable fallback correction adjudicated on the incumbent itself.

## T (repaired, replacement reading) + T2 (increment reading) — `results/iris/w/`

```json
t_typed_info_repaired: {"delta_participant_first": {"mean": -4.145,
  "bootstrap_low": -5.5926, "bootstrap_high": -2.9051, "n": 26,
  "positive_count": 1}, "gate": {"pass": false}}
t2_increment: {"delta_inc_participant_first": {"mean": 0.0683,
   "bootstrap_low": 0.0253, "bootstrap_high": 0.1102, "n": 26,
   "positive_count": 20},
  "delta_null_participant_first": {"mean": -0.0067, "bootstrap_low": -0.0105,
   "bootstrap_high": -0.0029},
  "inc_minus_null_participant_first": {"mean": 0.075, "bootstrap_low": 0.0343,
   "bootstrap_high": 0.1154},
  "gate": {"pass": true}}
```

The two readings together are the finding: typed/optical drives **cannot replace** the
EOG waveform reference (−4.145, replicated under the repaired encoding) and **do add**
~7% held-out artifact-window information on top of it (+0.0683, ≥0.05 floor, 20/26,
clearing a regressor-count null that itself runs slightly negative). F9's open clause
— "the family question is open only where the reference is genuinely richer" —
resolves YES, additive-only. The typed family proceeds to F2 strictly as an addition
to the EOG columns, which also preserves the incumbent-as-exact-sub-model property.

Repair robustness: W3 repaired 0.0806 [0.0575, 0.1044] ≈ banked 0.0812; W1 repaired
0.8998 ≈ banked 0.8955 (inside CI width → banked stays primary); W2 untouched.

## Campaign state after M-E

| DoD item | State |
| --- | --- |
| D1 MobileBCI fight | NEXT: confirm the adopted fallback on the diffusion path (F1) |
| D2 EEGEyeNet dual-reading + instrument items | typed family qualified (additive); W1/W2/W3 rows banked; F2 prereg next |
| D3 gate properties | substantially served (see digest scorecard) |
| D4 UQ | pending (F4; inflation's surviving role) |
| D5 sealed | frozen, quarantined, untouched |
| D6 digest | extended at every milestone |

Budget: 0 GPU-h spent; planned F1 ≤ 20, F2 ≤ 40, F4 ≤ 80.
