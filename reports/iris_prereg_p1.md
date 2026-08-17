# IRIS P1 preregistration — covariance-inflation gate pilot (MobileBCI dev-15)

Committed BEFORE any P1 execution. Charter 5cc8e05; K verdicts cc98a3d (K1 set the
reclamation bar via the charter-frozen formula; K2 removed the drift term). CPU only —
the pilot adjudicates the gate on the LINEAR subtraction path; no diffusion sampling.
Decision JSON → `results/iris/p1/p1_decision.json`; per-cell rows banked beside it.

## Question

The incumbent's binary EB gate abstains 6/93 cells (all within-outlier, full support —
K1: 100% reliability-class, 90.1% of the 0.13707 gate-shrinkage row). Does replacing
the hard λ-zeroing with covariance inflation reclaim ≥ 30% of the abstained-cell loss
without reopening the wrong-donor hole or ever falling below NO_A0?

## Machinery (frozen — V44-S1 verbatim where shared)

5 folds × 3 seeds (20261201/2/3); `TransferRegistry(data, fold, 30, 0.05)`;
`EBTransferRegistry(..., 120)`; episodes `TransferEpisodeSampler(data, fold, "test",
seed+3, registry30).sample_balanced(8)`; drive = `pinv(C_query) @ artifact`
(`_bank_drives` convention); metric `paired_metrics` → `rrmse_temporal` (program
primary); cell = (participant, session, task); per-cell aggregation = mean over the
cell's episodes, then pooling across folds/seeds; bootstrap 5000 draws seed 420.

## Arms (linear path: x̂ = y − C̃ψ, or weighted where stated)

| Arm | Operator | Note |
| --- | --- | --- |
| BINARY | `eb120.operator(key,"EB")` | incumbent gate (hard λ-zeroing) |
| INFLATION | see below | the IRIS gate |
| INFLATION_SOFT | C_soft, no Wiener weighting | declared descriptive decomposition |
| POP | `C0` population | |
| NO_A0 | zero subtraction | safety floor |
| ORACLE | `C_query` | evaluator-only; reclamation denominator |
| WRONG_binary | wrong-donor gated | V44 wrong-donor convention (first eligible other owner) |
| WRONG_inflation | wrong-donor soft + its own V | harm gate arm |

**INFLATION construction (frozen).** λ_soft = τ²/(τ² + within/4) with NO hard gate;
C_soft = C_pop + λ_soft (C_own − C_pop). Per-coefficient posterior variance V from
`_posterior_variance` (U0-b object), NO drift term (K2 verdict). Per episode and
channel i: â_i(t) = (C_soft ψ)_i(t); Var_i(t) = Σ_j V_ij ψ_j(t)²; Wiener weight
w_i = Ē_t[â_i²] / (Ē_t[â_i²] + Ē_t[Var_i]); subtraction x̂_i = y_i − w_i â_i.
The same construction with the wrong donor's C_soft and V gives WRONG_inflation.

## Gates (frozen)

- **GATE-P1a reclamation** (abstained cells only, i.e. cells whose BINARY hard gate
  fired; pooled over folds/seeds):
  R = (RRMSE_BINARY − RRMSE_INFLATION) / (RRMSE_BINARY − RRMSE_ORACLE),
  computed on abstained-cell means. PASS = R ≥ **0.30** (the K1-priced bar,
  f_conv = 1.00), bootstrap CI reported. If the denominator < 0.005 the gate is
  reported VACUOUS (nothing to reclaim) and treated as FAIL for the ladder.
- **GATE-P1b never-worse-than-NO_A0** (every cell, deployable INFLATION arm):
  RRMSE_INFLATION ≤ RRMSE_NO_A0 + 0.005 per cell. PASS = zero violations.
- **GATE-P1c wrong-donor harm**: mean(RRMSE_WRONG_inflation − RRMSE_POP) ≤ +0.005
  (the V44-G2 contrast form and margin; the binary gate's banked value is −0.000223).

## Decision ladder (frozen)

- P1a AND P1b AND P1c pass → the inflation gate is ADOPTED as IRIS's gate for the
  fights (F1/F2); the gate-row claim proceeds.
- P1a fail, P1b AND P1c pass → HYBRID: the binary hard gate is retained and inflation
  operates only inside the gate-open region; the abstention-reclamation claim is
  DROPPED; fights proceed.
- P1b or P1c fail → the inflation gate is DEAD for point estimates (binary gate kept
  unchanged); covariance inflation survives only as a UQ-width mechanism in F4.

Honest-power note: the reclamation gate reads on 6 abstained cells (each appearing in
exactly one fold's test split; 3 seeds give 18 cell-instances). Small-n is inherent —
these six cells ARE the phenomenon (90.1% of the row). The CI is reported; the verdict
reads on the pooled point estimate against the frozen bar, per program convention.
