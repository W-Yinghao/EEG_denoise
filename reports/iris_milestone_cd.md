# IRIS milestones M-C (P1) + M-D (W/T) — verdicts, diagnostics, repairs

Branch `codex/iris`. Preregs 4543b37 committed before execution; raw results banked
pre-analysis at 93821c6. Still 0 GPU-h spent. Slurm jobs 946814 (P1), 946819 (W/T),
partition CPU.

## M-C — P1 decision JSON verbatim (`results/iris/p1/p1_decision.json`)

```json
"GATE_P1a_reclamation": {"reclamation": 0.09117604964867738, "bar": 0.3,
  "bootstrap_low": 0.0015609980288091, "bootstrap_high": 0.39307065757948956,
  "binary_mean": 1.8781846012054177, "inflation_mean": 1.7069391551999173,
  "no_a0_mean": 1.6244404096079188, "pop_mean": 1.8781846012054177,
  "vacuous": false, "pass": false}
"GATE_P1b_never_worse": {"violations": 6, "worst_excess": 0.31439634044867804,
  "pass": false}
"GATE_P1c_wrong_donor_harm": {"value": -0.062311027751067, "margin": 0.005,
  "binary_reference_banked": -0.000223, "pass": true}
"ladder": "DEAD_POINT_ESTIMATES"
```

The frozen ladder verdict stands: **the inflation gate is dead for point estimates**;
the incumbent binary gate is retained; inflation survives only as a UQ-width mechanism
(F4). Three post-bank diagnostics that must be read WITH the verdict:

1. **P1b's failure is pathway-inherited, not inflation-specific.** The incumbent
   violates never-worse-than-NO_A0 on exactly the same 6 cells, and inflation is
   equal-or-better than the incumbent on every one of them; pooled over all 90 cells
   INFLATION 0.4246 < BINARY 0.4340. The absolute finding is real — subtraction (with
   ANY operator) is worse than no subtraction on ~7% of cells — but it indicts the
   deployed fallback, not the inflation mechanism. The gate as frozen measured an
   absolute property the incumbent itself fails; it adjudicated honestly and the
   verdict is banked, but the fight-relevant comparison (vs incumbent) was not what it
   measured.
2. **The abstention row is NOT convertible in practice.** λ_soft ∈ [0.0014, 0.020] on
   three of the four abstained cells — the reliability signal that fired the hard gate
   also zeroes the soft gate, so there is nothing to move. K1's f_conv = 1.00
   ("reliability-class in cause") resolves to ~0.09 in deliverable practice. The one
   confident abstained cell (sub-09|ses-04, λ_soft 0.937) reclaimed 64% of its span
   (0.7821 → 0.2795 vs NO_A0 0.2739) — the mechanism works exactly where the gate is
   confident, and that set is nearly empty. The 35.4% ledger row survives as
   abstention-priced SAFETY cost, not as recoverable headroom.
3. **Fallback discovery (deployable):** on abstained cells NO_A0 1.6244 beats the
   deployed POP-subtraction fallback 1.8782; pooled NO_A0 0.5973 < POP 0.7877. The
   right fallback for hard-gated cells is NO SUBTRACTION, not population subtraction.
   This is a one-line deployment change with its own prereg (P1R below).

## M-D — W/T decision JSONs verbatim (`results/iris/w/*.json`)

```json
w1_a4:     {"d_rms_participant_first": {"mean": 0.8955, "bootstrap_low": 0.8417,
             "bootstrap_high": 0.9524, "n": 30, "positive_count": 30},
            "d_corr_participant_first": {"mean": 0.5067}}
w2_kappa:  {"pooled_kappa": 0.3409, "n_events": 28927, "accuracy": 0.7987,
            "bar": 0.6, "pass": false}
w3_readout:{"included": 95, "excluded_referee": 82,
            "best_gain_participant_first": {"mean": 0.0812, "bootstrap_low": 0.0582,
             "bootstrap_high": 0.1056, "n": 26, "positive_count": 26},
            "verdict": "SIZED"}
t_typed_info: {"delta_participant_first": {"mean": -4.1569,
               "bootstrap_low": -5.5666, "bootstrap_high": -2.9498, "n": 26,
               "positive_count": 1}, "gate": {"pass": false}}
```

- **W1** replaces WAVE4's axis-limited A4 bound with a true-axis measurement at
  roughly half the size (0.8955 vs 1.7673) — entered beside, never over.
- **W2**: κ 0.3409 < 0.60 — Class-E (EOG-morphology) typing stays invalid; Class-G
  typing (EyeLink labels directly) is what IRIS uses and is unaffected.
- **W3**: the first readout-bound measurement with the CI clear of 0.03 — best-family
  gain 0.0812 [0.0582, 0.1056] on an operator-independent oracle, n=26 participants
  (vs WAVE4's n=3 straddle). Two caveats: 82/177 referee exclusions (counted), and the
  oracle shares the defective gaze encoding below → repair rerun reported beside.
- **T — banked FAIL with a confirmed instrument defect.** Gaze drive channels carry
  0-snaps at tracking loss, and loss concentrates exactly in the artifact-rich eval
  windows (verified: EP39_DOTS2 39.9% loss inside rich windows vs 12.4% outside,
  1312-px steps; the only positive recording has 2.2%). The typed-information verdict
  is therefore NOT yet earned in either direction; banked unedited, repair
  preregistered before any rerun (`reports/iris_prereg_p1r_trepair.md`), no gate
  constant moves.

## Sealed / budget / queue

Sealed quarantine untouched (mode 000). 0 GPU-h of 400 spent. Job monitoring switched
from sacct to squeue (slurmdbd connection refused; two stale sacct-waiters killed).
