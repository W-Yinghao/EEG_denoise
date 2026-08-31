# V43-S3 — Natural-route repair, downstream endpoint, cross-panel floor
### Server execution instructions (Claude Code)

S2 banked the definitive floor (D-F1..D-F4 all PASS) plus two surprises this stage must
consume: (1) the retrained gated model is state-reliant (ungated WRONG +0.654; branch
+0.092) with the gate providing complete protection — S3 carries that framing forward;
(2) duration-randomized gated training improved the absolute POP route 0.632 → 0.526 —
S3a retrains FROM THE S2a RECIPE, not from V42R's. S3 closes the V43 arc's last
submission gate: natural-recording validity.

---

# 1. Workspace and Git

```text
continue : worktree /home/infres/yinwang/denoiseNet_rgcc_v43, branch codex/rgcc-v43
tip      : e42158c (the S2 decisions commit)
```

Harness unchanged (V43 §2): Slurm only, CPU/cpu-high + H100/A100/L40S/A40/V100
(A100/H100 `--time=24:00:00`), no verification ceremonies, sealed-8 never read, frozen
V42R/S1/S2 artifacts read-only. V44 and flagship-m0 worktrees untouched.

---

# 2. Preregistration addendum (append to reports/v43_preregistration.md BEFORE submissions)

```text
V43-S3 addendum — frozen before the first S3 submission.

S3a NATURAL-ROUTE REPAIR (primary):
  Diagnosed cause of the frozen invalidity (POP remaining ratio 1.082, attenuation
  −0.133 dB): train/natural severity-prevalence mismatch — paired training injects
  artifact gains {0.35,0.70,1.15}×U(0.85,1.15) with only 15% zero-artifact episodes,
  while natural query windows are dominated by low/no-artifact content.
  PRIMARY repair (one registered change to the S2a recipe): continuous severity
  augmentation — per episode, gain g ~ mixture: 40% exactly 0, 60% LogUniform(0.05, 1.3);
  everything else identical to S2a (duration-randomized gated states, 80k updates,
  checkpoint by validation POP RRMSE).
  SECONDARY (inference-only, evaluated on the same checkpoints): support-estimated
  per-window output scaling of Delta by a support-only artifact-level statistic.
  Learned severity predictors are NOT permitted (N7 discipline).

  GATES:
  N-G1 natural validity (the frozen V42R criteria, unchanged): POP arm
       heldout-EOG remaining ratio < 1 AND artifact attenuation > 0 dB AND
       output/input RMS q99 < 3, participant-first.
  N-G2 paired non-degradation: repaired POP paired RRMSE ≤ S2a POP (0.526) + 0.010.
  N-G3 floor preservation on the repaired model: D-F1 and D-F3 re-evaluated with
       S2 margins (wrong-gated ≤ +0.005 with reduction CI-low > 0;
       MATCH_EB120 − POP mean ≤ +0.002, upper95 ≤ +0.005).
  If N-G1 passes: natural MATCH_EB120 − POP utilities reported with CIs (still
  development evidence). If N-G1 fails on the primary repair: apply the secondary
  scaling and re-test ONCE; a second failure closes the natural route for the V43 arc
  (no further repairs; the flagship's K2 rule inherits the verdict).

S3b DOWNSTREAM ENDPOINT (conditional on N-G1):
  SSVEP spectral SNR on natural task windows: SNR at the stimulation frequencies
  (reuse the mobile_bci_headroom_v4 readout machinery) computed on denoised vs raw
  natural windows; endpoint = participant-first SNR improvement of the gated MATCH arm
  and the POP arm over RAW, plus MATCH − POP (descriptive). ERP readout as a secondary
  descriptive row. No decoding-accuracy claim; no label-dependent tuning.

S3c CROSS-PANEL FLOOR PROBES (CPU, subtraction class):
  On Klados v4 (54 records) and BCI2b (9 participants): the V44-S0-style subtraction
  probe with gated operators — endpoints: gain(C_gated vs C0), wrong-donor safety under
  the gate, duration flatness. Floor rules as S2 margins; gain rows descriptive.
  (These panels are the flagship transport-GO panels; this data feeds both papers.)

S3d PRIVACY ONSET GRID (CPU):
  Extend the lambda-privacy curve with lambda in {0.05, 0.10, 0.15, 0.20}; report
  top-1/AUROC and locate the linkage onset. Descriptive; the paper's privacy framing
  becomes "abstention (lambda=0) is the privacy mechanism; any subject content pays
  most of the linkage cost" — no privacy-safe claim at any lambda > 0.

Statistics: participant-first (record-first on Klados), 5000-draw bootstrap; Holm over
{N-G1's two POP criteria treated as one gate, N-G2, N-G3}. No threshold changes after
this commit.
```

---

# 3. S3a — repair training and evaluation

```text
cells     : 5 folds x seeds {20261201, 20261202, 20261203} = 15, S2a recipe + severity mix
training  : v43_stage3_train.sbatch, A100/H100 --time=24:00:00 (V100 fallback 36 h), array 0-14
paired inference (floor re-check arms): POP, MATCH_EB120, WRONG, WRONG_EB120,
            NO_TRANSFER_BRANCH — common noise, S1 seed convention
natural   : frozen-evaluator flow as V42R natural (outputs frozen before evaluator opens
            query EOG); arms POP, MATCH_EB120; frozen gate criteria
aggregate : CPU; decision JSON results/rgcc_v43/stage3/decision.json with
            {N-G1, N-G2, N-G3, natural utilities if valid, secondary_applied: bool}
```

# 4. S3b/S3c/S3d

S3b runs only if N-G1 passes (GPU inference on natural task windows + CPU readouts).
S3c and S3d are CPU-only, run immediately in parallel with S3a training:

```text
S3c: v43_stage3c.sbatch, partition cpu-high — Klados v4 + BCI2b subtraction probes
     (reuse eb_transfer_v43 raw-operator accessor; montage-appropriate ridge transfers;
     population operators = recipient-excluded fold-train means)
S3d: single CPU job extending stage2c lambda grid
```

---

# 5. Deliverables and stop point

```text
reports/v43_preregistration.md (addendum), reports/v43_stage3.md
results/rgcc_v43/stage3/{decision.json, natural results, floor re-check}
results/rgcc_v43/stage3b_ssvep/ (if run), stage3c_crosspanel/, stage3d_privacy_onset/
one short ledger section
```

Stop after the S3 decision JSON (and S3b if triggered). Report N-G1/N-G2/N-G3 verdicts,
the natural utilities (or the closure statement), the SSVEP rows, the cross-panel floor
tables, and the privacy-onset grid verbatim. This completes the V43 arc; what follows
(flagship Month 1-3) will be issued after the M0 ceiling matrix lands.

Prohibitions: no sealed reads; no gain-claim endpoints; no new conditioning mechanisms;
no threshold changes post-commit; V42R/S1/S2 artifacts read-only.

---

# 6. Kickoff prompt

```text
Read V43_S3_Natural_Repair_Server_Instructions.md in full and execute it. Continue on
codex/rgcc-v43 (tip e42158c). Commit the preregistration addendum BEFORE any submission.
Launch S3c and S3d (CPU) immediately; train S3a (15 cells, S2a recipe + the registered
severity mixture) on A100/H100 at --time=24:00:00; run the paired floor re-check and the
frozen-gate natural evaluation; run S3b only if the natural gate passes; aggregate, write
the decision JSON and report, commit, and stop. Slurm only; no verification ceremonies;
no sealed reads.
```
