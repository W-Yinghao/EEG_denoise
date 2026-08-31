# V43-S2 — RGCC floor-definitive round
### Server execution instructions (Claude Code)

Operator has read the S1 and S1.5 decisions. Under the S1.5 NO-GO, S2 runs **floor-only**:
train the deployable gated system on its own state distribution and convert the provisional
S1 floor into definitive claims, complete the oracle-trained ceiling picture to the full
panel, and measure the lambda-privacy curve. No gain claim is pursued; the S1.5 NO-GO
stands and is not reopened by anything in this round.

---

# 1. Workspace and Git

Continue on:

```text
worktree:  /home/infres/yinwang/denoiseNet_rgcc_v43
branch:    codex/rgcc-v43        (tip = 64f3c34, the S1/S1.5 decisions commit)
```

Commit per stage, push at will. No PR, no master merge.

Harness: identical to V43 §2 — Slurm for all compute (CPU/cpu-high; H100/A100/L40S/A40/V100
by availability; note the 24 h cap on A100/H100 observed in S1.5 — use `--time=24:00:00`
there), no verification ceremonies, sealed-8 never read, V42R artifacts read-only,
query EOG only in evaluator/ORACLE/teacher roles. One short ledger section at the end.

---

# 2. Preregistration addendum (write BEFORE any submission)

Append to `reports/v43_preregistration.md`:

```text
V43-S2 addendum — frozen before the first S2 submission.

S2a DEFINITIVE FLOOR ENDPOINTS (on the retrained gated model, participant-first, n=15):
  D-F1 wrong-donor safety:
       mean[RRMSE(WRONG_EB120) - RRMSE(POP)] <= +0.005
       AND reduction vs the ungated WRONG arm has bootstrap CI-low > 0.
  D-F2 short-support safety:
       the hard gate routes <60-s support to the bit-identical POP state (assert, all cells).
  D-F3 definitive non-inferiority:
       mean[RRMSE(MATCH_EB120) - RRMSE(POP)] <= +0.002
       AND one-sided 95% bootstrap upper bound <= +0.005.
  D-F4 duration safety (no spike at any budget):
       for every d in {10, 30, 60, 120}: mean[RRMSE(MATCH_EBd) - RRMSE(POP)] <= +0.002.
       The duration curve shape is reported descriptively; no monotone-benefit claim.
  Holm over {D-F1, D-F3, D-F4}. D-F2 is a construction check, not a statistical test.

S2b CEILING COMPLETION: descriptive only. The S1.5 NO-GO is final for this project;
  no pooled reanalysis may reopen the gain leg. If folds 1/3/4 show strong sign
  heterogeneity, report it as heterogeneity; reopening would require a new
  preregistered protocol (v19->v20 discipline).

S2c PRIVACY CURVE: descriptive; report top-1 linkage and verification AUROC per lambda;
  no "privacy-safe" claim at any lambda.

DET twin / LINEAR-EOG positioning: descriptive competitive positioning only; no
  superiority claim in either direction (ledger C05). LINEAR-EOG is labeled
  "requires query EOG at inference; not information-matched".
```

---

# 3. S2a — gated retraining (floor-definitive)

Training (15 cells = 5 folds x seeds {20261201, 20261202, 20261203}):

```text
recipe   : clone the V42R training recipe (80k updates, AdamW 1e-4, EMA 0.999,
           20% population-context dropout, checkpoint by validation POP RRMSE)
change 1 : conditioning states during training come from the EB builder with
           SUPPORT-DURATION RANDOMIZATION: each training episode draws its support
           duration from {10, 30, 60, 120} s uniformly, builds the gated state
           (hard gate included, so 10-s draws train the POP route via the gate)
change 2 : quality features under the frozen clamp contract (unchanged from S1)
```

Inference arms per cell (common noise, reuse the sample_bank seed convention):

```text
POP, MATCH_EB120 (primary), MATCH_EB10/EB30/EB60 (duration set),
WRONG (ungated, for the reduction contrast), WRONG_EB120,
NO_TRANSFER_BRANCH (branch-necessity check on the retrained model)
```

DET twin (capacity-matched one-step, for positioning): clone the V42R backbone, remove the
diffusion state input (feed x_t := y, fixed t=0 embedding), train with direct MSE to the
clean target, identical optimizer/updates/dropout/conditioning; 10 cells
(5 folds x 2 seeds) suffice. Evaluate POP and MATCH_EB120 arms.

LINEAR-EOG reference (CPU): per-subject gated transfer applied as direct subtraction
clean_hat = y - C_gated * E_bipolar(query) on the paired panel; labeled non-matched
(uses query EOG at inference). One CPU job.

Aggregate: participant-first contrasts per the addendum; decision JSON
`results/rgcc_v43/stage2/decision.json`; report `reports/v43_stage2.md` including:
the D-F1..D-F4 table, the duration curve, DET/LINEAR positioning rows, branch-necessity
on the retrained model, and the lambda distribution of the retrained-state builder.

Slurm:

```text
training : v43_stage2_train.sbatch, partition A100 or H100 (--time=24:00:00), array 0-14
           (V100 fallback --time=36:00:00); DET twin array 0-9 same template
inference: reuse the S1 replay template (any GPU partition, --time=04:00:00), array 0-14
aggregate: partition CPU, --time=00:30:00
```

Budget expectation: ~25-50 GPU-hours total given the ~1 h/cell wall time observed in S1.5.

---

# 4. S2b — ceiling completion (descriptive)

Train the remaining oracle-trained cells: folds {1, 3, 4}, seed 20261201, identical S1.5
recipe. Evaluate ORACLE vs POP on each fold's held-out participants. Pool with the S1.5
cells into the n=15 panel-complete figure: per-participant POP - ORACLE forest plot.
Write `results/rgcc_v43/stage2b/ceiling_panel.json` and a short section in
`reports/v43_stage2.md`. The NO-GO is not revisited.

```text
v43_stage2b.sbatch, partition A100/H100, --time=24:00:00, array 0-2
```

---

# 5. S2c — lambda-privacy curve (CPU only)

For lambda in {0, 0.25, 0.5, 0.75, lambda_hat (the EB value), 1.0}: build the 120-s gated
signatures for all development participants, run the existing linkage machinery
(`src/eeg_scad/evaluation/linkage_diagnostic.py` pattern from V30) on the stored states:
top-1 participant identification and same/different verification AUROC, participant-first.
Also record state size in bytes. Output `results/rgcc_v43/stage2c/lambda_privacy.csv`
+ one figure-ready table in the report. Partition CPU, single job.

---

# 6. Prohibitions

```text
no sealed reads
no natural-route training, tuning, or evaluation (that is S3, separate instructions)
no gain-claim endpoints anywhere in S2; no pooled reopening of S1.5
no modification of frozen V42R or V43-S1 artifacts
no manuscript edits
no threshold changes after the addendum is committed
```

---

# 7. Deliverables and stop point

```text
reports/v43_preregistration.md   (addendum appended, committed before submissions)
results/rgcc_v43/stage2/decision.json + per-cell results
results/rgcc_v43/stage2b/ceiling_panel.json
results/rgcc_v43/stage2c/lambda_privacy.csv
reports/v43_stage2.md            (D-F table, duration curve, DET/LINEAR rows,
                                  n=15 ceiling forest data, privacy curve)
one short ledger section
```

Stop after the S2 decision JSON is written. Report the D-F1..D-F4 verdicts, the DET/LINEAR
positioning rows, the pooled ceiling numbers, and the privacy curve back to the operator
verbatim. S3 (natural-route repair + BCI2b MI-kappa endpoint) will be issued separately.

---

# 8. Kickoff prompt

```text
Read V43_S2_Floor_Definitive_Server_Instructions.md in full and execute it. Continue on
branch codex/rgcc-v43 in the existing worktree. Commit the preregistration addendum
BEFORE any submission. Run S2a (15 gated-retraining cells + 10 DET-twin cells + LINEAR-EOG
CPU reference), S2b (3 oracle-trained completion cells), and S2c (lambda-privacy curve)
concurrently where capacity allows; aggregate; write the decision JSON and report; commit;
stop. Slurm only (CPU/cpu-high; A100/H100 at --time=24:00:00, V100 fallback); no
verification ceremonies; no sealed reads; no gain endpoints.
```
