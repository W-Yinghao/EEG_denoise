# FLAGSHIP-M13R — One registered prior repair + the decoupled transport factorial
### Server execution instructions (Claude Code)

P0's stop is consumed. House rule applies: diagnose → ONE registered repair round → close
either way. The diagnosis is heterogeneous and both fixes are precedented:
(a) Klados/BCI2b amplitude collapse (q99 0.812/0.457) = the V41R full-generation failure
mode reappearing in canonical space under sparse-montage inpainting — the repair is the
program's only ever SUPPORTED diffusion design (P3 artifact-subspace, complement
identity), generalized to canonical coordinates; (b) MobileBCI PV-1 failure = transport
quality (kappa median 209, heterogeneity-dominated frames) — the repair is the W1
diagnosis-driven per-panel transport config. W3 is DECOUPLED from the prior: the analytic
cleaner is already valid and carries the primary transport rows regardless.

---

# 1. Workspace

```text
continue : codex/flagship-m13 (tip 3c2649c), same worktree
compute  : repair pilots ~35 min/cell A100; W3 CPU-heavy; total well under 30 GPU-h
```

# 2. Preregistration addendum (commit BEFORE submissions)

```text
FLAGSHIP-M13R addendum — frozen before first submission.

HONESTY RULE (immediate, unconditional): P0's Klados/BCI2b PV-1 utilities (+0.164, +0.401)
  are amplitude-shrinkage-confounded (q99 0.812/0.457) and may not be cited as denoising
  gains anywhere. PV-2 is reported as having performed its designed function.

R-A CANONICAL ARTIFACT-SUBSPACE READOUT (the amplitude fix, P3-precedented):
  the posterior updates ONLY the ocular canonical coordinates: correction c ∈ span(U°)
  (rank 3); every other canonical coordinate is pinned to the observation (hard data
  consistency). Sensor readout x̂ = y − T(ρ)^+ (U° û). Complement identity holds by
  construction: outside the sensor image of span(U°), x̂ = y bit-exactly. Expected
  q99 ≈ 1 structurally. The diffusion prior's role: the clean-EEG prior constrains the
  rank-3 coefficient trajectory via the posterior step (the P3 design in canonical
  coordinates). Training loss unchanged; only the sampler/readout change.

R-B PER-PANEL TRANSPORT CONFIG (frozen from the W1 diagnosis, not tunable):
  heterogeneity_dominated panels (MobileBCI, BCI2b): covariance alignment OFF (G = I);
  transport = ocular Procrustes ∘ montage lift only, with the split-half abstention rule.
  estimation_noise_dominated panel (Klados): G active for non-abstained subjects only.
  Rank-truncated whitening stays undeployed (failed its own target).

REPAIR GATES: PV-1 and PV-2 per panel, margins unchanged. PASS = both gates on ≥ 2 of 3
  panels including at least one cross-montage panel. FAIL → the pooled-prior axis is
  CLOSED as an honest negative with the two-mode diagnosis; no further repairs; the
  flagship descopes to {matrix + V43/V44 legs + UQ + per-panel/analytic transport rows}.

W3 TRANSPORT FACTORIAL (decoupled; runs regardless of the repair outcome):
  primary backbone = the validated analytic canonical cleaner; panels = Klados, BCI2b
  (GO panels); arms per the M13 prereg (T-POP / T-MATCH(ρ̂) / T-WRONG(gated) / T-ORACLE /
  GAUGE-NULL); TG-1 primary MATCH−POP with CI-low > 0, TOST ±0.005 alternative;
  TG-2 controls; LINEAR/DET rows as the backbone itself is deterministic — the DIFF row
  is added ONLY if the repair passes (repaired prior, same arms, common noise).
  Transport configs per R-B; abstentions counted in the denominator (ITT).

If repair passes: run a reduced P1 before any DIFF claims — pooled 3 seeds + LODO ×2
  (hold out Klados; hold out BCI2b). PV-3 and the LODO gate as originally frozen.
No threshold changes after this commit. Sealed cohorts untouched. W4 is banked; no
further UQ work this round.
```

# 3. Execution order

```text
1. commit addendum
2. R-A/R-B implementation (sampler/readout + transport configs; extend unit tests:
   complement identity in canonical coordinates, q99 structural check on synthetic)
3. repair pilots: 1 cell per panel (3 cells, ~35 min each) -> PV gates -> repair decision
4. W3 factorial (CPU/cpu-high arrays) with the analytic backbone on both GO panels
5. if repair passed: reduced P1 (3 pooled seeds + LODO x2, ~6-8 cells) -> DIFF rows in W3
6. aggregate: results/flagship_m13/{repair/decision.json, w3_transport/decision.json};
   reports/m13r_report.md; ledger section; commit, push, STOP
```

Report verbatim: the repair decision per panel (PV-1/PV-2 with q99), the W3 TG table per
panel (analytic rows, and DIFF rows if present), abstention counts, and the compute
ledger. Months 3–5 instructions follow from the operator's read.

# 4. Kickoff prompt

```text
Read FLAGSHIP_M13R_Prior_Repair_And_W3_Server_Instructions.md in full and execute it.
Continue on codex/flagship-m13. Commit the preregistration addendum first. Implement the
canonical artifact-subspace readout (R-A) and the per-panel transport configs (R-B); run
the three repair pilot cells; apply the frozen repair gates; run the W3 transport
factorial with the analytic backbone on Klados and BCI2b regardless; if the repair
passed, run the reduced P1 and add the DIFF rows. Write both decision JSONs and the
report; commit, push, stop. Slurm only; no verification ceremonies; no sealed reads.
```
