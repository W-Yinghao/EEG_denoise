# IRIS Campaign Charter — master preregistration

Branch `codex/iris` (from `codex/eegeyenet-acquisition` tip 50b9e7f, which contains
master 8c54fc1 + the EEGEyeNet acquisition). Committed BEFORE any campaign compute.
Mission: `docs/IRIS_MISSION_BRIEF.md`. Facts F1–F10 and the digest closures are binding;
every deviation from `docs/EEG_denoise_IRIS_method_design.md` is argued in section 7.

## 1. Definition of done (from the brief, restated as my staging)

| Item | Serviced by stage | Status at charter time |
| --- | --- | --- |
| D1 mechanism-pure MobileBCI fight | F1 | pending |
| D2 EEGEyeNet dual-reading + 4 dangling instrument items | W + F2 | pending (data landed: 28 antisaccade + 30 dots dev subjects) |
| D3 gate properties | P1 + F1 arms | pending |
| D4 UQ vs DET-ensemble bar | F4 | pending |
| D5 sealed EEGEyeNet block frozen before query contact | done at this commit | **FROZEN: 55 subjects** |
| D6 digest extended with every verdict | every milestone | ongoing |

## 2. Staging (cheapest-first; each stage carries its own preregistration commit)

```text
K  (CPU only)  K1 abstention-cause taxonomy  → prices the gate-shrinkage bonus BEFORE
               the pilot bar is chosen
               K2 Σ_drift prior-predictive recheck in corrected (per-window) units
               → drift layer ON/OFF
               K3 EEGEyeNet instrument validity (true-VEOG correspondence gates)
               → unlocks or re-closes the four WAVE4 dangling items
P1 (≤5 GPU-h)  covariance-inflation linear-Gaussian pilot, MobileBCI dev-15, frozen
               V44-S1 assets — the marginalization thesis bet
T  (≤5 GPU-h)  typed-reference information kill, EEGEyeNet dots: typed-drive family
               ceiling vs static family ceiling, disjoint-segment oracles, VEOG
               cross-referee — the "rich reference is new information" thesis bet
W  (CPU)       the four instrument items, if K3 passes: true-VEOG A4 row; typed-label
               κ; event-level oracle; readout re-bound
F1 (≤40 GPU-h) IRIS vs incumbent, MobileBCI EOG-only, identical reference and prior
F2 (≤40 GPU-h) EEGEyeNet dual-reading fight (same-reference AND native-reference,
               both preregistered) — DET twin carries point claims (F8)
F4 (≤80 GPU-h) UQ: operator-posterior K-chain + inflation; bar = calibrated 80/90
               bands at < 3.0× CRPS (beat the current 3×; stretch target 1.5×)
S  (≤10 GPU-h) single sealed opening, C-1 protocol, OPERATOR SIGN-OFF REQUIRED
```

Every thesis-level bet (K1–K3, P1, T) is adjudicated in <10 GPU-h total. Any component
death leaves a functioning method whose floor is the incumbent (exact sub-model).

## 3. Budget ledger

Soft cap 400 GPU-h. Planned: K 0 / P1 5 / T 5 / W 0 / F1 40 / F2 40 / F4 80 / S 10 =
**180 GPU-h planned, 220 reserve**. Actuals tracked in `results/iris/budget.json`,
updated at every milestone. No single planned spend exceeds 150 GPU-h; crossing the cap
or any >150 GPU-h single spend = operator check-in first.

## 4. Frozen campaign-wide rules

- **Incumbent floor**: the V44-S1 frozen system (MATCH_gated; dev +0.143 [15/15],
  sealed +0.1537 [7/8]) is IRIS's exact sub-model. No IRIS variant that removes the
  pseudo-EOG columns ships. All MobileBCI comparisons use the identical frozen diffusion
  checkpoints, identical EOG reference, identical priors as V44-S1.
- **Strongest-baseline honesty**: gains vs NO_A0 (not vs POP) wherever the claim is
  "subject-awareness pays"; C05 wording (diffusion "competitive", never superior);
  retention metric named "low-EOG observation retention" (C08); Dev/Sealed never mix.
- **Exogeneity standing hazard gate**: any typed-subtraction arm must pass post-saccadic
  retention ≥ 0.84 on posterior channels (λ-wave protection). An arm that fails retention
  is invalid regardless of its RRMSE.
- **No covariance whitening anywhere** (F7, κ≈4600 pathology structurally deleted).
- **Closed forms first** (F5): any learned component (NPE heads, Kalman drift) enters
  only behind a preregistered beats-closed-form kill.
- **Oracle discipline**: oracles fit on disjoint segments, cross-refereed by held-out
  reference channels; degenerate-oracle numbers are never claimable (M0 lesson).
- **ITT**: excluded cells/subjects are counted and reported, never silently dropped.
- **Banking**: raw per-cell results committed before aggregates are viewed; defective
  runs banked unedited beside corrections; no frozen constant moves post-data.
- **Digest**: every verdict appends to `docs/EEG_denoise_arc_results_digest.md` in its
  verbatim-numbers style, same commit as the decision JSON.

## 5. Sealed discipline (D5) — executed at this commit

- **Frozen**: `results/iris/sealed/sealed_freeze.json` — EEGEyeNet antisaccade
  `synchronised_min` **listing positions 31–90** (deterministic Drive listing order,
  the continuation of the dev slice 1–30): 60 folders, **55 non-empty subjects**
  (AH0, AH6, AH9, AJ8, AJ9 empty upstream — recorded, counted, never replaced).
- Zero contact of any kind so far: these subjects were never downloaded, never read.
  Fetch goes to `/projects/EEG-foundation-model/eegeyenet/eegeyenet_sealed/` (Slurm),
  header-verified at acquisition (integrity, not analysis), then `chmod 000`.
- **Opening rule**: at most once, only with operator sign-off, single inference pass,
  outputs digest-frozen before evaluation, reported regardless of outcome (C-1
  protocol). No IRIS or incumbent asset may be tuned after sealed contact.
- **Cross-paradigm caveat (honest limitation)**: dots (EP*) and antisaccade (A*/B*)
  subject IDs are not mappable from released metadata, so a sealed antisaccade subject
  could in principle be a dev dots participant. Therefore: all sealed-fight support,
  prior, and atlas assets are built from antisaccade-dev + non-EEGEyeNet panels only;
  dots data never enters any asset that the sealed pass consumes.
- Remaining ~280 antisaccade folders (positions 91+) are DEV-CLASS if ever fetched
  (e.g. the 356-subject scale probe); they can never join the sealed block.
- Other sealed assets (BrainID Day-200, PhysioMotion-10, SHU Day-4/5) stay reserved for
  paper time; MobileBCI-8 is spent and closed. This campaign touches none of them.

## 6. Milestones and reporting

M-A charter+freeze+K-prereg (this commit block) · M-B K verdicts · M-C P1 pilot verdict
· M-D T + W verdicts · M-E F1 fight · M-F F2 fight · M-G UQ · M-H sealed (sign-off).
Each: decision JSON verbatim in the report, digest appended, commit+push. Between
milestones I run free.

## 7. Deviations from the method doc (argued, as the brief requires)

1. **Template-BEM physics (G_phys) deferred, not built now.** The hostile review already
   demoted it to a gated prior-mean hypothesis (orbital anatomy is where template physics
   is worst; the lid is a sliding shunt, not a dipole — method doc §three). The typed
   EMPIRICAL family (event-kernel drives + rank-constrained Δ_typed with EB atlas
   shrinkage) tests the same information question — is the typed rich reference new
   information? — without inheriting the canonical-object failure class (F7). BEM enters
   only if T shows typed headroom AND the empirical typed family underdelivers on it;
   then it faces the S0b gate (G(θ̂) ≥ 0.75× free-fit frontal artifact variance)
   exactly as specified. This violates no F-fact; it re-orders a component the method
   doc itself flagged as its weakest physical risk.
2. **Step-3 "≥100-subject large-grid" physics validation is impossible as written** —
   the dots release has 30 subjects total. Wherever the method doc assumed ≥100, the
   preregistration states the actual n and the power consequence honestly.
3. **K1 runs before P1's bar is frozen** (the method doc lists them as parallel).
   Reason: P1's reclamation bar should not be chosen blind to how much of the
   abstention mass is convertible in principle — freezing a 30% bar against a 20%
   convertible mass would manufacture a NO-GO. K1's output feeds P1's bar via a rule
   frozen HERE: P1 reclamation bar = min(0.30, 0.75 × f_conv). f_conv is measured by
   K1 before P1's prereg commit; the formula itself is frozen now and does not move.

## 8. What I am NOT doing

No manuscript text. No touching `taas_submission/**`. No verification ceremonies. No
sealed contact without sign-off. No re-litigation of closed routes (pooled canonical
prior, score-network personalization, RLS-replaces-calibration, operator-feature
ownership) — IRIS builds on their corpses, it does not exhume them.
