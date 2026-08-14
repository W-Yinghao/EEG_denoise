# FLAGSHIP-M35 preregistration

```text
FLAGSHIP-M35 preregistration — frozen before first submission.

U-1 UNIFICATION FACTORIAL (Klados + BCI2b, analytic backbone, frozen R-B transport
  configs, ITT abstention):
  arms = {T-POP, T-MATCH(ρ̂)} × {A0-none, A0-POP (C0·e), A0-MATCH (C_gated·e)}, common
  windows; WRONG rows for both channels descriptive only.
  Endpoints (participant/record-first, 5000-draw bootstrap, Holm over the three tests):
  UF-1 transport gain with the anchor active: [T-POP,A0-MATCH] − [T-MATCH,A0-MATCH] > 0
  UF-2 anchor gain with transport active:    [T-MATCH,A0-POP] − [T-MATCH,A0-MATCH] > 0
  UF-3 composition: JOINT [T-MATCH,A0-MATCH] vs best single leg (report the additivity
       index (joint − base) / ((legA − base) + (legB − base)) with CI; no super-additivity
       claim — the estimand is whether the channels are complementary or redundant).

C-1 SEALED CONFIRMATION — MobileBCI sealed-8, opened ONCE (first legitimate opening in
  program history; log everything; report regardless of outcome):
  model     = the frozen V44-S1 checkpoints, 5-fold ensemble mean (seed 20261201),
              chosen and frozen HERE, before any sealed byte is read
  support   = each sealed subject's own S120 prefix (V31 contract, 120-s gated states)
  PRIMARY   = MATCH_gated − NO_A0 paired temporal-RRMSE utility on the sealed
              semi-simulated construction (identical V19-style recipe, built fresh for
              sealed subjects), mean/CI/positive count over n=8
  PRECONDITION (reported first): backbone sanity on sealed subjects — NO_A0 beats RAW
              and q99 ∈ [0.90, 1.10]; if the precondition fails, the primary is reported
              as non-adjudicable, not suppressed
  SECOND ROW = natural G4 validity (attenuation > 0, retention ≥ 0.75) + natural
              MATCH_gated − POP descriptive
  NO other endpoints. Outputs digest-frozen before the evaluator opens query EOG.

C-2 FRESH-DATASET TRANSPORT + CROSS-DAY (BrainID, Day-200 stays sealed):
  transport config (frozen): covariance alignment ONLY (G_s with the split-half
  abstention rule; Q fixed at the population base — no per-subject ocular Procrustes,
  BrainID has no dedicated EOG channels), 57-ch montage lift shared.
  support = Day-1; queries = Day-7 (primary), Day-80 (secondary/stress).
  Estimand = TG-1 (T-POP − T-MATCH > 0, analytic backbone, paired semi-sim construction
  from BrainID's own blink-rich segments); controls = WRONG donor + GAUGE-NULL; cross-day
  stability = Day-7 vs Day-80 gain ratio (descriptive). This doubles as the
  acquisition-shortcut test: a transport that survives a cap remount across 80 days is
  carrying anatomy, not acquisition.

D-1 DOWNSTREAM ROW (BCI2b): MI kappa for [T-MATCH,A0-MATCH]-cleaned vs [T-POP,A0-POP]-
  cleaned natural trials via the existing CSP-LDA/kappa harness; descriptive with CI.

P-1 TRANSPORT-STATE PRIVACY (CPU): linkage (top-1, verification AUROC) of stored
  transport states (Σ̂_s summaries + frame parameters) at ρ ∈ {0, 0.25, 0.5, 0.75, ρ̂, 1},
  Klados + BCI2b; completes the two-leg privacy story alongside the λ curve.

OWNERSHIP: no verification attempts in any leg (fourth replication documented; the
  deployment answer is abstention + the certified floor). No threshold changes after
  this commit. BrainID Day-200, PhysioMotion-10, SHU Day-4/5 remain sealed.
```

## Registered C-1 implementation choices (frozen HERE, before any sealed byte is read)

1. Ensemble: output-space mean of the 5 V44-S1 fold models (seed 20261201), each fed the
   same inputs; DDIM noise shared across folds per episode (seed rule below).
2. Population/gate objects for sealed subjects: computed from the FULL 16-subject
   development cohort (population transfer/quality per (session, task) = mean over dev
   owners' 30-s prefix transfers; tau2, within-threshold (95th pct), and the quality
   clamp from the dev cohort; the frozen EB λ rule and 60-s hard gate unchanged).
   Signatures normalized with the dev-cohort continuous center/scale.
3. Sealed prepared records: built fresh from the MobileBCI BIDS source for the 8 sealed
   subjects by replicating the frozen V19 preparation pipeline exactly (code retrieved
   from the frozen protocol history); only sealed subjects' own recordings are read, at
   C-1 time, in the single isolated job.
4. Paired construction: TransferEpisodeSampler semantics with recipient = sealed subject;
   clean carriers and EOG donors drawn from DEV participants only; sampler seed 20269001;
   8 episodes per available (session, task) cell; gains per the V42R distribution.
5. Noise seeds: paired DDIM 421000 + sealed_index (order sub-01,04,08,10,13,16,20,22);
   natural 611000 + sealed_index; 4 natural windows per cell from qnatural onward.
6. Digest freeze: sha256 of the outputs npz recorded in the manifest before the
   evaluator opens sealed query EOG (the registered exception to the no-ceremonies rule).
7. C-2 blink-drive convention (no EOG channels): pseudo-drive = first two principal
   components of the two most frontal channels' band-passed (0.5-8 Hz) signal over
   blink-rich segments (top-decile frontal RMS), standardized on Day-1 support; the
   per-subject blink operator is the ridge regression of all channels on the pseudo-drive
   over Day-1 blink-rich segments (V41R ridge ratio 0.05).
```
