# FLAGSHIP-M0 preregistration

```text
FLAGSHIP-M0 preregistration — frozen before first submission.

U0-a TRANSPORT GEOMETRY GATES (per docs/prop5_endpoint_correction.md §6, fail-closed):
  round-trip max_ρ ‖T(ρ)+T(ρ) − I‖max ≤ 1e-10 at ρ ∈ {0, 0.5, 1}
  ρ=0 bit-identity to the POP arm (transport, projector, outputs; array-equal)
  locality: Q(ρ) is the identity on W⊥ to 1e-10 (dim W ≤ 9)
  ocular-frame concentration: post-Q principal angle(Ã_s, U°) ≤ 15° for ≥ 80% of subjects
  within-subject split-half transport distance < between-subject cohort distance (median)
  κ(T_s(ρ)) reported; subjects with κ > the frozen cap flagged, not silently dropped

U0-b OPERATOR-POSTERIOR COVERAGE GATE:
  EB posterior over 46x2 operators (mean = the V43 gated operator; covariance from the
  hierarchical EB model). Held-out support-block coverage at 80% nominal must fall in
  [0.70, 0.90] per fold (participant-first).

U1 CEILING GO RULE (per channel, per preregistered stratum):
  GO iff oracle-ceiling mean ≥ +0.020 AND bootstrap CI-low > +0.005.
  Preregistered strata: {all windows} x {high-severity tercile} x {high-EOG windows},
  on each of the three panels (MobileBCI 46-ch, Klados v4 19-ch, BCI2b 3-ch).
  The conditioning channel's banked value (+0.006 [−0.0016, +0.0197], V42R) enters the
  matrix as-is and is not re-measured.

K1: if no channel GOes anywhere → gain thesis dead; write the decision and STOP.
Statistics: participant-first (or record-first for Klados), 5000-draw bootstrap,
no post-hoc strata, no threshold changes after this commit.
```
