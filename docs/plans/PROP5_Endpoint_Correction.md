# Proposition 5′ — Corrected nested-POP endpoint for the CHART transport family
### Math note (paper-appendix seed + implementation contract for Stage 0)

Status: corrects the gauge bookkeeping error found by the design review in Prop. 5 of the
CHART design (`EEG_denoise_design_panel/broad/design-geometry.md`). Must be adopted verbatim
in the Stage-0 preregistration; the ρ = 0 bit-identity unit test below is mandatory.

---

## 1. Setup (unchanged)

Per subject s with montage M_s (C_s electrodes), the transport into the canonical
K-dimensional head space (K = 121 SH coefficients) is

    T_s = Q_s · G_s · L_s

- L_s ∈ R^{K×C_s}: montage lift (regularized SH least squares from electrode positions);
  full column rank, L_s^+ L_s = I_{C_s} exactly. Hardware descriptor, zero subject content.
- G_s = Σ̄^{1/2} Σ̂_s^{−1/2} ∈ R^{K×K}: covariance alignment (subject-estimated
  canonical covariance Σ̂_s, population AIRM Fréchet mean Σ̄).
- Q_s ∈ SO(K): minimal rotation carrying the subject's whitened lifted ocular frame
  U_s = orth(G_s L_s A_s) onto the fixed canonical frame U° (r = 3 columns, physically
  pinned order VEOG/HEOG/blink).

## 2. The bug

Prop. 5 as originally written shrinks each factor along a geodesic anchored at the
IDENTITY:

    Σ_s(ρ) = Σ̄^{1/2} (Σ̄^{−1/2} Σ̂_s Σ̄^{−1/2})^ρ Σ̄^{1/2},    Q_s(ρ) = exp(ρ log Q_s)

giving T_s(0) = I · I · L_s = L_s. But the declared strong-POP arm — the comparator the
whole MATCH−POP estimand is defined against — is

    T_pop^{(s)} = Q̄_s · I · L_s ,

which already receives the population ocular frame. L_s ≠ Q̄_s L_s in general, so the
"exact nested POP endpoint" claim fails: at ρ → 0 the shrunk method does NOT converge to
the POP arm, mis-adaptation harm is no longer structurally bounded, and the fail-closed
property (the V43-style bit-identity) breaks. The covariance factor is NOT affected:
Σ_s(0) = Σ̄ ⟹ G(0) = Σ̄^{1/2} Σ̄^{−1/2} = I, which is the correct population behavior
(the prior already lives at Σ̄; the POP arm performs no covariance alignment). The bug is
purely in the rotation factor.

## 3. The population base rotation Q̄_s (definition)

Let Ā_{M_s} be the population-mean ocular mixing topography expressed on montage M_s
(estimated once on the training cohort, per montage cell; zero subject content). Define
the population lifted frame and base rotation, both with G = I (consistent with ρ = 0):

    Ū_s = orth(L_s Ā_{M_s}) ∈ R^{K×r},
    Q̄_s = minimal rotation carrying Ū_s → U°   (same minimal-rotation formula as Q_s,
           identity on (span Ū_s + span U°)^⊥).

Q̄_s depends only on cap geometry and population topography — it is a per-montage object
shared by every subject wearing the same cap, exactly like the montage mask.

## 4. Proposition 5′ (corrected family)

Define, for ρ ∈ [0,1]:

    Σ_s(ρ) = Σ̄^{1/2} (Σ̄^{−1/2} Σ̂_s Σ̄^{−1/2})^ρ Σ̄^{1/2}          (AIRM geodesic, unchanged)
    G_s(ρ) = Σ̄^{1/2} Σ_s(ρ)^{−1/2}
    Q_s(ρ) = exp( ρ · log( Q_s Q̄_s^⊤ ) ) · Q̄_s                    (SO(K) geodesic FROM Q̄_s TO Q_s)
    T_s(ρ) = Q_s(ρ) · G_s(ρ) · L_s

Then:

(a) **Endpoints.** T_s(0) = Q̄_s · I · L_s = T_pop^{(s)} exactly, and T_s(1) = Q_s G_s L_s = T_s.
    Proof: Q_s(0) = exp(0) Q̄_s = Q̄_s; Q_s(1) = exp(log(Q_s Q̄_s^⊤)) Q̄_s = Q_s Q̄_s^⊤ Q̄_s = Q_s;
    G_s(0) = I, G_s(1) = G_s as above. ∎
    MATCH−POP is therefore a genuinely nested comparison and ρ → 0 degenerates
    bit-identically to the population sampler (fail-closed as a theorem, restored).

(b) **Round-trip identity for all ρ (Lemma 4 preserved).** With the arm-consistent
    pseudo-inverse T_s(ρ)^+ := L_s^+ G_s(ρ)^{−1} Q_s(ρ)^⊤,

        T_s(ρ)^+ T_s(ρ) = L_s^+ G(ρ)^{−1} Q(ρ)^⊤ Q(ρ) G(ρ) L_s = L_s^+ L_s = I_{C_s}   ∀ρ.

    The no-correction path (ẑ = P(ρ) ỹ) returns x̂ = y bit-exactly at every ρ; the
    amplitude-safety bound uses κ(T_s(ρ)), which interpolates between κ(Q̄_s L_s) and κ(T_s).

(c) **Locality (Lemma 3, weakened by one frame).** log(Q_s Q̄_s^⊤) is supported on
    W = span(U_s) + span(U°) + span(Ū_s), dim W ≤ 3r = 9: both rotations act as the
    identity on W^⊥, hence so does their product and (on the principal branch) its
    logarithm. The corrected statement: the ocular canonicalization perturbs at most
    **9 of 121** canonical coordinates (was 6). All downstream uses of Lemma 3 must cite 9.

(d) **Well-definedness / angle cap.** The principal logarithm log(Q_s Q̄_s^⊤) is defined
    iff no principal angle equals π. Preregistered cap: if the largest principal angle of
    Q_s Q̄_s^⊤ exceeds π/2, set ρ_s := 0 (ABSTAIN, reported) — this folds into the existing
    abstention rule and prevents geodesic wraparound pathologies.

## 5. Consequential contract changes (all arms, frozen before Stage 0)

1. **POP arm** := T(0) = Q̄_s L_s, with its likelihood projector built from the SAME T(0).
   Every arm uses its own arm-consistent (T(ρ), T(ρ)^+, P(ρ)); the canonical artifact
   frame U° in the likelihood is fixed and arm-independent.
2. **ρ = 0 bit-identity unit test (mandatory)**: the implementation must produce, at
   ρ = 0, arrays bit-identical to the POP arm's transport, projector, and outputs —
   including that no ρ-modulating feature leaks into the ρ = 0 path. This is the direct
   analogue of V43's λ = 0 clamp contract, which S1 verified at 465/465 cells.
3. **WRONG arm**: donor's (Σ̂_w, U_w) substituted into the recipient's construction
   (same L_s, same Q̄_s base, geodesic from Q̄_s toward the donor's Q_w), gated by the
   donor's own ρ̂_w.
4. **ORACLE-T arm**: transport estimated from the query window itself; evaluator-only,
   opened after output freeze.
5. **GAUGE-NULL control (new, recommended)**: a random rotation supported on W with the
   same principal-angle spectrum as Q_s Q̄_s^⊤ — tests that any gain comes from CORRECT
   alignment rather than from perturbing the ocular subspace per se.
6. **ρ_s EB rule, rotation factor**: between-subject variance τ² = cohort Fréchet variance
   of the geodesic distances d(Q_i, Q̄) = ‖log(Q_i Q̄^⊤)‖_F / √2; within-subject v_s from
   the split-half distance d(Q_s^{(1)}, Q_s^{(2)}). Closed form, no learned predictor (N7).

## 6. What to verify in Stage 0 (numeric gates, fail-closed)

```text
round-trip:        max_ρ∈{0,0.5,1} ‖T(ρ)^+ T(ρ) − I‖_max ≤ 1e-10
ρ=0 bit-identity:  array-equal to the POP arm (transport, projector, outputs)
locality:          ‖(Q(ρ) − I) restricted to W^⊥‖_max ≤ 1e-10
frame concentration: post-Q principal angle(Ã_s , U°) ≤ 15°  (per subject, reported)
angle cap:         principal angles of Q_s Q̄_s^⊤ reported; abstentions counted
conditioning:      κ(T_s(ρ)) reported per subject/montage; cap per prereg
```
