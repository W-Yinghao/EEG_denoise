# FLAGSHIP-M35 report — the program's final experimental round

Preregistration: `reports/m35_preregistration.md` (frozen before submission; sealed
choices frozen before any sealed byte was read).

## C-1 SEALED CONFIRMATION — PASS

First and only opening of the MobileBCI sealed-8. Single inference pass, outputs
digest-frozen (sha256 857a6713a7a64534...), 96 sealed reads logged; the
evaluator crashed once post-freeze on a code bug and was repaired and run once against
the digest-verified frozen outputs (inference never re-ran).

- Precondition PASS: NO_A0 beats RAW by 0.0865; q99 = 0.977 in [0.90, 1.10].
- **PRIMARY: MATCH_gated − NO_A0 = +0.1537 [+0.0725, +0.2504], 7/8 positive**
  (development reference: +0.1428 [15/15]). The likelihood-leg subject-aware system
  gain REPLICATES on sealed subjects at development magnitude.
- Natural second row: G4 bar met by both arms (attenuation +0.19/+0.20 dB, retention
  0.92); MATCH−POP natural +0.009 dB (descriptive; natural arms carried a0 = 0 in this
  pass per the registered note).
- Condition means: MATCH_gated 0.535, NO_A0 0.689, POP 0.766, RAW 0.775.

## U-1 unification factorial — channels partially REDUNDANT

| panel | UF-1 (transport gain, anchor active) | UF-2 (anchor gain, transport active) | UF-3 (joint vs best single) | additivity |
|---|---|---|---|---|
| klados | **PASS** +0.0134 (Holm p 0.0096) | 0 exactly (anchor 100% abstained under the frozen gate) | FAIL (−0.0033) | 1.0 (degenerate) |
| bci2b | FAIL +0.0064 (p 0.125) | FAIL −0.0028 | FAIL (−0.0072) | 0.596 |

The two physics channels remove overlapping ocular content: composition is
sub-additive; on BCI2b the joint arm over-subtracts below the best single leg.

## C-2 BrainID fresh-dataset transport — preregistered NEGATIVE

TG-1 fails on both days at pathological magnitude (Day-7 −8.59 [−14.95, −4.41];
Day-80 −11.45): the 121-dim covariance-only whitening from ~10-min 57-ch supports is
numerically unstable (the W1 kappa failure mode on a fresh dataset). Wrong/gauge rows
confirm breakdown rather than subtle effects. Cross-day ratio 1.33 (both negative).
57/57 montage labels resolved; Day-200 never dereferenced.

## D-1 downstream kappa row (descriptive)

joint-cleaned 0.129 vs pop-cleaned 0.219 (delta −0.090 [−0.375, +0.171]) — consistent
with the U-1 redundancy/over-subtraction finding.

## P-1 transport-state privacy (descriptive)

                  top1_accuracy  same_different_auroc
panel  rho_label                                     
bci2b  0.00               0.111                 0.500
       0.25               0.222                 0.769
       0.50               0.333                 0.819
       0.75               0.444                 0.866
       1.00               0.556                 0.895
       EB                 0.556                 0.798
klados 0.00               0.019                 0.500
       0.25               0.537                 0.836
       0.50               0.759                 0.942
       0.75               0.907                 0.975
       1.00               0.926                 0.988
       EB                 0.407                 0.744

rho = 0 is exactly chance on both panels; linkage rises steeply with rho (klados
rho=1 AUROC 0.988). Same story as the lambda curve: abstention is the privacy
mechanism on both legs.

## Ownership

No verification attempts anywhere in this round (fourth replication of the
reliability-not-identity limitation documented; deployment answer = abstention +
certified floor).

## Compute ledger
U-1+P-1 CPU ~50 min; C-2+D-1 CPU ~65 min; sealed chain 1 GPU pass (~40 min A100)
+ evaluator-only rerun (~15 min). Program total across all arcs ≈ 60 GPU-h.
