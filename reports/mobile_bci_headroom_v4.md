# MobileBCI participant headroom v4

This is a development-only headroom screen. The eight sealed participants were never opened.

## Coverage

- Development participants: 16; sealed participants: 8.
- Science-ready records: 91/96.
- Eligible protocol units: 150/151.
- Headroom inference: all 16 development participants in four participant-held-out folds.

## Primary participant-level effects

Positive values favor matching temporal support. Intervals are 20,000-draw participant bootstraps.

| Protocol | MATCH−NULL | MATCH−STRONG-POP | MATCH−mean WRONG | MATCH−temporal-shuffled |
|---|---:|---:|---:|---:|
| S0_STATIC_XSESSION | -0.00230 [-0.00478, +0.00011] | -0.01439 [-0.02025, -0.00912] | +0.00052 [-0.00197, +0.00279] | +0.00069 [-0.00206, +0.00337] |
| S1_MOTION_WITHIN_SESSION | -0.00251 [-0.00570, +0.00090] | -0.01547 [-0.02238, -0.00965] | +0.00049 [-0.00076, +0.00174] | -0.00005 [-0.00137, +0.00117] |
| S2_MOTION_XSPEED | -0.00192 [-0.00557, +0.00161] | -0.01344 [-0.02048, -0.00560] | +0.00057 [-0.00070, +0.00188] | +0.00066 [-0.00074, +0.00197] |

## Decision

Routing decision: `SUBJECT_HEADROOM_NO_GO`.

None of S0/S1/S2 established matching-support utility over either no-support or STRONG-POP. Matching-versus-wrong effects were small and their intervals crossed zero. S1 and S2 met the mean safety margins; S0 missed the ERP margin slightly. Therefore the pre-authorized temporal diffusion screen was not run.

Independent P-C constrained-oracle decision: `GO_MINIMAL_SELECTOR`; minimal selector: `completed_single_minimal_deployable_selector`.

This result constrains only the fixed temporal deterministic probe and these frozen MobileBCI support protocols. It is not a family-wide verdict on temporal support, personalization, or diffusion.
