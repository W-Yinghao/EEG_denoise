# V26 V25 diffusion forensic

The frozen V25 learned-basis representation is not a canonical generative coordinate system. In the exact rotation fixture, an equivalent orthogonal rotation changed the latent target by relative norm `1.606667` while preserving the decoded 46-channel artifact to `1.33e-15` maximum error and its projector to `1.31e-15`.

Across 15 development participants, three deterministic support-subset resamples per test fold produced mean learned-basis projector distance `0.680930` (median `0.557249`). The mean Procrustes rotation magnitude was `0.104841`. Greedy column order and sign did not flip in these sampled cells, but the subspace and coefficient estimates varied; Procrustes alignment did not remove the temporal estimate discrepancy. Thus V25's failure remains frozen, while V26's sensor-coordinate artifact target removes the algebraic ambiguity from the active diffusion target.

This forensic diagnosis is mechanistic context, not a new scientific gate. It did not open sealed data or alter V25 artifacts.
