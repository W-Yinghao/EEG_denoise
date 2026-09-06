# CalibEnergy-SDEdit method specification

For contaminated EEG `Y` and query-disjoint raw EEG+EOG support `S`, the frozen V25 model produces
a support-conditioned deterministic artifact estimate `A_det`, a population estimate `A_pop`, and
a support context/projector. V26 performs 10-step, sensor-coordinate artifact SDEdit from a
deterministic warm start at `sigma_start=0.05`, conditioned on `Y`, `A_det`, `A_pop`, their clean
estimates/increment, and the support context.

V27 then applies a single closed-form partial-observation proximal map with `lambda_a=1` and
`lambda_y=0.5`. The support projector releases more correction inside the learned artifact span;
an EEG-only, training-fold-calibrated temporal mask reduces correction at low artifact confidence.
The cleaned signal is `Y-A_energy`. The energy is final-only and the registered point estimate uses
one stochastic draw. No query EOG, query operator, event, subject identity token, or test-time model
update enters inference.

The matched EnergyDET route receives the same energy and is a competitive mechanism control. The
method is an attenuation-oriented operating point, not a universally safe physiological restorer.

