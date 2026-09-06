# V39A project plan

V39A models ocular artifact variability for augmentation; it is neither a query-time diffusion
cleaner nor a representation privacy method. It reuses the corrected V24/V25 waveform generator,
five participant folds, frozen V25 DeepSets support encoder, and a V31-compliant 30-second
chronological non-overlap support prefix. Four artifact generators receive the same training rows and
conditioning. Five augmentation arms train the same deterministic support-conditioned U-Net with
identical initialization protocol, updates, carrier count, severity distribution, and eight
corruptions per carrier.

Paired known artifacts and natural synchronized-EOG proxy artifacts remain separately labeled and
reported. Test query EOG is unavailable to inference and opened only by the evaluator after output
generation. No confirmation or sealed cohort is opened.
