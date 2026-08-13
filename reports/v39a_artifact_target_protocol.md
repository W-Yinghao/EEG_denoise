# V39A artifact-target and calibration protocol

Paired targets are the corrected-coordinate injected artifact field `C_query Z_e`. Natural targets
are the registered synchronized-EOG/query-operator regression proxy and are explicitly marked
`teacher_proxy`, never clean physiological ground truth. Training generator banks contain outer-train
participants only.

Calibration context uses support EEG and EOG from the first native 30 seconds. Fifteen chronological,
non-overlapping 2-second windows begin at samples `0, 200, ..., 2800`; EOG center and scale use only
that 30-second prefix, while EEG uses training-fold population scale. No support sample is repeated,
no query sample enters normalization, and inference receives neither query EOG nor subject ID.
