# V28 natural metric audit

The historical `preservation` scalar is correction-based low-EOG observation retention, not ERP, SSVEP, or physiological ground truth. V28 renames it `low_eog_observation_retention`, reports its complement as `low_eog_observation_change`, and sets ERP/SSVEP/ERD-ERS to `unavailable`. No scalar aliases are active.

The corrected V24/V25 arrays used by the active V28 evaluator are already in
the registered STANDARD preprocessing coordinates. They do not contain the
RAW-like waveform condition, so the no-denoising row is named `STANDARD`.
V22 RAW and STANDARD rows are retained only in the frozen historical comparator
table and are marked pre-coordinate-correction/not directly comparable. V28
does not relabel a STANDARD array as RAW-like.
