# Corrected CGDR Klados v4 source fold

Evidence status: `exploratory_pre_repair_not_gate_evidence`.

This run is a complete real EEG/EOG-backed paired semi-simulation source-record computation. It uses all non-overlapping query samples after the frozen 30 s calibration and 1 s guard. The v4 release does not expose a reliable 54-to-27 participant map, so no leakage-safe Klados population operator or participant-level inference can be formed. The oracle-restoration direction check on this one source failed, but that is only a `single_source_exploratory_check_failed` diagnosis. Formal G1 was **NOT RUN/BLOCKED**; formal G2 was also **NOT RUN/BLOCKED**. This result cannot enter a formal gate or manuscript table. Slurm 919385 remains invalid inference evidence, while its independently trained clean-prior checkpoint was reused here. See `result_summary.json` and `metrics.csv` for debugging only.
