# CGDR Klados v4 source fold

Evidence status: `exploratory_pre_repair_not_gate_evidence`.

This run is a complete real EEG/EOG-backed paired semi-simulation source-record computation. It uses all non-overlapping query samples after the frozen 30 s calibration and 1 s guard, but its inference semantics are invalid as described in `INVALID_SEMANTICS.md`. The v4 release does not expose a reliable 54-to-27 participant map. Formal G1 and G2 were not run and remain blocked. This directory cannot pass or fail a formal gate and must not supply manuscript evidence. See `result_summary.json` and `metrics.csv` only for retained debugging provenance.
