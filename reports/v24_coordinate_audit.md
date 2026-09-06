# V24 operator/EOG/EEG coordinate audit

Verdict: `V23_COORDINATE_MISMATCH_CONFIRMED`.

The V19 producer stores source EOG in microvolts after common preprocessing (winsorization, 0.5–15 Hz filtering and resampling), without amplitude standardization. V23 then formed `D_y^-1 C_raw D_e`, multiplied it by centered physical EOG without `D_e^-1`, and divided by `D_y` a second time.

The mathematically equivalent raw and canonical routes agreed over 9000 real windows from 90 participant/session/task cells; maximum relative difference `2.571e-16`.
The V23 committed route differed from the correct raw route with median relative difference `1.173398` (range `0.096218`–`10.897984`).

No V23 file was changed. V24 will use corrected assets and will not use V23 coefficient statistics or checkpoints as scientific initialization.

Sealed reads: `0`. GPU jobs before this verdict: `0`.
