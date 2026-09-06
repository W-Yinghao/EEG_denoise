# V26 Round B

Round B completed 5 participant folds × 3 seeds for four model families: CalibRefineDET, PopRefineDET, CalibSDEdit, and PopSDEdit. All 60 model cells were retained. The operating point was frozen from validation before test evaluation: `sigma_start=0.05`, 10 DDIM steps, K=1, natural fraction 0.30.

Paired clean RRMSE was 0.70618 for CalibRefineDET-MATCH and 0.70975 for CalibSDEdit-MATCH. The SDEdit-minus-one-step utility was -0.003575 [95% bootstrap CI -0.005380, -0.001642], 2/15 positive. This establishes the one-step model as the stronger competitive point estimator, not an automatic rule for deleting diffusion.

Mechanistically, both second-stage models used support: one-step MATCH–WRONG utility was +0.011046 (10/15), and SDEdit MATCH–WRONG utility was +0.009850 (12/15). SDEdit support versus its independent PopSDEdit control was +0.009143 (10/15).

All statistics are participant-first development summaries. No test-fold result selected the operating point.
