# V23 Round B

All five development folds and three training seeds completed for OF-DET, OF-SCAD-K1, POP-MARGINAL-DET and POP-MARGINAL-SCAD-K1. Seeds were averaged within participant before the canonical n=15 summary.

- Paired clean RRMSE: OF-DET-MATCH 1.140326; POP-MARGINAL-DET 1.047880; OF-SCAD-K1-MATCH 1.143630; POP-MARGINAL-SCAD-K1 1.051032.
- Artifact-field RRMSE: OF-DET-MATCH 1.586844; POP-MARGINAL-DET 0.859185; OF-SCAD-K1-MATCH 1.759248; POP-MARGINAL-SCAD-K1 0.903636.
- Canonical effects (positive favors MATCH/diffusion): OF_DET_MATCH_POP_SWAP=+0.057644, OF_DET_MATCH_WRONG_SWAP=+0.008791, OF_DET_SUBJECT=-0.092447, OF_SCAD_MATCH_POP_SWAP=+0.167918, OF_SCAD_MATCH_WRONG_SWAP=+0.045371, OF_SCAD_SUBJECT=-0.092598, DIFF_K1_vs_DET1=-0.003304.
- K1 stability: seed means {'20260808': 0.009490572873668668, '20260810': -0.006744887627094084, '20260811': -0.012657201619650726}; K8/DET8 was not authorized because only one of three seeds favored diffusion.

Model exposure, checkpoint criteria, parameters, wall time, updates and hashes are recorded in `training_exposure.csv` and `checkpoint_manifest.csv`.
