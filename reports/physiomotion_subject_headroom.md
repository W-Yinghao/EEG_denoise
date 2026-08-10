# PhysioMotion subject clean-patch retrieval headroom

Decision: `PHYSIOMOTION_SUBJECT_RETRIEVAL_HEADROOM_NO_GO`. Scientific denominator: 20 development participants; retrieval available for 17, with 3 blocked participants retained as zero-effect ITT units.

Retrieval uses fixed K=8, two-second clean patches, z-normalized correlation on unmasked context, equal outer-participant bank quotas, and only outer-training artifact annotations for masks.

- H_P: mean -0.00142, median -0.00316, 6/20, one-sided exact p=0.537369.
- H_W: mean +0.02403, median +0.01166, 12/20, one-sided exact p=0.027908.

Participant-first restoration metrics among the 17 evaluable participants:
- MATCH: RRMSE 1.06858, correlation 0.09010, spectral error 0.44190, topography error 0.13588.
- POP: RRMSE 1.06692, correlation 0.21456, spectral error 0.43808, topography error 0.13357.
- mean-WRONG: RRMSE 1.09685, correlation 0.07375, spectral error 0.45763, topography error 0.14415.

Jointly positive primary artifact-family strata: 2/5.

If the frozen headroom gate fails, no diffusion or deterministic GPU model is trained. Such a failure closes only this fixed retrieval representation on the development split.
