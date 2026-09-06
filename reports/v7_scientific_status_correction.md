# v7 scientific-status correction

The v6/v7 historical files remain unchanged.  Their corrected interpretation for this development round is:

`CURRENT_V6_BACKBONE_AND_K8_ESTIMATOR_NOT_VALIDATED / CAUSE_NOT_FULLY_IDENTIFIED`

- The analytic oracle-v DDIM roundtrip passed (relative error approximately `7.4e-8`).
- The tested weighted-v, unweighted-v, and epsilon end-to-end estimators did not pass the frozen real-batch reconstruction criteria.
- Those observations do **not** isolate a mathematical error in the diffusion objective; they leave backbone optimization, finite-data behavior, conditioning, and estimator variance unresolved.
- The improvement observed at `K=32` diagnoses substantial posterior sampling variance.  It does not replace the frozen `K=8` primary estimator.
- Seeds `20260807` and `20260808` are not added to v6.  No further v6 sampler, K, loss-weight, or objective repair is attempted.
- The existing `eb_score_adapter.py` is not trained on the unvalidated v6 dynamic-transfer backbone.

This correction narrows the claim about one implementation.  It is not evidence that diffusion objectives, diffusion denoising, or subject adaptation fail as method families.
