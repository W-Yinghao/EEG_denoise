# Corrected neural geometry and oracle actionability

The historical covariance label is corrected to **robust spatial-correlation geometry** because each channel is median/MAD standardized. Support and later-query evaluator windows both use the frozen lowest-30% EOG-energy rule. AIRM populations use a true affine-invariant Karcher barycenter; log-Euclidean populations use a log-domain barycenter. Historical outputs remain unchanged.

The scientific unit is the participant (n=9), after session aggregation; support and later-query raw time blocks are disjoint.

- H_P_airm: mean +0.224603, median +0.134036, 8/9, one-sided exact p=0.005859.
- H_W_airm: mean +0.377347, median +0.321536, 9/9, one-sided exact p=0.001953.
- H_P_logeuclidean: mean +0.227330, median +0.135691, 8/9, one-sided exact p=0.005859.
- H_W_logeuclidean: mean +0.378205, median +0.326009, 9/9, one-sided exact p=0.001953.

Oracle route: `COVARIANCE_ACTIONABILITY_ROUTE_CLOSED`. The query-clean covariance context is evaluator-only and is not deployable.
- U_O: mean -0.000438, median -0.000205, 3/9.
- U_OW: mean +0.000691, median +0.000898, 6/9.
The corrected geometry is subject-identifiable, but its oracle clean-target actionability gate fails; no whitening/alignment model was trained.
