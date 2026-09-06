# BCI2b base-bridge oracle audit V11

Development audit; both support and later-query deployment are EOG-guided.

Decision: `BASE_BRIDGE_VALID_J1_AUTHORIZED`.

FULL oracle RRMSE: 2.76919e-08; query-transfer oracle: 0.0000; EOG regression: 0.0000; support gamma zero fraction: 0.000.

Same-session and cross-session results are stored separately in `base_bridge_summary.csv`. Query-optimal gamma values are evaluator-only ceilings; primary operator comparisons use the support-derived gamma.

`QUERY-TRANSFER-ORACLE` and `EOG-REGRESSION` are evaluator-only query-fitted ceilings. Their near-zero error is expected because the frozen V10 paired artifact was generated with that query transfer; they validate paired construction and the evaluator bridge, not deployable denoising. Deployable evidence starts with support/population/wrong operators and the later V11 model gates.
