# V25 Natural Development

Inference used query EEG plus query-disjoint S120 EEG+EOG support only. V24's frozen auxiliary-free query bundle and a separately materialized S120 support bank were used. The 15 inference outputs were digest-frozen before the evaluator opened V24 query EOG-derived targets. Inference read counts were query EOG 0, query operator 0, event 0, sealed 0.

```json
{
  "DIFF_MATCH_POP_artifact": {
    "mean": -0.08012665056313688,
    "median": -0.06408361144309493,
    "positive": 1,
    "participants": 15,
    "bootstrap_low": -0.11942708101783009,
    "bootstrap_high": -0.043084296640780166
  },
  "DIFF_MATCH_POP_preservation": {
    "mean": -0.12929952311099308,
    "median": -0.13477503400279844,
    "positive": 0,
    "participants": 15,
    "bootstrap_low": -0.14349867851930448,
    "bootstrap_high": -0.11491412305752197
  }
}
```

Both dimensions oppose advancement: the model neither improves the artifact endpoint nor preserves low-artifact EEG relative to the strong population anchor. This natural result is evaluator-based development evidence without a clean counterfactual target.
