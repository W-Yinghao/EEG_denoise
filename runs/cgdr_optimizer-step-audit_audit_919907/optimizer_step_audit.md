# Frozen Stage-3 optimizer-step audit

This retrospective audit only reads three deterministic `best.pt` and three conditional `final.pt` checkpoints. It does not read EEG or resume training.

Status: `failed_optimizer_step_contract`. Expected AdamW step: `6000`.

| Family | Operator scope | Global step | Adam states | Min | Max | Unique | Scaler | Status |
|---|---|---:|---:|---:|---:|---|---|---|
| deterministic_best | population_projector | 6000 | 178 | 6000 | 6000 | [6000] | _growth_tracker, backoff_factor, growth_factor, growth_interval, scale | passed |
| deterministic_best | matching_p0 | 6000 | 178 | 6000 | 6000 | [6000] | _growth_tracker, backoff_factor, growth_factor, growth_interval, scale | passed |
| deterministic_best | query_derived_oracle_projector | 6000 | 178 | 6000 | 6000 | [6000] | _growth_tracker, backoff_factor, growth_factor, growth_interval, scale | passed |
| conditional_final | population_projector | 6000 | 178 | 5999 | 5999 | [5999] | _growth_tracker, backoff_factor, growth_factor, growth_interval, scale | failed |

- `conditional_final/population_projector`: AdamW parameter steps are not all equal to 6000: observed [5999]
| conditional_final | matching_p0 | 6000 | 178 | 5999 | 5999 | [5999] | _growth_tracker, backoff_factor, growth_factor, growth_interval, scale | failed |

- `conditional_final/matching_p0`: AdamW parameter steps are not all equal to 6000: observed [5999]
| conditional_final | query_derived_oracle_projector | 6000 | 178 | 6000 | 6000 | [6000] | _growth_tracker, backoff_factor, growth_factor, growth_interval, scale | passed |

Checked `6` of exactly `6` configured checkpoints.
