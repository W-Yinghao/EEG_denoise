# Conditional v2 optimizer-step audit

The original v2 artifacts are retained as history, but they are not eligible
for the matched incremental-value decision. Slurm job 919907 inspected the
checkpointed AdamW state for all six learned endpoints. All three deterministic
v4 scopes and the conditional query-derived-oracle scope reached 6000 actual
optimizer updates. Conditional population and matching scopes each reached
5999 while their legacy loop cursor reported 6000 because one AMP-overflow skip
was counted as an update.

No checkpoint was edited or resumed backward. The replacement v3 protocol
uses a new result root and increments its update cursor only after a real
optimizer step. Thresholds, source-record splits, input fields, objectives,
architecture, random seeds, and the fixed 6000-update endpoint are otherwise
unchanged.
