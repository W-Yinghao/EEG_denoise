# Conditional diffusion v3 optimizer-step repair

Slurm job 919907 found that the historical conditional v2 population and
matching checkpoints contained 5999 AdamW updates even though the old loop
cursor reported 6000. The query-derived-oracle conditional checkpoint and all
three deterministic v4 checkpoints contained 6000 actual updates. This was an
AMP-overflow accounting defect, not an outcome-based model selection result.

The v2 checkpoints and record-level outputs remain unchanged under
`results/cgdr/klados_stage3_conditional_diffusion_matched_v2/` and are excluded
from the incremental-value decision. Protocol v3 uses a new result root and
counts an update only when `GradScaler` executes `optimizer.step`. It retrains
all three conditional operator scopes from scratch to 6000 successful updates.
The AMP initial scale is frozen at 1024 and any skipped optimizer step fails the
run, so the replacement cannot silently gain extra batch exposure.

No scientific threshold, source-record split, visible input, objective,
architecture, sampler, seed set, or deterministic comparator changed. Klados
remains an exploratory source-record diagnostic and cannot establish formal G1
or G3 or a diffusion-family conclusion by itself.
