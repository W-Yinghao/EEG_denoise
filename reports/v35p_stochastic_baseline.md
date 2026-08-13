# V35P Fiber-Stratified-Resample

Fiber-Stratified-Resample is a non-neural empirical population channel. It stores
only the six outer-training participants' Session-T fibers and training-derived
centered-logit strata. A query supplies centered logits only. The channel samples
from the matching predicted-class and training confidence-tertile stratum, then
combines that donor fiber with the query head component.

Across six fold/seed cells and train/gallery/query releases, all `20,736` queries had
an exact populated stratum. Class and global fallback counts were both zero. Donor
indices always referred to the outer-training Session-T pool. No query fiber or query
subject was used to select a donor.

All resampled outputs preserved the frozen task function: zero prediction mismatches,
zero fixed-head BA difference, maximum centered-logit/H-recovery error `2.625e-7`,
and maximum softmax error `7.702e-8` across the complete 24-row strong-output audit.
