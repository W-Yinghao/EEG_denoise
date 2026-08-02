# Frozen diffusion incremental-value decision protocol

This protocol was written before reading any fixed-endpoint Klados v4/v3 or
full EEGDfus evaluation output.  It does not alter the retained conclusion
`current_M2_no_incremental_value` and cannot turn the SGEYESUB deterministic
audit into diffusion evidence.

The original conditional v2 endpoints were subsequently excluded by the
pre-decision optimizer-state audit: two scopes contained 5999 rather than 6000
actual AdamW updates. Conditional v3 changes only truthful optimizer-step
accounting and the result root; the decision thresholds below remain frozen.

The Klados comparison uses source records, not verified independent
participants, and only the already designated development records.  It is
therefore exploratory.  A conditional-vs-deterministic effect is called stable
there only when at least 75% of registered records are paired, at least 75% have
lower e_parallel, median e_perp harm is at most 0.05, median RRMSE decreases,
median correlation increases, and the failure rate is at most 10%.  Seeds are
averaged within records and never counted as independent evidence.

The EEGDfus strict source-epoch comparison is evaluated separately for EOG and
EMG.  Each must contain all 11 frozen SNR levels, use identical source manifests
and optimizer-update budgets across arms, and favor conditional diffusion on
the mean of SNR improvement, correlation, and temporal RRMSE.  Conditional
diffusion must win at least 8/11 SNR levels on each of those three metrics, and
the explicitly corrected spectral RRMSE mean may not worsen.  The broken
upstream official spectral field remains empty and blocked; it is never silently
replaced.

`conditional_diffusion_supported` is allowed only when both strict EEGDfus
artifact cells meet that rule.  Its scope remains paired single-channel
EOG/EMG stress testing, with Klados providing at most exploratory source-record
support.  `diffusion_no_detectable_incremental_value_under_tested_protocols`
requires every frozen matrix to complete and requires M1, M4, the matched
operator-conditioned DDIM100 model, and EEGDfus conditional diffusion all to
miss their matched stability rules.  The phrase remains limited to the tested
datasets, splits, tasks, and objectives.  Every partial, mixed, unsafe, or
underpowered outcome is `inconclusive`.

Formal G1 and G3 remain `NOT_RUN_BLOCKED`; none of these rules licenses a claim
about EEG diffusion as a whole.
