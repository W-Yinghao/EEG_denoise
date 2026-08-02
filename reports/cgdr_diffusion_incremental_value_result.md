# CGDR diffusion incremental-value result

Status: `draft_waiting_for_EEGDfus_full_matrix`; this file is not yet a final
scientific decision.

## Corrected starting point

| Item | Corrected status or retained result |
|---|---|
| Original historical classifier | `A_limited` (retained, not overwritten) |
| Absolute-baseline interpretation | `B_geometry_only` |
| Klados current M2 | `current_M2_no_incremental_value` |
| Diffusion family before the new matched runs | `not_tested` |
| SGE operator specificity | `hard_Q_P0_tradeoff_inconclusive` |
| Engineering priority | `deterministic_first_diffusion_open` |
| Formal G1/G3 | `NOT_RUN_BLOCKED` |

The retained absolute-baseline audit found median oracle-projector M2 minus
same-sampler POP `delta e_parallel=-0.004969`, but oracle-projector M2 minus
deterministic oracle Qy was approximately `+1.145` for `e_parallel`, `+0.700`
for RRMSE and `-0.178` for correlation. In this post-hoc descriptive audit,
Qy had a better value on each of the four audited waveform metrics for all
16 source records.

All Klados `e_parallel` values in this report use the dimensionless
neural-normalized definition `||P(x_hat-x)||_F / ||Px||_F`. They are not
pooled or directly compared with legacy artifact-normalized `e_parallel`.

The retained 16-record result only shows that the tested unconditional clean
prior + deterministic DDIM100 + final-hard-Q M2 instance adds no value over
the query-oracle `Qy` diagnostic. It does not reject conditional diffusion,
dual-prior diffusion, other samplers, other objectives, or EEG diffusion as a
class. `Qy` is query-derived, oracle, and non-deployable.

## SGEYESUB real-EEG boundary

The corrected post-hoc release-internal block1-to-block2 audit retained all 44
registered stems as the feasibility denominator. Matching and population
methods were jointly successful for 43 compatible stems;
`study05/study05_p42` retained successful standalone matching-Qy metrics, but
its population arm remained `blocked_no_population`, so it was excluded from
the matching-versus-population paired comparison.

For matching minus population, held-out EOG remaining improved by a mean
`-0.014066` (95% descriptive bootstrap CI `[-0.021331, -0.006442]`, 31/43
wins) and EOG coherence reduction improved by `+0.079048`
(`[0.057199, 0.100998]`, 41/43 wins). Non-artifact preservation, PSD and
covariance differences were small with intervals spanning zero, while the ERP
proxy was worse by `-0.004741` (`[-0.007960, -0.000672]`, 5/41 wins). All
hard-Q methods failed the frozen absolute safety thresholds. This is therefore
a trade-off/inconclusive post-hoc operator result, not diffusion evidence.

All deltas below are matching minus population. Confidence intervals are the
fixed-seed 20,000-draw participant-stem bootstrap of the mean and are
descriptive, not preregistered inference.

| Metric (favorable direction) | Finite pairs | Mean delta | Median delta | Descriptive 95% CI | W/T/L |
|---|---:|---:|---:|---:|---:|
| EOG remaining ratio (lower) | 43 | -0.014066 | -0.012717 | [-0.021331, -0.006442] | 31/0/12 |
| EOG coherence reduction (higher) | 43 | +0.079048 | +0.069832 | [+0.057199, +0.100998] | 41/0/2 |
| Non-artifact preservation (higher) | 43 | +0.001324 | +0.002729 | [-0.005682, +0.008077] | 22/0/21 |
| PSD distortion (lower) | 43 | -0.003810 | -0.005637 | [-0.014734, +0.007498] | 24/0/19 |
| Covariance distortion (lower) | 43 | -0.003936 | -0.002126 | [-0.010558, +0.002731] | 23/0/20 |
| ERP observation-preservation proxy (higher) | 41 | -0.004741 | -0.004343 | [-0.007960, -0.000672] | 5/0/36 |
| Observation change (lower) | 43 | +0.003196 | +0.002106 | [-0.000721, +0.006656] | 17/0/26 |

Coverage keeps every registered stem in the denominator; no blocked identity
row contributes a performance value.

| Method | Success | Fallback | Blocked | Ineligible | Failed | Denominator |
|---|---:|---:|---:|---:|---:|---:|
| matching Qy | 44 | 0 | 0 | 0 | 0 | 44 |
| population Qy | 43 | 0 | 1 | 0 | 0 | 44 |
| B6 Qy, frozen gamma=0 | 43 | 0 | 1 | 0 | 0 | 44 |
| B6 soft proximal, frozen gamma=0 | 43 | 0 | 1 | 0 | 0 | 44 |
| wrong Qy | 43 | 0 | 1 | 0 | 0 | 44 |
| shuffled Qy | 0 | 43 | 0 | 1 | 0 | 44 |
| source-faithful Python SGEYESUB port | 44 | 0 | 0 | 0 | 0 | 44 |

No independent downstream-task classifier was implemented or validated under
this release-internal protocol, so downstream task preservation is `N/A`; the
ERP quantity remains a proxy. Method-specific latency was not aggregated and
is likewise not comparable.

The frozen gamma=0 endpoint came only from the support-side development
objective and was not reselected after reading evaluation. At gamma=0 the full
and both split-half shrinkage projectors equal the population projector, so the
stability term is structurally zero; this is a conservative endpoint property,
not a held-out personalization-failure verdict. Study heterogeneity was
retained rather than pooled away: mean matching-minus-population EOG-remaining
deltas were -0.0090/-0.0303/-0.0012 and coherence-reduction deltas were
+0.0835/+0.0951/+0.0554 in study02/04/05, respectively. The complete 27-row
study table is in `reports/cgdr_sgeyesub_corrected_audit.md`.

MATLAB numerical parity remains blocked: CPU job 919775 found no MATLAB
executable/module, while job 919777 checked out official source commit
`2c95b4f46f37670d25399ac0fdd705ae18248b25`. The Python implementation is
`source_faithful_not_numerically_cross_validated`, not an exact official
reproduction.

## Klados matched exploratory comparator

Conditional-v3 and the operator-scope-isolated deterministic multichannel
U-Net used the same paired windows, targets, corruption exposure, and
deployment-visible conditioning. Conditional diffusion additionally consumed
its algorithm-internal diffused state and timestep. Both used 6000 actual
optimizer updates per scope. All three conditional checkpoints had exactly
6000 attempted and successful updates, zero AMP skips, and `resumed=false`.
The checkpoint audit passed 6/6 deterministic/conditional endpoints.

Six of eight development source records were commonly eligible
(`sim31, sim33, sim34, sim36, sim44, sim45`); `sim32` and `sim35` remained
ineligible. The aggregate contained the exact 18/18 expected eligible
record-by-scope conditional cells (6 records by 3 scopes) with no missing,
unexpected, or failed eligible cell. The other two records, equivalently six
possible record-scope cells, remain explicitly ineligible in the 8-record
coverage denominator. These are source records, not verified independent
participants, and all results are exploratory.

Every row below uses the exact same six commonly eligible source records; the
remaining two of eight are retained as ineligible, and no successful arm is
missing or failed. Values are medians over source records after each iterative
arm's five algorithmic seeds have been combined by its frozen output rule.

| Operator scope | Arm | n/8 | e_parallel | e_perp | RRMSE | Correlation | PSD distortion | Failed/ineligible |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| population | conditional DDIM100 | 6/8 | 1.336499 | 2.042543 | 1.784544 | 0.524246 | 0.583208 | 0/2 |
| population | M1 warm start | 6/8 | 1.429758 | 0.302132 | 0.906294 | 0.839562 | 1.023181 | 0/2 |
| population | current M2 final hard-Q | 6/8 | 2.101045 | 0.036367 | 1.329672 | 0.754692 | 0.460802 | 0/2 |
| population | M4 stepwise proximal | 6/8 | 5.665473 | 0.066052 | 3.340256 | 0.809208 | 8.254769 | 0/2 |
| population | deterministic hard-Q | 6/8 | 0.997237 | 0.025712 | 0.611976 | 0.931779 | 0.432551 | 0/2 |
| population | deterministic soft proximal | 6/8 | 0.891149 | 0.012856 | 0.529691 | 0.949164 | 0.225207 | 0/2 |
| population | matched multichannel U-Net | 6/8 | 0.437619 | 0.006840 | 0.250479 | 0.988751 | 0.117406 | 0/2 |
| matching P0 | conditional DDIM100 | 6/8 | 1.495480 | 2.100908 | 1.842459 | 0.521488 | 0.682966 | 0/2 |
| matching P0 | M1 warm start | 6/8 | 1.429758 | 0.302132 | 0.906294 | 0.839562 | 1.023182 | 0/2 |
| matching P0 | current M2 final hard-Q | 6/8 | 2.082083 | 0.251221 | 1.333888 | 0.740136 | 0.468542 | 0/2 |
| matching P0 | M4 stepwise proximal | 6/8 | 5.570334 | 0.622933 | 3.320811 | 0.785214 | 9.175758 | 0/2 |
| matching P0 | deterministic hard-Q | 6/8 | 0.999078 | 0.182197 | 0.621291 | 0.911477 | 0.425308 | 0/2 |
| matching P0 | deterministic soft proximal | 6/8 | 0.897473 | 0.091098 | 0.550480 | 0.940688 | 0.236949 | 0/2 |
| matching P0 | matched multichannel U-Net | 6/8 | 0.427280 | 0.007333 | 0.251056 | 0.988743 | 0.109629 | 0/2 |
| query-derived oracle | conditional DDIM100 | 6/8 | 1.472510 | 2.023539 | 1.814604 | 0.503322 | 0.570232 | 0/2 |
| query-derived oracle | M1 warm start | 6/8 | 1.429758 | 0.302132 | 0.906294 | 0.839562 | 1.023182 | 0/2 |
| query-derived oracle | current M2 final hard-Q | 6/8 | 2.096519 | 1.801174e-7 | 1.326292 | 0.747608 | 0.463068 | 0/2 |
| query-derived oracle | M4 stepwise proximal | 6/8 | 5.688950 | 0.012159 | 3.353533 | 0.803980 | 8.349044 | 0/2 |
| query-derived oracle | deterministic oracle Qy | 6/8 | 1.000000 | 5.660542e-8 | 0.613398 | 0.929887 | 0.436191 | 0/2 |
| query-derived oracle | deterministic soft proximal | 6/8 | 0.891709 | 5.660542e-8 | 0.529688 | 0.950724 | 0.228190 | 0/2 |
| query-derived oracle | matched multichannel U-Net | 6/8 | 0.435234 | 0.007105 | 0.249433 | 0.989125 | 0.117401 | 0/2 |

The compute ledger is method-aware; blank wall time means it was not captured,
not zero cost.

| Arm/scope | Parameters | Training count and semantics | Training wall time (s) | Training peak MB | Median inference latency (s) | Median inference peak MB | Network evaluations |
|---|---:|---|---:|---:|---:|---:|---|
| conditional DDIM100, population/matching/oracle | 4,563,667 | 6000 audited successful updates each | 686.235 / 691.616 / 835.807 | 688.756 / 688.756 / 688.533 | 25.901 / 25.242 / 25.280 | 404.711 | 100/seed/window; 500 total |
| matched U-Net, population/matching/oracle | 4,557,587 | 6000 audited successful updates each | 881.016 / 946.551 / 833.747 | 551.446 / 554.044 / 554.044 | 0.722 / 0.074 / 0.077 | 256.941 / 265.066 / 265.066 | no algorithmic seed; 1/window total |
| M1, population/matching/oracle | 4,435,667 shared prior | 3000 training-history steps; AMP skips not audited | N/A | N/A | 71.476 / 67.118 / 70.204 | 1541.650 | 100/seed/window; 500 total |
| current M2, population/matching/oracle | 4,435,667 shared prior | 3000 training-history steps; AMP skips not audited | N/A | N/A | 68.798 / 67.018 / 67.771 | 1541.650 | 100/seed/window; 500 total |
| M4, population/matching/oracle | 4,435,667 shared prior | 3000 training-history steps; AMP skips not audited | N/A | N/A | 68.668 / 69.610 / 67.932 | 1541.650 | 100/seed/window; 500 total |
| hard-Q / soft proximal | no learned model | N/A | N/A | N/A | <=0.0011 | 0 | 0 |

The conditional model had 4,563,667 parameters versus 4,557,587 for the
matched U-Net. Training objectives differed by design (masked epsilon MSE
versus paired task loss); deployment-visible conditioning, paired target
exposure, 154 training windows, 68 development windows, and update budget were
equal. The training objective and diffusion-internal state differed by design.
The frozen paired classifier, rather than these marginal medians alone,
determines the exploratory outcome.
Latency is the descriptive wall time for restoring one complete eligible
source record (all windows and, where applicable, five algorithmic seeds).
Because `gpu-any` allocated heterogeneous GPU models, training wall time,
inference latency and memory are not controlled cross-scope or cross-record
hardware comparisons.

## EEGDfus full benchmark

Jobs 919809 (eight full cells) and 919833 (aggregate) are still running or
dependency-pending. No final conclusion is permitted until all official-native
and strict-source EOG/EMG cells complete and the frozen mean-direction,
8-of-11-win, spectral-safety, source-manifest, and update-budget checks pass.

The official-native arm is a seeded source-faithful wrapper with disclosed
validation source overlap. The strict arm is source-epoch separated but not
participant-independent. The matched comparator is single-channel and
same-backbone; it is distinct from the Klados multichannel U-Net. The corrected
spectral RRMSE is a compatibility metric, not the broken official value.

## Evidence roles

- Post-hoc: current-M2 absolute baseline audit and SGE corrected audit.
- Exploratory: Klados development source records, including M1/M2/M4,
  deterministic hard/soft consistency, matched U-Net and conditional DDIM100.
- Frozen benchmark: EEGDfus official-native and strict-source matrices; neither
  is formal G1/G3 or natural participant-level multichannel EEG evidence.

The frozen EEGDfus benchmark can report only a local stability outcome for its
paired single-channel EOG/EMG stress test. It cannot itself emit the top-level
`conditional_diffusion_supported` or category-negative label. Because the
completed SGEYESUB audit contains no diffusion arm, natural multichannel
real-EEG diffusion incremental value remains `not_tested`, and the v1
top-level decision is forced to `inconclusive`.

## Key execution record

| Stage | Slurm job(s) | Profile | Terminal state / role |
|---|---|---|---|
| SGE corrected validation and audit | 919825, 919826 | cpu, cpu-high | completed; 43 paired stems / 44 feasibility stems |
| MATLAB runtime probe | 919775 | cpu | completed blocker probe; executable/module unavailable, license test not run |
| Official SGE source checkout | 919777 | cpu | completed at commit `2c95b4f46f37670d25399ac0fdd705ae18248b25` |
| Deterministic-v4 validation/train/development/aggregate | 919850, 919854, 919857, 919876 | cpu, gpu-any, cpu-high | completed |
| Conditional-v2 endpoint audit | 919907 | cpu | contract failed: population/matching 5999, oracle 6000; invalid for matched decision |
| Conditional-v3 validation/train/audit/development | 919944, 919945, 919946, 919947 | cpu, gpu-any | completed; replacement protocol |
| Conditional-v3 exact aggregate | 919988 | cpu-high | completed 18/18 expected eligible cells |
| Final common-record validation/audit/aggregate | 920047, 920048, 920049 | cpu, cpu-high | completed; 246 tests, 6/6 endpoints, 21 arm-scope rows |
| EEGDfus validation/smoke/full/aggregate | 919807, 919808, 919809, 919833 | cpu, gpu-any | validation/smoke complete; full running, aggregate dependency-pending |
| Frozen incremental-value decision | 920060 | cpu | dependency-pending after 919833 |
| Final full test at terminal report HEAD | 920061 | cpu | dependency-pending after 920060; required before push |

The first EEGDfus smoke (919798) failed all eight cells because the released
official `RRMSE_s` compared a 400-bin clean PSD with a 512-sample zero
denominator; 919799 therefore never ran. The replacement protocol keeps that
official metric blocked and separately reports the corrected PSD-denominator
compatibility metric. Conditional-v2 is likewise preserved as invalid
history, not silently folded into v3.

Stable non-Git checkpoints are under
`/home/infres/yinwang/denoiseNet/results/cgdr/klados_stage3_deterministic_scope_isolated_v4/checkpoints/<operator_scope>/last.pt`
and
`/home/infres/yinwang/denoiseNet/results/cgdr/klados_stage3_conditional_diffusion_matched_v3/checkpoints/<operator_scope>/last.pt`.
The existing `train-deterministic` and `train-conditional` commands resume from
those paths without changing scope or the fixed endpoint. Checkpoints are not
staged in Git.

## Unresolved boundaries and blockers

| Item | Status / consequence |
|---|---|
| MATLAB / official SGE parity | blocked; no executable/module, no license test, all numerical parity metrics N/A |
| Python SGE port | source-faithful but not numerically cross-validated; not exact official reproduction |
| Official EEGDfus spectral RRMSE | blocked by released 400-vs-512 denominator shape; corrected metric is compatibility-only |
| EEGdenoiseNet identities | no participant IDs; strict unit is source epoch, not participant |
| Klados independence | source records, not verified independent participants |
| SGE natural real EEG | full block2 operator/preservation audit exists, but no diffusion arm |
| Natural multichannel real-EEG diffusion value | not tested |
| Formal G1/G3 | `NOT_RUN_BLOCKED` |
| Timing | heterogeneous `gpu-any`; Klados timing is descriptive, not cross-hardware evidence |
| D4PM | not tested; verified-input and released-source defects remain blockers |

Git started from `cd516ed5484253fbca1c4d144abf63a36e7498d2` on
`master`. Final result commit, pushed `origin/master` commit and the retained
unrelated dirty-worktree state will be filled only after the terminal decision
and final tests.
