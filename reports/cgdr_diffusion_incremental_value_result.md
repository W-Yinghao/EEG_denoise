# CGDR diffusion incremental-value result

Status: `completed_protocol_scoped_inconclusive`.

The frozen comparisons are complete. They do not support a protocol-scoped
incremental-value claim, but they also do not test or reject diffusion as a
method family.

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

## Frozen natural SGEYESUB diffusion comparison

The prospective release-internal block1-to-block2 comparison completed all 15
frozen folds. Study02, study04 and study05 remained separate layout/reference
cells. The availability denominator was 44 stems: all 43 compatible stems
completed all six methods, while `study05/study05_p42` remained the single
pre-registered `blocked_no_population` row for every method. It contributed
no performance values. All 30 learned endpoints completed exactly 6000
successful updates using the verified shared minibatch sequences.

Conditional DDIM100 and the task-matched multichannel deterministic U-Net used
the same outer-training stems, weak low-artifact targets, input fields,
windowing, channel layout, normalization and operator conditioning. Neither
method read query EOG or query labels during fitting or inference. The target
is low-artifact observed EEG, not paired clean truth.

All deltas below are conditional diffusion minus matched U-Net. Intervals are
fixed-seed participant-stem bootstrap descriptions and did not alter the
frozen point rule.

| Metric (favorable direction) | Finite pairs | Mean delta | Wins | Descriptive 95% CI |
|---|---:|---:|---:|---:|
| EOG coherence reduction (higher) | 43 | +0.094844 | 43/43 | [+0.080219, +0.110352] |
| Matching-projector attenuation, dB (higher) | 43 | -30.898542 | 0/43 | [-32.069700, -29.610501] |
| Non-artifact preservation (higher) | 43 | -53.234719 | 0/43 | [-59.836502, -46.864287] |
| PSD distortion (lower) | 43 | +3680.228239 | 0/43 | [+2836.423719, +4630.615785] |
| Covariance distortion (lower) | 43 | +1662.045435 | 0/43 | [+1295.652746, +2070.402964] |
| ERP observation-preservation proxy (higher) | 41 | -30.469225 | 0/41 | [-34.225833, -26.756511] |

The two missing ERP proxy values are `study05_p44/p45`, where the frozen
condition-trial requirement was not met; they are N/A, not failures or ties.
All 43 paired learned-arm comparisons (86 inference rows) otherwise succeeded. Conditional inference
used 100 network calls per window versus one for the U-Net; mean recorded
latency was 0.3543 s versus 0.00426 s, with timing interpreted descriptively
under scheduler-selected GPUs.

Coverage and primary-metric completeness passed, but the attenuation point
condition, every preservation/safety point condition, and joint primary wins
failed. ERP safety coverage was 41/43. The frozen natural result is therefore
`inconclusive`, not `conditional_diffusion_supported` and not a
diffusion-family-negative verdict.

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

Jobs 919809 and 919833 completed all 8/8 cells, 88/88 metric rows and
44/44 paired SNR rows. Every paired arm used the same reconstructed ordered
inputs, source membership, supervision exposure, optimizer schedule and
update count. Conditional inference used 500 network calls per output versus
one for the matched deterministic arm.

The official-native arm is a seeded source-faithful wrapper with disclosed
validation source overlap. The strict arm is source-epoch separated but not
participant-independent. The matched comparator is single-channel and
same-backbone; it is distinct from the Klados multichannel U-Net. The corrected
spectral RRMSE is a compatibility metric, not the broken official value.

| Protocol | Noise | ΔSNR improvement | Δcorrelation | Δtemporal RRMSE | Δcorrected spectral RRMSE |
|---|---|---:|---:|---:|---:|
| official native | EOG | -0.124059 | +0.008955 | -0.022176 | -0.023442 |
| official native | EMG | -0.490040 | -0.000240 | +0.014817 | +0.313852 |
| strict source epoch | EOG | +0.324183 | -0.004320 | +0.012350 | +0.009928 |
| strict source epoch | EMG | -0.411911 | -0.008063 | +0.017893 | +0.012339 |

The official-native rows remain descriptive because the released split has
disclosed train/validation source overlap. In the strict source-epoch rows,
conditional diffusion did not meet the frozen stability/safety rule: strict
EOG improved SNR at all 11 levels but lost all 11 correlation, temporal and
spectral comparisons; strict EMG lost all four comparisons at all 11 levels.
The local EEGDfus outcome is `inconclusive`. This is a paired single-channel
EOG/EMG mixture stress test without participant identities, not natural
multichannel EEG or formal G1/G3.

## Evidence roles

- Post-hoc: current-M2 absolute baseline audit and SGE corrected audit.
- Exploratory: Klados development source records, including M1/M2/M4,
  deterministic hard/soft consistency, matched U-Net and conditional DDIM100.
- Frozen benchmark: EEGDfus official-native and strict-source matrices.
- Prospective frozen natural comparison: SGEYESUB block1-to-block2 conditional
  DDIM100 versus matched multichannel U-Net. This is release-internal and
  weak-target, not a clean-waveform or original-paper reproduction.
- Confirmatory formal G1/G3: none; both remain `NOT_RUN_BLOCKED`.

The frozen EEGDfus benchmark can report only a local stability outcome for its
paired single-channel EOG/EMG stress test. It cannot itself emit the top-level
`conditional_diffusion_supported` or category-negative label. The completed
natural SGE comparison is also `inconclusive`: it shows a strong tested-arm
failure outside coherence, while two ERP proxies are legitimately N/A.
The combined v2 protocol-scoped decision is therefore `inconclusive`;
`diffusion_family_wide_status` remains `not_tested`.

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
| EEGDfus validation/smoke/full/aggregate | 919807, 919808, 919809, 919833 | cpu, gpu-any | completed; 8 cells, 88 metric rows, 44 paired rows |
| Frozen v1 decision and post-v1 full test | 920060, 920061 | cpu | completed inconclusive; 333 tests passed |
| Natural SGE validation/development/aggregate | 920231, 920256, 920192, 920193 | cpu, gpu-any, cpu-high | completed; 10 folds, 15 stems, 20 endpoints |
| Natural SGE evaluation/aggregate | 920476, 920485 | gpu-any, cpu-high | completed; 15 folds, 43 successes plus one blocked stem |
| Metric-specific ERP coverage repair | 920555, 920583 | cpu | first test exposed one stale expectation; replacement passed 336 tests |
| Frozen v2 decision | 920585 | cpu | `completed_frozen_v2_decision`; protocol-scoped inconclusive |
| Final full test at terminal committed HEAD | reported in final handoff | cpu | run after this report commit and before push, so embedding its ID cannot change the tested HEAD |

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

Natural SGE fold checkpoints are retained at
`/home/infres/yinwang/denoiseNet/results/cgdr/sgeyesub_diffusion_incremental/{development,evaluation}/<fold_id>/checkpoints/<arm>/{last,final}.pt`.
Re-running the same fold command under the same committed HEAD reloads the
terminal endpoint; no checkpoint is committed to Git.

## Unresolved boundaries and blockers

| Item | Status / consequence |
|---|---|
| MATLAB / official SGE parity | blocked; no executable/module, no license test, all numerical parity metrics N/A |
| Python SGE port | source-faithful but not numerically cross-validated; not exact official reproduction |
| Official EEGDfus spectral RRMSE | blocked by released 400-vs-512 denominator shape; corrected metric is compatibility-only |
| EEGdenoiseNet identities | no participant IDs; strict unit is source epoch, not participant |
| Klados independence | source records, not verified independent participants |
| SGE natural real EEG | frozen 15-fold block2 conditional-vs-U-Net comparison completed; no clean target |
| Natural multichannel real-EEG diffusion value | tested protocol is inconclusive; family-wide status remains `not_tested` |
| Formal G1/G3 | `NOT_RUN_BLOCKED` |
| Timing | heterogeneous `gpu-any`; Klados timing is descriptive, not cross-hardware evidence |
| D4PM | not tested; verified-input and released-source defects remain blockers |

Git started from `cd516ed5484253fbca1c4d144abf63a36e7498d2` on
`master`. The final result commit and pushed `origin/master` identity are
recorded after the terminal committed-HEAD test; unrelated dirty-worktree
content remains retained and unmodified.
