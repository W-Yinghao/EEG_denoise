# Repository audit

Recorded: 2026-08-01 (Europe/Paris)

Status: **complete as a read-only source/config/result inventory; scientific chain frozen**

This report audits the repository that was present when the current server contract began. It does
not validate, reproduce, or endorse any historical result. No project Python, test, data parser,
training, evaluation, or plotting command was run for this repository audit; line references below
come from read-only text inspection and Git control-plane observations.

## Decisive finding

The worktree is not an empty CSPD repository. It is a pre-contract SADDPM/BCI-IV-2a worktree with
historical models, scripts, configs, tests, checkpoints/results, figures, and manuscript material.
Its scientific interfaces and execution assumptions do not implement the current
population-posterior, support-only calibration, `NULL`, P0, G1--G5, or deferred-backup contracts.

Accordingly:

- all existing M0--M12 claims, `PASS` labels, tables, figures, checkpoints, job IDs, and values are
  **legacy provenance only** and are inadmissible as evidence for the current protocol;
- no old milestone is equivalent to current Stage A/B/C or G1--G5, even where both use words such as
  “gate”, “null”, “oracle”, “baseline”, “B1”, or “diffusion”;
- no population base, P0 slice, gate, or B1--B6 job may be submitted from the legacy entry points;
- low-level utilities may be reused only behind new contract-facing adapters and only after the
  required Slurm tests establish their semantics and provenance.

This conclusion is independent of whether any historical number is accurate under its original
protocol. The issue is that the current protocol requires different estimands, split authority,
visible inputs, execution records, and failure behavior.

## Git and preservation state

At audit time:

- code root: `/home/infres/yinwang/denoiseNet`;
- branch: `master`;
- HEAD: `2f408f3cdc4895347b6a567b913159b1c60a0b50`, subject
  `Improve conditional denoiser: x0-prediction + conditional-SDEdit sampling`;
- remote: `origin -> https://github.com/W-Yinghao/EEG_denoise.git` for fetch and push;
- the specified remote had no heads or tags when bootstrap checked it; `origin` was added locally,
  and no push, default-branch change, release, or upload was performed;
- `git status --porcelain=v1` contained 10 tracked-change entries and 48 untracked top-level entries.
  These counts describe porcelain entries, not recursively counted files.

The dirty worktree includes user/concurrent changes in source, results, figures, manuscript files,
and newly added audit infrastructure. It is preserved in place. This audit does not infer authorship
or validity from tracked/untracked state and does not clean, reset, revert, mass-format, stage, or
commit those changes. The detailed root and remote observations are retained in
`reports/bootstrap_state.md`.

No applicable `AGENTS.md` was found in the repository file inventory used for this review. The
reviewed legacy project-level narrative files include `README.md`, `PLAN.md`,
`SADDPM_IMPLEMENTATION_HANDOFF.md`, `environment.yml`, and `RESULTS.md`; current task attachments and
the server execution contract take authority over them.

## Legacy architecture and authority mismatch

The legacy package is organized around:

- BCI Competition IV-2a loading and overlapping window construction in `saddpm/data/`;
- participant-ID embeddings and FiLM conditioning in `saddpm/models/`;
- unconditional, SDEdit, and noisy-conditioned diffusion code in `saddpm/diffusion/`;
- ICA, neural denoisers, EEGNet/FBCSP, pairwise accuracy, and denoising metrics in
  `saddpm/baselines/` and `saddpm/eval/`;
- milestone scripts `scripts/m0_*` through `scripts/m12_*` and old per-job Slurm files;
- legacy configs directly under `configs/`, with historical outputs under `artifacts/`, `results/`,
  `RESULTS.md`, and manuscript directories.

That dependency graph is not the required one-way graph
data/splits -> population posterior -> support operator/mask -> energy correction -> sampler and
baselines -> gates -> evaluation. Repository-wide symbol inspection found no implementation of the
required `PopulationPosteriorBase`, immutable `SupportBatch`, `TransferEstimate`,
`PrecisionAttenuation`, `CalibrationCorrection`, `NullSampler`, visible-field audit, or current gate
status contracts. Absence of a matching symbol is not alone proof that no reusable mathematics
exists; it is proof that the auditable contract-facing interfaces are absent.

### Population posterior and `NULL`

- `saddpm/models/subject_embed.py:1-29` defines a participant embedding table plus a “null” embedding
  slot. `saddpm/models/unet1d.py:156-176,218-224` uses that slot when a participant ID is omitted.
  This is a no-subject ablation in a legacy network, not a direct call to an observation-conditioned
  `PopulationPosteriorBase.sample(y, z_d, seed)`.
- `scripts/m8_benchmark.py:44-62` explicitly trains an unconditional clean DDPM, while
  `scripts/m8_benchmark.py:65-80` and `saddpm/diffusion/conditional.py:27-87` implement a separate
  noisy-conditioned network. Neither split representation preserves a separately auditable clean
  score and population observation energy `E0`, nor does it establish `EC-E0` semantics.
- `saddpm/diffusion/conditional.py:63-70` permits full generation or a noisy SDEdit warm start; this
  is not evidence that `rho=0` returns the exact same-query population base or that individualized
  components have zero calls.

Therefore every legacy “null”, unconditional-prior, no-subject, or full-generation result is
ineligible for current `NULL` evidence.

### Participant identity, split, and query leakage boundary

The following are conflicts with the current interfaces; they are not accusations about the intent
of the older protocol:

- `saddpm/data/bcic2a.py:23-51` stores raw subject ID and a session role in each window object;
  `saddpm/data/datasets.py:17-36` exposes `(window, subject_id, mi_label)` directly to models.
- `saddpm/data/bcic2a.py:208-225` assigns train/evaluation roles by sorted session position and writes
  participant IDs for embedding lookup. It does not provide immutable outer-participant, session,
  support-context, query-context IDs or an intersection assertion.
- `saddpm/data/preprocessing.py:14-49` creates 2 s windows by an arbitrary configured step; the
  legacy config uses a 0.5 s step (`configs/data.yaml:27-31`). Adjacent windows therefore overlap.
  Such windows cannot be treated as independent support duration or independent statistical units
  without a new context/split construction.
- `scripts/m3_subject_conditioning.py:94-104` pools all configured participants for training, and
  `scripts/m4_saddpm.py:91-100` does the same. There is no frozen outer-participant exclusion at
  these entry points.
- `scripts/m4_saddpm.py:54-69,169-188` reads Session-E participant identity labels and uses its
  accuracy in the old M4 pass/fail rule. That is incompatible with the current prohibition on test
  participant identity and query outcomes entering fitting, selection, gating, or stopping.
- `scripts/m5_sdedit.py:68-84,112-118` sweeps `t*` on Session-E inputs and declares an old gate from
  the resulting trend. `scripts/m10_ablation.py:90-93,126-152,169-178` evaluates a `t*` sweep against
  `clean_test` and writes the sweep table. These are not preregistered untouched outer-test
  comparisons under the current contract.
- `scripts/m9_train.py:87-97,135-143` constructs Session-E synthetic test pairs for all configured
  subjects and exposes correct/wrong/null participant embeddings; `scripts/m9_reeval.py:78-93`
  compares legacy operating points on clean test targets. `scripts/m12_subject_rescue.py:2-14,95-104,
  141-147` explicitly “rescues” a subject-aware claim using synthetic subject-specific corruption.
  These paths cannot establish current operator specificity or real-EEG personalization.
- `scripts/run_pairwise_matrix.py:87-108` trains and evaluates a 9x9 source/target protocol using
  participant-indexed embeddings. It is not an outer-participant support/query calibration split.

No legacy split manifest records a frozen outer fold, disjoint support/query context IDs, split
hash, visible-field list, or rejection of a query-clean-target type. Until new split objects and
leakage tests exist, legacy loaders may not feed current fitting or gate code.

### Data-role and data-plane conflicts

- `configs/data.yaml:35-38` leaves the MOABB root null, permitting the home-directory default, while
  its comments describe a symlink into a local cache. `scripts/check_env.py:89-123` inspects
  `~/mne_data` and says absent data may download on first use. `scripts/link_local_dataset.py:24-70`
  can create cache directories and symlinks. These behaviors are incompatible with the fixed data
  root and Slurm-only download/inventory contract unless replaced by a registered, scheduled,
  read-only/transactional path.
- `saddpm/data/cache.py:1-6,21-50` writes preprocessed subject objects under
  `artifacts/cache/` in the code worktree. Current large derived arrays must instead be written under
  the approved versioned data-root derived namespace with raw/config/code/channel-order provenance.
- `configs/eegdenoise.yaml:1-9` points to `EEGdenoiseNet/data` inside the worktree, and
  `saddpm/data/eegdenoisenet.py:92-98` loads three `.npy` arrays there. The untracked
  `EEGdenoiseNet/` tree is preserved, but neither its data contents nor its license/version/source
  status is inferred from its directory name.
- `saddpm/data/eegdenoisenet.py:101-150` builds random row-level train/validation/test mixtures from
  single-channel clean/artifact arrays; the returned object has no participant/session provenance.
  It may support a historical semi-simulated benchmark, but cannot establish individualized
  multichannel transfer, outer-participant evidence, or a dataset-specific clean prior.
- `saddpm/data/bcic2a.py:111-125` drops EOG before the main SADDPM path, while the current protocol
  requires explicit separation of external-mask and EEG-only regimes and an audited external
  reference construction. Existing ICA use of EOG does not supply that missing contract.

The historical statement in `RESULTS.md:6-20` that BCI-IV-2a was pre-staged and hash-matched is not
accepted as current inventory, license, integrity, or readability evidence. At this audit point the
scheduled full `/projects/EEG-foundation-model` inventory and candidate-specific license/sample-read
review were not complete. Thus no candidate dataset is promoted here to `verified_available` or
`missing`, and no download or data-root write is authorized. The fixed data root was observed at 96%
utilization, so any later derived-data plan also requires a disk-increment estimate.

### Slurm and Conda conflicts

- `README.md:33-44` instructs direct login-node Python, dataset parsing, and `pytest`, and names a
  `V100` partition. `PLAN.md:43-48` describes an additive `moabb` install, V100 jobs, and automatic
  home-cache downloading. `environment.yml:3-10` documents modifying `eeg2025` and creating a third
  `saddpm` environment. None of those actions are authorized now.
- Legacy jobs such as `scripts/slurm/m4.sbatch:1-16` and `scripts/slurm/m8.sbatch:1-16` request the
  unregistered `V100` partition, select `eeg2025` for GPU model work, inherit an arbitrary submit
  directory, and omit the current immutable run/allocation/status contract. They must not be used.
- The audited current logical profiles map to `CPU`, `cpu-high`, `A100`, `H100`, and `L40S`, as
  recorded in `configs/cluster/slurm.yaml` and `reports/slurm/control_plane_audit.md`. SlurmDBD/
  `sacct` was unavailable and a client/controller configuration-hash mismatch was observed; missing
  accounting evidence must remain missing.
- Initial scheduled environment probes (jobs 918736 and 918737) reported useful package and GPU
  capabilities, but the first-generation audit/submit scripts were subsequently found to have
  provenance and sanitization weaknesses. Their scientific use remains frozen pending a hardened
  rerun; they must not silently validate legacy jobs or historical environment claims.

No environment creation, package install/upgrade, direct project execution on the login node, or
fallback from one environment/GPU/scientific configuration to another is permitted.

## Historical results are not current evidence

`RESULTS.md` contains concrete historical numbers and many `PASS` declarations, for example old
environment/data assertions (`RESULTS.md:6-20`), V100 jobs and unit-test counts
(`RESULTS.md:58-100`), joint all-participant training (`RESULTS.md:103-135`), Session-E/t* claims
(`RESULTS.md:137-160`), and M7/M8--M12 outcomes (`RESULTS.md:155-249` and later). The worktree also
contains result CSVs/arrays, figures, checkpoints, manuscript tables, and scripts that can render or
restate them.

They remain inadmissible because the current run contract cannot be reconstructed merely from those
files: there is no required resolved-config snapshot, environment lock for the actual job, immutable
data/split/prior/checkpoint hashes, visible-field audit, allocation record, failure-retaining context
results, or gate-threshold hash. In addition, the old scientific semantics and splits differ as
described above. `scripts/revision/sig_downstream.py:39` explicitly operates on published table
numbers, and `scripts/reproduce_manuscript.py:114-119` prints manuscript reference values alongside
its outputs; neither is an authorized provenance bridge.

The current attachment review also found 230 `TBD` tokens and no admissible completed empirical
table in the authoritative CSPD manuscript bundle. No repository value may be copied into those
placeholders. Old artifacts are retained for forensic comparison only; they must not be renamed,
relabelled, averaged with new runs, or used to choose current thresholds.

## Reuse, adaptation, and isolation boundary

| Classification | Candidate | Required handling |
|---|---|---|
| Reuse after focused verification | Pure array shape helpers in `saddpm/data/preprocessing.py`; seed helper in `saddpm/utils/seed.py`; beta schedule and selected numerical primitives in `saddpm/diffusion/schedule.py` / `gaussian_diffusion.py` | Import through a contract-facing adapter only; add endpoint/property tests, fixed dtype/device behavior, config/provenance capture, and execute tests through Slurm. Window helpers do not define statistical independence. |
| Adapt substantially | MNE loading/channel handling, diffusion backbone/sampler, ICA/EEGNet and deterministic denoisers, checkpoint/logging helpers | Remove participant/query leakage; use dataset registry and immutable split types; enforce fixed montage/reference/channel order; add `E0`, `EC-E0`, `rho`, attenuation, visible-input and information-matching semantics. Checkpoints/logs need atomic writes plus data/split/config/environment/Slurm hashes. |
| Isolate from current science | Participant embeddings/FiLM and old dual decoder; all M0--M12 scientific entry points; old configs and V100 sbatch files; home/code-tree caches; W&B paths; manuscript reproduction/revision scripts | Keep as legacy provenance. Do not call from Stage B/C or G1--G5, do not use their pass/fail flags, and do not consume their results/checkpoints. |
| Never reinterpret | Unconditional DDPM as `NULL`; subject null embedding as `NULL`; SDEdit sweep as a gate; synthetic “subject rescue” as G2; legacy “B1” dropout-fix wording as formal `B1 ROBUST-M`; ICA as native SGEYESUB or oracle subtraction | These are semantic category errors under the current contract, not adapter tasks. |

Specific utility limits:

- `saddpm/utils/seed.py:15-38` seeds common RNGs and toggles cuDNN determinism, but a current run must
  also record the requested and effective determinism/precision policy and all seeds.
- `saddpm/utils/checkpoint.py:28-43` stores model state, a caller-provided config, and Git commit only;
  it is not atomic and omits dirty-tree, data, split, prior, environment, allocation, and artifact
  hashes.
- `saddpm/utils/logging.py:14-78` creates a mutable CSV and optionally initializes W&B. It does not
  implement the required immutable run directory, JSONL failures, allocation, resource usage, or
  secret-safe visible-field provenance. External W&B communication is not implicitly authorized.

The preferred integration boundary is a new `src/eeg_cspd/` contract layer. It may wrap verified
legacy primitives but must not make `saddpm` types the authority for support/query, population base,
operator, attenuation, gate, or run provenance semantics.

## Test audit and missing assertions

The existing `tests/` suite targets the historical implementation. Examples include DDPM schedule
and shape checks (`tests/test_diffusion.py:21-93`), participant embedding/FiLM behavior
(`tests/test_subject.py:30-80`), and toy EEGNet accuracy (`tests/test_eval.py:13-40`). These may be
useful regression tests for isolated primitives, but historical statements that they passed are not
a current Slurm test record, and they do not cover the present contracts.

Required missing test families include:

- attachment completeness, archive traversal/link/bomb protection, full-root pre-download inventory,
  final-directory collision, and transactional publication;
- participant/session/train-validation-test exclusivity, support/query context non-overlap,
  participant-pseudonym boundary, query-target type rejection, and visible-field auditing;
- population-base dependence on the same query `y`, exclusion of test identity/support parameters,
  prior/checkpoint/clean-target provenance, calibration/proper-score boundaries;
- direct `NULL` equality to the base at the same seed and mock call counts of zero for operator,
  mask, attenuation, correction, `rho`, and unconditional-prior components;
- `C'E'=CE`, image-space invariance, projector symmetry/idempotence/rank/failure states, attenuation
  `a=0/1`, PSD, `rho=0/1`, `EC-E0`, likelihood-ratio/generalized-Bayes labels, and no observation
  double-counting;
- external versus EEG-only input isolation, stop-gradient choice, confidence/abstention and exact
  rollback to explicit `NULL`;
- correct/population/wrong/shuffled/oracle source provenance, information-field matching, native
  SGEYESUB versus oracle separation, participant-level statistics, and retention of failed samples;
- drift-only degradation actions, no online family search, stale-input detection, immutable run
  records, allocation/config/environment hashes, checkpoint/resume, and retry classification.

No existing test should be deleted merely because it is insufficient. It should be classified as a
legacy regression test, run only through a registered Slurm profile, and supplemented by the new
contract suite. Synthetic fixtures may validate these engineering properties but cannot enter gate
decisions or scientific tables.

## Scientific-chain freeze and next legal boundary

The current scientific chain is frozen before Stage B for two independent reasons:

1. Stage A is not complete until the full scheduled data-root inventory, source/license/sample-read
   audit, at least one verified real-EEG path, and immutable outer/support/query splits are available.
2. `reports/attachment_review.md` records unresolved `CONFLICT-SCI-001` through
   `CONFLICT-SCI-003`: population-energy versus `NULL`/mask isolation, different placement of
   mandatory real-EEG evidence in the ordered gates, and incompatible backup activation order.

Thus no old or new population base, P0 fit, P0 full-fold slice, G1--G5 comparison, outer-test table,
or B1--B6 fit/search/comparison is approved. B1--B6 may exist only as `enabled: false` schemas and
import-safe interfaces. A legacy script or result cannot “rescue” a failed or blocked upstream gate.

Legal work while frozen is limited to independent audit and safety infrastructure: hardened Slurm
submission/runtime provenance, full-root read-only inventory, dataset registries, source/license and
sample-read audits, immutable split/provenance schemas, disabled config stubs, and Slurm-run
administrative/leakage/math tests that do not choose scientific semantics. The implementation map is
maintained in `reports/implementation_plan.md`, and requirement status/conflicts are maintained in
`reports/requirement_traceability.csv`.

This audit makes no claim that a dataset, clean target, license, checkpoint, GPU result, statistical
effect, or current gate exists merely because a similarly named legacy file is present.
