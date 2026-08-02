# CGDR stage-3 deterministic-first static audit

## Scope-isolated v3 contract amendment

Protocol v3 is a new, not-yet-executed contract and output root. It does not
modify or relabel the v2 jobs or files listed below. The v3 change holds the
training and validation record/window sets constant across population,
matching-P0, and query-derived-oracle U-Net scopes by first applying one common
matching-P0 eligibility rule. Every scope reports requested, included, and
skipped source-record counts and IDs; checkpoints from the three scopes must
agree on those sets before inference.

The v3 aggregate reports two matching estimands separately:

- `matching_p0_eligible_only` measures the matching operator only on records
  where matching P0 was eligible;
- `matching_request_fallback_policy` measures the intention-to-deploy policy
  over all requested records, including explicit POP fallback.

Failed methods and failed algorithmic seeds remain in method/seed status tables
and `failures.json`; only complete five-seed successes enter performance
summaries. Systemic configuration, checkpoint, shape, device, and OOM errors
still fail the job. The aggregate additionally reports within-record paired
deltas, latency, peak memory, actual/planned calls, selected/training updates,
and cumulative training wall time. An early-stop checkpoint carries a terminal
reason so resubmission returns terminal-complete without performing more
updates.

The multichannel U-Net is paired-clean-target supervised. It is therefore a
stronger, differently supervised exploratory comparator, not a same-supervision
baseline and not formal G3 evidence. The v3 config and stable output root are:

```text
configs/cgdr/klados_stage3_deterministic_comparison_v3.yaml
/home/infres/yinwang/denoiseNet/results/cgdr/klados_stage3_deterministic_scope_isolated_v3/
```

No v3 Slurm result is claimed by this static amendment.

Status: scope-isolated v2 implementation and routing validated by the aggregate
CPU Slurm test job 919825 (`195 passed`). The earlier shared-scope training job 919785 is
retained only as `invalid_pre_operator_scope_isolation`; its downstream jobs
919786/919787 were cancelled by the coordinator before scientific use. The old
checkpoint/result directory remains untouched.

## Superseding operator-scope isolation

The original Stage-3 draft trained one U-Net on population, matching, and
query-clean-derived oracle projector cells together. Although source-record
splits were disjoint, oracle-conditioned development cells could therefore
influence the checkpoint later used for deployable population/matching rows.
That is not acceptable as a strict same-information G3 baseline.

Protocol v2 trains and selects three independent checkpoints, one each for
`population_projector`, `matching_p0`, and
`query_derived_oracle_projector`. Every scope uses the same 3000--6000 update
budget and source-record split, but its training and validation bundles contain
only that scope. The oracle checkpoint is explicitly nondeployable. If matching
calibration is rejected for a record, inference uses both the population
projector and the population-scope U-Net and retains requested/effective source
plus fallback labels.

## Frozen interpretation boundary

The protocol records these exact fields before any new development outcome is
read:

```yaml
Klados: current_M2_no_incremental_value
diffusion_family: not_tested
SGE: hard_Q_P0_tradeoff_inconclusive
priority: deterministic_first_diffusion_open
```

This is not a new A/B/C classifier. Existing Klados results remain historical
and no formal G1 or G3 claim is created by this protocol.

## Existing sampler audit and frozen matrix

| ID | Existing semantics | Stage-3 role |
|---|---|---|
| M0 | full-generation guided DDIM | excluded from this deterministic-first comparison |
| M1 | observation warm-start/SDEdit followed by guided DDIM | frozen comparison arm |
| M2 | final hard-Q consistency | frozen comparison arm |
| M3 | per-step hard-Q consistency | excluded; it is a different hard enforcement regime |
| M4 | per-step quadratic proximal Q consistency | frozen comparison arm |
| M5 | single-prior-evaluation proximal heuristic | retained in the repository, but not represented as a strong matched baseline |

The v2 matrix is exactly M1, M2, M4, deterministic Qy, deterministic soft
proximal, and a paired-supervised multichannel deterministic U-Net. It is crossed
with population, matching P0, and query-derived oracle projectors. The oracle
projector arm is a nondeployable mechanism upper bound.

The sampler budget is explicit rather than described as equal compute: M1, M2,
and M4 use 100 network evaluations per seed and five fixed algorithmic seeds
(500 evaluations per window before batching); the U-Net uses one evaluation per
window; Qy and soft proximal use no network. The runner records per-seed and
total evaluations, actual forward invocations, latency, peak GPU memory,
parameter counts, training updates, and wall time. With `t_start=250`, M1 can
form an exact 100-step strictly decreasing DDIM sequence because 100 is at most
251; the existing sampler also asserts the observed call count at runtime.

## Split and target boundary

- Training source records: sim01--sim30.
- Development source records: sim31--sim36, sim44, sim45.
- Historical records already used in diagnosis: sim37--sim43, sim46--sim54.
- These are source records, not participants.
- The old sixteen records are never called untouched and are available only
  through explicitly labelled exploratory replay stages.
- Clean targets and query-derived oracle projectors may be used on training and
  development source records to train/select the strong nondeployable
  oracle-conditioned baseline arm. Historical clean targets do not enter
  fitting or selection.

The paired-supervised model receives only observed query EEG, the full frozen
operator projector, framewise external-EOG attenuation, and the valid-time
mask. Its input features contain `y`, `Pi*y`, attenuation, and all entries of
`Pi` broadcast over time, so it is not disadvantaged by seeing only the
projector action on one observation. The backbone is the repository's masked
three-level multichannel U-Net; invalid/padded frames are suppressed throughout
normalization, convolution, resampling, and attention. Training uses aligned
Klados contaminated/clean pairs, a paired task loss in P/Q subspaces plus a
first-difference term, at least 3000 and at most 6000 optimizer updates, and
development-only checkpoint selection. Its paired clean-target supervision is
stronger and different from clean-prior training, so this comparison cannot be
described as same-supervision G3 evidence.

The stable checkpoint layout is:

```text
/home/infres/yinwang/denoiseNet/results/cgdr/klados_stage3_deterministic_scope_isolated_v2/checkpoints/<operator_scope>/last.pt
/home/infres/yinwang/denoiseNet/results/cgdr/klados_stage3_deterministic_scope_isolated_v2/checkpoints/<operator_scope>/best.pt
```

They are not Git artifacts. The loader rejects checkpoints below 3000 updates,
with a different frozen contract, or with normalization outside sim01--sim30.

## Real-record integration and Slurm routes

A distinct engineering-only integration stage traverses every window and all
three operator sources for complete development record sim31, performs multiple
optimizer updates, saves a run-local checkpoint, reloads it, and requires exact
output equality. That checkpoint is explicitly ineligible as the trained
baseline. It exists to catch real-loader/model/checkpoint integration faults;
it is not scientific evidence.

Submitted dependency chain:

```bash
scripts/slurm/submit.sh gpu-any cgdr stage3-deterministic configs/cgdr/klados_stage3_deterministic_comparison.yaml real-record-integration
# 919827, array 0-2%3, afterok:919825
scripts/slurm/submit.sh gpu-any cgdr --afterok 919825 --array '0-2%3' stage3-deterministic configs/cgdr/klados_stage3_deterministic_comparison.yaml train-deterministic
# 919830, array 0-7%8, afterok:919827
scripts/slurm/submit.sh gpu-any cgdr --afterok 919827 --array '0-7%8' stage3-deterministic configs/cgdr/klados_stage3_deterministic_comparison.yaml development-record
# 919831, afterok:919830
scripts/slurm/submit.sh cpu-high cgdr --afterok 919830 stage3-deterministic configs/cgdr/klados_stage3_deterministic_comparison.yaml aggregate-development
```

`gpu-any` delegates among the registered GPU constraints. Explicit L40S,
V100-32GB, A100, and H100 profiles are also accepted. A40 remains available via
the site's `gpu-any` profile. The historical 16-record replay routes exist but
are deliberately not part of the command chain above. Training requests an
initial 12-hour wall time, 32 GB or more GPU memory, 8 CPUs and 64 GB host
memory; the final parameter count, observed wall time, peak memory and latency
must come from the Slurm outputs and are not guessed here.

The aggregate J0 test wildcard will include
`tests/unit/test_cgdr_stage3_deterministic.py`, covering input isolation, full
operator exposure, internal padding invariance, exact split/status/method
freezing, minimum-checkpoint-update semantics, M1 timestep/call compatibility,
and presence of the real-record Slurm route. Job 919825 completed this aggregate
suite with 195 passing tests; the final committed-tree replay remains required
before push.

## Targeted EEGDfus/D4PM availability check

No full data-root scan or download was performed. Targeted checkout reports now
show:

- EEGDfus source: `.external/EEGDfus`, commit
  `a19a652b3b6346188ae77067e1daf8b90cad005f` (checkout job 919778).
- D4PM source: `.external/D4PM`, commit
  `5be2b3c72973fea6c879e63cd83067ff66aace13` (checkout job 919779).
- No `/projects/EEG-foundation-model/{EEGDfus,eegdfus,D4PM,d4pm}` target exists.
- The registered EEGdenoiseNet path remains
  `/projects/EEG-foundation-model/eegdenoisenet/github-8d290661146c7189c98cc04812d37371d4b9426c`.

The EEGDfus native loader is single-channel and randomly permutes epoch arrays
before a positional split. It therefore cannot serve as the Klados
multichannel/source-record-safe strict baseline without an explicit adapter.
Native and strict variants must stay separately labelled; native reproduction
can only support its original stress-test semantics, while strict adaptation
must freeze sim01--30 training and the eight development records before using
the historical records exploratorily. No EEGDfus/D4PM result is claimed here.
