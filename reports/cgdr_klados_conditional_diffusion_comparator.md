# Klados matched conditional-diffusion comparator

Status: implementation prepared, not executed. This document defines an
exploratory source-record comparison and is not formal G1 or G3 evidence.

## Scientific question

The existing M1/M2/M4 arms use an unconditional clean-EEG prior followed by an
observation/sampler mechanism. Their result cannot determine whether a model
trained directly as a conditional diffusion model adds value over a matched
deterministic network. The new arm therefore estimates epsilon conditionally on
the observed EEG, the selected projector, framewise external-EOG attenuation,
and the valid-time mask. It is a different model from current CGDR/M2, while
reusing the same masked multichannel U-Net and Gaussian diffusion code.

## Frozen fairness contract

- Training source records are sim01--sim30. Development diagnostics are
  sim31--sim36, sim44, and sim45. No historical/evaluation record is accepted.
- Records are source records, not participants; all output remains exploratory.
- The deterministic v3 common matching-P0 eligibility set is reused for every
  operator scope. Population, matching, and query-derived-oracle checkpoints
  therefore see the same source records and window counts within a partition.
- The exact deterministic window-bundle builder supplies observed EEG, paired
  clean target, projector, attenuation, and mask to both models.
- Each operator scope has an independent conditional checkpoint. A matching
  rejection cannot silently become a population-conditioned training sample.
- The conditional target optimizer-update count is read from the validated
  same-scope deterministic `best.pt`; it is not approximated by the nominal
  3000--6000 range. Batch size, order-seed rule, optimizer, learning rate,
  weight decay, AMP policy, and worker count are inherited from that protocol.
- The training objectives are intentionally different and are always labelled:
  valid-time-masked epsilon MSE versus the deterministic paired task loss.
- Development loss is diagnostic only. It does not early-stop or select the
  conditional checkpoint; the terminal checkpoint is the exact matched-update
  checkpoint.
- Development inference uses deterministic DDIM100 (`eta=0`) and the same five
  algorithmic seeds. The posterior mean is scored once per source record; seeds
  are not independent statistical units.
- Query-derived oracle projectors remain nondeployable mechanism upper bounds.
  They cannot be presented as a practical method.

## Implementation

- `src/eeg_cgdr/models/conditional_diffusion.py` defines the masked joint
  multichannel conditional epsilon model. Its conditioning stack is byte-for-
  byte constructed in the same order as the deterministic U-Net stack; only
  `x_t` is prepended as diffusion algorithm state.
- `src/eeg_cgdr/experiments/stage3_conditional_diffusion.py` validates the
  deterministic v3 checkpoint/split, constructs common-eligible bundles,
  trains with atomic checkpoint/resume, runs development DDIM, and reports
  complete runtime/budget fields.
- `configs/cgdr/klados_stage3_conditional_diffusion.yaml` freezes the protocol.
- `tests/unit/test_cgdr_conditional_diffusion.py` covers visible-input equality,
  exact conditioning-stack equality, internal padding invariance, masked loss,
  exact DDIM-call accounting, and the development-only protocol boundary.
- `src/eeg_cgdr/cli/main.py`, `scripts/slurm/submit.sh`, and
  `scripts/slurm/jobs/cgdr.sbatch` expose one fail-closed
  `stage3-conditional-diffusion` route. Static route tests are in
  `tests/unit/test_cgdr_conditional_diffusion_routes.py`.

Every result row reports training updates, model size, latency, peak memory,
network calls, algorithmic seed count, eligibility, and operator scope. The
development aggregate pairs conditional and deterministic U-Net cells only
when record and operator scope both match and asserts equal optimizer updates.

## Required execution order

The intended dependency chain is:

```text
deterministic v3 CPU validation
  -> deterministic v3 three-scope training
  -> conditional three-scope training
  -> deterministic and conditional development arrays
  -> conditional paired development aggregation after both arrays
```

Training and inference must use the GPU Conda environment through Slurm; the
aggregate must use the CPU environment through Slurm. Checkpoints remain
outside Git under
`results/cgdr/klados_stage3_conditional_diffusion_matched_v1/checkpoints/`.

## Interpretation boundary

Even a complete run can only say whether this tested operator-conditioned
diffusion configuration shows incremental development-record value relative to
the matched deterministic U-Net. It cannot establish participant-level
generalization, formal G3, or a claim about the full EEG diffusion family. A
fresh frozen evaluation protocol would still be required for a confirmatory
claim.
