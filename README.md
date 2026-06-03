# SADDPM — Subject-Aware Diffusion for Cross-Subject EEG Denoising (BCI-IV-2a)

Re-implementation of **SADDPM** (subject-conditional DDPM with a dual decoder + three losses)
for cross-subject EEG denoising on **BCI Competition IV-2a**, following
[SADDPM_IMPLEMENTATION_HANDOFF.md](SADDPM_IMPLEMENTATION_HANDOFF.md) (authoritative spec).
See [PLAN.md](PLAN.md) for the milestone roadmap and [RESULTS.md](RESULTS.md) for the
assumptions ledger and logged numbers.

## Status

- [x] **M0** — env + data: `check_env.py` passes; BCI-IV-2a loaded via MOABB; §3 preprocessing;
      one preprocessed window plotted.
- [x] **M1** — diffusion core: schedule + `q_sample`; numerical forward-marginal check passes.
- [x] **M2** — 1D U-Net (single decoder) + DDPM sampling; overfit one batch (loss 1.10→0.03) on V100.
- [x] **M3** — subject conditioning (embeddings + FiLM); conditioning changes generated samples.
- [x] **M4** — dual decoder + 3 losses + ArcFace; subject acc 0.937 on held-out Session-E.
- [x] **M5** — SDEdit denoising; t* sweep regularises monotonically.
- [x] **M6** — EEGNet downstream + ICA baseline (one pair, both end-to-end).
- [x] **M7** — full 9×9 sweep + subject-correlation matrices. **Phase 1 complete.**

### Phase-1 headline (M7, 4-class, chance 0.25; honest re-implementation — see [RESULTS.md](RESULTS.md))

| Denoiser | grand mean | within-subject diag | spread (per-target std) |
|----------|-----------|---------------------|--------------------------|
| SADDPM | 0.276 | 0.344 | **0.033** (lower spread) |
| ICA | 0.284 | 0.393 | 0.047 |

Within-subject accuracy is well above chance; cross-subject is near chance for both. SADDPM shows
the lower spread the paper claims; ICA is marginally ahead on mean accuracy. Not a bit-exact repro.

## Environment

On this server we use the conda env **`eeg2025`** (Python 3.13.7) with `moabb` added
(`pip install moabb`, purely additive). A portable spec is in [environment.yml](environment.yml).

```bash
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
$PY scripts/check_env.py                 # env + dataset sanity (login node)
$PY scripts/check_env.py --probe-subject 1   # also parse a full subject via MOABB
$PY scripts/m0_load_subject.py --subject 1    # M0: load + print shapes + plot a window
$PY -m pytest tests/ -q                        # unit tests
```

GPU training runs as **Slurm** jobs on partition `V100` (the login node `nodecpu11` has no GPU).
Inside GPU jobs, `check_env.py --require-cuda` fails loudly if CUDA is unavailable.

## Layout

```
configs/    YAML configs (data; model/train/eval added per milestone)
saddpm/     library: data/ models/ diffusion/ losses/ baselines/ eval/ utils/
scripts/    check_env, m0_load_subject, (training/eval added per milestone), slurm/
tests/      unit tests + numerical sanity checks
artifacts/  figures, checkpoints, run CSVs (gitignored except small figures)
```
