# SADDPM — Subject-Aware Diffusion for Cross-Subject EEG Denoising (BCI-IV-2a)

Re-implementation of **SADDPM** (subject-conditional DDPM with a dual decoder + three losses)
for cross-subject EEG denoising on **BCI Competition IV-2a**, following
[SADDPM_IMPLEMENTATION_HANDOFF.md](SADDPM_IMPLEMENTATION_HANDOFF.md) (authoritative spec).
See [PLAN.md](PLAN.md) for the milestone roadmap and [RESULTS.md](RESULTS.md) for the
assumptions ledger and logged numbers.

## Status

- [x] **M0** — env + data: `check_env.py` passes; BCI-IV-2a loaded via MOABB; §3 preprocessing;
      one preprocessed window plotted.
- [ ] M1 — diffusion core (schedule + `q_sample` + numerical marginal check)
- [ ] M2 — 1D U-Net (single decoder), overfit one batch
- [ ] M3 — subject conditioning (embeddings + FiLM)
- [ ] M4 — dual decoder + 3 losses + ArcFace
- [ ] M5 — SDEdit denoising
- [ ] M6 — EEGNet downstream + ICA baseline
- [ ] M7 — full 9×9 sweep + subject-correlation matrices

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
