#!/usr/bin/env python
"""Probe: does posterior-mean ENSEMBLING close the diffusion-vs-CNN gap? (no retraining)

Loads an existing M9 subject-conditional denoiser checkpoint (x0-parameterized) and denoises the
held-out test pairs K times with independent sampling noise, then AVERAGES the K denoised outputs.
Averaging over the stochastic sampler noise is a Monte-Carlo estimate of the posterior mean
E[clean | corrupted], which is exactly what RRMSE_temporal / CC reward. If CC rises with K, the
sample-vs-mean gap is confirmed as the dominant cause of the diffusion denoiser underperforming the
supervised CNNs — and ensembling (or training a mean estimator) is the fix.

Sweeps (t*, K) on the correct subject embedding. Read-only w.r.t. the model; cheap. GPU node.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from saddpm.data.config import DataConfig  # noqa: E402
from saddpm.data.eegdenoisenet import EEGDenoiseConfig, load_components  # noqa: E402
from saddpm.data.synthetic_artifacts import build_synthetic_pairs  # noqa: E402
from saddpm.diffusion.gaussian_diffusion import GaussianDiffusion  # noqa: E402
from saddpm.diffusion.schedule import DiffusionConfig  # noqa: E402
from saddpm.eval.denoise_metrics import correlation_coefficient, rrmse_spectral, rrmse_temporal  # noqa: E402
from saddpm.models.cond_denoiser import SubjectConditionalDenoiser  # noqa: E402
from saddpm.models.config import ModelConfig  # noqa: E402
from saddpm.models.unet1d import UNet1D  # noqa: E402
from saddpm.utils.checkpoint import load_checkpoint  # noqa: E402
from saddpm.utils.seed import seed_everything  # noqa: E402

CONFIGS = REPO_ROOT / "configs"


def _metrics(denoised: np.ndarray, clean: np.ndarray, fs: int) -> dict:
    d = denoised.reshape(-1, denoised.shape[-1])
    c = clean.reshape(-1, clean.shape[-1])
    return {"rrmse_t": float(rrmse_temporal(d, c).mean()),
            "rrmse_s": float(rrmse_spectral(d, c, fs).mean()),
            "cc": float(correlation_coefficient(d, c).mean())}


@torch.no_grad()
def _denoise_avg(model, corr, sid, k, t_star, ddim_steps, device, batch=128):
    """Average k independent denoise() draws (fresh sampler noise each) -> posterior-mean estimate."""
    ts = None if t_star <= 0 else t_star  # t_star<=0 => full generation from pure noise
    acc = np.zeros_like(corr)
    for _ in range(k):
        out = []
        for i in range(0, len(corr), batch):
            cb = torch.from_numpy(corr[i:i + batch]).float().to(device)
            sb = torch.from_numpy(sid[i:i + batch]).long().to(device)
            out.append(model.denoise(cb, sb, ddim_steps=ddim_steps, t_star=ts).cpu().numpy())
        acc += np.concatenate(out)
    return acc / k


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise", choices=["EOG", "EMG"], default="EOG")
    parser.add_argument("--n-eval", type=int, default=800)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--t-stars", type=int, nargs="+", default=[200, 400])
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 4, 8, 16])
    args = parser.parse_args()

    data_cfg = DataConfig.from_yaml(CONFIGS / "data.yaml")
    eegdn_cfg = EEGDenoiseConfig.from_yaml(CONFIGS / "eegdenoise.yaml")
    seed_everything(data_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fs = data_cfg.preprocess.resample_hz

    ck = load_checkpoint(REPO_ROOT / f"artifacts/checkpoints/m9_{args.noise}.pt", map_location=str(device))
    mcfg = ModelConfig(**ck["config"]["model"])
    unet = UNet1D(mcfg, subject_conditioned=True).to(device)
    model = SubjectConditionalDenoiser(unet, GaussianDiffusion(DiffusionConfig(**ck["config"]["diffusion"])).to(device)).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    eeg, eog, emg = load_components(eegdn_cfg)
    artifacts = {"EOG": eog, "EMG": emg}[args.noise]
    subjects = list(range(1, data_cfg.dataset.n_subjects + 1))
    test = build_synthetic_pairs(subjects, data_cfg, artifacts, args.noise, "E",
                                 tuple(eegdn_cfg.snr_train_db), data_cfg.seed + 1,
                                 fixed_snr_levels=[-7.0, -4.0, -1.0, 2.0])
    rng = np.random.default_rng(0)
    idx = rng.choice(len(test.clean), size=min(args.n_eval, len(test.clean)), replace=False)
    corr, clean, sid = test.corrupted[idx], test.clean[idx], test.subject_ids[idx]
    print(f"[probe:{args.noise}] device={device} eval {corr.shape} ddim={args.ddim_steps}")

    base = _metrics(corr, clean, fs)
    print(f"  Corrupted(in)              RRMSE_t={base['rrmse_t']:.3f} RRMSE_s={base['rrmse_s']:.3f} CC={base['cc']:.3f}")
    print("  --- denoised: averaging K independent sampler draws (posterior-mean estimate) ---")
    print(f"  {'t*':>4} {'K':>3} | {'RRMSE_t':>8} {'RRMSE_s':>8} {'CC':>7}")
    for t_star in args.t_stars:
        for k in args.ks:
            d = _denoise_avg(model, corr, sid, k, t_star, args.ddim_steps, device)
            m = _metrics(d, clean, fs)
            print(f"  {t_star:>4} {k:>3} | {m['rrmse_t']:>8.4f} {m['rrmse_s']:>8.4f} {m['cc']:>7.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
