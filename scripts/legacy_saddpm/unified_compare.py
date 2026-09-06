#!/usr/bin/env python
"""Unified SADDPM ↔ EEGdenoiseNet head-to-head on common multi-channel BCI ground truth.

Builds synthetic-artifact BCI-IV-2a pairs (multi-channel BCI + injected EEGdenoiseNet artifact at
controlled SNR = ground truth), then denoises with EVERY method and scores RRMSE_t/s + CC:
  * Noisy            — no-op reference
  * SADDPM-SDEdit    — the manuscript's method (M4 checkpoint, SDEdit, subject-conditioned)
  * SADDPM-Cond      — our conditional subject-aware denoiser (M9 checkpoint)
  * EEGdenoiseNet nets (SimpleCNN, NovelCNN) — trained per-channel on the same pairs
This is the apples-to-apples comparison the two papers otherwise never share. Run on a GPU node.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from saddpm.baselines.dl_denoisers import denoise_with, make_denoiser, train_denoiser  # noqa: E402
from saddpm.data.config import DataConfig  # noqa: E402
from saddpm.data.eegdenoisenet import EEGDenoiseConfig, load_components  # noqa: E402
from saddpm.data.synthetic_artifacts import build_synthetic_pairs  # noqa: E402
from saddpm.diffusion.gaussian_diffusion import GaussianDiffusion  # noqa: E402
from saddpm.diffusion.schedule import DiffusionConfig  # noqa: E402
from saddpm.eval.denoise_metrics import correlation_coefficient, rrmse_spectral, rrmse_temporal  # noqa: E402
from saddpm.eval.pairwise import saddpm_denoise_windows  # noqa: E402
from saddpm.models.cond_denoiser import SubjectConditionalDenoiser  # noqa: E402
from saddpm.models.config import ModelConfig  # noqa: E402
from saddpm.models.dual_decoder import DualDecoderSADDPM  # noqa: E402
from saddpm.models.unet1d import UNet1D  # noqa: E402
from saddpm.utils.checkpoint import load_checkpoint  # noqa: E402
from saddpm.utils.seed import seed_everything  # noqa: E402

CONFIGS = REPO_ROOT / "configs"


def _metrics(denoised: np.ndarray, clean: np.ndarray, fs: int) -> dict:
    d, c = denoised.reshape(-1, denoised.shape[-1]), clean.reshape(-1, clean.shape[-1])
    return {"rrmse_t": float(rrmse_temporal(d, c).mean()),
            "rrmse_s": float(rrmse_spectral(d, c, fs).mean()),
            "cc": float(correlation_coefficient(d, c).mean())}


def _per_channel_denoise(arch, train_co, train_cl, test_co, epochs, device, seed):
    """Train an EEGdenoiseNet single-channel net on flattened per-channel pairs; denoise test."""
    n, c, length = train_co.shape
    model = make_denoiser(arch, length)
    model = train_denoiser(model, train_co.reshape(-1, length), train_cl.reshape(-1, length),
                           epochs, 128, 1e-4, device, seed)
    out = denoise_with(model, test_co.reshape(-1, length), device)
    return out.reshape(test_co.shape)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise", choices=["EOG", "EMG"], default="EOG")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--eval-subsample", type=int, default=1800)
    args = parser.parse_args()

    data_cfg = DataConfig.from_yaml(CONFIGS / "data.yaml")
    eegdn_cfg = EEGDenoiseConfig.from_yaml(CONFIGS / "eegdenoise.yaml")
    seed_everything(data_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fs = data_cfg.preprocess.resample_hz
    subjects = list(range(1, data_cfg.dataset.n_subjects + 1))

    eeg, eog, emg = load_components(eegdn_cfg)
    artifacts = {"EOG": eog, "EMG": emg}[args.noise]
    train = build_synthetic_pairs(subjects, data_cfg, artifacts, args.noise, "T",
                                  tuple(eegdn_cfg.snr_train_db), data_cfg.seed)
    test = build_synthetic_pairs(subjects, data_cfg, artifacts, args.noise, "E",
                                 tuple(eegdn_cfg.snr_train_db), data_cfg.seed + 1,
                                 fixed_snr_levels=[-7.0, -4.0, -1.0, 2.0])
    rng = np.random.default_rng(0)
    idx = rng.choice(len(test.clean), size=min(args.eval_subsample, len(test.clean)), replace=False)
    co_te, cl_te, sid_te = test.corrupted[idx], test.clean[idx], test.subject_ids[idx]
    print(f"[unified:{args.noise}] device={device} train {train.clean.shape} test(eval) {co_te.shape}")

    denoised = {"Noisy": co_te}

    # SADDPM-SDEdit (manuscript method, M4 checkpoint).
    ck = load_checkpoint(REPO_ROOT / "artifacts/checkpoints/m4_saddpm.pt", map_location=str(device))
    saddpm = DualDecoderSADDPM(ModelConfig(**ck["config"]["model"])).to(device).eval()
    saddpm.load_state_dict(ck["model_state"])
    diff = GaussianDiffusion(DiffusionConfig(**ck["config"]["diffusion"])).to(device)
    # denoise each subject's slice with its own embedding.
    out = np.empty_like(co_te)
    for s in range(data_cfg.dataset.n_subjects):
        m = sid_te == s
        if m.any():
            out[m] = saddpm_denoise_windows(saddpm, diff, co_te[m], s, 200, 50, device)
    denoised["SADDPM-SDEdit"] = out

    # SADDPM-Conditional (M9 checkpoint), if present.
    m9_path = REPO_ROOT / f"artifacts/checkpoints/m9_{args.noise}.pt"
    if m9_path.exists():
        ck9 = load_checkpoint(m9_path, map_location=str(device))
        unet = UNet1D(ModelConfig(**ck9["config"]["model"]), subject_conditioned=True).to(device)
        cdn = SubjectConditionalDenoiser(unet, GaussianDiffusion(DiffusionConfig(**ck9["config"]["diffusion"])).to(device)).to(device)
        cdn.load_state_dict(ck9["model_state"])
        cdn.eval()
        co_t = torch.from_numpy(co_te).float()
        sid_t = torch.from_numpy(sid_te).long()
        with torch.no_grad():
            cond_out = np.concatenate([cdn.denoise(co_t[i:i+128].to(device), sid_t[i:i+128].to(device), 50).cpu().numpy()
                                       for i in range(0, len(co_t), 128)])
        denoised["SADDPM-Cond"] = cond_out

    # EEGdenoiseNet nets (per-channel).
    for arch, label in [("simple_cnn", "EEGdN-SimpleCNN"), ("novel_cnn", "EEGdN-NovelCNN")]:
        denoised[label] = _per_channel_denoise(arch, train.corrupted, train.clean, co_te, args.epochs, device, data_cfg.seed)
        print(f"[unified:{args.noise}] {label} trained+denoised")

    out_dir = REPO_ROOT / "results" / "unified"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# Unified SADDPM vs EEGdenoiseNet on synthetic-artifact BCI ({args.noise}), RRMSE/CC vs ground truth\n",
             "method,RRMSE_temporal,RRMSE_spectral,CC\n"]
    print("\n========== UNIFIED COMPARISON ==========")
    for name, d in denoised.items():
        mt = _metrics(d, cl_te, fs)
        lines.append(f"{name},{mt['rrmse_t']:.4f},{mt['rrmse_s']:.4f},{mt['cc']:.4f}\n")
        print(f"  {name:18s} RRMSE_t={mt['rrmse_t']:.3f} RRMSE_s={mt['rrmse_s']:.3f} CC={mt['cc']:.3f}")
    (out_dir / f"{args.noise}_unified.csv").write_text("".join(lines))
    print(f"saved -> results/unified/{args.noise}_unified.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
