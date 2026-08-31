#!/usr/bin/env python
"""M11 — fair MULTI-CHANNEL baselines for the conditional-diffusion denoiser.

The multi-channel head-to-head (m9 reeval: SADDPM-Cond full-gen EOG 0.988 / EMG 0.994) beat the
EEGdenoiseNet CNNs (~0.85), but those CNNs are SINGLE-channel (per-channel) and cannot exploit the
artifact's spatial topography. This script asks the honest question: is the win from DIFFUSION, or
just from being MULTI-CHANNEL? It trains, on the same multi-channel pairs and ground truth:

  * RegUNet      — the SAME U-Net backbone as the diffusion denoiser, trained as a one-shot MSE
                   regressor (noisy -> clean). Isolates diffusion-vs-regression at equal capacity.
  * MC-ConvNet   — a multi-channel fully-convolutional regressor (22->22), an EEGdenoiseNet-style CNN
                   that DOES see all channels.
  * SADDPM-Cond  — the existing m9 diffusion checkpoint, full conditional generation (reference).
  * Noisy        — input reference.
All evaluated on the identical test windows (rng(0), 1800) used by m9_reeval. GPU node.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

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


class MCConvNet(nn.Module):
    """Multi-channel fully-convolutional denoiser (C->C), EEGdenoiseNet-style but spatial-aware."""

    def __init__(self, c: int, width: int = 64, k: int = 9) -> None:
        super().__init__()
        p = k // 2
        layers = [nn.Conv1d(c, width, k, padding=p), nn.BatchNorm1d(width), nn.ReLU()]
        for _ in range(4):
            layers += [nn.Conv1d(width, width, k, padding=p), nn.BatchNorm1d(width), nn.ReLU()]
        layers += [nn.Conv1d(width, c, k, padding=p)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _metrics(d, c, fs):
    d, c = d.reshape(-1, d.shape[-1]), c.reshape(-1, c.shape[-1])
    return {"rrmse_t": float(rrmse_temporal(d, c).mean()),
            "rrmse_s": float(rrmse_spectral(d, c, fs).mean()),
            "cc": float(correlation_coefficient(d, c).mean())}


def _train_regressor(model, noisy, clean, epochs, bs, lr, device, t_zero=False):
    """Train a direct MSE regressor (N,C,L)->(N,C,L). t_zero: model is a UNet1D needing a t arg."""
    x = torch.from_numpy(noisy).float()
    y = torch.from_numpy(clean).float()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=bs, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb, torch.zeros(xb.shape[0], device=device, dtype=torch.long)) if t_zero else model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model


@torch.no_grad()
def _apply(model, noisy, device, t_zero=False, batch=256):
    model.eval()
    out = []
    for i in range(0, len(noisy), batch):
        xb = torch.from_numpy(noisy[i:i + batch]).float().to(device)
        pred = model(xb, torch.zeros(xb.shape[0], device=device, dtype=torch.long)) if t_zero else model(xb)
        out.append(pred.cpu().numpy())
    return np.concatenate(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--noise", choices=["EOG", "EMG"], default="EOG")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-eval", type=int, default=1800)
    args = ap.parse_args()

    data_cfg = DataConfig.from_yaml(CONFIGS / "data.yaml")
    eegdn_cfg = EEGDenoiseConfig.from_yaml(CONFIGS / "eegdenoise.yaml")
    seed_everything(data_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fs = data_cfg.preprocess.resample_hz
    n_subj = data_cfg.dataset.n_subjects
    subjects = list(range(1, n_subj + 1))

    eeg, eog, emg = load_components(eegdn_cfg)
    artifacts = {"EOG": eog, "EMG": emg}[args.noise]
    train = build_synthetic_pairs(subjects, data_cfg, artifacts, args.noise, "T", tuple(eegdn_cfg.snr_train_db), data_cfg.seed)
    test = build_synthetic_pairs(subjects, data_cfg, artifacts, args.noise, "E", tuple(eegdn_cfg.snr_train_db),
                                 data_cfg.seed + 1, fixed_snr_levels=[-7.0, -4.0, -1.0, 2.0])
    c, L = train.clean.shape[1], train.clean.shape[2]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(test.clean), size=min(args.n_eval, len(test.clean)), replace=False)
    co, cl, sid = test.corrupted[idx], test.clean[idx], test.subject_ids[idx]
    print(f"[M11:{args.noise}] device={device} train{train.clean.shape} eval{co.shape}")

    rows = [("Noisy(in)", _metrics(co, cl, fs))]

    # RegUNet: same U-Net backbone as the diffusion denoiser, trained as a one-shot regressor.
    reg_unet = UNet1D(ModelConfig(in_channels=c, out_channels=c, signal_length=L), subject_conditioned=False).to(device)
    n_reg = sum(p.numel() for p in reg_unet.parameters())
    reg_unet = _train_regressor(reg_unet, train.corrupted, train.clean, args.epochs, 64, 1e-4, device, t_zero=True)
    rows.append((f"RegUNet(sameArch,{n_reg/1e6:.1f}M)", _metrics(_apply(reg_unet, co, device, t_zero=True), cl, fs)))
    print(f"[M11:{args.noise}] RegUNet trained+eval")

    # MC-ConvNet: multi-channel fully-conv regressor.
    mc = MCConvNet(c).to(device)
    mc = _train_regressor(mc, train.corrupted, train.clean, args.epochs, 128, 1e-4, device, t_zero=False)
    rows.append(("MC-ConvNet", _metrics(_apply(mc, co, device, t_zero=False), cl, fs)))
    print(f"[M11:{args.noise}] MC-ConvNet trained+eval")

    # SADDPM-Cond (existing m9 checkpoint), full conditional generation.
    ck = load_checkpoint(REPO_ROOT / f"artifacts/checkpoints/m9_{args.noise}.pt", map_location=str(device))
    unet = UNet1D(ModelConfig(**ck["config"]["model"]), subject_conditioned=True).to(device)
    diff = SubjectConditionalDenoiser(unet, GaussianDiffusion(DiffusionConfig(**ck["config"]["diffusion"])).to(device)).to(device)
    diff.load_state_dict(ck["model_state"]); diff.eval()
    out = []
    for i in range(0, len(co), 128):
        cb = torch.from_numpy(co[i:i + 128]).float().to(device)
        sb = torch.from_numpy(sid[i:i + 128]).long().to(device)
        out.append(diff.denoise(cb, sb, ddim_steps=50, t_star=None).cpu().numpy())
    rows.append(("SADDPM-Cond(full-gen)", _metrics(np.concatenate(out), cl, fs)))
    print(f"[M11:{args.noise}] SADDPM-Cond eval")

    out_dir = REPO_ROOT / "results" / "m11"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# M11 {args.noise}: fair multi-channel baselines (RRMSE/CC vs clean, same test windows)\n",
             "method,RRMSE_temporal,RRMSE_spectral,CC\n"]
    print("\n========== M11 MULTI-CHANNEL ==========")
    for name, m in rows:
        lines.append(f"{name},{m['rrmse_t']:.4f},{m['rrmse_s']:.4f},{m['cc']:.4f}\n")
        print(f"  {name:28s} RRMSE_t={m['rrmse_t']:.4f} RRMSE_s={m['rrmse_s']:.4f} CC={m['cc']:.4f}")
    (out_dir / f"{args.noise}_mc_baselines.csv").write_text("".join(lines))
    print(f"[M11:{args.noise}] done -> results/m11/{args.noise}_mc_baselines.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
