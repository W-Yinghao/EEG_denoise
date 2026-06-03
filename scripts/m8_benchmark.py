#!/usr/bin/env python
"""M8 EEGdenoiseNet benchmark for one artifact type (EOG or EMG).

Trains and evaluates, on identical paired data with ground truth:
  * SDEdit         — unconditional diffusion prior over clean EEG + SDEdit denoising (our Phase-1 method)
  * CondDiff       — noisy-conditioned diffusion denoiser
  * SimpleCNN, NovelCNN — supervised DL baselines (EEGdenoiseNet)
  * Noisy          — no-op reference
Reports RRMSE_temporal, RRMSE_spectral and CC per SNR level (−7…2 dB). Run on a GPU node.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from saddpm.baselines.dl_denoisers import denoise_with, make_denoiser, train_denoiser  # noqa: E402
from saddpm.data.eegdenoisenet import EEGDenoiseConfig, prepare_pairs  # noqa: E402
from saddpm.diffusion.conditional import ConditionalDiffusionDenoiser  # noqa: E402
from saddpm.diffusion.gaussian_diffusion import GaussianDiffusion  # noqa: E402
from saddpm.diffusion.schedule import DiffusionConfig  # noqa: E402
from saddpm.eval.denoise_metrics import benchmark_by_snr  # noqa: E402
from saddpm.losses.recon import reconstruction_loss  # noqa: E402
from saddpm.models.config import ModelConfig  # noqa: E402
from saddpm.models.unet1d import UNet1D  # noqa: E402
from saddpm.utils.logging import RunLogger  # noqa: E402
from saddpm.utils.seed import seed_everything  # noqa: E402

CONFIGS = REPO_ROOT / "configs"


def _train_prior(clean_train, cfg, diff_cfg, device, logger):
    """Unconditional 1-ch DDPM over clean EEG (for SDEdit)."""
    model = UNet1D(ModelConfig(in_channels=1, out_channels=1, signal_length=cfg.segment_len),
                   subject_conditioned=False).to(device)
    diffusion = GaussianDiffusion(diff_cfg).to(device)
    x = torch.from_numpy(clean_train).float().unsqueeze(1)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x), batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    step = 0
    for ep in range(cfg.epochs):
        for (xb,) in loader:
            xb = xb.to(device)
            t = torch.randint(0, diffusion.num_timesteps, (xb.shape[0],), device=device)
            noise = torch.randn_like(xb)
            loss = reconstruction_loss(model(diffusion.q_sample(xb, t, noise), t), noise)
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
            if step % 200 == 0:
                logger.log(step, {"prior_loss": loss.item()})
    return model, diffusion


def _train_cond(clean_train, noisy_train, cfg, diff_cfg, device, logger):
    unet = UNet1D(ModelConfig(in_channels=2, out_channels=1, signal_length=cfg.segment_len),
                  subject_conditioned=False).to(device)
    cdd = ConditionalDiffusionDenoiser(unet, GaussianDiffusion(diff_cfg).to(device)).to(device)
    c = torch.from_numpy(clean_train).float().unsqueeze(1)
    nz = torch.from_numpy(noisy_train).float().unsqueeze(1)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(c, nz), batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(cdd.parameters(), lr=cfg.lr)
    step = 0
    for ep in range(cfg.epochs):
        for cb, nb in loader:
            loss = cdd.loss(cb.to(device), nb.to(device))
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
            if step % 200 == 0:
                logger.log(step, {"cond_loss": loss.item()})
    return cdd


@torch.no_grad()
def _sdedit_denoise(model, diffusion, noisy, cfg, device, batch=256):
    out = []
    for i in range(0, len(noisy), batch):
        y = torch.from_numpy(noisy[i:i + batch]).float().unsqueeze(1).to(device)
        d = diffusion.sdedit(lambda x, t: model(x, t), y, t_star=cfg.sdedit_t_star, ddim_steps=cfg.ddim_steps)
        out.append(d.squeeze(1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def _cond_denoise(cdd, noisy, cfg, device, batch=256):
    out = []
    for i in range(0, len(noisy), batch):
        y = torch.from_numpy(noisy[i:i + batch]).float().unsqueeze(1).to(device)
        out.append(cdd.denoise(y, ddim_steps=cfg.ddim_steps).squeeze(1).cpu().numpy())
    return np.concatenate(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise", choices=["EOG", "EMG"], default="EOG")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="tiny CPU end-to-end check")
    args = parser.parse_args()

    cfg = EEGDenoiseConfig.from_yaml(CONFIGS / "eegdenoise.yaml")
    if args.epochs is not None:
        cfg = EEGDenoiseConfig(**{**cfg.__dict__, "epochs": args.epochs})
    if args.smoke:
        cfg = EEGDenoiseConfig(**{**cfg.__dict__, "epochs": 1, "sdedit_t_star": 20, "ddim_steps": 4})
        diff_cfg = DiffusionConfig(num_timesteps=50)
    else:
        diff_cfg = DiffusionConfig.from_yaml(CONFIGS / "diffusion.yaml")
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[M8:{args.noise}] device={device} epochs={cfg.epochs} t*={cfg.sdedit_t_star}")

    p = prepare_pairs(args.noise, cfg)
    if args.smoke:  # subsample a few epochs from each of 3 SNR levels for a fast glue check
        p.noisy_train, p.clean_train = p.noisy_train[:256], p.clean_train[:256]
        levels = p.snr_levels_db[:3]
        sub = np.concatenate([np.where(p.test_snr_db == db)[0][:40] for db in levels])
        p.noisy_test, p.clean_test, p.test_snr_db = p.noisy_test[sub], p.clean_test[sub], p.test_snr_db[sub]
        p.snr_levels_db = levels
    print(f"[M8:{args.noise}] train {p.noisy_train.shape} test {p.noisy_test.shape}")
    logger = RunLogger(REPO_ROOT / "artifacts" / "runs" / f"m8_{args.noise}", use_wandb=args.wandb,
                       wandb_project="saddpm", wandb_run_name=f"m8-{args.noise}", wandb_mode="offline",
                       config={"milestone": "M8", "noise": args.noise, "epochs": cfg.epochs})

    prior, diffusion = _train_prior(p.clean_train, cfg, diff_cfg, device, logger)
    print(f"[M8:{args.noise}] prior trained")
    cdd = _train_cond(p.clean_train, p.noisy_train, cfg, diff_cfg, device, logger)
    print(f"[M8:{args.noise}] cond-diffusion trained")
    # All 5 EEGdenoiseNet DL baselines (paper-specified epochs per architecture).
    baseline_epochs = {"fcnn": 60, "simple_cnn": 40, "complex_cnn": 40, "rnn_lstm": 30, "novel_cnn": 60}
    baselines = {}
    for arch in ("fcnn", "simple_cnn", "complex_cnn", "rnn_lstm", "novel_cnn"):
        ep = 1 if args.smoke else baseline_epochs[arch]
        m = make_denoiser(arch, cfg.segment_len)
        baselines[arch] = train_denoiser(m, p.noisy_train, p.clean_train, ep, cfg.batch_size, cfg.lr, device, cfg.seed)
        print(f"[M8:{args.noise}] baseline {arch} trained ({ep} ep)")

    denoised = {
        "Noisy": p.noisy_test,
        "SDEdit": _sdedit_denoise(prior, diffusion, p.noisy_test, cfg, device),
        "CondDiff": _cond_denoise(cdd, p.noisy_test, cfg, device),
        "FCNN": denoise_with(baselines["fcnn"], p.noisy_test, device),
        "SimpleCNN": denoise_with(baselines["simple_cnn"], p.noisy_test, device),
        "ComplexCNN": denoise_with(baselines["complex_cnn"], p.noisy_test, device),
        "RNN_LSTM": denoise_with(baselines["rnn_lstm"], p.noisy_test, device),
        "NovelCNN": denoise_with(baselines["novel_cnn"], p.noisy_test, device),
    }

    out_dir = REPO_ROOT / "results" / "m8"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_lines = [f"# M8 EEGdenoiseNet — {args.noise} (overall mean over SNR −7..2 dB)\n",
                     "method,RRMSE_temporal,RRMSE_spectral,CC\n"]
    results = {}
    for name, d in denoised.items():
        r = benchmark_by_snr(d, p.clean_test, p.test_snr_db, p.snr_levels_db, cfg.fs, cfg.psd_max_hz)
        results[name] = r
        summary_lines.append(f"{name},{r.overall['rrmse_t']:.4f},{r.overall['rrmse_s']:.4f},{r.overall['cc']:.4f}\n")
        print(f"  {name:10s} RRMSE_t={r.overall['rrmse_t']:.3f} RRMSE_s={r.overall['rrmse_s']:.3f} CC={r.overall['cc']:.3f}")
    (out_dir / f"{args.noise}_summary.csv").write_text("".join(summary_lines))

    # per-SNR figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, r in results.items():
        axes[0].plot(r.snr_levels_db, r.rrmse_t, "o-", label=name)
        axes[1].plot(r.snr_levels_db, r.rrmse_s, "o-", label=name)
        axes[2].plot(r.snr_levels_db, r.cc, "o-", label=name)
    for ax, title in zip(axes, ["RRMSE_temporal", "RRMSE_spectral", "CC"]):
        ax.set_xlabel("input SNR (dB)"); ax.set_title(f"{args.noise}: {title}"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(REPO_ROOT / "artifacts" / "figures" / f"m8_{args.noise}_metrics.png", dpi=130)
    plt.close(fig)
    logger.finish()
    print(f"[M8:{args.noise}] done -> results/m8/{args.noise}_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
