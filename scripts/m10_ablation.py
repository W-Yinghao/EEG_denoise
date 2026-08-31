#!/usr/bin/env python
"""M10 — what makes the conditional diffusion denoiser match the supervised CNNs? (single-channel EOG/EMG)

A controlled ablation on the EEGdenoiseNet paired benchmark (RRMSE_t/RRMSE_s/CC vs clean ground truth),
isolating each factor that separated our diffusion denoiser (CondDiff CC 0.842) from SimpleCNN (0.913):

  A1  eps,  TRAIN-mode sampling (dropout ON)   full-gen  K=1   — reproduces the buggy status quo
  A2  eps,  EVAL-mode  sampling (dropout OFF)   full-gen  K=1   — isolates the B1 dropout bug/unfairness
  A3  x0,   eval                                full-gen  K=1   — isolates eps -> x0 parameterization
  A4  x0 + EMA, eval                            full-gen  K=1   — isolates EMA of weights
  A5  x0 + EMA, eval   t* sweep {full,800,600,400,200}         — warm-start vs full conditional generation
  A6  x0 + EMA, unit-variance-clean training    full-gen  K=1   — isolates the clean-RMS-<<1 off-design
  R1/R2  SimpleCNN / ComplexCNN (supervised regressors)        — the reference to match/beat

Trains 3 diffusion U-Nets (eps; x0+EMA; x0+EMA in scaled space) + 2 CNNs; every other arm is an
inference-time toggle on those. NOTE (from the M9 ensemble probe): K-sample averaging is a no-op here
(the conditional sampler is already near the posterior mean), so K is fixed at 1. GPU node.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from saddpm.baselines.dl_denoisers import denoise_with, make_denoiser, train_denoiser  # noqa: E402
from saddpm.data.eegdenoisenet import EEGDenoiseConfig, prepare_pairs  # noqa: E402
from saddpm.diffusion.conditional import ConditionalDiffusionDenoiser  # noqa: E402
from saddpm.diffusion.gaussian_diffusion import GaussianDiffusion  # noqa: E402
from saddpm.diffusion.schedule import DiffusionConfig  # noqa: E402
from saddpm.eval.denoise_metrics import benchmark_by_snr  # noqa: E402
from saddpm.models.config import ModelConfig  # noqa: E402
from saddpm.models.unet1d import UNet1D  # noqa: E402
from saddpm.utils.checkpoint import save_checkpoint  # noqa: E402
from saddpm.utils.ema import EMA  # noqa: E402
from saddpm.utils.seed import seed_everything  # noqa: E402

CONFIGS = REPO_ROOT / "configs"


def _make_cond(param, seg_len, diff_cfg, device):
    unet = UNet1D(ModelConfig(in_channels=2, out_channels=1, signal_length=seg_len), subject_conditioned=False)
    return ConditionalDiffusionDenoiser(unet.to(device), GaussianDiffusion(diff_cfg).to(device), parameterization=param).to(device)


def train_cond(clean, noisy, param, cfg, diff_cfg, device, epochs, cosine=True, ema_decay=0.999):
    """Train a conditional denoiser; return (model, ema). EMA shadow tracked from step 0."""
    model = _make_cond(param, cfg.segment_len, diff_cfg, device)
    c = torch.from_numpy(clean).float().unsqueeze(1)
    nz = torch.from_numpy(noisy).float().unsqueeze(1)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(c, nz),
                                         batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader)) if cosine else None
    ema = EMA(model, decay=ema_decay)
    model.train()
    for _ in range(epochs):
        for cb, nb in loader:
            loss = model.loss(cb.to(device), nb.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            if sched is not None:
                sched.step()
            ema.update(model)
    return model, ema


@torch.no_grad()
def cond_denoise(model, noisy, device, t_star, ddim_steps=50, k=1, scale=1.0, train_mode=False, batch=256):
    """Denoise (N,L) noisy with the conditional model. scale: divide noisy/output for unit-var arm."""
    model.train() if train_mode else model.eval()
    out = []
    for i in range(0, len(noisy), batch):
        y = torch.from_numpy(noisy[i:i + batch]).float().unsqueeze(1).to(device) * scale
        d = model.denoise(y, ddim_steps=ddim_steps, t_star=t_star, k=k)
        out.append((d.squeeze(1) / scale).cpu().numpy())
    return np.concatenate(out)


def evaluate(name, denoised, p, cfg, rows):
    r = benchmark_by_snr(denoised, p.clean_test, p.test_snr_db, p.snr_levels_db, cfg.fs, cfg.psd_max_hz)
    rows.append((name, r.overall["rrmse_t"], r.overall["rrmse_s"], r.overall["cc"]))
    print(f"  {name:34s} RRMSE_t={r.overall['rrmse_t']:.4f} RRMSE_s={r.overall['rrmse_s']:.4f} CC={r.overall['cc']:.4f}")
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--noise", choices=["EOG", "EMG"], default="EOG")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = EEGDenoiseConfig.from_yaml(CONFIGS / "eegdenoise.yaml")
    if args.epochs is not None:
        cfg = EEGDenoiseConfig(**{**cfg.__dict__, "epochs": args.epochs})
    if args.smoke:
        cfg = EEGDenoiseConfig(**{**cfg.__dict__, "epochs": 1})
        diff_cfg = DiffusionConfig(num_timesteps=50)
    else:
        diff_cfg = DiffusionConfig.from_yaml(CONFIGS / "diffusion.yaml")
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = diff_cfg.num_timesteps
    full = None  # t_star=None -> full generation from pure noise

    p = prepare_pairs(args.noise, cfg)
    if args.smoke:
        p.noisy_train, p.clean_train = p.noisy_train[:256], p.clean_train[:256]
        keep = np.concatenate([np.where(p.test_snr_db == db)[0][:30] for db in p.snr_levels_db[:3]])
        p.noisy_test, p.clean_test, p.test_snr_db = p.noisy_test[keep], p.clean_test[keep], p.test_snr_db[keep]
        p.snr_levels_db = p.snr_levels_db[:3]
    print(f"[M10:{args.noise}] device={device} epochs={cfg.epochs} train{p.noisy_train.shape} test{p.noisy_test.shape}")
    scale = float(1.0 / np.median(np.sqrt((p.clean_train ** 2).mean(-1))))  # unit-variance-clean scale
    print(f"[M10:{args.noise}] clean-RMS unit-variance scale = {scale:.3f}")
    ddim = 4 if args.smoke else 50
    t_sweep = ([int(0.6 * T)] if args.smoke
               else [int(round(0.95 * T)), int(0.8 * T), int(0.6 * T), int(0.4 * T), int(0.2 * T)])
    rows: list = []

    # ---- A1/A2: eps model (reproduce status quo + isolate the dropout bug) ----
    eps_model, _ = train_cond(p.clean_train, p.noisy_train, "eps", cfg, diff_cfg, device, cfg.epochs, cosine=False)
    save_checkpoint(REPO_ROOT / f"artifacts/checkpoints/m10_{args.noise}_eps.pt", eps_model,
                    {"model": {"in_channels": 2, "out_channels": 1, "signal_length": cfg.segment_len},
                     "diffusion": diff_cfg.__dict__, "parameterization": "eps"})
    print("[M10] eps model trained")
    evaluate("A1 eps,train-mode(dropout),fullgen", cond_denoise(eps_model, p.noisy_test, device, full, ddim_steps=ddim, train_mode=True), p, cfg, rows)
    evaluate("A2 eps,eval-mode,fullgen", cond_denoise(eps_model, p.noisy_test, device, full, ddim_steps=ddim), p, cfg, rows)

    # ---- A3/A4/A5: x0 model + EMA ----
    x0_model, x0_ema = train_cond(p.clean_train, p.noisy_train, "x0", cfg, diff_cfg, device, cfg.epochs, cosine=True)
    save_checkpoint(REPO_ROOT / f"artifacts/checkpoints/m10_{args.noise}_x0.pt", x0_model,
                    {"model": {"in_channels": 2, "out_channels": 1, "signal_length": cfg.segment_len},
                     "diffusion": diff_cfg.__dict__, "parameterization": "x0"})
    print("[M10] x0 model trained")
    evaluate("A3 x0,eval,fullgen", cond_denoise(x0_model, p.noisy_test, device, full, ddim_steps=ddim), p, cfg, rows)
    x0_ema.copy_to(x0_model)  # swap in EMA weights for all subsequent x0 arms
    evaluate("A4 x0+EMA,eval,fullgen", cond_denoise(x0_model, p.noisy_test, device, full, ddim_steps=ddim), p, cfg, rows)
    sweep_rows = []
    for ts in [full] + t_sweep:
        label = "full" if ts is None else str(ts)
        r = evaluate(f"A5 x0+EMA,t*={label}", cond_denoise(x0_model, p.noisy_test, device, ts, ddim_steps=ddim), p, cfg, rows)
        sweep_rows.append((label, r.overall["rrmse_t"], r.overall["rrmse_s"], r.overall["cc"]))

    # ---- A6: x0+EMA trained in unit-variance-clean space ----
    cs, ns = p.clean_train * scale, p.noisy_train * scale
    norm_model, norm_ema = train_cond(cs, ns, "x0", cfg, diff_cfg, device, cfg.epochs, cosine=True)
    norm_ema.copy_to(norm_model)
    print("[M10] x0+norm model trained")
    best_ts = full  # full-gen is the natural single-channel setting; report it (sweep done above for A5)
    evaluate("A6 x0+EMA+unitvar,eval,fullgen", cond_denoise(norm_model, p.noisy_test, device, best_ts, ddim_steps=ddim, scale=scale), p, cfg, rows)

    # ---- R1/R2: supervised CNN references ----
    for arch, label, ep in [("simple_cnn", "R1 SimpleCNN", 40), ("complex_cnn", "R2 ComplexCNN", 40)]:
        ep = 1 if args.smoke else ep
        m = train_denoiser(make_denoiser(arch, cfg.segment_len), p.noisy_train, p.clean_train, ep, cfg.batch_size, cfg.lr, device, cfg.seed)
        evaluate(label, denoise_with(m, p.noisy_test, device), p, cfg, rows)
    evaluate("A0 Noisy(in)", p.noisy_test, p, cfg, rows)

    # ---- write results ----
    out_dir = REPO_ROOT / "results" / "m10"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# M10 {args.noise}: conditional-diffusion ablation vs supervised CNNs (RRMSE/CC vs clean)\n",
             "arm,RRMSE_temporal,RRMSE_spectral,CC\n"]
    lines += [f"{n},{rt:.4f},{rs:.4f},{cc:.4f}\n" for n, rt, rs, cc in rows]
    (out_dir / f"{args.noise}_ablation.csv").write_text("".join(lines))
    sweep_lines = ["# M10 t* sweep (x0+EMA), CC vs clean\nt_star,RRMSE_temporal,RRMSE_spectral,CC\n"]
    sweep_lines += [f"{ts},{rt:.4f},{rs:.4f},{cc:.4f}\n" for ts, rt, rs, cc in sweep_rows]
    (out_dir / f"{args.noise}_tstar_sweep.csv").write_text("".join(sweep_lines))

    fig, ax = plt.subplots(figsize=(7, 4))
    xs = [0 if ts == "full" else int(ts) for ts, *_ in sweep_rows]
    ax.plot(xs, [cc for *_, cc in sweep_rows], "o-", label="x0+EMA diffusion (t* sweep)")
    cnn_cc = [cc for n, *_, cc in rows if n.startswith("R1")]
    if cnn_cc:
        ax.axhline(cnn_cc[0], ls="--", color="k", label="SimpleCNN (regressor)")
    ax.set_xlabel("t* (0 = full generation from noise)"); ax.set_ylabel("CC"); ax.set_title(f"M10 {args.noise}: CC vs t*"); ax.legend()
    fig.tight_layout(); fig.savefig(REPO_ROOT / "artifacts" / "figures" / f"m10_{args.noise}_tstar.png", dpi=130)
    print(f"[M10:{args.noise}] done -> results/m10/{args.noise}_ablation.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
