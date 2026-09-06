#!/usr/bin/env python
"""M2 milestone gate: overfit a single batch with the plain DDPM (handoff §11).

Loads one fixed batch of subject windows, trains the 1D U-Net to predict the diffusion noise
(L_simple) until the loss collapses toward 0, then draws samples via full ancestral sampling to
check they look EEG-like. Logs the loss curve to CSV (+ optional W&B) and saves figures.

Usage (run on a GPU node, e.g. via scripts/slurm/m2_overfit.sbatch):
    python scripts/m2_overfit_batch.py
    python scripts/m2_overfit_batch.py --steps 8000 --wandb
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

from saddpm.data.bcic2a import load_subject  # noqa: E402
from saddpm.data.config import DataConfig  # noqa: E402
from saddpm.diffusion.gaussian_diffusion import GaussianDiffusion  # noqa: E402
from saddpm.diffusion.schedule import DiffusionConfig  # noqa: E402
from saddpm.losses.recon import reconstruction_loss  # noqa: E402
from saddpm.models.config import ModelConfig  # noqa: E402
from saddpm.models.unet1d import UNet1D  # noqa: E402
from saddpm.utils.logging import RunLogger  # noqa: E402
from saddpm.utils.seed import seed_everything  # noqa: E402

CONFIGS = REPO_ROOT / "configs"


def _plot_loss(steps: list[int], losses: list[float], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, losses, lw=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("L_simple (MSE, log scale)")
    ax.set_title("M2 overfit-one-batch: noise-prediction loss")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _plot_samples(real: np.ndarray, gen: np.ndarray, out: Path, n_ch: int = 4) -> None:
    """Overlay a few channels of real vs generated windows."""
    n = min(real.shape[0], gen.shape[0])
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 5), sharex=True, sharey=True)
    for j in range(n):
        for c in range(n_ch):
            axes[0, j].plot(real[j, c] + 3 * c, lw=0.6)
            axes[1, j].plot(gen[j, c] + 3 * c, lw=0.6)
        axes[0, j].set_title(f"real #{j}", fontsize=9)
        axes[1, j].set_title(f"generated #{j}", fontsize=9)
    axes[0, 0].set_ylabel("real (first 4 ch)")
    axes[1, 0].set_ylabel("generated")
    fig.suptitle("M2: real vs DDPM-generated windows (overfit model)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=str(CONFIGS / "overfit.yaml"))
    parser.add_argument("--steps", type=int, default=None, help="override step count")
    parser.add_argument("--wandb", action="store_true", help="also log to W&B")
    parser.add_argument("--smoke", action="store_true", help="tiny CPU smoke (few steps, small T)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        oc = yaml.safe_load(fh)
    if args.smoke:  # fast end-to-end glue check on CPU
        oc = {**oc, "batch_size": 4, "steps": 3, "n_gen_samples": 2, "log_every": 1}
    steps = args.steps if args.steps is not None else int(oc["steps"])

    data_cfg = DataConfig.from_yaml(CONFIGS / "data.yaml")
    model_cfg = ModelConfig.from_yaml(CONFIGS / "model.yaml")
    diff_cfg = (
        DiffusionConfig(num_timesteps=20) if args.smoke else DiffusionConfig.from_yaml(CONFIGS / "diffusion.yaml")
    )
    seed_everything(data_cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[M2] device={device} steps={steps} batch={oc['batch_size']} lr={oc['lr']}")
    if device.type != "cuda":
        print("[M2] WARNING: running on CPU; this gate is intended for a Slurm V100 node.")

    # One fixed batch from the training session of the chosen subject.
    per_session = load_subject(int(oc["subject"]), data_cfg)
    train_sw = next(v for v in per_session.values() if v.session_role == "T")
    rng = np.random.default_rng(data_cfg.seed)
    idx = rng.choice(len(train_sw.windows), size=int(oc["batch_size"]), replace=False)
    x0 = torch.from_numpy(train_sw.windows[idx]).to(device)  # (B, 22, 512)
    print(f"[M2] fixed batch: {tuple(x0.shape)} from A{oc['subject']:02d} {train_sw.session}")

    model = UNet1D(model_cfg).to(device)
    diffusion = GaussianDiffusion(diff_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[M2] U-Net params: {n_params/1e6:.2f}M")
    opt = torch.optim.Adam(model.parameters(), lr=float(oc["lr"]))

    run_dir = REPO_ROOT / "artifacts" / "runs" / "m2_overfit"
    logger = RunLogger(
        run_dir,
        use_wandb=args.wandb,
        wandb_project="saddpm",
        wandb_run_name="m2-overfit",
        config={"milestone": "M2", **oc, "n_params": n_params},
    )

    model.train()
    log_steps, log_losses, recent = [], [], []
    for step in range(1, steps + 1):
        t = torch.randint(0, diffusion.num_timesteps, (x0.shape[0],), device=device)
        noise = torch.randn_like(x0)
        xt = diffusion.q_sample(x0, t, noise)
        loss = reconstruction_loss(model(xt, t), noise)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(oc["grad_clip"]))
        opt.step()

        recent.append(loss.item())
        recent = recent[-100:]
        if step % int(oc["log_every"]) == 0 or step == 1:
            trailing = float(np.mean(recent))
            log_steps.append(step)
            log_losses.append(loss.item())
            logger.log(step, {"loss": loss.item(), "trailing_mean_loss": trailing})
            print(f"  step {step:5d}  loss={loss.item():.5f}  trailing={trailing:.5f}")

    trailing_final = float(np.mean(recent))
    passed = trailing_final < float(oc["pass_threshold"])

    # Generate EEG-like samples via full ancestral sampling.
    model.eval()
    gen = diffusion.p_sample_loop(
        lambda x, tt: model(x, tt),
        shape=(int(oc["n_gen_samples"]), model_cfg.in_channels, model_cfg.signal_length),
        device=device,
    )
    gen_np = gen.detach().cpu().numpy()
    real_np = x0[: int(oc["n_gen_samples"])].detach().cpu().numpy()

    fig_dir = REPO_ROOT / "artifacts" / "figures"
    _plot_loss(log_steps, log_losses, fig_dir / "m2_overfit_loss.png")
    _plot_samples(real_np, gen_np, fig_dir / "m2_overfit_samples.png")
    logger.log_image("loss_curve", fig_dir / "m2_overfit_loss.png")
    logger.log_image("samples", fig_dir / "m2_overfit_samples.png")
    logger.finish()

    print(
        f"[M2] initial loss ~ {log_losses[0]:.4f} -> trailing-mean final {trailing_final:.5f} "
        f"(threshold {oc['pass_threshold']}) -> {'PASS' if passed else 'FAIL'}"
    )
    print(f"[M2] generated sample stats: mean={gen_np.mean():+.3f} std={gen_np.std():.3f}")
    print(f"[M2] figures: artifacts/figures/m2_overfit_loss.png, m2_overfit_samples.png")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
