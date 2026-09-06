#!/usr/bin/env python
"""M5 milestone gate: SDEdit denoising + t* sweep ([DD-2], handoff §7, §11).

Loads the trained M4 SADDPM, denoises held-out Session-E segments of a subject via SDEdit
(forward-diffuse to t*, subject-conditioned reverse to 0), sweeps t*, and visualises input vs
denoised. Larger t* should regularise more (lower correlation to the input). Run on a GPU node.
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

from saddpm.data.config import DataConfig  # noqa: E402
from saddpm.data.datasets import load_session_windows  # noqa: E402
from saddpm.diffusion.gaussian_diffusion import GaussianDiffusion  # noqa: E402
from saddpm.diffusion.schedule import DiffusionConfig  # noqa: E402
from saddpm.models.config import ModelConfig  # noqa: E402
from saddpm.models.dual_decoder import DualDecoderSADDPM  # noqa: E402
from saddpm.utils.checkpoint import load_checkpoint  # noqa: E402
from saddpm.utils.seed import seed_everything  # noqa: E402

CONFIGS = REPO_ROOT / "configs"


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-(segment,channel) Pearson correlation between two (N, C, L) arrays."""
    af = a - a.mean(axis=-1, keepdims=True)
    bf = b - b.mean(axis=-1, keepdims=True)
    num = (af * bf).sum(axis=-1)
    den = np.sqrt((af ** 2).sum(axis=-1) * (bf ** 2).sum(axis=-1)) + 1e-12
    return float((num / den).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    with open(CONFIGS / "m5.yaml", "r", encoding="utf-8") as fh:
        m5 = yaml.safe_load(fh)
    data_cfg = DataConfig.from_yaml(CONFIGS / "data.yaml")
    seed_everything(data_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = load_checkpoint(REPO_ROOT / m5["checkpoint"], map_location=str(device))
    model_cfg = ModelConfig(**ckpt["config"]["model"])
    diff_cfg = DiffusionConfig(**ckpt["config"]["diffusion"])
    model = DualDecoderSADDPM(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    diffusion = GaussianDiffusion(diff_cfg).to(device)
    print(f"[M5] loaded {m5['checkpoint']} (git {ckpt.get('git_commit','?')[:8]}) device={device}")

    subject = int(m5["subject"])
    sw = load_session_windows(subject, data_cfg, "E")
    rng = np.random.default_rng(data_cfg.seed)
    idx = rng.choice(len(sw.windows), size=int(m5["n_segments"]), replace=False)
    y = torch.from_numpy(sw.windows[idx]).float().to(device)  # (n, 22, 512)

    def eps_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sid = torch.full((x.shape[0],), subject - 1, device=x.device, dtype=torch.long)
        return model.predict_eps(x, t, sid)

    t_stars = list(m5["t_star_sweep"])
    denoised = {}
    corrs = []
    for ts in t_stars:
        out = diffusion.sdedit(eps_fn, y, t_star=int(ts), ddim_steps=int(m5["ddim_steps"]))
        denoised[ts] = out.detach().cpu().numpy()
        corrs.append(_corr(sw.windows[idx], denoised[ts]))
        print(f"  t*={ts:4d}: corr(denoised, input) = {corrs[-1]:.3f}")

    # Figure: input vs denoised for the t* sweep (first segment, first 3 channels).
    fig_dir = REPO_ROOT / "artifacts" / "figures"
    n_show_ch = 3
    fig, axes = plt.subplots(len(t_stars) + 1, 1, figsize=(10, 2.0 * (len(t_stars) + 1)), sharex=True)
    for c in range(n_show_ch):
        axes[0].plot(sw.windows[idx][0, c] + 3 * c, lw=0.7)
    axes[0].set_ylabel("input", fontsize=8)
    for k, ts in enumerate(t_stars):
        for c in range(n_show_ch):
            axes[k + 1].plot(denoised[ts][0, c] + 3 * c, lw=0.7)
        axes[k + 1].set_ylabel(f"t*={ts}\nr={corrs[k]:.2f}", fontsize=8)
    fig.suptitle(f"M5 SDEdit denoising t* sweep (A{subject:02d} Session-E, segment 0)")
    fig.tight_layout()
    fig.savefig(fig_dir / "m5_sdedit_sweep.png", dpi=130)
    plt.close(fig)

    # correlation-vs-t* curve.
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(t_stars, corrs, "o-")
    ax2.set_xlabel("t* (forward-diffusion strength)")
    ax2.set_ylabel("corr(denoised, input)")
    ax2.set_title("M5: SDEdit regularisation strength vs t*")
    fig2.tight_layout()
    fig2.savefig(fig_dir / "m5_sdedit_corr.png", dpi=130)
    plt.close(fig2)

    # Gate: denoising ran and t* monotonically regularises (corr decreases with t*).
    monotone = all(corrs[i] >= corrs[i + 1] - 0.05 for i in range(len(corrs) - 1))
    finite = all(np.all(np.isfinite(denoised[ts])) for ts in t_stars)
    passed = finite and monotone and corrs[0] > corrs[-1]
    print(f"[M5] corr decreases with t*: {monotone}; default t*={m5['t_star_default']}")
    print(f"[M5] figures: m5_sdedit_sweep.png, m5_sdedit_corr.png")
    print(f"[M5] {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
