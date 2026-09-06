#!/usr/bin/env python
"""M1 deliverable: numerical forward-marginal sanity check (handoff §1.3).

Verifies q(x_t|x_0) = N(√ᾱ_t x_0, (1-ᾱ_t) I) by comparing iterated single steps against the
one-shot reparameterization, for several t. Prints an error table and saves a variance-vs-t plot.

Usage:
    python scripts/m1_marginal_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from saddpm.diffusion.gaussian_diffusion import GaussianDiffusion  # noqa: E402
from saddpm.diffusion.marginal_check import run_marginal_check  # noqa: E402
from saddpm.diffusion.schedule import DiffusionConfig  # noqa: E402
from saddpm.utils.seed import seed_everything  # noqa: E402

TOL = 0.05  # pass threshold on every error metric (Monte-Carlo with n_samples below)
N_SAMPLES = 20000


def main() -> int:
    seed_everything(42)
    cfg = DiffusionConfig.from_yaml(REPO_ROOT / "configs" / "diffusion.yaml")
    diff = GaussianDiffusion(cfg)
    print(f"[M1] schedule: T={cfg.num_timesteps} {cfg.schedule} beta[{cfg.beta_start},{cfg.beta_end}]")
    print(f"     abar[0]={diff.alphas_cumprod[0]:.5f}  abar[T-1]={diff.alphas_cumprod[-1]:.5f}")

    x0_vec = torch.linspace(-2.0, 2.0, 16)
    t_indices = [0, 9, 99, 299, 499, 799, 999]
    results = run_marginal_check(diff, x0_vec, t_indices, n_samples=N_SAMPLES, seed=42)

    header = (
        f"{'t':>4} {'abar':>8} | {'os_mean':>8} {'os_var':>8} {'os_off':>8} "
        f"| {'sw_mean':>8} {'sw_var':>8} {'sw_off':>8} | {'maxerr':>8}"
    )
    print(header)
    print("-" * len(header))
    worst = 0.0
    for r in results:
        worst = max(worst, r.max_error())
        print(
            f"{r.t_index:>4} {r.abar:>8.4f} | {r.oneshot_mean_err:>8.4f} {r.oneshot_var_err:>8.4f} "
            f"{r.oneshot_offdiag:>8.4f} | {r.stepwise_mean_err:>8.4f} {r.stepwise_var_err:>8.4f} "
            f"{r.stepwise_offdiag:>8.4f} | {r.max_error():>8.4f}"
        )

    # Figure: (left) the schedule curves from the buffers; (right) check errors vs tolerance.
    t_axis = torch.arange(cfg.num_timesteps)
    grid = [0, 9, 99, 299, 499, 799, 999]
    grid_results = run_marginal_check(diff, x0_vec, grid, n_samples=N_SAMPLES, seed=7)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))
    ax0.plot(t_axis, diff.sqrt_alphas_cumprod.cpu(), label="signal scale √ᾱ_t")
    ax0.plot(t_axis, (1.0 - diff.alphas_cumprod).cpu(), label="noise variance (1-ᾱ_t)")
    ax0.set_xlabel("timestep t")
    ax0.set_ylabel("value")
    ax0.set_title(f"Linear schedule (T={cfg.num_timesteps}, β∈[{cfg.beta_start},{cfg.beta_end}])")
    ax0.legend(fontsize=9)

    gt = [r.t_index for r in grid_results]
    ax1.plot(gt, [r.oneshot_mean_err for r in grid_results], "o-", label="one-shot mean err")
    ax1.plot(gt, [r.oneshot_var_err for r in grid_results], "s-", label="one-shot var err")
    ax1.plot(gt, [r.stepwise_mean_err for r in grid_results], "o--", label="stepwise mean err")
    ax1.plot(gt, [r.stepwise_var_err for r in grid_results], "s--", label="stepwise var err")
    ax1.plot(gt, [r.stepwise_offdiag for r in grid_results], "^:", label="stepwise off-diag cov")
    ax1.axhline(TOL, color="k", ls=":", label=f"tolerance {TOL}")
    ax1.set_xlabel("timestep t")
    ax1.set_ylabel("|empirical − closed-form|")
    ax1.set_title(f"§1.3 marginal check (n={N_SAMPLES}): stepwise vs one-shot")
    ax1.legend(fontsize=8)
    fig.tight_layout()
    out = REPO_ROOT / "artifacts" / "figures" / "m1_marginal_check.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[M1] saved schedule + check plot -> {out.relative_to(REPO_ROOT)}")

    ok = worst < TOL
    print(f"[M1] worst error over all t = {worst:.4f} (tol {TOL}) -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
