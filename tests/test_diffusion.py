"""Unit tests + numerical sanity checks for the diffusion forward process (handoff §1, §5)."""

from __future__ import annotations

from pathlib import Path

import torch

from saddpm.diffusion.gaussian_diffusion import GaussianDiffusion
from saddpm.diffusion.marginal_check import run_marginal_check
from saddpm.diffusion.schedule import DiffusionConfig, make_betas

REPO_ROOT = Path(__file__).resolve().parents[1]


def _diffusion() -> GaussianDiffusion:
    cfg = DiffusionConfig.from_yaml(REPO_ROOT / "configs" / "diffusion.yaml")
    return GaussianDiffusion(cfg)


def test_schedule_endpoints_and_length() -> None:
    cfg = DiffusionConfig(num_timesteps=1000, beta_start=1e-4, beta_end=0.02, schedule="linear")
    betas = make_betas(cfg)
    assert betas.shape == (1000,)
    assert torch.isclose(betas[0], torch.tensor(1e-4, dtype=torch.float64))
    assert torch.isclose(betas[-1], torch.tensor(0.02, dtype=torch.float64))
    # strictly increasing for a linear schedule.
    assert torch.all(betas[1:] > betas[:-1])


def test_alphas_cumprod_monotone_in_unit_interval() -> None:
    diff = _diffusion()
    abar = diff.alphas_cumprod
    assert torch.all(abar > 0) and torch.all(abar <= 1.0 + 1e-6)
    assert torch.all(abar[1:] < abar[:-1])  # decreasing
    assert abar[0] > 0.999  # barely noised at t=0
    assert abar[-1] < 0.05  # almost pure noise at t=T-1


def test_q_sample_zero_noise_is_scaled_signal() -> None:
    diff = _diffusion()
    x0 = torch.randn(4, 22, 512)
    t = torch.tensor([0, 10, 100, 999])
    xt = diff.q_sample(x0, t, noise=torch.zeros_like(x0))
    expected = diff.sqrt_alphas_cumprod[t].reshape(4, 1, 1) * x0
    assert torch.allclose(xt, expected, atol=1e-6)


def test_q_sample_shape() -> None:
    diff = _diffusion()
    x0 = torch.randn(8, 22, 512)
    t = torch.randint(0, diff.num_timesteps, (8,))
    assert diff.q_sample(x0, t).shape == (8, 22, 512)


def test_forward_marginal_matches_closed_form() -> None:
    """Core §1.3 check: stepwise and one-shot both match N(√ᾱ x0, (1-ᾱ)I)."""
    diff = _diffusion()
    torch.manual_seed(0)
    x0_vec = torch.linspace(-2.0, 2.0, 8)
    results = run_marginal_check(diff, x0_vec, t_indices=[0, 9, 99, 499, 999], n_samples=12000, seed=0)
    for r in results:
        assert r.max_error() < 0.08, (
            f"t={r.t_index}: mean/var/offdiag mismatch "
            f"(oneshot {r.oneshot_mean_err:.3f}/{r.oneshot_var_err:.3f}/{r.oneshot_offdiag:.3f}, "
            f"stepwise {r.stepwise_mean_err:.3f}/{r.stepwise_var_err:.3f}/{r.stepwise_offdiag:.3f})"
        )
