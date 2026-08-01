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
    assert abar[-1] <= 1.0e-4  # scientific CGDR terminal marginal is effectively N(0,I)


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


def test_sdedit_shape_and_finite() -> None:
    """SDEdit forward-to-t* then reverse-to-0 preserves shape (uses a dummy eps_fn)."""
    diff = GaussianDiffusion(DiffusionConfig(num_timesteps=100))
    y = torch.randn(2, 4, 64)
    out = diff.sdedit(lambda x, t: torch.zeros_like(x), y, t_star=40, ddim_steps=5)
    assert out.shape == y.shape and torch.all(torch.isfinite(out))


def test_predict_xstart_inverts_q_sample() -> None:
    """Reverse-process buffers are consistent with the forward: x̂_0(q_sample(x_0,t,ε),t,ε)=x_0."""
    diff = _diffusion()
    torch.manual_seed(0)
    x0 = torch.randn(4, 22, 512)
    eps = torch.randn_like(x0)
    t = torch.tensor([5, 50, 500, 999])
    xt = diff.q_sample(x0, t, eps)
    x0_rec = diff.predict_xstart_from_eps(xt, t, eps)
    assert torch.allclose(x0_rec, x0, atol=1e-3)


def test_posterior_variance_nonnegative_and_bounded() -> None:
    diff = _diffusion()
    assert torch.all(diff.posterior_variance >= 0)
    # posterior variance never exceeds the per-step beta.
    assert torch.all(diff.posterior_variance <= diff.betas + 1e-6)


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
