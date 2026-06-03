"""Numerical verification of the forward marginal (handoff §1.3).

For several timesteps ``t`` we sample ``x_t`` two ways — iterated single steps
(:meth:`GaussianDiffusion.q_sample_stepwise`) vs. the one-shot reparameterization
(:meth:`GaussianDiffusion.q_sample`) — and check that the empirical mean and covariance of
both match the closed form ``N(√ᾱ_t x_0, (1-ᾱ_t) I)``. This catches schedule/indexing bugs.

Shared by ``tests/test_diffusion.py`` and ``scripts/m1_marginal_check.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch import Tensor

from .gaussian_diffusion import GaussianDiffusion


@dataclass
class MarginalCheckResult:
    """Per-timestep comparison of empirical statistics to the closed-form marginal."""

    t_index: int
    abar: float
    oneshot_mean_err: float  # max_i |E[x_t]_i - √ᾱ_t x0_i|
    oneshot_var_err: float  # max_i |Var[x_t]_i - (1-ᾱ_t)|
    oneshot_offdiag: float  # max off-diagonal |Cov[x_t]|
    stepwise_mean_err: float
    stepwise_var_err: float
    stepwise_offdiag: float

    def max_error(self) -> float:
        """Largest of all six error metrics (used as the scalar pass/fail criterion)."""
        return max(
            self.oneshot_mean_err,
            self.oneshot_var_err,
            self.oneshot_offdiag,
            self.stepwise_mean_err,
            self.stepwise_var_err,
            self.stepwise_offdiag,
        )


def _stats(xt: Tensor, mean_cf: Tensor, var_cf: float) -> tuple[float, float, float]:
    """Return (max mean error, max diagonal-variance error, max off-diagonal covariance)."""
    emp_mean = xt.mean(dim=0)
    emp_var = xt.var(dim=0, unbiased=True)
    cov = torch.cov(xt.t())  # (D, D)
    offdiag = (cov - torch.diag(torch.diag(cov))).abs().max()
    mean_err = (emp_mean - mean_cf).abs().max().item()
    var_err = (emp_var - var_cf).abs().max().item()
    return mean_err, var_err, offdiag.item()


def run_marginal_check(
    diffusion: GaussianDiffusion,
    x0_vec: Tensor,
    t_indices: Sequence[int],
    n_samples: int = 20000,
    seed: int = 0,
) -> List[MarginalCheckResult]:
    """Run the §1.3 forward-marginal check for each timestep index.

    Args:
        diffusion: the diffusion process providing the schedule + samplers.
        x0_vec: a single clean vector ``(D,)``; ``n_samples`` noisy copies are drawn from it.
        t_indices: timestep indices to check (in ``[0, T-1]``).
        n_samples: Monte-Carlo sample count per timestep.
        seed: RNG seed for reproducibility.

    Returns:
        One :class:`MarginalCheckResult` per timestep index.
    """
    x0_vec = x0_vec.flatten().float()
    dim = x0_vec.numel()
    x0_batch = x0_vec.reshape(1, dim).expand(n_samples, dim).contiguous()

    results: List[MarginalCheckResult] = []
    for t in t_indices:
        abar = diffusion.alphas_cumprod[t].item()
        mean_cf = (abar ** 0.5) * x0_vec
        var_cf = 1.0 - abar

        gen = torch.Generator().manual_seed(seed + t)
        noise = torch.randn(n_samples, dim, generator=gen)
        t_tensor = torch.full((n_samples,), t, dtype=torch.long)
        xt_oneshot = diffusion.q_sample(x0_batch, t_tensor, noise=noise)

        gen_sw = torch.Generator().manual_seed(seed + t + 100000)
        xt_stepwise = diffusion.q_sample_stepwise(x0_batch, t, generator=gen_sw)

        os_mean, os_var, os_off = _stats(xt_oneshot, mean_cf, var_cf)
        sw_mean, sw_var, sw_off = _stats(xt_stepwise, mean_cf, var_cf)
        results.append(
            MarginalCheckResult(
                t_index=t,
                abar=abar,
                oneshot_mean_err=os_mean,
                oneshot_var_err=os_var,
                oneshot_offdiag=os_off,
                stepwise_mean_err=sw_mean,
                stepwise_var_err=sw_var,
                stepwise_offdiag=sw_off,
            )
        )
    return results
