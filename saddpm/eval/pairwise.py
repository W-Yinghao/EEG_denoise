"""SADDPM SDEdit denoising of full sessions for the pairwise 9×9 protocol (handoff §8.1)."""

from __future__ import annotations

import numpy as np
import torch

from ..data.preprocessing import zscore_per_channel
from ..diffusion.gaussian_diffusion import GaussianDiffusion
from ..models.dual_decoder import DualDecoderSADDPM


@torch.no_grad()
def saddpm_denoise_windows(
    model: DualDecoderSADDPM,
    diffusion: GaussianDiffusion,
    windows: np.ndarray,
    embed_subject: int,
    t_star: int,
    ddim_steps: int,
    device: torch.device,
    batch_size: int = 256,
    renormalize: bool = True,
) -> np.ndarray:
    """SDEdit-denoise a session's windows, conditioned on ``embed_subject``'s embedding ([DD-4]).

    SDEdit attenuates amplitude (the reverse process regresses toward a lower-variance prior, e.g.
    std 1.0 -> ~0.56 at t*=200). ``renormalize`` re-applies the per-channel z-score so the denoised
    windows reach EEGNet in the same normalized space as the (z-scored) input and the ICA baseline,
    keeping the comparison fair (audit finding #1).

    Args:
        model: trained SADDPM.
        diffusion: diffusion process.
        windows: ``(N, 22, 512)`` z-scored windows to denoise.
        embed_subject: 0-based subject id whose embedding conditions the reverse process.
        t_star: SDEdit forward-diffusion strength.
        ddim_steps: DDIM reverse steps.
        device: device.
        batch_size: windows per batch.

    Returns:
        ``(N, 22, 512)`` denoised windows.
    """
    model.eval()

    def eps_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sid = torch.full((x.shape[0],), embed_subject, device=x.device, dtype=torch.long)
        return model.predict_eps(x, t, sid)

    out = []
    for i in range(0, len(windows), batch_size):
        y = torch.from_numpy(np.ascontiguousarray(windows[i : i + batch_size])).float().to(device)
        d = diffusion.sdedit(eps_fn, y, t_star=t_star, ddim_steps=ddim_steps)
        out.append(d.detach().cpu().numpy())
    denoised = np.concatenate(out, axis=0)
    return zscore_per_channel(denoised) if renormalize else denoised


def matrix_summary(matrix: np.ndarray) -> dict:
    """Grand mean, per-target std, and diagonal (within-subject) mean of a 9×9 accuracy matrix."""
    return {
        "grand_mean": float(matrix.mean()),
        "diag_mean": float(np.diag(matrix).mean()),
        "mean_per_target_std": float(matrix.std(axis=0).mean()),  # spread over sources, per target
        "source_mean_row": matrix.mean(axis=1).tolist(),  # mean over targets, per source
    }
