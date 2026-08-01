"""Auditable clean-EEG diffusion prior.

The prior is deliberately independent of the query observation, calibration
operator and participant identity. Scientific CGDR uses a joint multichannel
network whose input/output montage is frozen. The historical one-channel
reshape remains available only through an explicitly labelled ablation mode;
it cannot satisfy the scientific prior contract.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
from torch import Tensor, nn

from saddpm.diffusion.gaussian_diffusion import GaussianDiffusion
from saddpm.diffusion.schedule import DiffusionConfig, validate_cgdr_schedule
from saddpm.models.config import ModelConfig
from saddpm.models.unet1d import UNet1D


PriorMode = Literal["joint_multichannel", "independent_channel_ablation"]


def canonical_valid_time_mask(signal: Tensor, valid_time_mask: Optional[Tensor]) -> Tensor:
    """Return a detached boolean mask with shape ``(B,1,L)``."""

    if valid_time_mask is None:
        return torch.ones(
            (signal.shape[0], 1, signal.shape[-1]),
            dtype=torch.bool,
            device=signal.device,
        )
    value = torch.as_tensor(valid_time_mask, device=signal.device)
    if value.ndim == 2:
        value = value[:, None, :]
    if value.shape != (signal.shape[0], 1, signal.shape[-1]):
        raise ValueError(
            "valid_time_mask must have shape (B,L) or (B,1,L); "
            f"got {tuple(value.shape)}"
        )
    if value.dtype != torch.bool:
        if not bool(((value == 0) | (value == 1)).all()):
            raise ValueError("numeric valid_time_mask must contain only 0/1")
        value = value.bool()
    if not bool(value.flatten(start_dim=1).any(dim=1).all()):
        raise ValueError("every signal must contain at least one valid time point")
    return value.detach()


class CleanEEGDiffusionPrior(nn.Module):
    """Unconditional epsilon-prediction prior over clean EEG.

    Args:
        model_config: U-Net configuration. Scientific use requires at least
            two montage channels and equal input/output channel counts.
        diffusion_config: Variance-preserving diffusion schedule.
        prior_mode: Joint montage model, or an explicitly non-scientific
            independent-channel compatibility ablation.
        enforce_scientific_schedule: Require the frozen T=1000 linear schedule
            and terminal alpha-bar contract. This may be disabled only for
            tests or compatibility loading, never for scientific results.

    Input and output tensors use ``(batch, channels, samples)``. The joint path
    never reshapes channels into the batch axis.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        diffusion_config: DiffusionConfig,
        *,
        prior_mode: PriorMode = "joint_multichannel",
        enforce_scientific_schedule: bool = True,
    ) -> None:
        super().__init__()
        model_channels = (
            model_config.in_channels
            if model_config.out_channels is None
            else model_config.out_channels
        )
        if model_config.in_channels < 1 or model_channels != model_config.in_channels:
            raise ValueError("a clean epsilon prior requires equal positive input/output channels")
        if model_config.signal_length < 8 or model_config.signal_length % 8 != 0:
            raise ValueError("UNet1D signal_length must be a positive multiple of 8")
        if prior_mode not in ("joint_multichannel", "independent_channel_ablation"):
            raise ValueError(f"unknown prior_mode: {prior_mode!r}")
        if prior_mode == "joint_multichannel" and model_config.in_channels < 2:
            raise ValueError(
                "scientific joint_multichannel prior requires at least two EEG channels"
            )
        if prior_mode == "independent_channel_ablation" and model_config.in_channels != 1:
            raise ValueError("independent_channel_ablation requires in_channels=1")
        terminal_alpha_bar = (
            validate_cgdr_schedule(diffusion_config)
            if enforce_scientific_schedule
            else None
        )

        self.model_config = model_config
        self.diffusion_config = diffusion_config
        self.prior_mode: PriorMode = prior_mode
        self.enforce_scientific_schedule = bool(enforce_scientific_schedule)
        self.terminal_alpha_bar = terminal_alpha_bar
        self.unet = UNet1D(model_config, subject_conditioned=False)
        self.diffusion = GaussianDiffusion(diffusion_config)

    def _validate_signal(self, signal: Tensor) -> None:
        if signal.ndim != 3:
            raise ValueError(f"expected signal shape (B,C,L), got {tuple(signal.shape)}")
        if any(size < 1 for size in signal.shape):
            raise ValueError("EEG tensors cannot contain an empty dimension")
        if not signal.dtype.is_floating_point:
            raise TypeError("EEG tensors must use a floating-point dtype")
        if signal.shape[-1] != self.model_config.signal_length:
            raise ValueError(
                f"expected signal length {self.model_config.signal_length}, "
                f"got {signal.shape[-1]}"
            )

    @staticmethod
    def _validate_timesteps(timesteps: Tensor, batch_size: int, device: torch.device) -> None:
        if timesteps.shape != (batch_size,):
            raise ValueError(
                f"expected timestep shape ({batch_size},), got {tuple(timesteps.shape)}"
            )
        if timesteps.dtype != torch.long:
            raise TypeError("diffusion timesteps must have dtype torch.long")
        if timesteps.device != device:
            raise ValueError("signal and timestep tensors must be on the same device")

    def predict_noise_channel_independent(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        *,
        valid_time_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply a one-channel prior independently to ``(B,C,L)`` EEG.

        Every channel from the same example receives the same diffusion
        timestep.  There is no channel mixing in this helper; population and
        calibrated observation precision remain responsible for EEG-space
        coupling during inference.
        """

        self._validate_signal(x_t)
        self._validate_timesteps(timesteps, x_t.shape[0], x_t.device)
        if self.prior_mode != "independent_channel_ablation":
            raise ValueError(
                "channel-independent prediction is only available as the explicit ablation"
            )
        mask = canonical_valid_time_mask(x_t, valid_time_mask)
        masked = x_t * mask.to(dtype=x_t.dtype)
        batch, channels, length = x_t.shape
        flat = masked.reshape(batch * channels, 1, length)
        flat_timesteps = timesteps.repeat_interleave(channels)
        prediction = self.unet(flat, flat_timesteps)
        return prediction.reshape(batch, channels, length) * mask.to(dtype=x_t.dtype)

    def predict_noise(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        *,
        valid_time_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict diffusion noise with no observation or identity input."""

        self._validate_signal(x_t)
        self._validate_timesteps(timesteps, x_t.shape[0], x_t.device)
        if self.prior_mode == "independent_channel_ablation":
            return self.predict_noise_channel_independent(
                x_t, timesteps, valid_time_mask=valid_time_mask
            )
        if x_t.shape[1] != self.model_config.in_channels:
            raise ValueError(
                f"joint prior configured for {self.model_config.in_channels} channels, "
                f"got {x_t.shape[1]}"
            )
        mask = canonical_valid_time_mask(x_t, valid_time_mask)
        masked = x_t * mask.to(dtype=x_t.dtype)
        return self.unet(masked, timesteps) * mask.to(dtype=x_t.dtype)

    def forward(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        valid_time_mask: Optional[Tensor] = None,
    ) -> Tensor:
        return self.predict_noise(x_t, timesteps, valid_time_mask=valid_time_mask)

    def noise_standard_deviation(self, timesteps: Tensor, ndim: int) -> Tensor:
        """Return ``sqrt(1-alpha_bar_t)`` broadcastable to an ``ndim`` signal."""

        values = self.diffusion.sqrt_one_minus_alphas_cumprod.gather(0, timesteps)
        return values.reshape(timesteps.shape[0], *((1,) * (ndim - 1)))

    def predict_clean(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        predicted_noise: Optional[Tensor] = None,
        valid_time_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Return the clean estimate implied by the epsilon prediction."""

        mask = canonical_valid_time_mask(x_t, valid_time_mask)
        masked_x_t = x_t * mask.to(dtype=x_t.dtype)
        if predicted_noise is not None and predicted_noise.shape != x_t.shape:
            raise ValueError("predicted noise and x_t must have identical shapes")
        eps = (
            self.predict_noise(masked_x_t, timesteps, valid_time_mask=mask)
            if predicted_noise is None
            else predicted_noise * mask.to(dtype=predicted_noise.dtype)
        )
        if eps.shape != x_t.shape:
            raise ValueError("predicted noise and x_t must have identical shapes")
        clean = self.diffusion.predict_xstart_from_eps(masked_x_t, timesteps, eps)
        return clean * mask.to(dtype=clean.dtype)

    def score_from_epsilon(self, epsilon: Tensor, timesteps: Tensor) -> Tensor:
        """Convert epsilon prediction to VP marginal score with explicit sign."""

        sigma = self.noise_standard_deviation(timesteps, epsilon.ndim)
        return -epsilon / sigma.clamp_min(torch.finfo(epsilon.dtype).eps)

    def epsilon_from_score(self, score: Tensor, timesteps: Tensor) -> Tensor:
        """Convert VP marginal score to epsilon prediction with explicit sign."""

        sigma = self.noise_standard_deviation(timesteps, score.ndim)
        return -sigma * score

    def score(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        *,
        valid_time_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Return the VP marginal score ``-epsilon/sqrt(1-alpha_bar_t)``."""

        eps = self.predict_noise(x_t, timesteps, valid_time_mask=valid_time_mask)
        return self.score_from_epsilon(eps, timesteps)

    def training_loss(
        self,
        clean: Tensor,
        *,
        timesteps: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
        valid_time_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Uniform-timestep epsilon MSE for clean EEG ``(B,C,L)``.

        Supplying ``timesteps`` and ``noise`` makes the loss deterministic for
        tests or paired diagnostics. Multi-channel tensors remain joint in the
        scientific mode; only the explicit ablation reshapes channels.
        """

        self._validate_signal(clean)
        mask = canonical_valid_time_mask(clean, valid_time_mask)
        mask_float = mask.to(dtype=clean.dtype)
        clean = clean * mask_float
        batch_size = clean.shape[0]
        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.diffusion.num_timesteps,
                (batch_size,),
                device=clean.device,
                dtype=torch.long,
                generator=generator,
            )
        else:
            self._validate_timesteps(timesteps, batch_size, clean.device)
        if noise is None:
            noise = torch.randn(
                clean.shape,
                device=clean.device,
                dtype=clean.dtype,
                generator=generator,
            )
        elif (
            noise.shape != clean.shape
            or noise.device != clean.device
            or noise.dtype != clean.dtype
        ):
            raise ValueError("training noise must match the clean tensor shape, device and dtype")
        noise = noise * mask_float
        x_t = self.diffusion.q_sample(clean, timesteps, noise)
        prediction = self.predict_noise(x_t, timesteps, valid_time_mask=mask)
        squared_error = (prediction - noise).square() * mask_float
        denominator = mask_float.sum() * clean.shape[1]
        return squared_error.sum() / denominator

    def loss(
        self,
        clean: Tensor,
        *,
        timesteps: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
        valid_time_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Alias for :meth:`training_loss`."""

        return self.training_loss(
            clean,
            timesteps=timesteps,
            noise=noise,
            generator=generator,
            valid_time_mask=valid_time_mask,
        )

    @torch.no_grad()
    def cross_channel_influence(
        self,
        probe: Tensor,
        timesteps: Tensor,
        *,
        valid_time_mask: Optional[Tensor] = None,
        perturbation: float = 1.0e-3,
    ) -> Tensor:
        """Return RMS output influence ``[output_channel,input_channel]``."""

        if self.prior_mode != "joint_multichannel":
            raise ValueError("cross-channel influence is undefined for the independent ablation")
        if self.training:
            raise ValueError("cross-channel dependency audit requires prior.eval()")
        if not 0.0 < perturbation < 1.0:
            raise ValueError("perturbation must lie in (0,1)")
        self._validate_signal(probe)
        mask = canonical_valid_time_mask(probe, valid_time_mask)
        base = self.predict_noise(probe, timesteps, valid_time_mask=mask)
        channels = probe.shape[1]
        matrix = torch.empty((channels, channels), device=probe.device, dtype=probe.dtype)
        for source in range(channels):
            perturbed = probe.clone()
            perturbed[:, source, :] = perturbed[:, source, :] + (
                float(perturbation) * mask[:, 0, :].to(dtype=probe.dtype)
            )
            changed = self.predict_noise(
                perturbed, timesteps, valid_time_mask=mask
            ) - base
            matrix[:, source] = torch.sqrt(
                changed.square().sum(dim=(0, 2))
                / mask.sum().clamp_min(1)
            )
        return matrix

    def assert_cross_channel_dependency(
        self,
        probe: Tensor,
        timesteps: Tensor,
        *,
        valid_time_mask: Optional[Tensor] = None,
        perturbation: float = 1.0e-3,
        minimum_influence: float = 1.0e-8,
    ) -> Tensor:
        """Fail unless every output channel depends on another input channel."""

        influence = self.cross_channel_influence(
            probe,
            timesteps,
            valid_time_mask=valid_time_mask,
            perturbation=perturbation,
        )
        if not math.isfinite(float(minimum_influence)) or minimum_influence < 0.0:
            raise ValueError("minimum_influence must be non-negative")
        off_diagonal = influence.clone()
        off_diagonal.fill_diagonal_(0.0)
        if bool((off_diagonal.max(dim=1).values <= float(minimum_influence)).any()):
            raise AssertionError(
                "joint prior failed cross-channel dependency: at least one output "
                "channel is insensitive to every other input channel"
            )
        return influence
