"""Auditable clean-EEG diffusion prior.

The prior is deliberately independent of the query observation, calibration
operator and participant identity.  The first CGDR model is trained on
single-channel EEGdenoiseNet epochs.  At inference time the same one-channel
network can be applied independently to every channel of ``(B, C, L)`` EEG.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from saddpm.diffusion.gaussian_diffusion import GaussianDiffusion
from saddpm.diffusion.schedule import DiffusionConfig
from saddpm.models.config import ModelConfig
from saddpm.models.unet1d import UNet1D


class CleanEEGDiffusionPrior(nn.Module):
    """Unconditional epsilon-prediction prior over clean EEG.

    Args:
        model_config: U-Net configuration.  ``in_channels=1`` is the formal
            first-stage configuration; ``out_channels`` must resolve to the
            same value as ``in_channels``.
        diffusion_config: Variance-preserving diffusion schedule.

    Input and output tensors use ``(batch, channels, samples)``.  When the
    configured model has one channel, :meth:`predict_noise_channel_independent`
    reshapes a multi-channel batch to ``(B*C, 1, L)`` and restores its original
    layout after prediction.
    """

    def __init__(self, model_config: ModelConfig, diffusion_config: DiffusionConfig) -> None:
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
        if diffusion_config.num_timesteps < 2:
            raise ValueError("diffusion requires at least two timesteps")
        if not (
            0.0 < diffusion_config.beta_start <= diffusion_config.beta_end < 1.0
        ):
            raise ValueError("diffusion betas must satisfy 0 < beta_start <= beta_end < 1")

        self.model_config = model_config
        self.diffusion_config = diffusion_config
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

    def predict_noise_channel_independent(self, x_t: Tensor, timesteps: Tensor) -> Tensor:
        """Apply a one-channel prior independently to ``(B,C,L)`` EEG.

        Every channel from the same example receives the same diffusion
        timestep.  There is no channel mixing in this helper; population and
        calibrated observation precision remain responsible for EEG-space
        coupling during inference.
        """

        self._validate_signal(x_t)
        self._validate_timesteps(timesteps, x_t.shape[0], x_t.device)
        if self.model_config.in_channels != 1:
            raise ValueError("channel-independent prediction requires a one-channel prior")
        batch, channels, length = x_t.shape
        flat = x_t.reshape(batch * channels, 1, length)
        flat_timesteps = timesteps.repeat_interleave(channels)
        prediction = self.unet(flat, flat_timesteps)
        return prediction.reshape(batch, channels, length)

    def predict_noise(self, x_t: Tensor, timesteps: Tensor) -> Tensor:
        """Predict diffusion noise with no observation or identity input."""

        self._validate_signal(x_t)
        self._validate_timesteps(timesteps, x_t.shape[0], x_t.device)
        if x_t.shape[1] == self.model_config.in_channels:
            return self.unet(x_t, timesteps)
        if self.model_config.in_channels == 1:
            return self.predict_noise_channel_independent(x_t, timesteps)
        raise ValueError(
            f"prior configured for {self.model_config.in_channels} channels, got {x_t.shape[1]}"
        )

    def forward(self, x_t: Tensor, timesteps: Tensor) -> Tensor:
        return self.predict_noise(x_t, timesteps)

    def noise_standard_deviation(self, timesteps: Tensor, ndim: int) -> Tensor:
        """Return ``sqrt(1-alpha_bar_t)`` broadcastable to an ``ndim`` signal."""

        values = self.diffusion.sqrt_one_minus_alphas_cumprod.gather(0, timesteps)
        return values.reshape(timesteps.shape[0], *((1,) * (ndim - 1)))

    def predict_clean(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        predicted_noise: Optional[Tensor] = None,
    ) -> Tensor:
        """Return the clean estimate implied by the epsilon prediction."""

        eps = self.predict_noise(x_t, timesteps) if predicted_noise is None else predicted_noise
        if eps.shape != x_t.shape:
            raise ValueError("predicted noise and x_t must have identical shapes")
        return self.diffusion.predict_xstart_from_eps(x_t, timesteps, eps)

    def score(self, x_t: Tensor, timesteps: Tensor) -> Tensor:
        """Return the VP marginal score ``-epsilon/sqrt(1-alpha_bar_t)``."""

        eps = self.predict_noise(x_t, timesteps)
        sigma = self.noise_standard_deviation(timesteps, x_t.ndim)
        return -eps / sigma.clamp_min(torch.finfo(eps.dtype).eps)

    def training_loss(
        self,
        clean: Tensor,
        *,
        timesteps: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Uniform-timestep epsilon MSE for clean EEG ``(B,C,L)``.

        Supplying ``timesteps`` and ``noise`` makes the loss deterministic for
        tests or paired diagnostics.  With an ``in_channels=1`` model, a
        multi-channel tensor is treated as independent channel examples.
        """

        self._validate_signal(clean)
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
        x_t = self.diffusion.q_sample(clean, timesteps, noise)
        prediction = self.predict_noise(x_t, timesteps)
        return nn.functional.mse_loss(prediction, noise)

    def loss(
        self,
        clean: Tensor,
        *,
        timesteps: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Alias for :meth:`training_loss`."""

        return self.training_loss(
            clean,
            timesteps=timesteps,
            noise=noise,
            generator=generator,
        )
