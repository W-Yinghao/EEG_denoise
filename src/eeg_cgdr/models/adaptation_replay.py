"""Explicit paired randomness for V9R support-only Score-LoRA adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch


def seed_all(seed: int) -> None:
    """Seed every RNG used by model construction, dropout, and adaptation."""

    value = int(seed)
    random.seed(value)
    np.random.seed(value % (2**32 - 1))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


@dataclass(frozen=True)
class AdaptationReplay:
    """Frozen optimizer and inference randomness for one recipient/seed.

    Gaussian training noise is stored in standard-normal form.  D01 and D11
    consume the same rows, timesteps, dropout seeds, initialization seed, and
    inference bank; only the protocol-defined basis/target may differ.
    """

    initialization_seed: int
    minibatch_indices: np.ndarray
    timesteps: np.ndarray
    gaussian_noise: np.ndarray
    dropout_seeds: np.ndarray
    checkpoint_steps: np.ndarray
    inference_noise_bank: np.ndarray

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        pair_count: int,
        validation_count: int,
        updates: int,
        batch_size: int,
        timesteps: int,
        signal_length: int,
        posterior_samples: int = 8,
        checkpoint_steps: tuple[int, ...] = (0, 50, 100, 200, 400, 800, 1000),
    ) -> "AdaptationReplay":
        if pair_count < 1 or validation_count < 1 or updates < 1:
            raise ValueError("adaptation replay requires non-empty real support pairs")
        if posterior_samples != 8 or signal_length < 1 or timesteps < 2:
            raise ValueError("V9R freezes K=8 and requires valid diffusion dimensions")
        size = min(int(batch_size), int(pair_count))
        rng = np.random.default_rng(int(seed))
        indices = rng.integers(0, pair_count, size=(updates, size), dtype=np.int64)
        steps = rng.integers(0, timesteps, size=(updates, size), dtype=np.int64)
        noise = rng.standard_normal((updates, size, 2, signal_length), dtype=np.float32)
        dropout = rng.integers(1, 2**31 - 1, size=updates, dtype=np.int64)
        inference = rng.standard_normal(
            (posterior_samples, validation_count, 2, signal_length), dtype=np.float32
        )
        checkpoints = np.asarray(sorted(set(map(int, checkpoint_steps))), dtype=np.int64)
        if checkpoints[0] != 0 or checkpoints[-1] != updates:
            raise ValueError("checkpoint steps must span step zero through final update")
        return cls(int(seed), indices, steps, noise, dropout, checkpoints, inference)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            initialization_seed=np.int64(self.initialization_seed),
            minibatch_indices=self.minibatch_indices,
            timesteps=self.timesteps,
            gaussian_noise=self.gaussian_noise,
            dropout_seeds=self.dropout_seeds,
            checkpoint_steps=self.checkpoint_steps,
            inference_noise_bank=self.inference_noise_bank,
        )

    @classmethod
    def load(cls, path: Path) -> "AdaptationReplay":
        values = np.load(path)
        return cls(
            int(values["initialization_seed"]),
            np.asarray(values["minibatch_indices"], np.int64),
            np.asarray(values["timesteps"], np.int64),
            np.asarray(values["gaussian_noise"], np.float32),
            np.asarray(values["dropout_seeds"], np.int64),
            np.asarray(values["checkpoint_steps"], np.int64),
            np.asarray(values["inference_noise_bank"], np.float32),
        )

    def validate(self, *, pair_count: int, validation_count: int, signal_length: int) -> None:
        updates, batch = self.minibatch_indices.shape
        if self.timesteps.shape != (updates, batch):
            raise ValueError("timestep schedule differs from minibatch schedule")
        if self.gaussian_noise.shape != (updates, batch, 2, signal_length):
            raise ValueError("training noise has the wrong shape")
        if self.dropout_seeds.shape != (updates,):
            raise ValueError("dropout schedule has the wrong shape")
        if int(self.minibatch_indices.max()) >= pair_count:
            raise ValueError("replay indexes outside the adaptation pairs")
        if self.inference_noise_bank.shape != (8, validation_count, 2, signal_length):
            raise ValueError("validation inference bank has the wrong shape")


__all__ = ["AdaptationReplay", "seed_all"]
