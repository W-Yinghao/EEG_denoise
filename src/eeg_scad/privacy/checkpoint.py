"""Complete resumable training state for V32P development jobs."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def save_resume_state(path: str | Path, *, model, optimizer, epoch: int, global_step: int, metadata: dict | None = None) -> None:
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "metadata": metadata or {},
    }, Path(path))


def load_resume_state(path: str | Path, *, model, optimizer, map_location="cpu") -> dict:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"])
    random.setstate(payload["python_rng"]); np.random.set_state(payload["numpy_rng"]); torch.set_rng_state(payload["torch_rng"])
    if torch.cuda.is_available() and payload["cuda_rng"]: torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return {key: payload[key] for key in ("epoch", "global_step", "metadata")}


__all__ = ["load_resume_state", "save_resume_state"]
