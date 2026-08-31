"""Checkpoint save/load with config + git provenance (handoff §6)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn


def git_commit_hash() -> str:
    """Return the current git commit hash, or 'unknown' if unavailable."""
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model weights + config + git hash to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "config": config,
        "git_commit": git_commit_hash(),
        **(extra or {}),
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
    """Load a checkpoint payload saved by :func:`save_checkpoint`."""
    return torch.load(path, map_location=map_location, weights_only=False)
