"""On-disk cache of preprocessed subject windows.

Preprocessing (MOABB parse + MNE filtering + windowing) takes ~10-20 s per subject; many training
and evaluation jobs reload the same windows. This caches each subject's :class:`SubjectWindows`
(both sessions) to ``artifacts/cache/`` keyed by a hash of the preprocessing config, so a stale
config never silently reuses old windows.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict

import torch

from .bcic2a import SubjectWindows, load_subject
from .config import DataConfig

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "cache"


def config_hash(cfg: DataConfig) -> str:
    """Short hash of the preprocessing-relevant config fields (8 hex chars)."""
    payload = json.dumps(
        {
            "dataset": cfg.dataset.__dict__,
            "preprocess": cfg.preprocess.__dict__,
            "epoch": cfg.epoch.__dict__,
            "window": cfg.window.__dict__,
        },
        sort_keys=True,
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]


def load_subject_cached(
    subject: int,
    cfg: DataConfig,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Dict[str, SubjectWindows]:
    """Load a subject's windows from cache, computing + caching on a miss."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"A{subject:02d}_{config_hash(cfg)}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    per_session = load_subject(subject, cfg)
    torch.save(per_session, path)
    return per_session
