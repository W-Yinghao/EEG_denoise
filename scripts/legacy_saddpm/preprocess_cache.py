#!/usr/bin/env python
"""Preprocess and cache all 9 subjects' windows once (handoff §3) for reuse by training/eval jobs.

Usage:
    python scripts/preprocess_cache.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from saddpm.data.cache import config_hash, load_subject_cached  # noqa: E402
from saddpm.data.config import DataConfig  # noqa: E402


def main() -> int:
    cfg = DataConfig.from_yaml(REPO_ROOT / "configs" / "data.yaml")
    print(f"[cache] preprocessing config hash = {config_hash(cfg)}")
    for s in range(1, cfg.dataset.n_subjects + 1):
        per_session = load_subject_cached(s, cfg)
        for sw in per_session.values():
            print(f"  {sw.summary()}")
    print("[cache] done; cached under artifacts/cache/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
