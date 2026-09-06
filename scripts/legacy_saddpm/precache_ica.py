#!/usr/bin/env python
"""Precompute + cache ICA-denoised windows for all 9 subjects × {T, E} (handoff §8.1).

Runs on CPU so the M7 GPU job loads ICA results from cache instead of fitting ICA on the GPU node.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from saddpm.baselines.ica import ICAConfig, ica_denoise_session  # noqa: E402
from saddpm.data.config import DataConfig  # noqa: E402


def main() -> int:
    data_cfg = DataConfig.from_yaml(REPO_ROOT / "configs" / "data.yaml")
    ica_cfg = ICAConfig.from_yaml(REPO_ROOT / "configs" / "ica.yaml")
    for s in range(1, data_cfg.dataset.n_subjects + 1):
        for role in ("T", "E"):
            r = ica_denoise_session(s, role, data_cfg, ica_cfg)
            print(f"  A{s:02d}-{role}: windows {r.windows.shape}, EOG comps excluded {r.n_excluded}")
    print("[ica-cache] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
