#!/usr/bin/env python
"""M0 deliverable: load one subject via MOABB, print shapes, plot one preprocessed window.

Usage:
    python scripts/m0_load_subject.py --subject 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / login node
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from saddpm.data.config import DataConfig  # noqa: E402
from saddpm.data.bcic2a import load_subject  # noqa: E402
from saddpm.utils.seed import seed_everything  # noqa: E402


def plot_window(window: np.ndarray, ch_names: list[str], pad: tuple[int, int], out_path: Path, title: str) -> None:
    """Plot a single (n_channels, length) window with channels stacked vertically.

    The zero-padded edges ([DD-6]) are shaded so the 500 real samples are distinguishable.
    """
    n_channels, length = window.shape
    offset = 4.0  # vertical spacing between channels (z-scored signals are ~unit variance)
    fig, ax = plt.subplots(figsize=(11, 9))
    for c in range(n_channels):
        ax.plot(np.arange(length), window[c] + c * offset, linewidth=0.6)
    left, right = pad
    if left:
        ax.axvspan(0, left, color="0.85", alpha=0.6)
    if right:
        ax.axvspan(length - right, length, color="0.85", alpha=0.6, label="zero-pad ([DD-6])")
    ax.set_yticks([c * offset for c in range(n_channels)])
    ax.set_yticklabels(ch_names, fontsize=6)
    ax.set_xlabel("sample (512 = 500 real + zero-pad)")
    ax.set_ylabel("EEG channel (z-scored, offset)")
    ax.set_title(title)
    ax.set_xlim(0, length - 1)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "data.yaml"))
    parser.add_argument("--window-index", type=int, default=0, help="which window to plot")
    args = parser.parse_args()

    cfg = DataConfig.from_yaml(args.config)
    seed_everything(cfg.seed)
    print(f"[M0] seed={cfg.seed} | loading subject A{args.subject:02d} via MOABB ({cfg.dataset.name})")

    per_session = load_subject(args.subject, cfg)
    for sw in per_session.values():
        print(f"  {sw.summary()}")
        print(
            f"    windows dtype={sw.windows.dtype} "
            f"mean={sw.windows.mean():+.4f} std={sw.windows.std():.4f} "
            f"min={sw.windows.min():+.3f} max={sw.windows.max():+.3f}"
        )

    # Plot one preprocessed window from the training session.
    train_key = next(k for k, v in per_session.items() if v.session_role == "T")
    sw = per_session[train_key]
    idx = args.window_index
    out_path = REPO_ROOT / "artifacts" / "figures" / f"m0_subject{args.subject:02d}_window{idx}.png"
    title = (
        f"BCI-IV-2a A{args.subject:02d} {sw.session} | window {idx} "
        f"(MI={sw.class_names[sw.mi_labels[idx]]}) | shape={tuple(sw.windows[idx].shape)}"
    )
    plot_window(sw.windows[idx], sw.ch_names, sw.pad, out_path, title)
    print(f"[M0] saved preprocessed-window figure -> {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
