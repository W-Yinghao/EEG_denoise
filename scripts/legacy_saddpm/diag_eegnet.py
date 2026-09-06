#!/usr/bin/env python
"""Diagnostic: within-subject EEGNet accuracy on RAW (non-denoised) BCI-IV-2a windows.

Isolates the downstream classifier quality from denoising. The manuscript reports ~74-92%
within-subject 4-class accuracy; our M7 reproduction got ~34-47%. Tests kernel_length 64 vs 125
(125 = sfreq/2 for 250 Hz, the standard EEGNet setting). Also tries per-trial vote aggregation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from saddpm.data.config import DataConfig  # noqa: E402
from saddpm.data.datasets import load_session_windows  # noqa: E402
from saddpm.eval.downstream import train_eegnet  # noqa: E402
from saddpm.models.eegnet import EEGNet, EEGNetConfig  # noqa: E402
from saddpm.utils.seed import seed_everything  # noqa: E402


@torch.no_grad()
def _trial_vote_acc(model, windows, labels, trial_index, device):
    """Per-trial accuracy by averaging window logits within each trial."""
    model.eval()
    x = torch.from_numpy(windows).float()
    logits = []
    for i in range(0, len(x), 256):
        logits.append(model(x[i:i + 256].to(device)).cpu())
    logits = torch.cat(logits).numpy()
    correct = total = 0
    for tr in np.unique(trial_index):
        m = trial_index == tr
        pred = logits[m].mean(0).argmax()
        correct += int(pred == labels[m][0]); total += 1
    return correct / total


def main() -> int:
    cfg = DataConfig.from_yaml(REPO_ROOT / "configs" / "data.yaml")
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subjects = [1, 3, 7]  # representative
    for kern in (64, 125):
        eeg_cfg = EEGNetConfig(n_channels=22, n_times=512, kernel_length=kern, epochs=100, lr=1e-3)
        win_accs, trial_accs = [], []
        for s in subjects:
            T = load_session_windows(s, cfg, "T")
            E = load_session_windows(s, cfg, "E")
            model = train_eegnet(T.windows, T.mi_labels, eeg_cfg, device, cfg.seed)
            model.eval()
            x = torch.from_numpy(E.windows).float()
            preds = torch.cat([model(x[i:i + 256].to(device)).argmax(1).cpu() for i in range(0, len(x), 256)]).numpy()
            win_acc = float((preds == E.mi_labels).mean())
            trial_acc = _trial_vote_acc(model, E.windows, E.mi_labels, E.trial_index, device)
            win_accs.append(win_acc); trial_accs.append(trial_acc)
            print(f"  kern={kern} A{s:02d}: per-window {win_acc:.3f}  per-trial-vote {trial_acc:.3f}")
        print(f"[kern={kern}] mean within-subject: per-window {np.mean(win_accs):.3f}  per-trial {np.mean(trial_accs):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
