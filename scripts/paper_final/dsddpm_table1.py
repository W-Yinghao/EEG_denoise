#!/usr/bin/env python
"""F5 — DS-DDPM Table I reproduction: the 9x9 cross-subject EEGNet matrix.

Their Table I reports, per column j, an EEGNet trained on subject j's data and
evaluated on every subject i's data (diagonal = within-subject), with M = the
column mean; columns come in an ICA arm and a DS-DDPM arm. Their repo ships no
classifier script, so the protocol here is frozen from what the repo does
document, before any numbers are seen:
  windows     their DatasetLoader verbatim: slice [:, :, 750:1500], transpose,
              224-sample windows / stride 75  ->  one [8, 224, 22] stack/trial
  backbone    their src/EEGNet.py EEG_Net_8_Stack (112-d), + Linear(112, 4)
  recipe      from their own max_acc.pth path string ("batchsize256_lr0.05_
              gamma0.5_step20_maxepoch100"): SGD lr .05, StepLR(20, .5),
              100 epochs, batch 256; no input normalization (theirs is
              commented out); seed 20260831
  split       train on subject j session T, test on subject i session E
  arms        raw (single_sep), ica (ica_sep), dsddpm (dsddpm_sep, after F3)
GPU node.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
UPSTREAM = REPO / ".external/DS-DDPM"
DATA_ROOT = Path("/projects/EEG-foundation-model/derived/denoiseNet/dsddpm_repro/data")
OUT_DIR = REPO / "results/paper_final/dsddpm"
ARM_DIRS = {"raw": "single_sep", "ica": "ica_sep", "dsddpm": "dsddpm_sep"}
SEED = 20260831
EPOCHS, BATCH, LR, STEP, GAMMA = 100, 256, 0.05, 20, 0.5
WINDOW, STRIDE, SLICE_LO, SLICE_HI = 224, 75, 750, 1500


def _stacks(mat_path, prefix):
    from scipy.io import loadmat
    data = loadmat(mat_path)
    x = np.asarray(data[f"{prefix}_x"], np.float32)[:, :, SLICE_LO:SLICE_HI]
    y = np.asarray(data[f"{prefix}_y"], np.float64).ravel().astype(int) - 1
    x = np.transpose(x, (0, 2, 1))                     # [trials, 750, 22]
    starts = range(0, x.shape[1] - WINDOW, STRIDE)     # their generator: 8 windows
    stacks = np.stack([np.stack([t[s:s + WINDOW] for s in starts]) for t in x])
    return _apply_norm(stacks.astype(np.float32)), y   # [trials, 8, 224, 22]


RECIPES = {"paper": dict(opt="sgd", lr=0.05, epochs=100, batch=256),
           "adam": dict(opt="adam", lr=1e-3, epochs=200, batch=64)}
RECIPE = "paper"
NORM = "none"


def _apply_norm(stacks):
    if NORM == "l2":       # their commented loader line, verbatim semantics:
        import torch       # F.normalize(Tensor(data[i]), p=2, dim=2) on [8,224,22]
        import torch.nn.functional as F
        return F.normalize(torch.from_numpy(stacks), p=2, dim=3).numpy()
    if NORM == "zscore":
        mu = stacks.mean(axis=2, keepdims=True)
        sd = stacks.std(axis=2, keepdims=True).clip(1e-6)
        return ((stacks - mu) / sd).astype(np.float32)
    return stacks


def _fit(train_x, train_y, device):
    import torch
    from torch import nn
    sys.path.insert(0, str(UPSTREAM))
    from src.EEGNet import EEG_Net_8_Stack

    r = RECIPES[RECIPE]
    torch.manual_seed(SEED)
    backbone = EEG_Net_8_Stack().to(device)
    head = nn.Linear(4 * 2 * 14, 4).to(device)
    params = list(backbone.parameters()) + list(head.parameters())
    if r["opt"] == "sgd":
        opt = torch.optim.SGD(params, lr=r["lr"])
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=STEP, gamma=GAMMA)
    else:
        opt = torch.optim.Adam(params, lr=r["lr"])
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10**9, gamma=1.0)
    criterion = nn.CrossEntropyLoss()
    xt = torch.from_numpy(train_x).to(device)
    yt = torch.from_numpy(train_y).long().to(device)
    generator = torch.Generator().manual_seed(SEED)
    for _ in range(r["epochs"]):
        backbone.train(); head.train()
        order = torch.randperm(len(xt), generator=generator).to(device)
        for s in range(0, len(order), r["batch"]):
            i = order[s:s + r["batch"]]
            opt.zero_grad()
            loss = criterion(head(backbone(xt[i])), yt[i])
            loss.backward(); opt.step()
        sched.step()
    return backbone, head


def run(arm: str) -> None:
    import torch
    device = torch.device("cuda")
    root = DATA_ROOT / ARM_DIRS[arm]
    train_sets, test_sets = {}, {}
    for j in range(1, 10):
        p = root / f"single_subject_data_{j}.mat"
        train_sets[j] = _stacks(p, "train")
        test_sets[j] = _stacks(p, "test")
    matrix = np.zeros((9, 9))
    for j in range(1, 10):
        backbone, head = _fit(*train_sets[j], device)
        backbone.eval(); head.eval()
        with torch.no_grad():
            for i in range(1, 10):
                x, y = test_sets[i]
                preds = []
                for s in range(0, len(x), BATCH):
                    logits = head(backbone(
                        torch.from_numpy(x[s:s + BATCH]).to(device)))
                    preds.append(logits.argmax(1).cpu().numpy())
                matrix[i - 1, j - 1] = float(
                    np.mean(np.concatenate(preds) == y))
        print(json.dumps({"arm": arm, "trained_on": j,
                          "col": [round(v, 4) for v in matrix[:, j - 1]]}),
              flush=True)
    out = {"arm": arm, "matrix_rows_test_cols_train": matrix.round(4).tolist(),
           "column_means_M": matrix.mean(axis=0).round(4).tolist(),
           "grand_mean": round(float(matrix.mean()), 4),
           "diagonal": np.diag(matrix).round(4).tolist(),
           "published_reference": {"ICA_M_mean": 50.58, "DSDDPM_M_mean": 52.87,
                                   "units": "percent, their Table I"},
           "protocol": {"recipe": RECIPE, "norm": NORM, **RECIPES[RECIPE], "seed": SEED}}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"table1_{arm}_{RECIPE}_{NORM}.json").write_text(
        json.dumps(out, indent=2) + "\n")
    print(json.dumps({"arm": arm, "M": out["column_means_M"],
                      "grand": out["grand_mean"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=list(ARM_DIRS))
    parser.add_argument("--recipe", choices=list(RECIPES), default="paper")
    parser.add_argument("--norm", choices=["none", "l2", "zscore"], default="none")
    args = parser.parse_args()
    global RECIPE, NORM
    RECIPE = args.recipe
    NORM = args.norm
    run(args.arm)


if __name__ == "__main__":
    main()
