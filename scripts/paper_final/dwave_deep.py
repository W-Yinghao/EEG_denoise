#!/usr/bin/env python3
"""D-wave deep-decoder endpoint — EEGNet on the already-denoised waveforms.

The shallow endpoints (CCA for SSVEP, shrinkage-LDA for ERP) answer "does cleaning
change decodability under the field-standard decoder for each paradigm". They do not
answer whether a decoder with enough capacity to exploit the cleaning would see a
difference the standard decoder misses. This module adds that second endpoint.

No diffusion inference is re-run: every arm's denoised waveform is read from the banked
D-wave arrays. Only the decoder changes.

Protocol (frozen here, before the numbers are seen)
  architecture   EEGNet-8,2 (F1=8, D=2, F2=16), 46 channels, the paradigm's window
  training       within participant, 5-fold stratified CV over that participant's
                 trials pooled across sessions, identical folds for every arm
                 (the fold assignment is a function of the trial index, not of the arm)
  budget         200 epochs, AdamW 1e-3, batch 32, early stop on the fold's validation
                 loss with patience 20; identical for every arm
  seeds          fold seed 20260830 + trial-index hash; per-arm weights are freshly
                 initialised from the same seed so no arm inherits another's optimum
  endpoints      SSVEP: 3-class accuracy.  ERP: target/nontarget AUC.
  contrasts      arm minus RAW, participant-first, 5000-draw participant bootstrap
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pf_common import ARRAYS, OUT, stat

DWAVE = OUT / "dwave"
DEN = DWAVE / "denoised"
ARMS = ("RAW", "LINEAR", "NO_A0", "MATCH", "POP", "ICA", "ASR", "SGEYESUB")
SEED = 20260830
EPOCHS, BATCH, LR, PATIENCE = 200, 32, 1e-3, 20
FOLDS = 5
EPOCH_PRE, EPOCH_POST = 20, 80
CALIB = 12000
WINDOW = 512


def build_eegnet(n_classes: int, samples: int, channels: int = 46):
    import torch
    from torch import nn

    class EEGNet(nn.Module):
        def __init__(self, f1=8, d=2, f2=16):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(1, f1, (1, 64), padding=(0, 32), bias=False),
                nn.BatchNorm2d(f1),
                nn.Conv2d(f1, f1 * d, (channels, 1), groups=f1, bias=False),
                nn.BatchNorm2d(f1 * d), nn.ELU(),
                nn.AvgPool2d((1, 4)), nn.Dropout(0.25))
            self.block2 = nn.Sequential(
                nn.Conv2d(f1 * d, f1 * d, (1, 16), padding=(0, 8),
                          groups=f1 * d, bias=False),
                nn.Conv2d(f1 * d, f2, (1, 1), bias=False),
                nn.BatchNorm2d(f2), nn.ELU(),
                nn.AvgPool2d((1, 8)), nn.Dropout(0.25))
            with torch.no_grad():
                n = self.block2(self.block1(
                    torch.zeros(1, 1, channels, samples))).numel()
            self.head = nn.Linear(n, n_classes)

        def forward(self, x):
            h = self.block2(self.block1(x.unsqueeze(1)))
            return self.head(h.flatten(1))

    return EEGNet()


def _fit_predict(train_x, train_y, test_x, n_classes, seed, device):
    """One fold. Fresh initialisation from `seed` for every arm."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    model = build_eegnet(n_classes, train_x.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    # last 20% of the training fold is the early-stopping split
    cut = max(int(len(train_x) * 0.8), 1)
    xt = torch.from_numpy(train_x[:cut]).float().to(device)
    yt = torch.from_numpy(train_y[:cut]).long().to(device)
    xv = torch.from_numpy(train_x[cut:]).float().to(device)
    yv = torch.from_numpy(train_y[cut:]).long().to(device)
    best, best_state, waited = float("inf"), None, 0
    generator = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        model.train()
        order = torch.randperm(len(xt), generator=generator).to(device)
        for start in range(0, len(order), BATCH):
            index = order[start:start + BATCH]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xt[index]), yt[index])
            loss.backward()
            optimizer.step()
        if len(xv) == 0:
            continue
        model.eval()
        with torch.no_grad():
            score = float(criterion(model(xv), yv))
        if score < best - 1e-5:
            best, waited = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(test_x).float().to(device))
        return torch.softmax(logits, dim=1).cpu().numpy()


def _normalise(x):
    """Per-trial per-channel z-score; identical for every arm."""
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True).clip(1e-6)
    return ((x - mu) / sd).astype(np.float32)


def _collect(task: str, heldout: bool):
    """Pool each participant's trials across session-task cells, per arm."""
    from pf_common import SEALED
    prefix = "d1_" if task == "SSVEP" else "d2_"
    data = defaultdict(lambda: {"y": [], "arms": defaultdict(list)})
    for path in sorted(DEN.glob(f"{prefix}*.npz")):
        subject = path.stem.split("_", 1)[1].split("|")[0]
        if (subject in SEALED) != heldout:
            continue
        cpu_path = DEN / f"cpuarms_{path.stem.split('_', 1)[1]}.npz"
        with np.load(path, allow_pickle=False) as d:
            cpu = np.load(cpu_path, allow_pickle=False) if cpu_path.is_file() else None
            if task == "SSVEP":
                labels = d["labels"]
                if not len(labels):
                    continue
                keep = slice(None)
                y = np.searchsorted(np.unique(labels), labels)
            else:
                meta, y = _erp_epochs(path, d)
                if meta is None:
                    continue
                keep = meta
            data[subject]["y"].append(np.asarray(y))
            for arm in ARMS:
                source = d if arm in d.files else cpu
                if source is None or arm not in source.files:
                    continue
                windows = source[arm]
                if task == "SSVEP":
                    trials = windows[keep]
                else:
                    trials = np.stack([windows[t][:, o:o + EPOCH_PRE + EPOCH_POST]
                                       for t, o in keep])
                data[subject]["arms"][arm].append(_normalise(trials))
            if cpu is not None:
                cpu.close()
    return data


def _erp_epochs(path: Path, d):
    """Map ERP events onto the banked tiles; identical selection for every arm."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dwave import load_events
    cell = path.stem.split("_", 1)[1].split("|")
    events = load_events(cell[0], cell[1], cell[2])
    tiles = list(d["starts"])
    tile_of = {s: i for i, s in enumerate(tiles)}
    meta, labels = [], []
    for e in events:
        lo, hi = e["onset"] - EPOCH_PRE, e["onset"] + EPOCH_POST
        if lo < CALIB:
            continue
        tlo = (lo - CALIB) // WINDOW
        if tlo != (hi - 1 - CALIB) // WINDOW:
            continue
        start = CALIB + tlo * WINDOW
        if start not in tile_of:
            continue
        meta.append((tile_of[start], lo - start))
        labels.append(1 if e["value"] == 2 else 0)
    if len(meta) < 40:
        return None, None
    return meta, np.asarray(labels)


def run(task: str, heldout: bool) -> None:
    import torch
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    device = torch.device("cuda")
    data = _collect(task, heldout)
    n_classes = 3 if task == "SSVEP" else 2
    rows = []
    for subject, payload in sorted(data.items()):
        y = np.concatenate(payload["y"])
        arms = {a: np.concatenate(v) for a, v in payload["arms"].items()
                if len(v) == len(payload["y"])}
        if len(y) < 40 or "RAW" not in arms:
            continue
        splitter = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
        folds = list(splitter.split(np.zeros(len(y)), y))
        for arm, x in sorted(arms.items()):
            scores = np.zeros((len(y), n_classes))
            for fold_index, (train, test) in enumerate(folds):
                scores[test] = _fit_predict(x[train], y[train], x[test], n_classes,
                                            SEED + fold_index, device)
            if task == "SSVEP":
                value = float(np.mean(scores.argmax(1) == y))
            else:
                value = float(roc_auc_score(y, scores[:, 1]))
            rows.append({"participant": subject, "arm": arm, "metric": value,
                         "n_trials": int(len(y))})
            print(json.dumps(rows[-1]), flush=True)

    per = {a: {r["participant"]: r["metric"] for r in rows if r["arm"] == a}
           for a in ARMS}
    per = {a: v for a, v in per.items() if v}
    reference = per["RAW"]
    contrasts = {}
    for arm, values in per.items():
        if arm == "RAW":
            continue
        common = [p for p in values if p in reference]
        if common:
            contrasts[f"{arm}_minus_RAW"] = stat([values[p] - reference[p]
                                                  for p in common])
    tag = f"{task.lower()}_{'heldout' if heldout else 'dev'}"
    decision = {
        "endpoint": f"EEGNet-8,2 within-participant {FOLDS}-fold CV",
        "task": task, "cohort": "heldout" if heldout else "dev",
        "metric": "accuracy" if task == "SSVEP" else "auc",
        "means": {a: float(np.mean(list(v.values()))) for a, v in per.items()},
        "contrasts_vs_RAW": contrasts,
        "participants": len(reference), "rows": rows,
    }
    DWAVE.mkdir(parents=True, exist_ok=True)
    (DWAVE / f"deep_{tag}.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(ARRAYS / f"dwave_deep_{tag}.npz",
                        decision=np.asarray(json.dumps(decision)))
    print(json.dumps({"task": task, "cohort": decision["cohort"],
                      "means": decision["means"]}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["SSVEP", "ERP"])
    parser.add_argument("--heldout", action="store_true")
    args = parser.parse_args()
    run(args.task, args.heldout)


if __name__ == "__main__":
    main()
