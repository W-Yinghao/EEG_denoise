#!/usr/bin/env python3
"""BCI Competition IV-2a — does calibrated ocular denoising help 4-class MI decoding?

The ambulatory panel's own tasks sit near their ceiling (ERP AUC 0.75, SSVEP CCA 0.77),
so a downstream benefit has little room to appear. 2a is the opposite regime: 4-class
motor imagery, chance 0.25, published accuracies around 0.5-0.7 — plenty of headroom.
It also ships exactly what this method needs: 22 EEG + 3 EOG channels and a dedicated
pre-task EOG block (eyes open 0-119 s, eyes closed 119-200 s, eye movements 200-366 s)
that the dataset documentation designates for ocular-artifact correction.

Two routes, sharing this preprocessing and these operators:
  route1  calibrated linear operator only (montage-agnostic, no training)
  route2  full method: a 22-channel population prior trained on 2a, guided by the
          same calibration operator

Decoding is end-to-end ours (our preprocessing, our decoder, the official T/E split);
the collaborator's downstream numbers from the SADDPM manuscript are never touched and
never compared against — every arm is compared only to our own RAW baseline.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from pf_common import ARRAYS, OUT, stat

RAW_ROOT = Path("/projects/EEG-foundation-model/BCI-IV")
DERIVED = Path("/projects/EEG-foundation-model/derived/denoiseNet/bci2a_ocular")
OUT_DIR = OUT / "bci2a"
SUBJECTS = tuple(f"A{i:02d}" for i in range(1, 10))
FS_OUT = 100
WINDOW = 512                      # 5.12 s at 100 Hz; the model requires 512
N_EEG = 22
RIDGE = 0.05
CALIB_SECONDS = 120
SEED = 20260830
MI_CODES = {"769": 0, "770": 1, "771": 2, "772": 3}   # left, right, foot, tongue
SUB_BLOCKS = 4


def _bipolar(eog: np.ndarray) -> np.ndarray:
    """2a EOG is left/central/right. VEOG = central - mean(left,right);
    HEOG = left - right. Two regressors, matching the rest of the project."""
    left, central, right = eog
    return np.stack([central - 0.5 * (left + right), left - right])


def _ridge(block: np.ndarray, latent: np.ndarray):
    y_c = block - block.mean(axis=1, keepdims=True)
    e_c = latent - latent.mean(axis=1, keepdims=True)
    gram = e_c @ e_c.T
    ridge = RIDGE * max(float(np.trace(gram) / len(gram)), 1e-12)
    operator = (y_c @ e_c.T) @ np.linalg.inv(gram + ridge * np.eye(len(gram)))
    fitted = operator @ e_c
    r2 = 1 - float(np.sum((y_c - fitted) ** 2) / max(np.sum(y_c ** 2), 1e-12))
    cond = float(np.linalg.cond(gram + ridge * np.eye(len(gram))))
    return operator, r2, cond


def prep() -> None:
    """Resample to 100 Hz, split the EOG calibration block from the MI trials."""
    import mne
    warnings.filterwarnings("ignore")
    mne.set_log_level("ERROR")
    DERIVED.mkdir(parents=True, exist_ok=True)
    report = []
    for subject in SUBJECTS:
        for session in ("T", "E"):
            out = DERIVED / f"{subject}{session}.npz"
            if out.is_file():
                report.append({"cell": f"{subject}{session}", "state": "cached"})
                continue
            path = RAW_ROOT / f"{subject}{session}.gdf"
            raw = mne.io.read_raw_gdf(path, preload=True)
            raw.resample(FS_OUT)
            names = raw.ch_names
            eeg_idx = [i for i, n in enumerate(names) if n.startswith("EEG")]
            eog_idx = [names.index(n) for n in
                       ("EOG-left", "EOG-central", "EOG-right")]
            data = raw.get_data()
            eeg = data[eeg_idx]
            eog = _bipolar(data[eog_idx])
            events, event_id = mne.events_from_annotations(raw)
            inverse = {v: k for k, v in event_id.items()}
            # calibration: the eye-movement block (code 1072) if present, else the
            # whole pre-trial segment; always disjoint from every decoded trial
            movement = [e[0] for e in events if inverse[e[2]] == "1072"]
            trials = [(e[0], MI_CODES[inverse[e[2]]]) for e in events
                      if inverse[e[2]] in MI_CODES]
            first_trial = min(t[0] for t in trials) if trials else len(eeg[0])
            calib_start = movement[0] if movement else 0
            calib_stop = min(calib_start + CALIB_SECONDS * FS_OUT, first_trial)
            # session E labels live in the companion .mat
            if not trials or session == "E":
                from scipy.io import loadmat
                mat = loadmat(RAW_ROOT / f"{subject}{session}.mat")
                labels = np.asarray(mat["classlabel"]).ravel().astype(int) - 1
                cues = [e[0] for e in events if inverse[e[2]] == "783"] or \
                       [e[0] for e in events if inverse[e[2]] == "768"]
                trials = list(zip(cues[:len(labels)], labels))
            starts = np.asarray([t[0] for t in trials])
            labels = np.asarray([t[1] for t in trials])
            keep = (starts + WINDOW) <= eeg.shape[1]
            np.savez_compressed(
                out, eeg=eeg.astype(np.float32), eog=eog.astype(np.float32),
                trial_start=starts[keep], trial_label=labels[keep],
                calib=np.asarray([calib_start, calib_stop]),
                first_trial=np.asarray(first_trial))
            report.append({"cell": f"{subject}{session}", "state": "built",
                           "trials": int(keep.sum()),
                           "calib_s": float((calib_stop - calib_start) / FS_OUT),
                           "classes": sorted(set(labels.tolist()))})
            print(json.dumps(report[-1]), flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "prep_report.json").write_text(json.dumps(report, indent=1) + "\n")


def _load(cell: str) -> dict:
    with np.load(DERIVED / f"{cell}.npz") as d:
        return {k: d[k] for k in d.files}


def _scale_and_operator(cell: dict):
    """Amplitude scale from the calibration block, then the 22x2 ridge operator and
    its four sub-block refits (for the EB posterior used by route 2)."""
    lo, hi = cell["calib"]
    eeg = cell["eeg"][:, lo:hi].astype(np.float64)
    eog = cell["eog"][:, lo:hi].astype(np.float64)
    centre = np.median(eog, axis=1, keepdims=True)
    scale = 1.4826 * np.median(np.abs(eog - centre), axis=1, keepdims=True)
    latent = (eog - centre) / np.maximum(scale, 1e-9)
    eeg_scale = float(np.sqrt(np.mean(eeg ** 2)))
    block = eeg / max(eeg_scale, 1e-9)
    operator, r2, cond = _ridge(block, latent)
    span = block.shape[1] // SUB_BLOCKS
    blocks = np.stack([_ridge(block[:, i * span:(i + 1) * span],
                              latent[:, i * span:(i + 1) * span])[0]
                       for i in range(SUB_BLOCKS)])
    rms = np.sqrt(np.mean(latent ** 2, axis=1)).clip(1e-8)
    quality = np.array([np.log(rms[0]), np.log(rms[1]), r2, np.log1p(cond)])
    return {"operator": operator, "sub_blocks": blocks, "quality": quality,
            "eeg_scale": eeg_scale, "eog_centre": centre, "eog_scale": scale}


def _trials(cell: dict, state: dict):
    """Decoded windows in the scaled coordinates the operator lives in."""
    eeg = cell["eeg"].astype(np.float64) / max(state["eeg_scale"], 1e-9)
    latent = ((cell["eog"].astype(np.float64) - state["eog_centre"])
              / np.maximum(state["eog_scale"], 1e-9))
    xs, es = [], []
    for start in cell["trial_start"]:
        xs.append(eeg[:, start:start + WINDOW])
        es.append(latent[:, start:start + WINDOW])
    return np.stack(xs), np.stack(es), cell["trial_label"]


def operators() -> None:
    states = {}
    for subject in SUBJECTS:
        for session in ("T", "E"):
            cell = f"{subject}{session}"
            state = _scale_and_operator(_load(cell))
            states[cell] = state
            print(json.dumps({"cell": cell, "operator_norm":
                              float(np.linalg.norm(state["operator"])),
                              "fit_r2": float(state["quality"][2])}), flush=True)
    np.savez_compressed(DERIVED / "operators.npz",
                        **{f"{k}_{f}": v[f] for k, v in states.items()
                           for f in ("operator", "sub_blocks", "quality")},
                        **{f"{k}_eeg_scale": np.asarray(v["eeg_scale"])
                           for k, v in states.items()})
    print(json.dumps({"cells": len(states)}))


# --------------------------------------------------------------- decoding

def _eegnet(n_classes: int, channels: int, samples: int):
    import torch
    from torch import nn

    class EEGNet(nn.Module):
        def __init__(self, f1=8, d=2, f2=16):
            super().__init__()
            self.b1 = nn.Sequential(
                nn.Conv2d(1, f1, (1, 64), padding=(0, 32), bias=False),
                nn.BatchNorm2d(f1),
                nn.Conv2d(f1, f1 * d, (channels, 1), groups=f1, bias=False),
                nn.BatchNorm2d(f1 * d), nn.ELU(),
                nn.AvgPool2d((1, 4)), nn.Dropout(0.25))
            self.b2 = nn.Sequential(
                nn.Conv2d(f1 * d, f1 * d, (1, 16), padding=(0, 8),
                          groups=f1 * d, bias=False),
                nn.Conv2d(f1 * d, f2, (1, 1), bias=False),
                nn.BatchNorm2d(f2), nn.ELU(),
                nn.AvgPool2d((1, 8)), nn.Dropout(0.25))
            with torch.no_grad():
                n = self.b2(self.b1(torch.zeros(1, 1, channels, samples))).numel()
            self.head = nn.Linear(n, n_classes)

        def forward(self, x):
            return self.head(self.b2(self.b1(x.unsqueeze(1))).flatten(1))

    return EEGNet()


def _normalise(x):
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True).clip(1e-6)
    return ((x - mu) / sd).astype(np.float32)


def _decode(train_x, train_y, test_x, test_y, seed, device,
            epochs=300, batch=32, lr=1e-3, patience=30):
    """Official 2a protocol: train on session T, test on session E. Identical budget,
    identical folds and a fresh same-seed initialisation for every arm."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    model = _eegnet(4, train_x.shape[1], train_x.shape[2]).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    cut = int(len(train_x) * 0.85)
    order = np.random.default_rng(seed).permutation(len(train_x))
    tr, va = order[:cut], order[cut:]
    xt = torch.from_numpy(train_x[tr]).float().to(device)
    yt = torch.from_numpy(train_y[tr]).long().to(device)
    xv = torch.from_numpy(train_x[va]).float().to(device)
    yv = torch.from_numpy(train_y[va]).long().to(device)
    best, best_state, waited = float("inf"), None, 0
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        idx = torch.randperm(len(xt), generator=generator).to(device)
        for start in range(0, len(idx), batch):
            chunk = idx[start:start + batch]
            optimiser.zero_grad(set_to_none=True)
            criterion(model(xt[chunk]), yt[chunk]).backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            score = float(criterion(model(xv), yv))
        if score < best - 1e-5:
            best, waited = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(test_x).float().to(device)).argmax(1).cpu().numpy()
    accuracy = float(np.mean(pred == test_y))
    # Cohen's kappa against chance for a 4-class balanced problem
    confusion = np.zeros((4, 4))
    for t, p in zip(test_y, pred):
        confusion[t, p] += 1
    total = confusion.sum()
    expected = float((confusion.sum(0) * confusion.sum(1)).sum() / total ** 2)
    kappa = (accuracy - expected) / max(1 - expected, 1e-9)
    return accuracy, float(kappa)


def route1() -> None:
    """Calibrated linear operator only. No training, montage-agnostic."""
    import torch
    device = torch.device("cuda")
    states = {f"{s}{ss}": _scale_and_operator(_load(f"{s}{ss}"))
              for s in SUBJECTS for ss in ("T", "E")}
    rows = []
    for subject in SUBJECTS:
        arms = {}
        for session in ("T", "E"):
            cell = f"{subject}{session}"
            x, e, y = _trials(_load(cell), states[cell])
            own = states[cell]["operator"]
            pool = [states[f"{o}{session}"]["operator"] for o in SUBJECTS if o != subject]
            population = np.mean(pool, axis=0)
            wrong = pool[0]
            arms.setdefault("RAW", {})[session] = (x, y)
            for name, op in (("LINEAR_MATCH", own), ("LINEAR_POP", population),
                             ("LINEAR_WRONG", wrong)):
                cleaned = np.stack([x[i] - op @ e[i] for i in range(len(x))])
                arms.setdefault(name, {})[session] = (cleaned, y)
        for arm, per in sorted(arms.items()):
            xtr, ytr = per["T"]
            xte, yte = per["E"]
            accuracy, kappa = _decode(_normalise(xtr), ytr, _normalise(xte), yte,
                                      SEED, device)
            rows.append({"subject": subject, "arm": arm, "accuracy": accuracy,
                         "kappa": kappa, "n_train": int(len(ytr)),
                         "n_test": int(len(yte))})
            print(json.dumps(rows[-1]), flush=True)
    _report(rows, "route1", "calibrated linear operator, EEGNet, official T/E split")


def _report(rows, tag, description) -> None:
    per = {}
    for row in rows:
        per.setdefault(row["arm"], {})[row["subject"]] = row["accuracy"]
    reference = per.get("RAW", {})
    contrasts = {}
    for arm, values in per.items():
        if arm == "RAW":
            continue
        common = [s for s in values if s in reference]
        if common:
            contrasts[f"{arm}_minus_RAW"] = stat([values[s] - reference[s]
                                                  for s in common])
    decision = {"endpoint": description, "chance": 0.25,
                "accuracy_means": {a: float(np.mean(list(v.values())))
                                   for a, v in per.items()},
                "kappa_means": {a: float(np.mean(
                    [r["kappa"] for r in rows if r["arm"] == a])) for a in per},
                "contrasts_vs_RAW": contrasts,
                "subjects": len(reference), "rows": rows}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{tag}.json").write_text(json.dumps(decision, indent=2,
                                                    sort_keys=True) + "\n")
    np.savez_compressed(ARRAYS / f"bci2a_{tag}.npz",
                        decision=np.asarray(json.dumps(decision)))
    print(json.dumps({"accuracy": decision["accuracy_means"],
                      "contrasts": {k: round(v["mean"], 4)
                                    for k, v in contrasts.items()}}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prep", "operators", "route1",
                                         "episodes", "train", "route2"])
    parser.add_argument("--fold", type=int, default=None)
    args = parser.parse_args()
    if args.mode == "prep":
        prep()
    elif args.mode == "operators":
        operators()
    elif args.mode == "route1":
        route1()
    else:
        from bci2a_route2 import episodes, train, route2
        {"episodes": episodes, "train": lambda: train(args.fold),
         "route2": route2}[args.mode]()


if __name__ == "__main__":
    main()
