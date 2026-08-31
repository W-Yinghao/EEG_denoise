#!/usr/bin/env python3
"""EEGDfus SSED reproduction (their Tables I/VII column) from the local Klados v4 copy.

SSED in the EEGDfus paper IS the Klados semi-simulated dataset (54 x 19ch x 200Hz
recordings, V/HEOG-synthesized contamination) — we hold the official v4 MAT files.
The authoritative construction is upstream Data_Preparation/data_prepare_ssed.py:
  eeg  = per-channel standardized PURE signal
  eog  = per-channel standardized (CONTAMINATED - PURE)
  noise = eeg + eog        (components re-standardized, then summed)
  rows  = single channels unrolled across recordings
  split = 80/10/10 at row level, random_state=666

Upstream defect preserved and reported, never silently repaired: train_ssed.py line
43 splits `range(len(val_test_idx))` instead of `val_test_idx`, so the released test
rows are drawn from the first rows of the WHOLE set and overlap the training split.
We evaluate both ways: `released` (their indices verbatim) and `strict` (the split
they evidently intended), clearly labelled.

Modes: build | train (as-released driver, dynamic import, nothing vendored) | eval
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
UPSTREAM = REPO / ".external/EEGDfus"
KLADOS = Path("/projects/EEG-foundation-model/klados_bamidis/v4")
DERIVED = Path("/projects/EEG-foundation-model/derived/denoiseNet/eegdfus_ssed")
OUT_DIR = REPO / "results/paper_final/eegdfus_ssed"
SEED_SPLIT = 666          # theirs, verbatim


def build() -> None:
    from scipy.io import loadmat
    DERIVED.mkdir(parents=True, exist_ok=True)
    pure = loadmat(KLADOS / "Pure_Data.mat")
    cont = loadmat(KLADOS / "Contaminated_Data.mat")
    def stems(mat, suffix):
        out = {}
        for k in mat:
            if k.startswith("__"):
                continue
            assert k.endswith(suffix), f"unexpected key {k}"
            out[k[:-len(suffix)]] = np.asarray(mat[k], np.float64)
        return out
    pure_by = stems(pure, "_resampled")
    cont_by = stems(cont, "_con")
    assert set(pure_by) == set(cont_by), "record stems differ"
    pk = sorted(pure_by, key=lambda s: int(s.replace("sim", "")))
    prows = [pure_by[k] for k in pk]
    crows = [cont_by[k] for k in pk]
    length = min(min(r.shape[1] for r in prows), min(r.shape[1] for r in crows))
    # the upstream SSED model hardcodes 400-sample (2 s @ 200 Hz) inputs (FiLM beta
    # length, and Table X's '2-second duration' wording), so recordings are cut into
    # non-overlapping 400-sample windows before the upstream standardize-and-sum step
    SEG = 400
    n_seg = length // SEG
    eeg = np.stack([r[:, :n_seg * SEG].reshape(len(r), n_seg, SEG)
                    for r in prows]).transpose(0, 2, 1, 3).reshape(-1, 19, SEG)
    noisy = np.stack([r[:, :n_seg * SEG].reshape(len(r), n_seg, SEG)
                      for r in crows]).transpose(0, 2, 1, 3).reshape(-1, 19, SEG)
    np.save(DERIVED / "ssed_eeg.npy", eeg)
    np.save(DERIVED / "ssed_noise.npy", noisy)
    report = {"records": len(pk), "channels": int(eeg.shape[1]),
              "samples": int(length), "keys_sample": pk[:3]}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


def _dataset():
    sys.path.insert(0, str(UPSTREAM))
    from Data_Preparation.data_prepare_ssed import prepare_data
    return prepare_data(str(DERIVED / "ssed_noise.npy"),
                        str(DERIVED / "ssed_eeg.npy"))


def _splits(n):
    from sklearn.model_selection import train_test_split
    train_idx, val_test_idx = train_test_split(list(range(n)), test_size=0.2,
                                               random_state=SEED_SPLIT)
    released_test, released_val = train_test_split(
        list(range(len(val_test_idx))), test_size=0.5, random_state=SEED_SPLIT)
    strict_test, strict_val = train_test_split(val_test_idx, test_size=0.5,
                                               random_state=SEED_SPLIT)
    return {"train": train_idx,
            "released_test": released_test, "released_val": released_val,
            "strict_test": strict_test, "strict_val": strict_val}


def train() -> None:
    import os
    import torch
    import yaml
    from torch.utils.data import DataLoader, Subset, TensorDataset

    os.chdir(UPSTREAM)                      # their relative config/ imports
    sys.path.insert(0, str(UPSTREAM))
    from DDPM import DDPM
    from denoising_model_seed import DualBranchDenoisingModel
    from utils import train as upstream_train

    config = yaml.safe_load((UPSTREAM / "config/base.yaml").read_text())
    # paper: SSED uses feats 64 is the base; qkv/head adjustments are inside the
    # model defaults for the seed variant — base.yaml as released
    x, y = _dataset()
    x = torch.FloatTensor(np.asarray(x)).unsqueeze(1)
    y = torch.FloatTensor(np.asarray(y)).unsqueeze(1)
    dataset = TensorDataset(y, x)
    splits = _splits(len(dataset))
    torch.manual_seed(20260831)             # upstream sets none; disclosed
    folder = DERIVED / "check_points"
    folder.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(Subset(dataset, splits["train"]),
                              batch_size=config["train"]["batch_size"],
                              shuffle=True, drop_last=True, num_workers=8)
    val_loader = DataLoader(Subset(dataset, splits["released_val"]),
                            batch_size=config["train"]["batch_size"],
                            drop_last=True, num_workers=8)
    device = "cuda:0"
    base_model = DualBranchDenoisingModel(config["train"]["feats"]).to(device)
    model = DDPM(base_model, config, device)
    upstream_train(model, config["train"], train_loader, device,
                   valid_loader=val_loader, valid_epoch_interval=10,
                   foldername=str(folder) + "/")
    print(json.dumps({"trained": True, "folder": str(folder)}))


def evaluate() -> None:
    import os
    import torch
    from torch.utils.data import DataLoader, Subset, TensorDataset

    os.chdir(UPSTREAM)
    sys.path.insert(0, str(UPSTREAM))
    import yaml
    from DDPM import DDPM
    from denoising_model_seed import DualBranchDenoisingModel

    config = yaml.safe_load((UPSTREAM / "config/base.yaml").read_text())
    x, y = _dataset()
    x = torch.FloatTensor(np.asarray(x)).unsqueeze(1)
    y = torch.FloatTensor(np.asarray(y)).unsqueeze(1)
    dataset = TensorDataset(y, x)
    splits = _splits(len(dataset))
    device = "cuda:0"
    base_model = DualBranchDenoisingModel(config["train"]["feats"]).to(device)
    model = DDPM(base_model, config, device)
    ckpt = DERIVED / "check_points/model.pth"
    if not ckpt.is_file():
        candidates = sorted((DERIVED / "check_points").glob("*.pth"))
        assert candidates, "no checkpoint found"
        ckpt = candidates[-1]
    base_model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval() if hasattr(model, "eval") else None

    def rrmse_t(a, b):
        return float(np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-12))

    def rrmse_s(a, b, fs=200):
        from scipy.signal import welch
        _, pa = welch(a, fs=fs, nperseg=min(256, len(a)))
        _, pb = welch(b, fs=fs, nperseg=min(256, len(b)))
        return float(np.linalg.norm(np.sqrt(pb) - np.sqrt(pa))
                     / max(np.linalg.norm(np.sqrt(pa)), 1e-12))

    results = {}
    for name in ("released_test", "strict_test"):
        rows = []
        loader = DataLoader(Subset(dataset, splits[name]), batch_size=32)
        with torch.no_grad():
            for clean, noisy in loader:
                clean_np = clean.squeeze(1).numpy()
                denoised = model.denoising(noisy.to(device)) \
                    if hasattr(model, "denoising") else model.sample(noisy.to(device))
                denoised = denoised.squeeze(1).cpu().numpy()
                for i in range(len(clean_np)):
                    a, b = clean_np[i], denoised[i]
                    cc = float(np.corrcoef(a, b)[0, 1])
                    rows.append({"rrmse_t": rrmse_t(a, b),
                                 "rrmse_s": rrmse_s(a, b), "cc": cc})
        results[name] = {k: float(np.mean([r[k] for r in rows]))
                         for k in ("rrmse_t", "rrmse_s", "cc")}
        results[name]["n_rows"] = len(rows)
        print(json.dumps({name: results[name]}), flush=True)
    results["published_reference"] = {"rrmse_t": 0.121, "rrmse_s": 0.127,
                                      "cc": 0.992}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ssed_eval.json").write_text(json.dumps(results, indent=2,
                                                       sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["build", "train", "eval"])
    args = parser.parse_args()
    {"build": build, "train": train, "eval": evaluate}[args.mode]()


if __name__ == "__main__":
    main()
