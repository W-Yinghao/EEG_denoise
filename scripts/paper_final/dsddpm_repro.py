#!/usr/bin/env python3
"""DS-DDPM Table I / Table II reproduction on BCI-IV-2a.

Upstream: .external/DS-DDPM @ 12c339a (Apache-2.0). The release ships the training
entry (unet2d_overlap.py), the sampler (sample_save.py), an EEGNet subject-classifier
backbone (assets/max_acc.pth) and the dataset loaders, but NOT the preprocessing that
builds data/single_sep/single_subject_data_{id}.mat, and its README reports no
numbers — the reproduction targets are the paper's Table I (9x9 cross-subject MI
accuracy, ICA-denoised vs DS-DDPM-denoised training data) and Table II (subject-wise
correlation, real vs sampled), arXiv 2305.04200.

Modes
  data     rebuild the .mat files the loaders expect, from the registered BNCI mats:
           [trials, 22, 1500] at 250 Hz from trial onset (their slice 750:1500 is the
           3-6 s imagery window), labels 1..4, session T -> train_x, E -> test_x
  probe    import their module, build one loader, one forward/backward of their eps
           model + subject model on a tiny batch — the wiring gate before training
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
UPSTREAM = REPO / ".external/DS-DDPM"
RAW_ROOT = Path("/projects/EEG-foundation-model/BCI-IV")
DATA_ROOT = Path("/projects/EEG-foundation-model/derived/denoiseNet/dsddpm_repro/data")
OUT_DIR = REPO / "results/paper_final/dsddpm"
SUBJECTS = tuple(f"A{i:02d}" for i in range(1, 10))
FS = 250
TRIAL_SAMPLES = 1500          # 0-6 s at 250 Hz; their loaders slice 750:1500


def data() -> None:
    from scipy.io import loadmat, savemat
    (DATA_ROOT / "single_sep").mkdir(parents=True, exist_ok=True)
    report = []
    for index, subject in enumerate(SUBJECTS, start=1):
        out = DATA_ROOT / "single_sep" / f"single_subject_data_{index}.mat"
        if out.is_file():
            report.append({"subject": subject, "state": "cached"})
            continue
        payload = {}
        for session, prefix in (("T", "train"), ("E", "test")):
            runs = loadmat(RAW_ROOT / f"{subject}{session}.mat")["data"][0]
            xs, ys = [], []
            for run in runs:
                record = run[0, 0]
                onsets = np.asarray(record["trial"], np.float64).ravel().astype(int)
                if not len(onsets):
                    continue
                signal = np.nan_to_num(
                    np.asarray(record["X"], np.float64).T)      # 25 x T
                labels = np.asarray(record["y"], int).ravel()
                for k, onset in enumerate(onsets):
                    stop = onset + TRIAL_SAMPLES
                    if stop > signal.shape[1]:
                        continue
                    xs.append(signal[:22, onset:stop])
                    ys.append(labels[k])
            payload[f"{prefix}_x"] = np.stack(xs).astype(np.float64)
            payload[f"{prefix}_y"] = np.asarray(ys, np.float64)[:, None]
        savemat(out, payload)
        report.append({"subject": subject, "state": "built",
                       "train": int(payload["train_x"].shape[0]),
                       "test": int(payload["test_x"].shape[0])})
        print(json.dumps(report[-1]), flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "data_report.json").write_text(json.dumps(report, indent=1) + "\n")


def probe() -> None:
    os.chdir(UPSTREAM)
    sys.path.insert(0, str(UPSTREAM))
    import torch
    import unet2d_overlap as up

    loader = up.DatasetLoader_BCI_IV_mix_subjects("train", datafolder=str(DATA_ROOT))
    sample, label = loader[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = up.Configs()
    configs.eeg_channels = 22
    configs.device = device
    # bypass labml's option system: assign the dataset directly, then init
    checks = {"loader_len": len(loader),
              "sample_shape": list(np.asarray(sample).shape),
              "label": float(label)}
    configs.dataset = loader
    configs.epochs = 1
    configs.init()
    batch = next(iter(configs.data_loader))
    xb, yb = batch[0].to(device), batch[1].to(device)
    checks["batch_shape"] = list(xb.shape)
    loss = configs.loss_batch(xb, yb) if hasattr(configs, "loss_batch") else None
    if loss is None:
        # fall back to the diffusion object's loss
        subject_ids = yb.long()
        loss = configs.diffusion.loss(xb, subject_ids) \
            if hasattr(configs, "diffusion") else None
    checks["loss"] = float(loss) if loss is not None else "no unified loss entry"
    if loss is not None:
        loss.backward()
        checks["backward"] = True
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "probe.json").write_text(json.dumps(checks, indent=2,
                                                   default=str) + "\n")
    print(json.dumps(checks, default=str))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["data", "probe"])
    args = parser.parse_args()
    {"data": data, "probe": probe}[args.mode]()


if __name__ == "__main__":
    main()
