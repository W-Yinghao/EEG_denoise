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
    xb, yb = batch[0].to(device), batch[1].float().to(device)
    checks["batch_shape"] = list(xb.shape)
    xb = torch.permute(xb, (0, 3, 2, 1))     # their train(): unet2d_overlap.py:528
    checks["model_input_shape"] = list(xb.shape)
    # mirror their training step exactly (unet2d_overlap.py:533)
    loss, time_diff, noise_kl, sub_arc, loss_orth = \
        configs.diffusion.loss_with_diff_constraint(xb, yb)
    checks["loss"] = float(loss)
    checks["loss_components"] = {"time_diff": float(time_diff),
                                 "noise_kl": float(noise_kl),
                                 "sub_arc": float(sub_arc),
                                 "orth": float(loss_orth)}
    loss.backward()
    checks["backward"] = True
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "probe.json").write_text(json.dumps(checks, indent=2,
                                                   default=str) + "\n")
    print(json.dumps(checks, default=str))


def train(epochs: int) -> None:
    """F2 — their training loop, mirrored verbatim from Configs.train/run
    (unet2d_overlap.py:511-559). Only two execution-neutral changes, both
    disclosed in the report: (a) the metrics CSV goes to our results dir
    instead of their hardcoded /home/yiqduan path (which does not exist here
    and would crash open()); (b) checkpoints are torch.save'd every 10 epochs
    because their labml experiment plumbing needs a labml server config we do
    not run. Optimizer structure, the double sub_theta step, batch size,
    lr, permute — all theirs, untouched.
    """
    import csv

    os.chdir(UPSTREAM)
    sys.path.insert(0, str(UPSTREAM))
    import torch
    import unet2d_overlap as up

    device = torch.device("cuda")
    configs = up.Configs()
    configs.eeg_channels = 22
    configs.device = device
    configs.epochs = epochs
    configs.dataset = up.DatasetLoader_BCI_IV_mix_subjects(
        "train", datafolder=str(DATA_ROOT))
    configs.init()

    ckpt_dir = DATA_ROOT.parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260831)               # ours, disclosed (they set none)
    metric_path = OUT_DIR / "train_metrics.csv"
    with open(metric_path, "a+", newline="") as fh:
        writer = csv.writer(fh)
        for epoch in range(configs.epochs):
            for data_b, label_b in configs.data_loader:
                data_b = torch.permute(data_b, (0, 3, 2, 1)).to(device)
                label_b = label_b.float().to(device)
                configs.optimizer.zero_grad()
                loss, time_period_diff, noise_conent_kl, sub_arc_loss, \
                    loss_orth = configs.diffusion.loss_with_diff_constraint(
                        data_b, label_b)
                loss.backward()
                configs.optimizer.step()
                configs.optimizer_noise.step()
                writer.writerow([float(loss),
                                 float(loss - loss_orth - sub_arc_loss),
                                 float(time_period_diff),
                                 float(noise_conent_kl),
                                 float(sub_arc_loss), float(loss_orth)])
            fh.flush()
            print(json.dumps({"epoch": epoch, "loss": float(loss)}), flush=True)
            if (epoch + 1) % 10 == 0 or epoch == configs.epochs - 1:
                torch.save({"eps_model": configs.eps_model.state_dict(),
                            "sub_theta": configs.sub_theta.state_dict(),
                            "sub_archead": configs.sub_archead.state_dict(),
                            "epoch": epoch},
                           ckpt_dir / f"dsddpm_ep{epoch + 1:03d}.pt")
    print(json.dumps({"done": True, "epochs": configs.epochs,
                      "ckpt_dir": str(ckpt_dir)}))


def ica() -> None:
    """F4 — the ICA comparison arm of their Table I. Their paper names ICA as
    the baseline but publishes no ICA recipe, so the protocol here is the
    field-standard one and is frozen before any numbers are seen: per run,
    Infomax-family ICA (fastica, 22 components) fitted on a 1 Hz high-passed
    copy of the continuous 25-ch record, components rejected by the default
    mne find_bads_eog correlation test against the 3 EOG channels, the
    cleaned continuous record then cut into the SAME trials as data().
    """
    import mne
    from scipy.io import loadmat, savemat
    mne.set_log_level("ERROR")

    out_root = DATA_ROOT / "ica_sep"
    out_root.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(
        [f"EEG{i}" for i in range(22)] + [f"EOG{i}" for i in range(3)],
        FS, ["eeg"] * 22 + ["eog"] * 3)
    report = []
    for index, subject in enumerate(SUBJECTS, start=1):
        out = out_root / f"single_subject_data_{index}.mat"
        if out.is_file():
            report.append({"subject": subject, "state": "cached"})
            continue
        payload, n_rejected = {}, []
        for session, prefix in (("T", "train"), ("E", "test")):
            runs = loadmat(RAW_ROOT / f"{subject}{session}.mat")["data"][0]
            xs, ys = [], []
            for run in runs:
                record = run[0, 0]
                onsets = np.asarray(record["trial"],
                                    np.float64).ravel().astype(int)
                if not len(onsets):
                    continue
                signal = np.nan_to_num(
                    np.asarray(record["X"], np.float64).T)      # 25 x T
                labels = np.asarray(record["y"], int).ravel()
                raw = mne.io.RawArray(signal * 1e-6, info, verbose="ERROR")
                fit = raw.copy().filter(1.0, None, verbose="ERROR")
                ica_o = mne.preprocessing.ICA(
                    n_components=22, method="fastica",
                    random_state=97, max_iter="auto")
                ica_o.fit(fit, picks="eeg")
                bads, _ = ica_o.find_bads_eog(fit)
                ica_o.exclude = bads
                n_rejected.append(len(bads))
                cleaned = ica_o.apply(raw.copy()).get_data() * 1e6
                for k, onset in enumerate(onsets):
                    stop = onset + TRIAL_SAMPLES
                    if stop > cleaned.shape[1]:
                        continue
                    xs.append(cleaned[:22, onset:stop])
                    ys.append(labels[k])
            payload[f"{prefix}_x"] = np.stack(xs).astype(np.float64)
            payload[f"{prefix}_y"] = np.asarray(ys, np.float64)[:, None]
        savemat(out, payload)
        report.append({"subject": subject, "state": "built",
                       "rejected_per_run": n_rejected})
        print(json.dumps(report[-1]), flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ica_report.json").write_text(json.dumps(report, indent=1) + "\n")


def sample() -> None:
    """F3 — apply the trained separator to the REAL test/train trials.

    Reproduction finding, preserved: the released sample_save.py has no code
    path that touches recorded EEG — both sample_save_data and
    sample_animation start from torch.randn (dataroot only receives PNGs).
    The paper's stated inference (estimate the subject-noise component and
    keep the content estimate) is therefore reconstructed here from their own
    primitives, frozen before any numbers are seen:
      xt      = diffusion.q_sample(x_real, t*)          (their q_sample)
      content = (xt - sqrt(1-abar_t)*eps_model(xt,t*)) / sqrt(abar_t)
                                                (their Sampler.p_x0 formula)
    with t* = 20, their sample_animation apply_step default, windows/overlap
    reconstruction exactly as their overlap_cover loop (stride 75).
    """
    os.chdir(UPSTREAM)
    sys.path.insert(0, str(UPSTREAM))
    import torch
    import unet2d_overlap as up
    from scipy.io import loadmat, savemat

    device = torch.device("cuda")
    configs = up.Configs()
    configs.eeg_channels = 22
    configs.device = device
    configs.dataset = up.DatasetLoader_BCI_IV_mix_subjects(
        "train", datafolder=str(DATA_ROOT))
    configs.init()
    ckpts = sorted((DATA_ROOT.parent / "checkpoints").glob("dsddpm_ep*.pt"))
    assert ckpts, "no trained checkpoint"
    state = torch.load(ckpts[-1], map_location=device)
    configs.eps_model.load_state_dict(state["eps_model"])
    configs.sub_theta.load_state_dict(state["sub_theta"])
    configs.eps_model.eval()
    configs.sub_theta.eval()
    diffusion = configs.diffusion
    abar = diffusion.alpha_bar
    T_STAR, WINDOW, STRIDE = 20, 224, 75
    out_root = DATA_ROOT / "dsddpm_sep"
    out_root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260831)

    @torch.no_grad()
    def separate(trials, subject_index, batch=64):
        starts = list(range(0, 750 - WINDOW, STRIDE))          # 8 windows
        x = trials[:, :, 750:1500]                             # their slice
        stacks = np.stack([x[:, :, s0:s0 + WINDOW] for s0 in starts], axis=3)
        out = np.array(trials, np.float64, copy=True)
        for b in range(0, len(stacks), batch):
            xb = torch.from_numpy(stacks[b:b + batch]).float().to(device)
            n = len(xb)
            t = torch.full((n,), T_STAR, dtype=torch.long, device=device)
            xt = diffusion.q_sample(xb, t)
            tb = xt.new_full((1,), T_STAR, dtype=torch.long)
            eps = configs.eps_model(xt, tb)
            a = abar[T_STAR]
            content = (xt - (1 - a).sqrt() * eps) / a.sqrt()
            content = content.cpu().numpy()
            rec = np.zeros((n, 22, 750))
            for i, s0 in enumerate(starts):                    # overlap_cover
                rec[:, :, s0:s0 + WINDOW] = content[:, :, :, i]
            out[b:b + batch, :, 750:1500] = rec
        return out

    report = []
    for index in range(1, 10):
        src = loadmat(DATA_ROOT / "single_sep" / f"single_subject_data_{index}.mat")
        payload = {}
        for prefix in ("train", "test"):
            trials = np.asarray(src[f"{prefix}_x"], np.float64)
            payload[f"{prefix}_x"] = separate(trials, index - 1)
            payload[f"{prefix}_y"] = np.asarray(src[f"{prefix}_y"], np.float64)
        savemat(out_root / f"single_subject_data_{index}.mat", payload)
        report.append({"subject": index, "trials": int(payload["train_x"].shape[0])})
        print(json.dumps(report[-1]), flush=True)
    (OUT_DIR / "sample_report.json").write_text(json.dumps(
        {"t_star": T_STAR, "ckpt": str(ckpts[-1]),
         "note": "released code has no real-data path; paper-described "
                 "inference reconstructed from their q_sample/p_x0",
         "subjects": report}, indent=1) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["data", "probe", "train", "ica", "sample"])
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    if args.mode == "train":
        train(args.epochs)
    else:
        {"data": data, "probe": probe, "ica": ica, "sample": sample}[args.mode]()


if __name__ == "__main__":
    main()
