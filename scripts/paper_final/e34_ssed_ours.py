#!/usr/bin/env python
"""E3b/E4 — our conditional diffusion denoiser on SSED (Klados), and the
cross-dataset generalization row (EEGDfus Table VIII analogue).

mode train  (E3b): train our CondDiff (the M8 architecture, unchanged) on the
  SAME 13338 single-channel SSED rows and the SAME row-level split
  (random_state=666) as the EEGDfus-SSED reproduction, then evaluate on both
  the `released` and the `strict` test split with the SAME metric functions.
mode gen    (E4): take the EEGdenoiseNet-trained model (artifacts/checkpoints/
  e12_EOG.pt — trained by e12_regrid.py, never saw a Klados sample) and
  evaluate it zero-shot on the same two SSED test splits; then fine-tune on
  10% of the SSED train rows (10 epochs, lr 5e-5, seed 20260831) and
  re-evaluate — the EEGDfus_finetune analogue.

Sampling-rate bridge, disclosed: SSED rows are 400 samples @ 200 Hz (2 s), the
M8 model is fixed at 512 samples @ 256 Hz (2 s). Rows are resampled 400->512
(polyphase 32/25) on the way in and 512->400 on the way out; ALL metrics are
computed in the native 400 @ 200 Hz space against the untouched clean rows.
GPU node.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eegdfus_ssed as ES  # noqa: E402  (same rows, same split, same metrics)
from saddpm.diffusion.conditional import ConditionalDiffusionDenoiser  # noqa: E402
from saddpm.diffusion.gaussian_diffusion import GaussianDiffusion  # noqa: E402
from saddpm.diffusion.schedule import DiffusionConfig  # noqa: E402
from saddpm.models.config import ModelConfig  # noqa: E402
from saddpm.models.unet1d import UNet1D  # noqa: E402
from saddpm.utils.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from saddpm.utils.seed import seed_everything  # noqa: E402

CONFIGS = REPO_ROOT / "configs"
OUT_DIR = REPO_ROOT / "results/paper_final/e34"
SEED = 20260831
LEN_NATIVE, LEN_MODEL = 400, 512
EPOCHS, BATCH, LR = 60, 128, 1e-4
FT_EPOCHS, FT_LR, FT_FRACTION = 10, 5e-5, 0.10
DDIM_STEPS = 50


def _rrmse_t(a, b):
    return float(np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-12))


def _rrmse_s(a, b, fs=200):
    from scipy.signal import welch
    _, pa = welch(a, fs=fs, nperseg=min(256, len(a)))
    _, pb = welch(b, fs=fs, nperseg=min(256, len(b)))
    return float(np.linalg.norm(np.sqrt(pb) - np.sqrt(pa))
                 / max(np.linalg.norm(np.sqrt(pa)), 1e-12))


def _rows():
    """Their rows, their split. x=contaminated, y=clean, both (N, 400)."""
    x, y = ES._dataset()
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    return x, y, ES._splits(len(x))


def _up(rows):
    from scipy.signal import resample_poly
    return resample_poly(rows, 32, 25, axis=-1).astype(np.float32)


def _down(rows):
    from scipy.signal import resample_poly
    return resample_poly(rows, 25, 32, axis=-1)


def _scale(noisy512):
    sd = noisy512.std(axis=-1, keepdims=True)
    return np.clip(sd, 1e-8, None).astype(np.float32)


def _build_model(device):
    diff_cfg = DiffusionConfig.from_yaml(CONFIGS / "diffusion.yaml")
    unet = UNet1D(ModelConfig(in_channels=2, out_channels=1,
                              signal_length=LEN_MODEL),
                  subject_conditioned=False).to(device)
    return ConditionalDiffusionDenoiser(
        unet, GaussianDiffusion(diff_cfg).to(device)).to(device), diff_cfg


def _fit(cdd, noisy512, clean512, epochs, lr, device):
    c = torch.from_numpy(clean512).float().unsqueeze(1)
    nz = torch.from_numpy(noisy512).float().unsqueeze(1)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(c, nz), batch_size=BATCH,
        shuffle=True, drop_last=True)
    opt = torch.optim.Adam(cdd.parameters(), lr=lr)
    for ep in range(epochs):
        for cb, nb in loader:
            loss = cdd.loss(cb.to(device), nb.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        print(json.dumps({"epoch": ep, "loss": float(loss)}), flush=True)
    return cdd


@torch.no_grad()
def _denoise(cdd, noisy512, device, batch=256):
    cdd.eval()
    out = []
    for i in range(0, len(noisy512), batch):
        y = torch.from_numpy(noisy512[i:i + batch]).float().unsqueeze(1).to(device)
        out.append(cdd.denoise(y, ddim_steps=DDIM_STEPS).squeeze(1).cpu().numpy())
    return np.concatenate(out)


DERIVED = Path("/projects/EEG-foundation-model/derived/denoiseNet/e34_ours")


def _evaluate(cdd, x, y, splits, device, tag, save=False):
    results = {}
    for name in ("released_test", "strict_test"):
        idx = np.asarray(splits[name])
        noisy512 = _up(x[idx])
        sd = _scale(noisy512)
        den = _down(_denoise(cdd, noisy512 / sd, device) * sd)
        if save:
            DERIVED.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(DERIVED / f"denoised_{tag}_{name}.npz",
                                idx=idx, denoised=den)
        rows = [{"rrmse_t": _rrmse_t(y[i], d), "rrmse_s": _rrmse_s(y[i], d),
                 "cc": float(np.corrcoef(y[i], d)[0, 1])}
                for i, d in zip(idx, den)]
        results[name] = {k: float(np.mean([r[k] for r in rows]))
                         for k in ("rrmse_t", "rrmse_s", "cc")}
        results[name]["n_rows"] = len(rows)
        print(json.dumps({tag: {name: results[name]}}), flush=True)
    return results


def train() -> None:
    device = torch.device("cuda")
    seed_everything(SEED)
    x, y, splits = _rows()
    tr = np.asarray(splits["train"])
    noisy512, clean512 = _up(x[tr]), _up(y[tr])
    sd = _scale(noisy512)
    cdd, diff_cfg = _build_model(device)
    print(f"[E3b] train rows {len(tr)} model-len {LEN_MODEL}", flush=True)
    _fit(cdd, noisy512 / sd, clean512 / sd, EPOCHS, LR, device)
    save_checkpoint(REPO_ROOT / "artifacts/checkpoints/e3b_ssed.pt", cdd,
                    {"diffusion": diff_cfg.__dict__, "dataset": "SSED/Klados",
                     "rows": int(len(tr)), "seed": SEED})
    results = _evaluate(cdd, x, y, splits, device, "E3b")
    results["protocol"] = {"epochs": EPOCHS, "batch": BATCH, "lr": LR,
                           "seed": SEED, "resample": "400<->512 poly 32/25",
                           "metrics_space": "native 400 @ 200 Hz"}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "e3b_ours_ssed.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n")


def gen() -> None:
    device = torch.device("cuda")
    seed_everything(SEED)
    x, y, splits = _rows()
    cdd, _ = _build_model(device)
    ck = load_checkpoint(REPO_ROOT / "artifacts/checkpoints/e12_EOG.pt",
                         map_location=str(device))
    cdd.load_state_dict(ck["model_state"])
    out = {"zero_shot": _evaluate(cdd, x, y, splits, device, "E4-zero")}
    rng = np.random.default_rng(SEED)
    tr = np.asarray(splits["train"])
    sub = rng.choice(tr, size=max(1, int(len(tr) * FT_FRACTION)), replace=False)
    noisy512, clean512 = _up(x[sub]), _up(y[sub])
    sd = _scale(noisy512)
    print(f"[E4] finetune on {len(sub)} rows ({FT_FRACTION:.0%})", flush=True)
    _fit(cdd, noisy512 / sd, clean512 / sd, FT_EPOCHS, FT_LR, device)
    out["finetune_10pct"] = _evaluate(cdd, x, y, splits, device, "E4-ft")
    out["protocol"] = {"source_model": "e12_EOG (EEGdenoiseNet-trained)",
                      "ft_epochs": FT_EPOCHS, "ft_lr": FT_LR,
                      "ft_fraction": FT_FRACTION, "seed": SEED,
                      "resample": "400<->512 poly 32/25",
                      "metrics_space": "native 400 @ 200 Hz"}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "e4_generalization.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")


def resave() -> None:
    """Re-run the banked evaluations from the saved checkpoints, this time
    persisting the denoised waveforms for the E5 PLV analysis. The 10%
    fine-tune is re-fit (same seed/protocol) because gen() discarded it."""
    device = torch.device("cuda")
    seed_everything(SEED)
    x, y, splits = _rows()
    cdd, _ = _build_model(device)
    ck = load_checkpoint(REPO_ROOT / "artifacts/checkpoints/e3b_ssed.pt",
                         map_location=str(device))
    cdd.load_state_dict(ck["model_state"])
    _evaluate(cdd, x, y, splits, device, "e3b", save=True)
    cdd, _ = _build_model(device)
    ck = load_checkpoint(REPO_ROOT / "artifacts/checkpoints/e12_EOG.pt",
                         map_location=str(device))
    cdd.load_state_dict(ck["model_state"])
    _evaluate(cdd, x, y, splits, device, "e4zero", save=True)
    rng = np.random.default_rng(SEED)
    tr = np.asarray(splits["train"])
    sub = rng.choice(tr, size=max(1, int(len(tr) * FT_FRACTION)), replace=False)
    noisy512, clean512 = _up(x[sub]), _up(y[sub])
    sd = _scale(noisy512)
    _fit(cdd, noisy512 / sd, clean512 / sd, FT_EPOCHS, FT_LR, device)
    _evaluate(cdd, x, y, splits, device, "e4ft", save=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "gen", "resave"])
    args = parser.parse_args()
    {"train": train, "gen": gen, "resave": resave}[args.mode]()


if __name__ == "__main__":
    main()
