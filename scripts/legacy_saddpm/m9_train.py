#!/usr/bin/env python
"""M9 — subject-aware conditional diffusion denoiser (Option C).

Trains a conditional diffusion denoiser on synthetic (corrupted, clean) BCI-IV-2a pairs (EEGdenoiseNet
EOG/EMG injected with a spatial topography at controlled SNR), then evaluates RRMSE/CC against the
clean ground truth. KEY TEST (audit finding #2): does the subject embedding matter? — denoise the
test set conditioned on the CORRECT vs a WRONG vs the NULL subject embedding and compare. Run on GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from saddpm.data.config import DataConfig  # noqa: E402
from saddpm.data.eegdenoisenet import EEGDenoiseConfig, load_components  # noqa: E402
from saddpm.data.synthetic_artifacts import build_synthetic_pairs  # noqa: E402
from saddpm.diffusion.gaussian_diffusion import GaussianDiffusion  # noqa: E402
from saddpm.diffusion.schedule import DiffusionConfig  # noqa: E402
from saddpm.eval.denoise_metrics import correlation_coefficient, rrmse_spectral, rrmse_temporal  # noqa: E402
from saddpm.models.cond_denoiser import SubjectConditionalDenoiser  # noqa: E402
from saddpm.models.config import ModelConfig  # noqa: E402
from saddpm.models.unet1d import UNet1D  # noqa: E402
from saddpm.utils.checkpoint import save_checkpoint  # noqa: E402
from saddpm.utils.logging import RunLogger  # noqa: E402
from saddpm.utils.seed import seed_everything  # noqa: E402

CONFIGS = REPO_ROOT / "configs"


def _metrics(denoised: np.ndarray, clean: np.ndarray, fs: int) -> dict:
    """Multi-channel RRMSE_t/s + CC, computed per (window, channel) then averaged."""
    d = denoised.reshape(-1, denoised.shape[-1])
    c = clean.reshape(-1, clean.shape[-1])
    return {
        "rrmse_t": float(rrmse_temporal(d, c).mean()),
        "rrmse_s": float(rrmse_spectral(d, c, fs).mean()),
        "cc": float(correlation_coefficient(d, c).mean()),
    }


@torch.no_grad()
def _denoise(model, corrupted, subject_ids, ddim_steps, device, batch=128):
    out = []
    for i in range(0, len(corrupted), batch):
        cb = torch.from_numpy(corrupted[i:i + batch]).float().to(device)
        sb = torch.from_numpy(subject_ids[i:i + batch]).long().to(device)
        out.append(model.denoise(cb, sb, ddim_steps=ddim_steps).cpu().numpy())
    return np.concatenate(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise", choices=["EOG", "EMG"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="tiny CPU glue check (2 subjects, 1 epoch)")
    args = parser.parse_args()

    with open(CONFIGS / "m9.yaml", "r", encoding="utf-8") as fh:
        m9 = yaml.safe_load(fh)
    if args.smoke:
        m9 = {**m9, "epochs": 1, "ddim_steps": 4, "eval_subsample": 60}
    noise_type = args.noise or m9["noise_type"]
    epochs = args.epochs if args.epochs is not None else int(m9["epochs"])
    data_cfg = DataConfig.from_yaml(CONFIGS / "data.yaml")
    eegdn_cfg = EEGDenoiseConfig.from_yaml(CONFIGS / "eegdenoise.yaml")
    diff_cfg = DiffusionConfig.from_yaml(CONFIGS / "diffusion.yaml")
    seed_everything(data_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fs = data_cfg.preprocess.resample_hz

    eeg, eog, emg = load_components(eegdn_cfg)
    artifacts = {"EOG": eog, "EMG": emg}[noise_type]
    subjects = [1, 2] if args.smoke else list(range(1, data_cfg.dataset.n_subjects + 1))
    train = build_synthetic_pairs(subjects, data_cfg, artifacts, noise_type, "T",
                                  tuple(m9["snr_train_db"]), data_cfg.seed)
    test = build_synthetic_pairs(subjects, data_cfg, artifacts, noise_type, "E",
                                 tuple(m9["snr_train_db"]), data_cfg.seed + 1,
                                 fixed_snr_levels=m9["snr_test_levels"])
    c = train.clean.shape[1]
    print(f"[M9:{noise_type}] device={device} train {train.clean.shape} test {test.clean.shape} C={c}")

    model_cfg = ModelConfig(in_channels=2 * c, out_channels=c, signal_length=train.clean.shape[-1],
                            num_subjects=data_cfg.dataset.n_subjects)
    unet = UNet1D(model_cfg, subject_conditioned=True).to(device)
    model = SubjectConditionalDenoiser(unet, GaussianDiffusion(diff_cfg).to(device)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[M9:{noise_type}] params={n_params/1e6:.2f}M")

    emb_ids = {id(p) for p in unet.subject_embed.parameters()}
    other = [p for p in model.parameters() if id(p) not in emb_ids]
    opt = torch.optim.Adam([{"params": other, "weight_decay": 0.0},
                            {"params": list(unet.subject_embed.parameters()), "weight_decay": 1e-4}],
                           lr=float(m9["lr"]))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(train.clean).float(),
                                       torch.from_numpy(train.corrupted).float(),
                                       torch.from_numpy(train.subject_ids).long()),
        batch_size=int(m9["batch_size"]), shuffle=True, drop_last=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    logger = RunLogger(REPO_ROOT / "artifacts" / "runs" / f"m9_{noise_type}", use_wandb=args.wandb,
                       wandb_project="saddpm", wandb_run_name=f"m9-{noise_type}", wandb_mode="offline",
                       config={"milestone": "M9", "noise": noise_type, "epochs": epochs})

    step = 0
    model.train()
    for ep in range(epochs):
        for cl, co, sid in loader:
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss = model.loss(cl.to(device), co.to(device), sid.to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); step += 1
            if step % 100 == 0:
                logger.log(step, {"loss": loss.item(), "lr": sched.get_last_lr()[0]})
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:3d} step {step:6d} loss={loss.item():.4f}")

    save_checkpoint(REPO_ROOT / "artifacts" / "checkpoints" / f"m9_{noise_type}.pt", model,
                    {"model": model_cfg.__dict__, "diffusion": diff_cfg.__dict__, "noise": noise_type})

    # ---- eval: subject-ablation (correct vs wrong vs null embedding) ----
    model.eval()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(test.clean), size=min(int(m9["eval_subsample"]), len(test.clean)), replace=False)
    corr_te, clean_te, sid_te, snr_te = test.corrupted[idx], test.clean[idx], test.subject_ids[idx], test.snr_db[idx]
    null_id = data_cfg.dataset.n_subjects
    wrong_id = (sid_te + 1 + rng.integers(0, data_cfg.dataset.n_subjects - 1, size=len(sid_te))) % data_cfg.dataset.n_subjects

    variants = {
        "Corrupted(in)": corr_te,
        "Denoised[correct e(s)]": _denoise(model, corr_te, sid_te, m9["ddim_steps"], device),
        "Denoised[wrong e(s')]": _denoise(model, corr_te, wrong_id, m9["ddim_steps"], device),
        "Denoised[null e]": _denoise(model, corr_te, np.full_like(sid_te, null_id), m9["ddim_steps"], device),
    }
    out_dir = REPO_ROOT / "results" / "m9"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# M9 {noise_type}: subject-aware conditional denoiser (RRMSE/CC vs clean ground truth)\n",
             "variant,RRMSE_temporal,RRMSE_spectral,CC\n"]
    print("\n========== M9 RESULTS ==========")
    for name, d in variants.items():
        m = _metrics(d, clean_te, fs)
        lines.append(f"{name},{m['rrmse_t']:.4f},{m['rrmse_s']:.4f},{m['cc']:.4f}\n")
        print(f"  {name:26s} RRMSE_t={m['rrmse_t']:.3f} RRMSE_s={m['rrmse_s']:.3f} CC={m['cc']:.3f}")
    (out_dir / f"{noise_type}_ablation.csv").write_text("".join(lines))
    logger.finish()
    # subject-ablation verdict
    mc = _metrics(variants["Denoised[correct e(s)]"], clean_te, fs)
    mw = _metrics(variants["Denoised[wrong e(s')]"], clean_te, fs)
    print(f"[M9:{noise_type}] subject embedding load-bearing? correct CC {mc['cc']:.3f} vs wrong CC {mw['cc']:.3f} "
          f"(delta {mc['cc']-mw['cc']:+.3f})")
    print(f"[M9:{noise_type}] done -> results/m9/{noise_type}_ablation.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
