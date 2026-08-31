#!/usr/bin/env python
"""E1/E2 — the M8 single-channel EEGdenoiseNet benchmark re-run on the EEGDfus
SNR grid so the two papers' tables are directly comparable.

EEGDfus (Tables I-III) trains and tests at SNR -5..5 dB in 11 integer levels;
our banked M8 rows live on the EEGdenoiseNet-paper grid (-7..2, 10 levels), so
no row of ours was comparable to any row of theirs.  This re-runs the M8
protocol IDENTICALLY except for the grid: snr_train_db=(-5,5),
snr_test_levels=11.  Everything else (architectures, epochs, batch, lr, seed,
metrics) is the M8 code imported unchanged, and each arm's per-level
RRMSE_t / RRMSE_s / CC lands next to the corresponding EEGDfus table row.

The trained conditional model is also checkpointed (artifacts/checkpoints/
e12_{noise}.pt) because E4 (cross-dataset generalization, their Table VIII)
evaluates this same EEGdenoiseNet-trained model on SSED. GPU node.
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import m8_benchmark as m8  # noqa: E402  (helpers reused verbatim)
from saddpm.data.eegdenoisenet import EEGDenoiseConfig, prepare_pairs  # noqa: E402
from saddpm.diffusion.schedule import DiffusionConfig  # noqa: E402
from saddpm.eval.denoise_metrics import benchmark_by_snr  # noqa: E402
from saddpm.utils.checkpoint import save_checkpoint  # noqa: E402
from saddpm.utils.logging import RunLogger  # noqa: E402
from saddpm.utils.seed import seed_everything  # noqa: E402
from saddpm.baselines.dl_denoisers import denoise_with, make_denoiser, train_denoiser  # noqa: E402

CONFIGS = REPO_ROOT / "configs"
OUT_DIR = REPO_ROOT / "results/paper_final/e12"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise", choices=["EOG", "EMG"], required=True)
    args = parser.parse_args()

    cfg = EEGDenoiseConfig.from_yaml(CONFIGS / "eegdenoise.yaml")
    cfg = EEGDenoiseConfig(**{**cfg.__dict__,
                              "snr_train_db": (-5.0, 5.0),
                              "snr_test_levels": 11})
    diff_cfg = DiffusionConfig.from_yaml(CONFIGS / "diffusion.yaml")
    seed_everything(cfg.seed)
    device = torch.device("cuda")
    p = prepare_pairs(args.noise, cfg)
    print(f"[E12:{args.noise}] grid {p.snr_levels_db} train {p.noisy_train.shape} "
          f"test {p.noisy_test.shape}", flush=True)
    logger = RunLogger(REPO_ROOT / "artifacts/runs" / f"e12_{args.noise}",
                       use_wandb=False, config={"milestone": "E12",
                                                "noise": args.noise})

    prior, diffusion = m8._train_prior(p.clean_train, cfg, diff_cfg, device, logger)
    print(f"[E12:{args.noise}] prior trained", flush=True)
    cdd = m8._train_cond(p.clean_train, p.noisy_train, cfg, diff_cfg, device, logger)
    print(f"[E12:{args.noise}] cond trained", flush=True)
    save_checkpoint(REPO_ROOT / "artifacts/checkpoints" / f"e12_{args.noise}.pt",
                    cdd, {"eegdenoise": {**cfg.__dict__,
                                         "snr_train_db": list(cfg.snr_train_db)},
                          "diffusion": diff_cfg.__dict__, "noise": args.noise,
                          "grid": "-5..5 x 11 (EEGDfus alignment)"})

    baseline_epochs = {"fcnn": 60, "simple_cnn": 40, "complex_cnn": 40,
                       "rnn_lstm": 30, "novel_cnn": 60}
    denoised = {"Noisy": p.noisy_test,
                "SDEdit": m8._sdedit_denoise(prior, diffusion, p.noisy_test, cfg, device),
                "CondDiff": m8._cond_denoise(cdd, p.noisy_test, cfg, device)}
    for arch, label in (("fcnn", "FCNN"), ("simple_cnn", "SimpleCNN"),
                        ("complex_cnn", "ComplexCNN"), ("rnn_lstm", "RNN_LSTM"),
                        ("novel_cnn", "NovelCNN")):
        model = make_denoiser(arch, cfg.segment_len)
        trained = train_denoiser(model, p.noisy_train, p.clean_train,
                                 baseline_epochs[arch], cfg.batch_size, cfg.lr,
                                 device, cfg.seed)
        denoised[label] = denoise_with(trained, p.noisy_test, device)
        print(f"[E12:{args.noise}] {label} done", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    lines = ["method,snr_db,rrmse_t,rrmse_s,cc\n"]
    for name, d in denoised.items():
        r = benchmark_by_snr(d, p.clean_test, p.test_snr_db, p.snr_levels_db,
                             cfg.fs, cfg.psd_max_hz)
        payload = json.loads(json.dumps(r.__dict__,
                                        default=lambda o: np.asarray(o).tolist()))
        summary[name] = payload
        for i, db in enumerate(payload["snr_levels_db"]):
            lines.append(f"{name},{db:.0f},{payload['rrmse_t'][i]:.4f},"
                         f"{payload['rrmse_s'][i]:.4f},{payload['cc'][i]:.4f}\n")
        ov = payload["overall"]
        lines.append(f"{name},overall,{ov['rrmse_t']:.4f},"
                     f"{ov['rrmse_s']:.4f},{ov['cc']:.4f}\n")
        print(f"[E12:{args.noise}] {name}: RRMSE_t={ov['rrmse_t']:.4f} "
              f"CC={ov['cc']:.4f}", flush=True)
    (OUT_DIR / f"{args.noise}_grid.csv").write_text("".join(lines))
    (OUT_DIR / f"{args.noise}_grid.json").write_text(
        json.dumps({"grid": "-5..5 x 11", "noise": args.noise,
                    "arms": summary}, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
