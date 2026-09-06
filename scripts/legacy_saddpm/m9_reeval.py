#!/usr/bin/env python
"""M9 re-evaluation with the CORRECTED sampler (full conditional generation) + subject ablation.

The M9 ablation was run with the default conditional-SDEdit start t*=400, which the M10 t* sweep showed
is badly suboptimal for EOG (the warm-start re-injects the artifact). This re-evaluates the EXISTING
m9_{EOG,EMG}.pt checkpoints (no retraining) at:
  * t*=400      — the old operating point (for reference / to reproduce the published M9 number)
  * full-gen    — t_star=None, full conditional generation (the corrected default)
and, at each, runs the subject-embedding ablation (correct vs wrong vs null e(s)) to test whether the
subject conditioning is load-bearing AT THE OPERATING POINT WHERE THE DENOISER ACTUALLY WORKS. GPU node.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

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
from saddpm.utils.checkpoint import load_checkpoint  # noqa: E402
from saddpm.utils.seed import seed_everything  # noqa: E402

CONFIGS = REPO_ROOT / "configs"


def _metrics(d, c, fs):
    d, c = d.reshape(-1, d.shape[-1]), c.reshape(-1, c.shape[-1])
    return {"rrmse_t": float(rrmse_temporal(d, c).mean()),
            "rrmse_s": float(rrmse_spectral(d, c, fs).mean()),
            "cc": float(correlation_coefficient(d, c).mean())}


@torch.no_grad()
def _denoise(model, corr, sid, t_star, device, batch=128):
    out = []
    for i in range(0, len(corr), batch):
        cb = torch.from_numpy(corr[i:i + batch]).float().to(device)
        sb = torch.from_numpy(sid[i:i + batch]).long().to(device)
        out.append(model.denoise(cb, sb, ddim_steps=50, t_star=t_star).cpu().numpy())
    return np.concatenate(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--noise", choices=["EOG", "EMG"], default="EOG")
    ap.add_argument("--n-eval", type=int, default=1800)
    args = ap.parse_args()

    data_cfg = DataConfig.from_yaml(CONFIGS / "data.yaml")
    eegdn_cfg = EEGDenoiseConfig.from_yaml(CONFIGS / "eegdenoise.yaml")
    seed_everything(data_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fs = data_cfg.preprocess.resample_hz
    n_subj = data_cfg.dataset.n_subjects

    ck = load_checkpoint(REPO_ROOT / f"artifacts/checkpoints/m9_{args.noise}.pt", map_location=str(device))
    unet = UNet1D(ModelConfig(**ck["config"]["model"]), subject_conditioned=True).to(device)
    model = SubjectConditionalDenoiser(unet, GaussianDiffusion(DiffusionConfig(**ck["config"]["diffusion"])).to(device)).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    eeg, eog, emg = load_components(eegdn_cfg)
    artifacts = {"EOG": eog, "EMG": emg}[args.noise]
    test = build_synthetic_pairs(list(range(1, n_subj + 1)), data_cfg, artifacts, args.noise, "E",
                                 tuple(eegdn_cfg.snr_train_db), data_cfg.seed + 1,
                                 fixed_snr_levels=[-7.0, -4.0, -1.0, 2.0])
    rng = np.random.default_rng(0)
    idx = rng.choice(len(test.clean), size=min(args.n_eval, len(test.clean)), replace=False)
    corr, clean, sid = test.corrupted[idx], test.clean[idx], test.subject_ids[idx]
    wrong = (sid + 1 + rng.integers(0, n_subj - 1, size=len(sid))) % n_subj
    nullid = np.full_like(sid, n_subj)
    print(f"[m9reeval:{args.noise}] eval {corr.shape}  baseline corrupted CC={_metrics(corr, clean, fs)['cc']:.4f}")

    rows = [("Corrupted(in)", None, _metrics(corr, clean, fs))]
    for t_star, tag in [(400, "t*=400(old)"), (None, "full-gen(new)")]:
        for emb_name, ids in [("correct e(s)", sid), ("wrong e(s')", wrong), ("null e", nullid)]:
            m = _metrics(_denoise(model, corr, ids, t_star, device), clean, fs)
            rows.append((f"Denoised[{emb_name}]", tag, m))
            print(f"  {tag:14s} {emb_name:12s} RRMSE_t={m['rrmse_t']:.4f} RRMSE_s={m['rrmse_s']:.4f} CC={m['cc']:.4f}")

    out_dir = REPO_ROOT / "results" / "m9"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# M9 {args.noise} re-eval: corrected full-gen sampler + subject ablation (vs clean GT)\n",
             "variant,sampler,RRMSE_temporal,RRMSE_spectral,CC\n"]
    for name, tag, m in rows:
        lines.append(f"{name},{tag or '-'},{m['rrmse_t']:.4f},{m['rrmse_s']:.4f},{m['cc']:.4f}\n")
    (out_dir / f"{args.noise}_reeval.csv").write_text("".join(lines))
    print(f"[m9reeval:{args.noise}] done -> results/m9/{args.noise}_reeval.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
