#!/usr/bin/env python
"""M12 — make the subject embedding LOAD-BEARING (rescue the "subject-aware" claim) + controls.

M9 showed the subject embedding is not identity-specific (correct e(s) ~= wrong e(s')) because
additive, fully-observed denoising recovers the clean signal from the input without needing to know
WHO the subject is. This script tests the regimes where subject identity carries information the
corrupted window alone does not, for BOTH a diffusion denoiser and a same-architecture regressor:

  --subject-topo/--subject-pools : per-subject DISTINCT artifact topography + morphology pool
                                   (anatomy/electrode differences) — knowing the subject tells the
                                   model which artifact subspace to remove.  --topo-gain sets how
                                   subject-specific the topography is (log-normal sigma).
  --snr-low/--snr-high           : SNR regime (low SNR buries the signal).
  --model diffusion|regressor    : the subject-conditional x0 diffusion denoiser, OR the SAME U-Net
                                   trained as a one-shot MSE regressor — the control that asks whether
                                   the subject benefit is diffusion-specific or architecture-agnostic.

Evaluates the subject ablation (correct vs wrong vs null e(s)). correct >> wrong (ΔCC >= 0.02) ⇒
subject identity is load-bearing. GPU node.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

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
from saddpm.utils.ema import EMA  # noqa: E402
from saddpm.utils.seed import seed_everything  # noqa: E402

CONFIGS = REPO_ROOT / "configs"


def _metrics(d, c, fs):
    d, c = d.reshape(-1, d.shape[-1]), c.reshape(-1, c.shape[-1])
    return {"rrmse_t": float(rrmse_temporal(d, c).mean()),
            "rrmse_s": float(rrmse_spectral(d, c, fs).mean()),
            "cc": float(correlation_coefficient(d, c).mean())}


@torch.no_grad()
def _denoise(model, corr, sid, device, is_reg, batch=128):
    """Denoise with given subject ids. Diffusion: full-gen sampling. Regressor: one forward (t=0)."""
    out = []
    for i in range(0, len(corr), batch):
        cb = torch.from_numpy(corr[i:i + batch]).float().to(device)
        sb = torch.from_numpy(sid[i:i + batch]).long().to(device)
        if is_reg:
            t0 = torch.zeros(cb.shape[0], device=device, dtype=torch.long)
            out.append(model(cb, t0, sb).cpu().numpy())
        else:
            out.append(model.denoise(cb, sb, ddim_steps=50, t_star=None).cpu().numpy())
    return np.concatenate(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--noise", choices=["EOG", "EMG"], default="EOG")
    ap.add_argument("--model", choices=["diffusion", "regressor"], default="diffusion")
    ap.add_argument("--snr-low", type=float, default=-7.0)
    ap.add_argument("--snr-high", type=float, default=2.0)
    ap.add_argument("--subject-topo", action="store_true")
    ap.add_argument("--subject-pools", action="store_true")
    ap.add_argument("--topo-gain", type=float, default=0.6)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-eval", type=int, default=1800)
    ap.add_argument("--tag", type=str, required=True)
    args = ap.parse_args()
    is_reg = args.model == "regressor"

    data_cfg = DataConfig.from_yaml(CONFIGS / "data.yaml")
    eegdn_cfg = EEGDenoiseConfig.from_yaml(CONFIGS / "eegdenoise.yaml")
    diff_cfg = DiffusionConfig.from_yaml(CONFIGS / "diffusion.yaml")
    seed_everything(data_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fs = data_cfg.preprocess.resample_hz
    n_subj = data_cfg.dataset.n_subjects
    subjects = list(range(1, n_subj + 1))
    topo_seed = 7 if args.subject_topo else None

    eeg, eog, emg = load_components(eegdn_cfg)
    artifacts = {"EOG": eog, "EMG": emg}[args.noise]
    mk = dict(subject_topo_seed=topo_seed, subject_artifact_pools=args.subject_pools, subject_topo_gain=args.topo_gain)
    train = build_synthetic_pairs(subjects, data_cfg, artifacts, args.noise, "T", (args.snr_low, args.snr_high), data_cfg.seed, **mk)
    test_levels = list(np.linspace(args.snr_low, args.snr_high, 4))
    test = build_synthetic_pairs(subjects, data_cfg, artifacts, args.noise, "E", (args.snr_low, args.snr_high),
                                 data_cfg.seed + 1, fixed_snr_levels=test_levels, **mk)
    c = train.clean.shape[1]
    print(f"[M12:{args.tag}] model={args.model} noise={args.noise} snr=[{args.snr_low},{args.snr_high}] "
          f"topo={args.subject_topo}(g{args.topo_gain}) pools={args.subject_pools} train{train.clean.shape}")

    in_ch = c if is_reg else 2 * c
    model_cfg = ModelConfig(in_channels=in_ch, out_channels=c, signal_length=train.clean.shape[-1], num_subjects=n_subj)
    unet = UNet1D(model_cfg, subject_conditioned=True).to(device)
    model = unet if is_reg else SubjectConditionalDenoiser(unet, GaussianDiffusion(diff_cfg).to(device)).to(device)

    emb_ids = {id(p) for p in unet.subject_embed.parameters()}
    other = [p for p in model.parameters() if id(p) not in emb_ids]
    opt = torch.optim.Adam([{"params": other, "weight_decay": 0.0},
                            {"params": list(unet.subject_embed.parameters()), "weight_decay": 1e-4}], lr=float(eegdn_cfg.lr))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(train.clean).float(),
                                       torch.from_numpy(train.corrupted).float(),
                                       torch.from_numpy(train.subject_ids).long()),
        batch_size=64, shuffle=True, drop_last=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(loader))
    ema = EMA(model, 0.999)
    model.train()
    for ep in range(args.epochs):
        for cl, co, sid in loader:
            cl, co, sid = cl.to(device), co.to(device), sid.to(device)
            if is_reg:
                t0 = torch.zeros(cl.shape[0], device=device, dtype=torch.long)
                loss = nn.functional.mse_loss(model(co, t0, sid), cl)
            else:
                loss = model.loss(cl, co, sid)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); ema.update(model)
        if ep % 15 == 0 or ep == args.epochs - 1:
            print(f"  ep {ep:3d} loss={loss.item():.4f}")
    ema.copy_to(model); model.eval()
    save_checkpoint(REPO_ROOT / f"artifacts/checkpoints/m12_{args.tag}.pt", model,
                    {"model": model_cfg.__dict__, "diffusion": diff_cfg.__dict__, "tag": args.tag, "is_reg": is_reg})

    rng = np.random.default_rng(0)
    idx = rng.choice(len(test.clean), size=min(args.n_eval, len(test.clean)), replace=False)
    corr, clean, sid = test.corrupted[idx], test.clean[idx], test.subject_ids[idx]
    wrong = (sid + 1 + rng.integers(0, n_subj - 1, size=len(sid))) % n_subj
    nullid = np.full_like(sid, n_subj)

    variants = {"Corrupted(in)": corr,
                "correct e(s)": _denoise(model, corr, sid, device, is_reg),
                "wrong e(s')": _denoise(model, corr, wrong, device, is_reg),
                "null e": _denoise(model, corr, nullid, device, is_reg)}
    out_dir = REPO_ROOT / "results" / "m12"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# M12 {args.tag}: subject ablation | model={args.model} noise={args.noise} "
             f"snr=[{args.snr_low},{args.snr_high}] topo={args.subject_topo}(g{args.topo_gain}) pools={args.subject_pools}\n",
             "variant,RRMSE_temporal,RRMSE_spectral,CC\n"]
    res = {}
    print(f"\n========== M12 {args.tag} ==========")
    for name, d in variants.items():
        m = _metrics(d, clean, fs); res[name] = m
        lines.append(f"{name},{m['rrmse_t']:.4f},{m['rrmse_s']:.4f},{m['cc']:.4f}\n")
        print(f"  {name:16s} RRMSE_t={m['rrmse_t']:.4f} RRMSE_s={m['rrmse_s']:.4f} CC={m['cc']:.4f}")
    d_cw = res["correct e(s)"]["cc"] - res["wrong e(s')"]["cc"]
    d_cn = res["correct e(s)"]["cc"] - res["null e"]["cc"]
    lines.append(f"# delta CC correct-vs-wrong={d_cw:+.4f}  correct-vs-null={d_cn:+.4f}\n")
    (out_dir / f"{args.tag}.csv").write_text("".join(lines))
    verdict = "LOAD-BEARING" if d_cw >= 0.02 else ("weak" if d_cw >= 0.005 else "NOT load-bearing")
    print(f"[M12:{args.tag}] ({args.model}) subject identity: correct-vs-wrong ΔCC={d_cw:+.4f} ({verdict}); "
          f"correct-vs-null ΔCC={d_cn:+.4f}")
    print(f"[M12:{args.tag}] done -> results/m12/{args.tag}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
