#!/usr/bin/env python
"""Figure-data generator (EEGdenoiseNet-style). Produces the arrays/CSVs the plotters consume.

Tasks (one per Slurm job; F3/swap is sharded across GPUs by --regime/--subject):
  waveforms     F1 — example clean/contaminated/denoised traces (single- and multi-channel)
  per_snr       F2 — RRMSE_t/RRMSE_s/CC vs input SNR for our denoiser + EEGdenoiseNet CNNs (single-channel)
  swap          F3 — one ROW of the subject embedding-swap matrix: denoise subject i with every subject j's
                     embedding (+null), CC vs clean_i. --regime baseline|subjart, --subject i (1-based)
  multichannel  F5 — artifact spatial topography + a 22-channel example (clean/noisy/denoised)
F4 reuses results/m10/*tstar*; F6 (PSD) reuses the waveforms npz at plot time. Outputs under
results/figures/ (csv) and artifacts/figdata/ (npz).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from saddpm.baselines.dl_denoisers import denoise_with, make_denoiser, train_denoiser  # noqa: E402
from saddpm.data.config import DataConfig  # noqa: E402
from saddpm.data.eegdenoisenet import EEGDenoiseConfig, load_components, prepare_pairs  # noqa: E402
from saddpm.data.synthetic_artifacts import (build_synthetic_pairs, eog_topography,  # noqa: E402
                                             emg_topography)
from saddpm.diffusion.conditional import ConditionalDiffusionDenoiser  # noqa: E402
from saddpm.diffusion.gaussian_diffusion import GaussianDiffusion  # noqa: E402
from saddpm.diffusion.schedule import DiffusionConfig  # noqa: E402
from saddpm.eval.denoise_metrics import benchmark_by_snr, correlation_coefficient  # noqa: E402
from saddpm.models.cond_denoiser import SubjectConditionalDenoiser  # noqa: E402
from saddpm.models.config import ModelConfig  # noqa: E402
from saddpm.models.unet1d import UNet1D  # noqa: E402
from saddpm.utils.checkpoint import load_checkpoint  # noqa: E402
from saddpm.utils.seed import seed_everything  # noqa: E402

CONFIGS = REPO_ROOT / "configs"
FIGDATA = REPO_ROOT / "artifacts" / "figdata"
FIGCSV = REPO_ROOT / "results" / "figures"


def _load_single(noise, device):
    """Load the M10 single-channel improved (x0) denoiser."""
    ck = load_checkpoint(REPO_ROOT / f"artifacts/checkpoints/m10_{noise}_x0.pt", map_location=str(device))
    unet = UNet1D(ModelConfig(**ck["config"]["model"]), subject_conditioned=False).to(device)
    m = ConditionalDiffusionDenoiser(unet, GaussianDiffusion(DiffusionConfig(**ck["config"]["diffusion"])).to(device),
                                     parameterization=ck["config"].get("parameterization", "x0")).to(device)
    m.load_state_dict(ck["model_state"]); m.eval()
    return m


def _load_multi(path, device):
    ck = load_checkpoint(path, map_location=str(device))
    unet = UNet1D(ModelConfig(**ck["config"]["model"]), subject_conditioned=True).to(device)
    m = SubjectConditionalDenoiser(unet, GaussianDiffusion(DiffusionConfig(**ck["config"]["diffusion"])).to(device)).to(device)
    m.load_state_dict(ck["model_state"]); m.eval()
    return m


@torch.no_grad()
def _denoise_single(m, noisy, device, batch=256):
    out = []
    for i in range(0, len(noisy), batch):
        y = torch.from_numpy(noisy[i:i + batch]).float().unsqueeze(1).to(device)
        out.append(m.denoise(y, ddim_steps=50, t_star=None).squeeze(1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def _denoise_multi(m, corr, sid, device, batch=128):
    out = []
    for i in range(0, len(corr), batch):
        cb = torch.from_numpy(corr[i:i + batch]).float().to(device)
        sb = torch.from_numpy(sid[i:i + batch]).long().to(device)
        out.append(m.denoise(cb, sb, ddim_steps=50, t_star=None).cpu().numpy())
    return np.concatenate(out)


def task_waveforms(noise, device, data_cfg, eegdn_cfg):
    # single-channel examples (one per SNR level), EEGdenoiseNet
    p = prepare_pairs(noise, eegdn_cfg)
    levels = p.snr_levels_db
    m = _load_single(noise, device)
    # pick 1 example index per level (first of that level)
    pick = [int(np.where(p.test_snr_db == db)[0][0]) for db in levels]
    nz, cl = p.noisy_test[pick], p.clean_test[pick]
    dn = _denoise_single(m, nz, device)
    np.savez(FIGDATA / f"f1_{noise}_single.npz", clean=cl, noisy=nz, denoised=dn, snr=np.array(levels), fs=eegdn_cfg.fs)
    # multi-channel example (one window at a mid SNR), m9 checkpoint
    eeg, eog, emg = load_components(eegdn_cfg)
    art = {"EOG": eog, "EMG": emg}[noise]
    test = build_synthetic_pairs([1], data_cfg, art, noise, "E", tuple(eegdn_cfg.snr_train_db),
                                 data_cfg.seed + 1, fixed_snr_levels=[-4.0])
    mm = _load_multi(REPO_ROOT / f"artifacts/checkpoints/m9_{noise}.pt", device)
    k = 4  # a few example windows
    co, cln, sid = test.corrupted[:k], test.clean[:k], test.subject_ids[:k]
    dnm = _denoise_multi(mm, co, sid, device)
    np.savez(FIGDATA / f"f1_{noise}_multi.npz", clean=cln, noisy=co, denoised=dnm,
             ch_names=np.array(test.ch_names), fs=data_cfg.preprocess.resample_hz)
    print(f"[fig:waveforms:{noise}] saved f1_{noise}_single.npz + f1_{noise}_multi.npz")


def task_per_snr(noise, device, data_cfg, eegdn_cfg):
    p = prepare_pairs(noise, eegdn_cfg)
    m = _load_single(noise, device)
    methods = {"Noisy": p.noisy_test, "SADDPM (ours)": _denoise_single(m, p.noisy_test, device)}
    for arch, label, ep in [("simple_cnn", "SimpleCNN", 40), ("complex_cnn", "ComplexCNN", 40), ("novel_cnn", "NovelCNN", 60)]:
        net = train_denoiser(make_denoiser(arch, eegdn_cfg.segment_len), p.noisy_train, p.clean_train,
                             ep, eegdn_cfg.batch_size, eegdn_cfg.lr, device, eegdn_cfg.seed)
        methods[label] = denoise_with(net, p.noisy_test, device)
        print(f"[fig:per_snr:{noise}] {label} done")
    lines = ["method,snr_db,rrmse_t,rrmse_s,cc\n"]
    for name, d in methods.items():
        r = benchmark_by_snr(d, p.clean_test, p.test_snr_db, p.snr_levels_db, eegdn_cfg.fs, eegdn_cfg.psd_max_hz)
        for db, rt, rs, cc in zip(r.snr_levels_db, r.rrmse_t, r.rrmse_s, r.cc):
            lines.append(f"{name},{db:.2f},{rt:.5f},{rs:.5f},{cc:.5f}\n")
    FIGCSV.mkdir(parents=True, exist_ok=True)
    (FIGCSV / f"f2_{noise}_persnr.csv").write_text("".join(lines))
    print(f"[fig:per_snr:{noise}] saved results/figures/f2_{noise}_persnr.csv")


def task_swap(noise, regime, subject, device, data_cfg, eegdn_cfg):
    n_subj = data_cfg.dataset.n_subjects
    topo_seed = 7 if regime == "subjart" else None
    pools = regime == "subjart"
    ckpt = REPO_ROOT / f"artifacts/checkpoints/m12_{noise}_{'subjart' if regime=='subjart' else 'baseline'}.pt"
    m = _load_multi(ckpt, device)
    eeg, eog, emg = load_components(eegdn_cfg)
    art = {"EOG": eog, "EMG": emg}[noise]
    # build full 9-subject test data (needed for correct per-subject artifact pools), then slice subject i
    test = build_synthetic_pairs(list(range(1, n_subj + 1)), data_cfg, art, noise, "E", tuple(eegdn_cfg.snr_train_db),
                                 data_cfg.seed + 1, fixed_snr_levels=[-7.0, -4.0, -1.0, 2.0],
                                 subject_topo_seed=topo_seed, subject_artifact_pools=pools)
    sel = test.subject_ids == (subject - 1)
    co, cl = test.corrupted[sel], test.clean[sel]
    rng = np.random.default_rng(0)
    if len(co) > 300:
        keep = rng.choice(len(co), 300, replace=False); co, cl = co[keep], cl[keep]
    ccs = []
    for j in list(range(n_subj)) + [n_subj]:  # each subject embedding + null
        sid = np.full(len(co), j, dtype=np.int64)
        d = _denoise_multi(m, co, sid, device)
        ccs.append(float(correlation_coefficient(d.reshape(-1, d.shape[-1]), cl.reshape(-1, cl.shape[-1])).mean()))
    FIGCSV.mkdir(parents=True, exist_ok=True)
    hdr = ",".join([f"e{j+1}" for j in range(n_subj)] + ["null"])
    (FIGCSV / f"f3_{regime}_{noise}_s{subject}.csv").write_text(
        f"# row=source subject {subject}; cols=embedding used\nsource,{hdr}\n"
        f"s{subject}," + ",".join(f"{c:.4f}" for c in ccs) + "\n")
    print(f"[fig:swap:{noise}:{regime}:s{subject}] CC(correct e{subject})={ccs[subject-1]:.3f} "
          f"mean(wrong)={np.mean([c for k,c in enumerate(ccs[:n_subj]) if k!=subject-1]):.3f}")


def task_multichannel(noise, device, data_cfg, eegdn_cfg):
    eeg, eog, emg = load_components(eegdn_cfg)
    art = {"EOG": eog, "EMG": emg}[noise]
    test = build_synthetic_pairs([1], data_cfg, art, noise, "E", tuple(eegdn_cfg.snr_train_db),
                                 data_cfg.seed + 1, fixed_snr_levels=[-4.0])
    ch = test.ch_names
    topo = (eog_topography if noise == "EOG" else emg_topography)(ch)
    mm = _load_multi(REPO_ROOT / f"artifacts/checkpoints/m9_{noise}.pt", device)
    co, cl, sid = test.corrupted[:1], test.clean[:1], test.subject_ids[:1]
    dn = _denoise_multi(mm, co, sid, device)
    np.savez(FIGDATA / f"f5_{noise}.npz", topo=topo, ch_names=np.array(ch), clean=cl[0], noisy=co[0],
             denoised=dn[0], fs=data_cfg.preprocess.resample_hz)
    print(f"[fig:multichannel:{noise}] saved f5_{noise}.npz")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=["waveforms", "per_snr", "swap", "multichannel"])
    ap.add_argument("--noise", default="EOG", choices=["EOG", "EMG"])
    ap.add_argument("--regime", default="subjart", choices=["baseline", "subjart"])
    ap.add_argument("--subject", type=int, default=1)
    args = ap.parse_args()
    FIGDATA.mkdir(parents=True, exist_ok=True)
    data_cfg = DataConfig.from_yaml(CONFIGS / "data.yaml")
    eegdn_cfg = EEGDenoiseConfig.from_yaml(CONFIGS / "eegdenoise.yaml")
    seed_everything(data_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.task == "waveforms":
        task_waveforms(args.noise, device, data_cfg, eegdn_cfg)
    elif args.task == "per_snr":
        task_per_snr(args.noise, device, data_cfg, eegdn_cfg)
    elif args.task == "swap":
        task_swap(args.noise, args.regime, args.subject, device, data_cfg, eegdn_cfg)
    elif args.task == "multichannel":
        task_multichannel(args.noise, device, data_cfg, eegdn_cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
