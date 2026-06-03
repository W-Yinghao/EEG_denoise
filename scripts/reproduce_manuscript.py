#!/usr/bin/env python
"""Reproduce the manuscript's Tables 1-3 (BCI-IV-2a) with PER-TRIAL eval + trial-mean correlation.

Table 1: ICA 9x9 downstream accuracy (%). Table 2: SADDPM 9x9. Table 3: subject correlation
(a) real-real, (b) SADDPM-gen vs real, using the manuscript's trial-mean descriptor. Differs from
M7 by (i) per-trial vote evaluation (closer to the manuscript's "2 s segment") and (ii) the
trial-mean (not spectral) correlation descriptor. Run on a GPU node.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from saddpm.baselines.ica import ICAConfig, ica_denoise_session  # noqa: E402
from saddpm.data.config import DataConfig  # noqa: E402
from saddpm.data.datasets import load_session_windows  # noqa: E402
from saddpm.diffusion.gaussian_diffusion import GaussianDiffusion  # noqa: E402
from saddpm.diffusion.schedule import DiffusionConfig  # noqa: E402
from saddpm.eval.downstream import evaluate_eegnet_per_trial, train_eegnet  # noqa: E402
from saddpm.eval.pairwise import saddpm_denoise_windows  # noqa: E402
from saddpm.eval.subject_corr import descriptors_from_groups, pearson_matrix, trial_mean_descriptor  # noqa: E402
from saddpm.models.config import ModelConfig  # noqa: E402
from saddpm.models.dual_decoder import DualDecoderSADDPM  # noqa: E402
from saddpm.models.eegnet import EEGNetConfig  # noqa: E402
from saddpm.utils.checkpoint import load_checkpoint  # noqa: E402
from saddpm.utils.seed import seed_everything  # noqa: E402

CONFIGS = REPO_ROOT / "configs"
OUT = REPO_ROOT / "results" / "manuscript"


def _save(mat: np.ndarray, name: str, subjects, pct: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"{name}.npy", mat)
    labels = [f"s{s}" for s in subjects]
    scale = 100.0 if pct else 1.0
    with open(OUT / f"{name}.csv", "w", encoding="utf-8") as fh:
        fh.write("source/target," + ",".join(labels) + "\n")
        for r, lab in enumerate(labels):
            fh.write(lab + "," + ",".join(f"{mat[r, c]*scale:.2f}" for c in range(mat.shape[1])) + "\n")
        if pct:
            mean_row = mat.mean(axis=0) * scale
            fh.write("Mean," + ",".join(f"{v:.2f}" for v in mean_row) + "\n")


def main() -> int:
    with open(CONFIGS / "m7.yaml", "r", encoding="utf-8") as fh:
        mc = yaml.safe_load(fh)
    data_cfg = DataConfig.from_yaml(CONFIGS / "data.yaml")
    eeg_cfg = EEGNetConfig.from_yaml(CONFIGS / "eegnet.yaml")
    ica_cfg = ICAConfig.from_yaml(CONFIGS / "ica.yaml")
    seed_everything(data_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subjects = list(mc["subjects"])
    n = len(subjects)
    t_star, ddim_steps, batch = mc["t_star"], mc["ddim_steps"], mc["denoise_batch"]

    ckpt = load_checkpoint(REPO_ROOT / mc["checkpoint"], map_location=str(device))
    model = DualDecoderSADDPM(ModelConfig(**ckpt["config"]["model"])).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    diffusion = GaussianDiffusion(DiffusionConfig(**ckpt["config"]["diffusion"])).to(device)
    print(f"[repro] device={device} per-trial eval, trial-mean correlation")

    real_T = {s: load_session_windows(s, data_cfg, "T") for s in subjects}
    real_E = {s: load_session_windows(s, data_cfg, "E") for s in subjects}

    # ---- Table 2: SADDPM 9x9 (per-trial) ----
    acc_saddpm = np.zeros((n, n))
    for a, i in enumerate(subjects):
        dT = saddpm_denoise_windows(model, diffusion, real_T[i].windows, i - 1, t_star, ddim_steps, device, batch)
        clf = train_eegnet(dT, real_T[i].mi_labels, eeg_cfg, device, data_cfg.seed)
        for b, j in enumerate(subjects):
            dE = saddpm_denoise_windows(model, diffusion, real_E[j].windows, i - 1, t_star, ddim_steps, device, batch)
            acc_saddpm[a, b] = evaluate_eegnet_per_trial(clf, dE, real_E[j].mi_labels, real_E[j].trial_index, device)
        print(f"[repro][SADDPM] s{i} row mean={acc_saddpm[a].mean()*100:.1f}%")

    # ---- Table 1: ICA 9x9 (per-trial) ----
    ica_T = {s: ica_denoise_session(s, "T", data_cfg, ica_cfg) for s in subjects}
    ica_E = {s: ica_denoise_session(s, "E", data_cfg, ica_cfg) for s in subjects}
    acc_ica = np.zeros((n, n))
    for a, i in enumerate(subjects):
        clf = train_eegnet(ica_T[i].windows, ica_T[i].mi_labels, eeg_cfg, device, data_cfg.seed)
        for b, j in enumerate(subjects):
            # ICA windows keep the same per-trial structure as real_E (same epoching).
            acc_ica[a, b] = evaluate_eegnet_per_trial(clf, ica_E[j].windows, ica_E[j].mi_labels, real_E[j].trial_index, device)
        print(f"[repro][ICA] s{i} row mean={acc_ica[a].mean()*100:.1f}%")

    # ---- Table 3: trial-mean subject correlation ----
    real_desc = descriptors_from_groups([real_T[s].windows for s in subjects], descriptor=trial_mean_descriptor)
    gen_groups = []
    for i in subjects:
        def eps_fn(x, t, sid=i - 1):
            return model.predict_eps(x, t, torch.full((x.shape[0],), sid, device=x.device, dtype=torch.long))
        gen = diffusion.ddim_sample_loop(eps_fn, (32, 22, 512), device, ddim_steps=mc["gen_ddim_steps"])
        gen_groups.append(gen.cpu().numpy())
    gen_desc = descriptors_from_groups(gen_groups, descriptor=trial_mean_descriptor)
    corr_real = pearson_matrix(real_desc, real_desc)
    corr_gen = pearson_matrix(gen_desc, real_desc)

    for mat, name, pct in [(acc_ica, "table1_ica_acc", True), (acc_saddpm, "table2_saddpm_acc", True),
                           (corr_real, "table3a_real_corr", False), (corr_gen, "table3b_gen_corr", False)]:
        _save(mat, name, subjects, pct)

    def spread(m):
        return float((m.mean(axis=0) * 100).std())
    print("\n========== MANUSCRIPT REPRODUCTION (per-trial) ==========")
    print(f"Table 1 ICA   : grand {acc_ica.mean()*100:.2f}%  diag {np.diag(acc_ica).mean()*100:.2f}%  per-target-mean std {spread(acc_ica):.2f}")
    print(f"Table 2 SADDPM: grand {acc_saddpm.mean()*100:.2f}%  diag {np.diag(acc_saddpm).mean()*100:.2f}%  per-target-mean std {spread(acc_saddpm):.2f}")
    print(f"(manuscript: ICA grand 50.6 std 4.8 | SADDPM grand 52.8 std 3.3)")
    print(f"Table 3 diag-dominance: real {float(np.mean(np.argmax(corr_real,1)==np.arange(n))):.2f}  gen {float(np.mean(np.argmax(corr_gen,1)==np.arange(n))):.2f}")
    print("saved -> results/manuscript/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
