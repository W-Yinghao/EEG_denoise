#!/usr/bin/env python
"""Head-to-head: EEGNet vs FBCSP+LDA downstream classifier on the 9×9 protocol ([DD-5]).

For each source i, the SADDPM/ICA denoised sessions are computed ONCE, then both classifiers are
trained/evaluated on identical denoised windows -> 4 accuracy matrices:
  {SADDPM, ICA} × {EEGNet, FBCSP+LDA}. Run on a GPU node (scripts/slurm/compare.sbatch).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from saddpm.baselines.fbcsp import FBCSPConfig, downstream_accuracy_fbcsp  # noqa: E402
from saddpm.baselines.ica import ICAConfig, ica_denoise_session  # noqa: E402
from saddpm.data.config import DataConfig  # noqa: E402
from saddpm.data.datasets import load_session_windows  # noqa: E402
from saddpm.diffusion.gaussian_diffusion import GaussianDiffusion  # noqa: E402
from saddpm.diffusion.schedule import DiffusionConfig  # noqa: E402
from saddpm.eval.downstream import evaluate_eegnet, train_eegnet  # noqa: E402
from saddpm.eval.pairwise import matrix_summary, saddpm_denoise_windows  # noqa: E402
from saddpm.models.config import ModelConfig  # noqa: E402
from saddpm.models.dual_decoder import DualDecoderSADDPM  # noqa: E402
from saddpm.models.eegnet import EEGNetConfig  # noqa: E402
from saddpm.utils.checkpoint import load_checkpoint  # noqa: E402
from saddpm.utils.seed import seed_everything  # noqa: E402

CONFIGS = REPO_ROOT / "configs"
OUT = REPO_ROOT / "results" / "compare"


def _eegnet_row(train_w, train_y, tests, eeg_cfg, device, seed):
    clf = train_eegnet(train_w, train_y, eeg_cfg, device, seed)
    return [evaluate_eegnet(clf, w, y, device) for (w, y) in tests]


def _fbcsp_row(train_w, train_y, tests, fb_cfg):
    from saddpm.baselines.fbcsp import FBCSPLDA

    clf = FBCSPLDA(fb_cfg).fit(train_w, train_y)
    return [clf.score(w, y) for (w, y) in tests]


def main() -> int:
    with open(CONFIGS / "m7.yaml", "r", encoding="utf-8") as fh:
        mc = yaml.safe_load(fh)
    data_cfg = DataConfig.from_yaml(CONFIGS / "data.yaml")
    eeg_cfg = EEGNetConfig.from_yaml(CONFIGS / "eegnet.yaml")
    fb_cfg = FBCSPConfig.from_yaml(CONFIGS / "fbcsp.yaml")
    ica_cfg = ICAConfig.from_yaml(CONFIGS / "ica.yaml")
    seed_everything(data_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subjects = list(mc["subjects"])
    n = len(subjects)
    t_star, ddim_steps, batch, seed = mc["t_star"], mc["ddim_steps"], mc["denoise_batch"], data_cfg.seed

    ckpt = load_checkpoint(REPO_ROOT / mc["checkpoint"], map_location=str(device))
    model = DualDecoderSADDPM(ModelConfig(**ckpt["config"]["model"])).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    diffusion = GaussianDiffusion(DiffusionConfig(**ckpt["config"]["diffusion"])).to(device)
    print(f"[cmp] device={device} subjects={subjects} t*={t_star}")

    real_T = {s: load_session_windows(s, data_cfg, "T") for s in subjects}
    real_E = {s: load_session_windows(s, data_cfg, "E") for s in subjects}
    ica_T = {s: ica_denoise_session(s, "T", data_cfg, ica_cfg) for s in subjects}
    ica_E = {s: ica_denoise_session(s, "E", data_cfg, ica_cfg) for s in subjects}

    mats = {k: np.zeros((n, n)) for k in ("saddpm_eegnet", "saddpm_fbcsp", "ica_eegnet", "ica_fbcsp")}
    for a, i in enumerate(subjects):
        # Denoise this source's sessions ONCE (before any classifier touches the RNG).
        dT = saddpm_denoise_windows(model, diffusion, real_T[i].windows, i - 1, t_star, ddim_steps, device, batch)
        dE = [saddpm_denoise_windows(model, diffusion, real_E[j].windows, i - 1, t_star, ddim_steps, device, batch)
              for j in subjects]
        sad_tests = [(dE[b], real_E[subjects[b]].mi_labels) for b in range(n)]
        ica_tests = [(ica_E[j].windows, ica_E[j].mi_labels) for j in subjects]

        mats["saddpm_eegnet"][a] = _eegnet_row(dT, real_T[i].mi_labels, sad_tests, eeg_cfg, device, seed)
        mats["saddpm_fbcsp"][a] = _fbcsp_row(dT, real_T[i].mi_labels, sad_tests, fb_cfg)
        mats["ica_eegnet"][a] = _eegnet_row(ica_T[i].windows, ica_T[i].mi_labels, ica_tests, eeg_cfg, device, seed)
        mats["ica_fbcsp"][a] = _fbcsp_row(ica_T[i].windows, ica_T[i].mi_labels, ica_tests, fb_cfg)
        print(f"[cmp] source A{i:02d}: "
              f"SADDPM eegnet={mats['saddpm_eegnet'][a].mean():.3f} fbcsp={mats['saddpm_fbcsp'][a].mean():.3f} | "
              f"ICA eegnet={mats['ica_eegnet'][a].mean():.3f} fbcsp={mats['ica_fbcsp'][a].mean():.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    labels = [f"A{s:02d}" for s in subjects]
    print("\n========== EEGNet vs FBCSP+LDA (grand mean / diag / per-target std) ==========")
    for k, mat in mats.items():
        np.save(OUT / f"{k}.npy", mat)
        with open(OUT / f"{k}.csv", "w", encoding="utf-8") as fh:
            fh.write("source\\target," + ",".join(labels) + "\n")
            for r, lab in enumerate(labels):
                fh.write(lab + "," + ",".join(f"{mat[r,c]:.4f}" for c in range(n)) + "\n")
        s = matrix_summary(mat)
        print(f"  {k:16s} grand={s['grand_mean']:.3f} diag={s['diag_mean']:.3f} std={s['mean_per_target_std']:.3f}")
    print(f"matrices saved under results/compare/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
