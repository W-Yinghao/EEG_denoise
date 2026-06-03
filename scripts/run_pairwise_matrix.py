#!/usr/bin/env python
"""M7: full 9×9 cross-subject accuracy matrices (SADDPM vs ICA) + subject correlation (handoff §8).

For each (source i, target j): denoise i's Session-T and j's Session-E, train EEGNet on denoised
i-T, test on denoised j-E -> 9×9 accuracy matrix. Done for SADDPM (SDEdit, conditioned on e(i))
and ICA. EEGNet training depends only on the source, so 9 classifiers per pipeline are trained and
each is evaluated on all 9 targets. Also builds the §8.2 subject-correlation matrices (real-vs-real
and SADDPM-generated-vs-real). Run on a GPU node (scripts/slurm/m7.sbatch).
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

from saddpm.baselines.ica import ICAConfig, ica_denoise_session  # noqa: E402
from saddpm.data.config import DataConfig  # noqa: E402
from saddpm.data.datasets import load_session_windows  # noqa: E402
from saddpm.diffusion.gaussian_diffusion import GaussianDiffusion  # noqa: E402
from saddpm.diffusion.schedule import DiffusionConfig  # noqa: E402
from saddpm.eval.downstream import evaluate_eegnet, train_eegnet  # noqa: E402
from saddpm.eval.pairwise import matrix_summary, saddpm_denoise_windows  # noqa: E402
from saddpm.eval.subject_corr import descriptors_from_groups, diagonal_dominance, pearson_matrix  # noqa: E402
from saddpm.models.config import ModelConfig  # noqa: E402
from saddpm.models.dual_decoder import DualDecoderSADDPM  # noqa: E402
from saddpm.models.eegnet import EEGNetConfig  # noqa: E402
from saddpm.utils.checkpoint import load_checkpoint  # noqa: E402
from saddpm.utils.seed import seed_everything  # noqa: E402

CONFIGS = REPO_ROOT / "configs"
RUN_DIR = REPO_ROOT / "artifacts" / "runs" / "m7"
FIG_DIR = REPO_ROOT / "artifacts" / "figures"


def _save_matrix(mat: np.ndarray, name: str, subjects: list[int]) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    np.save(RUN_DIR / f"{name}.npy", mat)
    labels = [f"A{s:02d}" for s in subjects]
    with open(RUN_DIR / f"{name}.csv", "w", encoding="utf-8") as fh:
        fh.write("source\\target," + ",".join(labels) + "\n")
        for r, lab in enumerate(labels):
            fh.write(lab + "," + ",".join(f"{mat[r, c]:.4f}" for c in range(mat.shape[1])) + "\n")


def _heatmap(mat: np.ndarray, title: str, out: Path, subjects: list[int], vmin=0.0, vmax=1.0) -> None:
    labels = [f"A{s:02d}" for s in subjects]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax)
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            ax.text(c, r, f"{mat[r, c]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if mat[r, c] < (vmin + vmax) / 2 else "black")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("target j"); ax.set_ylabel("source i"); ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=None, help="override EEGNet epochs")
    args = parser.parse_args()

    with open(CONFIGS / "m7.yaml", "r", encoding="utf-8") as fh:
        mc = yaml.safe_load(fh)
    data_cfg = DataConfig.from_yaml(CONFIGS / "data.yaml")
    eeg_cfg = EEGNetConfig.from_yaml(CONFIGS / "eegnet.yaml")
    ica_cfg = ICAConfig.from_yaml(CONFIGS / "ica.yaml")
    if args.epochs is not None:
        eeg_cfg = EEGNetConfig(**{**eeg_cfg.__dict__, "epochs": args.epochs})
    seed_everything(data_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subjects = list(mc["subjects"])
    n = len(subjects)
    t_star, ddim_steps, batch = mc["t_star"], mc["ddim_steps"], mc["denoise_batch"]

    ckpt = load_checkpoint(REPO_ROOT / mc["checkpoint"], map_location=str(device))
    model = DualDecoderSADDPM(ModelConfig(**ckpt["config"]["model"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    diffusion = GaussianDiffusion(DiffusionConfig(**ckpt["config"]["diffusion"])).to(device)
    print(f"[M7] device={device} subjects={subjects} t*={t_star} eegnet_epochs={eeg_cfg.epochs}")

    real_T = {s: load_session_windows(s, data_cfg, "T") for s in subjects}
    real_E = {s: load_session_windows(s, data_cfg, "E") for s in subjects}

    # ---- SADDPM 9x9 ----
    acc_saddpm = np.zeros((n, n))
    for a, i in enumerate(subjects):
        dT = saddpm_denoise_windows(model, diffusion, real_T[i].windows, i - 1, t_star, ddim_steps, device, batch)
        clf = train_eegnet(dT, real_T[i].mi_labels, eeg_cfg, device, data_cfg.seed)
        for b, j in enumerate(subjects):
            dE = saddpm_denoise_windows(model, diffusion, real_E[j].windows, i - 1, t_star, ddim_steps, device, batch)
            acc_saddpm[a, b] = evaluate_eegnet(clf, dE, real_E[j].mi_labels, device)
        print(f"[M7][SADDPM] source A{i:02d} done; row mean={acc_saddpm[a].mean():.3f}")

    # ---- ICA 9x9 ----
    ica_T = {s: ica_denoise_session(s, "T", data_cfg, ica_cfg) for s in subjects}
    ica_E = {s: ica_denoise_session(s, "E", data_cfg, ica_cfg) for s in subjects}
    acc_ica = np.zeros((n, n))
    for a, i in enumerate(subjects):
        clf = train_eegnet(ica_T[i].windows, ica_T[i].mi_labels, eeg_cfg, device, data_cfg.seed)
        for b, j in enumerate(subjects):
            acc_ica[a, b] = evaluate_eegnet(clf, ica_E[j].windows, ica_E[j].mi_labels, device)
        print(f"[M7][ICA] source A{i:02d} done; row mean={acc_ica[a].mean():.3f}")

    # ---- subject correlation (Table 3) ----
    real_groups = [real_T[s].windows for s in subjects]
    desc_real = descriptors_from_groups(real_groups)
    corr_real = pearson_matrix(desc_real, desc_real)
    gen_groups = []
    for i in subjects:
        def eps_fn(x, t, sid=i - 1):
            return model.predict_eps(x, t, torch.full((x.shape[0],), sid, device=x.device, dtype=torch.long))
        gen = diffusion.ddim_sample_loop(
            eps_fn, (mc["gen_per_subject"], 22, 512), device, ddim_steps=mc["gen_ddim_steps"])
        gen_groups.append(gen.detach().cpu().numpy())
    desc_gen = descriptors_from_groups(gen_groups)
    corr_gen = pearson_matrix(desc_gen, desc_real)

    # ---- save everything ----
    for mat, name in [(acc_saddpm, "acc_saddpm"), (acc_ica, "acc_ica"),
                      (corr_real, "subjcorr_real_real"), (corr_gen, "subjcorr_gen_real")]:
        _save_matrix(mat, name, subjects)
    _heatmap(acc_saddpm, "M7 SADDPM 9x9 accuracy", FIG_DIR / "m7_acc_saddpm.png", subjects)
    _heatmap(acc_ica, "M7 ICA 9x9 accuracy", FIG_DIR / "m7_acc_ica.png", subjects)
    _heatmap(corr_real, "M7 subject corr: real vs real (a)", FIG_DIR / "m7_subjcorr_real.png", subjects, -1, 1)
    _heatmap(corr_gen, "M7 subject corr: SADDPM-gen vs real (b)", FIG_DIR / "m7_subjcorr_gen.png", subjects, -1, 1)

    s_sum, i_sum = matrix_summary(acc_saddpm), matrix_summary(acc_ica)
    print("\n========== M7 RESULTS ==========")
    print(f"SADDPM: grand mean={s_sum['grand_mean']:.3f} diag={s_sum['diag_mean']:.3f} "
          f"mean per-target std={s_sum['mean_per_target_std']:.3f}")
    print(f"ICA:    grand mean={i_sum['grand_mean']:.3f} diag={i_sum['diag_mean']:.3f} "
          f"mean per-target std={i_sum['mean_per_target_std']:.3f}")
    print(f"subject-corr diagonal dominance: real-real={diagonal_dominance(corr_real):.2f} "
          f"gen-real={diagonal_dominance(corr_gen):.2f}")
    print(f"matrices/figures saved under artifacts/runs/m7 and artifacts/figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
