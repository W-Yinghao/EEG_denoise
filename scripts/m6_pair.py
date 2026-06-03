#!/usr/bin/env python
"""M6 milestone gate: one (source, target) pair end-to-end for both denoisers (handoff §11).

For source i and target j:
  * SADDPM: SDEdit-denoise i's Session-T and j's Session-E (conditioned on e(i)), train EEGNet on
    denoised i-T, test on denoised j-E.
  * ICA: same pipeline with ICA denoising instead.
Both produce a 4-class accuracy. Run on a GPU node.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from saddpm.baselines.ica import ICAConfig, ica_denoise_session  # noqa: E402
from saddpm.data.config import DataConfig  # noqa: E402
from saddpm.data.datasets import load_session_windows  # noqa: E402
from saddpm.diffusion.gaussian_diffusion import GaussianDiffusion  # noqa: E402
from saddpm.diffusion.schedule import DiffusionConfig  # noqa: E402
from saddpm.eval.downstream import downstream_accuracy  # noqa: E402
from saddpm.eval.pairwise import saddpm_denoise_windows  # noqa: E402
from saddpm.models.config import ModelConfig  # noqa: E402
from saddpm.models.dual_decoder import DualDecoderSADDPM  # noqa: E402
from saddpm.models.eegnet import EEGNetConfig  # noqa: E402
from saddpm.utils.checkpoint import load_checkpoint  # noqa: E402
from saddpm.utils.seed import seed_everything  # noqa: E402

CONFIGS = REPO_ROOT / "configs"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=int, default=1)
    parser.add_argument("--target", type=int, default=2)
    args = parser.parse_args()

    with open(CONFIGS / "m7.yaml", "r", encoding="utf-8") as fh:
        mc = yaml.safe_load(fh)
    data_cfg = DataConfig.from_yaml(CONFIGS / "data.yaml")
    eeg_cfg = EEGNetConfig.from_yaml(CONFIGS / "eegnet.yaml")
    ica_cfg = ICAConfig.from_yaml(CONFIGS / "ica.yaml")
    seed_everything(data_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    i, j = args.source, args.target

    ckpt = load_checkpoint(REPO_ROOT / mc["checkpoint"], map_location=str(device))
    model = DualDecoderSADDPM(ModelConfig(**ckpt["config"]["model"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    diffusion = GaussianDiffusion(DiffusionConfig(**ckpt["config"]["diffusion"])).to(device)
    print(f"[M6] device={device} source=A{i:02d} target=A{j:02d} t*={mc['t_star']}")

    src_T = load_session_windows(i, data_cfg, "T")
    tgt_E = load_session_windows(j, data_cfg, "E")

    # SADDPM pipeline (denoise with source embedding e(i)).
    dT = saddpm_denoise_windows(model, diffusion, src_T.windows, i - 1, mc["t_star"], mc["ddim_steps"], device, mc["denoise_batch"])
    dE = saddpm_denoise_windows(model, diffusion, tgt_E.windows, i - 1, mc["t_star"], mc["ddim_steps"], device, mc["denoise_batch"])
    acc_saddpm = downstream_accuracy(dT, src_T.mi_labels, dE, tgt_E.mi_labels, eeg_cfg, device, data_cfg.seed)

    # ICA pipeline.
    iT = ica_denoise_session(i, "T", data_cfg, ica_cfg)
    iE = ica_denoise_session(j, "E", data_cfg, ica_cfg)
    acc_ica = downstream_accuracy(iT.windows, iT.mi_labels, iE.windows, iE.mi_labels, eeg_cfg, device, data_cfg.seed)

    print(f"[M6] SADDPM (i->j) accuracy = {acc_saddpm:.3f}")
    print(f"[M6] ICA    (i->j) accuracy = {acc_ica:.3f} (EOG comps excluded: T={iT.n_excluded}, E={iE.n_excluded})")
    passed = (0.0 < acc_saddpm <= 1.0) and (0.0 < acc_ica <= 1.0)
    print(f"[M6] {'PASS' if passed else 'FAIL'} (both pipelines ran end-to-end)")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
