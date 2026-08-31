#!/usr/bin/env python
"""F6 — DS-DDPM Table II analogue: real-real vs generated-real subject
correlation matrices for the separated subject noise.

Their Table II computation is not in the release, so the signature choice is
frozen here, disclosed: a subject's noise signature = the channel-wise mean
Welch PSD (0-40 Hz, fs 250) of their separated subject-noise segments,
flattened; matrix cell (i, j) = Pearson r between signatures.
  real noise      = raw trial - F3-separated content (slice 750:1500, test set)
  generated noise = real_noises from the mirrored sample_save_data generation
CPU node.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/projects/EEG-foundation-model/derived/denoiseNet/dsddpm_repro/data")
OUT_DIR = REPO / "results/paper_final/dsddpm"
FS, FMAX = 250, 40.0


def _psd_signature(rows):
    """rows: [n, 22, 750] -> flattened mean PSD per channel up to FMAX."""
    from scipy.signal import welch
    f, p = welch(rows, fs=FS, nperseg=256, axis=-1)
    keep = f <= FMAX
    return p[:, :, keep].mean(axis=0).ravel()


def main() -> None:
    from scipy.io import loadmat
    real_sig, gen_sig = {}, {}
    for i in range(1, 10):
        raw = loadmat(DATA_ROOT / "single_sep" / f"single_subject_data_{i}.mat")
        sep = loadmat(DATA_ROOT / "dsddpm_sep" / f"single_subject_data_{i}.mat")
        noise = (np.asarray(raw["test_x"], np.float64)
                 - np.asarray(sep["test_x"], np.float64))[:, :, 750:1500]
        real_sig[i] = _psd_signature(noise)
        gen = loadmat(DATA_ROOT / "gen_undersampled" / f"single_subject_data_{i}.mat")
        gen_sig[i] = _psd_signature(np.asarray(gen["real_noises"], np.float64))
    rr = np.array([[np.corrcoef(real_sig[i], real_sig[j])[0, 1]
                    for j in range(1, 10)] for i in range(1, 10)])
    gr = np.array([[np.corrcoef(gen_sig[i], real_sig[j])[0, 1]
                    for j in range(1, 10)] for i in range(1, 10)])
    out = {"real_real": rr.round(4).tolist(),
           "gen_real": gr.round(4).tolist(),
           "real_real_diag_mean": round(float(np.diag(rr).mean()), 4),
           "gen_real_diag_mean": round(float(np.diag(gr).mean()), 4),
           "gen_real_offdiag_mean": round(float(
               (gr.sum() - np.trace(gr)) / 72.0), 4),
           "signature": "channel-wise mean Welch PSD 0-40 Hz, flattened",
           "note": "their Table II computation is unreleased; this choice "
                   "is frozen before numbers were seen"}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "table2_correlations.json").write_text(
        json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in
                      ("real_real_diag_mean", "gen_real_diag_mean",
                       "gen_real_offdiag_mean")}))


if __name__ == "__main__":
    main()
