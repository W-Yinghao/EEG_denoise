#!/usr/bin/env python
"""E5 — band-wise phase-locking value (PLV) between denoised and clean SSED
rows, the EEGDfus Tables IV/V analogue. PLV(row) = |mean_t exp(i(phi_den -
phi_clean))| from the Hilbert phase of the band-passed signals, averaged over
rows; identical rows/splits as every other SSED evaluation. CPU node.
Arms: identity (contaminated as-is), EEGDfus repro, ours-SSED (e3b),
ours zero-shot (e4zero), ours finetuned (e4ft).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eegdfus_ssed as ES  # noqa: E402

FS = 200
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 80)}
DFUS = Path("/projects/EEG-foundation-model/derived/denoiseNet/eegdfus_ssed")
OURS = Path("/projects/EEG-foundation-model/derived/denoiseNet/e34_ours")
OUT_DIR = Path(__file__).resolve().parents[2] / "results/paper_final/e5"


def _plv(a, b, lo, hi):
    from scipy.signal import butter, filtfilt, hilbert
    bb, aa = butter(4, [lo / (FS / 2), min(hi / (FS / 2), 0.99)], "bandpass")
    pa = np.angle(hilbert(filtfilt(bb, aa, a, axis=-1), axis=-1))
    pb = np.angle(hilbert(filtfilt(bb, aa, b, axis=-1), axis=-1))
    return np.abs(np.exp(1j * (pa - pb)).mean(axis=-1))


def main() -> None:
    x, y = ES._dataset()
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    splits = ES._splits(len(x))
    arms = {}
    for name in ("released_test", "strict_test"):
        idx = np.asarray(splits[name])
        arms[("identity", name)] = (idx, x[idx])
        f = DFUS / f"denoised_{name}.npz"
        if f.is_file():
            d = np.load(f)
            arms[("eegdfus", name)] = (d["idx"], d["denoised"])
        for tag in ("e3b", "e4zero", "e4ft"):
            f = OURS / f"denoised_{tag}_{name}.npz"
            if f.is_file():
                d = np.load(f)
                arms[(tag, name)] = (d["idx"], d["denoised"])
    results = {}
    for (arm, split), (idx, den) in sorted(arms.items()):
        clean = y[np.asarray(idx)]
        row = {band: float(np.mean(_plv(den, clean, lo, hi)))
               for band, (lo, hi) in BANDS.items()}
        results[f"{arm}/{split}"] = row
        print(json.dumps({f"{arm}/{split}": row}), flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "plv.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
