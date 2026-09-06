#!/usr/bin/env python
"""Publication figures (EEGdenoiseNet-style) from the figgen_eval.py outputs.

F1 example waveforms | F2 metric-vs-SNR | F3 subject embedding-swap heatmap (centerpiece) |
F4 sampling recipe | F5 multi-channel + artifact topography | F6 PSD. Each figure that has its data
is written to artifacts/figures/paper/{name}.pdf and .png. Run on the login node after the GPU jobs land:
    python scripts/figgen_plot.py --which all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
FIGDATA = REPO_ROOT / "artifacts" / "figdata"
FIGCSV = REPO_ROOT / "results" / "figures"
OUT = REPO_ROOT / "artifacts" / "figures" / "paper"

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 9,
                     "axes.labelsize": 9, "legend.fontsize": 7, "figure.dpi": 150,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})
C_CLEAN, C_NOISY, C_DEN = "#222222", "#cf3a3a", "#2c6fbb"
OURS = "SADDPM-Cond"  # unified method label across all figures


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {name}.pdf/.png")


def fig_f1():
    """Example denoising waveforms (single-channel, per SNR), EOG + EMG."""
    for noise in ["EOG", "EMG"]:
        f = FIGDATA / f"f1_{noise}_single.npz"
        if not f.exists():
            print(f"  [F1:{noise}] missing {f.name}"); continue
        d = np.load(f, allow_pickle=True)
        clean, noisy, den, snr, fs = d["clean"], d["noisy"], d["denoised"], d["snr"], int(d["fs"])
        sel = list(range(0, len(snr), max(1, len(snr) // 4)))[:4]
        t = np.arange(clean.shape[-1]) / fs
        fig, axes = plt.subplots(len(sel), 1, figsize=(6.4, 1.55 * len(sel)), sharex=True)
        for ax, k in zip(np.atleast_1d(axes), sel):
            ax.plot(t, noisy[k], color=C_NOISY, lw=0.6, alpha=0.55, label="contaminated")
            ax.plot(t, clean[k], color=C_CLEAN, lw=1.0, label="clean (GT)")
            ax.plot(t, den[k], color=C_DEN, lw=1.0, label=OURS)
            ax.set_ylabel(f"SNR {snr[k]:+.0f} dB"); ax.margins(x=0)
        axes[0].legend(ncol=3, loc="upper right", framealpha=0.9)
        axes[-1].set_xlabel("time (s)")
        fig.suptitle(f"{noise} denoising — example traces", y=0.995)
        _save(fig, f"F1_waveforms_{noise}")


def fig_f2():
    """Metric vs input SNR, our denoiser + EEGdenoiseNet CNNs."""
    import csv
    for noise in ["EOG", "EMG"]:
        f = FIGCSV / f"f2_{noise}_persnr.csv"
        if not f.exists():
            print(f"  [F2:{noise}] missing {f.name}"); continue
        rows = list(csv.DictReader(f.open()))
        methods = dict.fromkeys(r["method"] for r in rows)
        fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9))
        for mi, (metric, ylab) in enumerate([("rrmse_t", "RRMSE temporal"), ("rrmse_s", "RRMSE spectral"), ("cc", "CC")]):
            for name in methods:
                pts = sorted(((float(r["snr_db"]), float(r[metric])) for r in rows if r["method"] == name))
                xs, ys = zip(*pts)
                ours, noisy = name.startswith("SADDPM"), name == "Noisy"
                col = C_DEN if ours else ("#9a9a9a" if noisy else None)
                axes[mi].plot(xs, ys, "o-" if ours else (":" if noisy else "s--"),
                              lw=2.2 if ours else 1.2, ms=4, label=(OURS if ours else name),
                              zorder=3 if ours else 2, color=col)
            axes[mi].set_xlabel("input SNR (dB)"); axes[mi].set_ylabel(ylab)
        axes[2].legend(loc="lower right")
        fig.suptitle(f"{noise} — denoising metrics vs input SNR (single-channel EEGdenoiseNet)", y=1.02)
        fig.tight_layout()
        _save(fig, f"F2_metrics_vs_snr_{noise}")


def fig_f3():
    """Subject embedding-swap heatmap (CENTERPIECE): denoise subject i with embedding j."""
    import csv
    noise = "EOG"
    mats, regimes = {}, [("baseline", "shared artifact"), ("subjart", "subject-specific artifact")]
    n = 9
    for regime, _ in regimes:
        rows = []
        for i in range(1, n + 1):
            f = FIGCSV / f"f3_{regime}_{noise}_s{i}.csv"
            if not f.exists():
                rows = None; break
            line = [r for r in csv.reader(f.open()) if r and not r[0].startswith("#")][1]
            rows.append([float(x) for x in line[1:]])  # e1..e9, null
        mats[regime] = np.array(rows) if rows is not None else None
    if all(m is None for m in mats.values()):
        print("  [F3] missing swap data"); return
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for ax, (regime, title) in zip(axes, regimes):
        M = mats[regime]
        if M is None:
            ax.set_visible(False); continue
        im = ax.imshow(M, vmin=0.4, vmax=1.0, cmap="viridis", aspect="auto")
        ax.set_xticks(range(n + 1)); ax.set_xticklabels([f"e{j+1}" for j in range(n)] + ["null"], fontsize=7)
        ax.set_yticks(range(n)); ax.set_yticklabels([f"s{i+1}" for i in range(n)], fontsize=7)
        ax.set_xlabel("subject embedding used"); ax.set_ylabel("source subject (data)")
        for i in range(n):  # outline the correct (diagonal) cell
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec="white", lw=1.5))
        diag = np.mean([M[i, i] for i in range(n)])
        offd = np.mean([M[i, j] for i in range(n) for j in range(n) if i != j])
        ax.set_title(f"{title}\ncorrect (diag) {diag:.3f}  vs  wrong (off-diag) {offd:.3f}")
        fig.colorbar(im, ax=ax, fraction=0.046, label="CC vs clean")
    fig.suptitle("Is the subject embedding load-bearing? Denoise subject $i$ with subject $j$'s embedding", y=1.02)
    fig.tight_layout()
    _save(fig, "F3_subject_swap_heatmap")


def fig_f4():
    """Sampling recipe: CC vs t* + ablation bars."""
    import csv
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.1))
    # (a) CC vs t*
    for noise, col in [("EOG", C_DEN), ("EMG", "#e08a1e")]:
        f = FIGCSV.parent / "m10" / f"{noise}_tstar_sweep.csv"
        if not f.exists():
            continue
        rows = [r for r in csv.reader(f.open()) if r and not r[0].startswith("#")][1:]
        xs, ys = [], []
        for r in rows:
            ts = 1000 if r[0] == "full" else int(r[0])
            xs.append(ts); ys.append(float(r[3]))
        order = np.argsort(xs)
        axes[0].plot(np.array(xs)[order], np.array(ys)[order], "o-", color=col, label=noise)
    axes[0].set_xlabel("t* (1000 = full conditional generation)"); axes[0].set_ylabel("CC")
    axes[0].set_title("(a) sampler start: warm-start vs full generation"); axes[0].legend()
    # (b) ablation bars (EOG). NOTE: arm names contain commas, so split off the last 3 numeric fields.
    f = FIGCSV.parent / "m10" / "EOG_ablation.csv"
    if f.exists():
        rows = {}
        for line in f.read_text().splitlines():
            if not line or line.startswith("#") or line.startswith("arm,"):
                continue
            name, _rt, _rs, cc = line.rsplit(",", 3)
            rows[name] = float(cc)
        bars = [("eps\n(orig)", rows.get("A1 eps,train-mode(dropout),fullgen")),
                ("+eval", rows.get("A2 eps,eval-mode,fullgen")),
                ("+x0", rows.get("A3 x0,eval,fullgen")),
                (f"+full-gen\n({OURS})", rows.get("A5 x0+EMA,t*=full")),
                ("SimpleCNN", rows.get("R1 SimpleCNN"))]
        bars = [(k, v) for k, v in bars if v is not None]
        labels, vals = zip(*bars)
        cols = [C_DEN if OURS in l else ("#888" if "CNN" in l else "#9bb7d4") for l in labels]
        axes[1].bar(range(len(vals)), vals, color=cols)
        axes[1].set_xticks(range(len(labels))); axes[1].set_xticklabels(labels, fontsize=7)
        axes[1].set_ylim(0.5, 0.95); axes[1].set_ylabel("CC (EOG)")
        axes[1].set_title("(b) recipe ablation (single-channel)")
        for i, v in enumerate(vals):
            axes[1].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=6.5)
    fig.tight_layout()
    _save(fig, "F4_sampling_recipe")


def fig_f5():
    """Multi-channel: artifact topography + 22-channel example."""
    for noise in ["EOG", "EMG"]:
        f = FIGDATA / f"f5_{noise}.npz"
        if not f.exists():
            print(f"  [F5:{noise}] missing {f.name}"); continue
        d = np.load(f, allow_pickle=True)
        topo, ch, clean, noisy, den, fs = (d["topo"], list(d["ch_names"]), d["clean"], d["noisy"], d["denoised"], int(d["fs"]))
        fig = plt.figure(figsize=(9.6, 3.1))
        gs = fig.add_gridspec(1, 4, width_ratios=[1.1, 1, 1, 1])
        # topography (try MNE topomap; fall back to bar)
        axt = fig.add_subplot(gs[0, 0]); plotted = False
        try:
            import mne
            info = mne.create_info(ch, fs, "eeg"); info.set_montage("standard_1020", match_case=False, on_missing="ignore")
            mne.viz.plot_topomap(topo, info, axes=axt, show=False, cmap="Reds", contours=0)
            plotted = True
        except Exception as e:  # noqa: BLE001
            print(f"  [F5:{noise}] topomap fallback ({e})")
        if not plotted:
            axt.barh(range(len(ch)), topo, color="#cf3a3a"); axt.set_yticks(range(len(ch)))
            axt.set_yticklabels(ch, fontsize=5); axt.invert_yaxis()
        axt.set_title(f"{noise} artifact\ntopography")
        # noisy / denoised / clean as channels x time images
        vmax = np.percentile(np.abs(noisy), 99)
        for ax, M, ttl in zip([fig.add_subplot(gs[0, i]) for i in (1, 2, 3)],
                               [noisy, den, clean], ["contaminated", OURS, "clean (GT)"]):
            ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      extent=[0, M.shape[-1] / fs, M.shape[0], 0])
            ax.set_title(ttl); ax.set_xlabel("time (s)"); ax.set_yticks([])
            ax.grid(False)
        fig.text(0.30, 0.5, "channel", va="center", rotation="vertical", fontsize=8)
        fig.suptitle(f"{noise}: multi-channel joint denoising exploits artifact topography", y=1.02)
        fig.tight_layout()
        _save(fig, f"F5_multichannel_{noise}")


def fig_f6():
    """PSD of clean / contaminated / denoised."""
    from scipy.signal import welch
    for noise in ["EOG", "EMG"]:
        f = FIGDATA / f"f1_{noise}_single.npz"
        if not f.exists():
            print(f"  [F6:{noise}] missing {f.name}"); continue
        d = np.load(f, allow_pickle=True)
        clean, noisy, den, fs = d["clean"], d["noisy"], d["denoised"], int(d["fs"])

        def psd(X):
            fr, P = welch(X, fs=fs, nperseg=min(256, X.shape[-1]))
            return fr, P.mean(0)
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        for X, c, lab in [(noisy, C_NOISY, "contaminated"), (den, C_DEN, OURS), (clean, C_CLEAN, "clean (GT)")]:
            fr, P = psd(X); ax.semilogy(fr, P, color=c, lw=1.4, label=lab)
        ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("PSD"); ax.set_xlim(0, min(fs / 2, 80))
        _, P0 = psd(clean)
        ax.set_ylim(max(P0.min() * 0.2, 1e-6), None)
        ax.legend(); ax.set_title(f"{noise}: spectral recovery")
        fig.tight_layout()
        _save(fig, f"F6_psd_{noise}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all")
    args = ap.parse_args()
    figs = {"f1": fig_f1, "f2": fig_f2, "f3": fig_f3, "f4": fig_f4, "f5": fig_f5, "f6": fig_f6}
    sel = list(figs) if args.which == "all" else args.which.split(",")
    for k in sel:
        print(f"[{k.upper()}]")
        figs[k]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
