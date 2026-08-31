"""Shared style for every v3 paper figure.

House rules (user-frozen): high information density; NO confidence intervals,
error bars, or bootstrap bands anywhere — show per-participant points/lines
instead; minimal text (single-letter panel tags, short axis labels, direct
line labeling over legend boxes); every figure saved as vector PDF + PNG
preview. One arm = one color across ALL figures.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[3] / "artifacts/figures/v3"

# one arm = one color, everywhere (colorblind-safe)
C = {
    "RAW": "#7f7f7f", "Noisy": "#7f7f7f", "identity": "#7f7f7f",
    "reference": "#2b2b2b", "clean": "#2b2b2b",
    "MATCH": "#D55E00", "MATCH_gated": "#D55E00", "matched": "#D55E00",
    "CondDiff": "#D55E00", "ours": "#D55E00",
    "LINEAR": "#E69F00", "linear": "#E69F00",
    "POP": "#56B4E9", "population": "#56B4E9",
    "NO_A0": "#0072B2", "unguided": "#0072B2",
    "WRONG": "#CC79A7", "WRONG_gated": "#CC79A7", "SHUFFLED": "#b06592",
    "mismatched": "#CC79A7",
    "ICA": "#009E73", "ASR": "#117733", "SGEYESUB": "#882255",
    "SDEdit": "#999933", "EEGDfus": "#332288", "DSDDPM": "#88CCEE",
    "FCNN": "#a6bddb", "SimpleCNN": "#67a9cf", "ComplexCNN": "#3690c0",
    "RNN_LSTM": "#02818a", "NovelCNN": "#016c59",
    "eog": "#CC79A7",
}

FULL = 7.0   # inches, double-column width
HALF = 3.35  # single column


def setup():
    plt.rcParams.update({
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
        "ytick.labelsize": 6, "legend.fontsize": 6,
        "font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        "legend.frameon": False, "figure.dpi": 110,
    })


def panel(ax, tag):
    ax.text(-0.08, 1.08, tag, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="right")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02,
                dpi=200)
    print(f"saved {OUT}/{name}.pdf")
