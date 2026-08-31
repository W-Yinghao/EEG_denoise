#!/usr/bin/env python3
"""fig-bench-snr — single-channel EEGdenoiseNet benchmark across SNR (v3 appendix).

2x3 line-panel grid: rows = {EOG, EMG}, columns = {CC, RRMSE_t, RRMSE_s};
11 SNR levels (-5..+5 dB), one line per arm (all 8 arms plotted, none
dropped). Right of each panel, a marginal "all" strip shows the overall
(SNR-pooled) value per arm as a dot — these are the numbers quoted in the
caption; in the CC panels the two decision-relevant overall values
(CondDiff vs the best supervised arm) carry tiny numeric annotations.
Direct color-matched arm labels sit in the rightmost column; no legend box.

Claim scope (per verified plan + fix): on EOG, CondDiff (overall CC 0.8880)
beats 2/5 supervised arms (FCNN 0.8771, RNN_LSTM 0.8238) but not
SimpleCNN (0.9336); on EMG it beats 0/5 supervised arms (NovelCNN 0.8775
vs 0.7625) though it beats SDEdit and Noisy on both artifacts. Honesty:
SDEdit sits below even the Noisy input on EOG (CC 0.6083 vs 0.6691).
All of these are asserted against the CSVs at runtime.

House rules: no CIs / error bars / bands anywhere; every plotted number is
read from results/paper_final/e12/{EOG,EMG}_grid.csv at runtime.
"""
import csv
import sys

import numpy as np

sys.path.insert(0, "/home/infres/yinwang/denoiseNet/scripts/paper_final/figures")
import figstyle

figstyle.setup()

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

C = figstyle.C
E12 = "/home/infres/yinwang/denoiseNet/results/paper_final/e12"

METHODS = ["Noisy", "SDEdit", "CondDiff", "FCNN", "SimpleCNN",
           "ComplexCNN", "RNN_LSTM", "NovelCNN"]
LABEL = {"RNN_LSTM": "RNN-LSTM"}          # display tweaks only
SNRS = list(range(-5, 6))
METRICS = ["cc", "rrmse_t", "rrmse_s"]


def load(artifact):
    per, overall = {}, {}
    with open(f"{E12}/{artifact}_grid.csv") as f:
        for row in csv.DictReader(f):
            vals = {m: float(row[m]) for m in METRICS}
            if row["snr_db"] == "overall":
                overall[row["method"]] = vals
            else:
                per.setdefault(row["method"], {})[int(row["snr_db"])] = vals
    assert set(per) == set(overall) == set(METHODS)
    assert all(sorted(per[m]) == SNRS for m in METHODS)
    return per, overall


DATA = {a: load(a) for a in ("EOG", "EMG")}

# ---- caption numbers must match the CSVs (verdict.fix scoping included) ----
eog_o, emg_o = DATA["EOG"][1], DATA["EMG"][1]
assert eog_o["SimpleCNN"]["cc"] == 0.9336 and eog_o["CondDiff"]["cc"] == 0.8880
assert eog_o["FCNN"]["cc"] == 0.8771 and eog_o["RNN_LSTM"]["cc"] == 0.8238
assert emg_o["NovelCNN"]["cc"] == 0.8775 and emg_o["CondDiff"]["cc"] == 0.7625
assert eog_o["SDEdit"]["cc"] == 0.6083 < eog_o["Noisy"]["cc"] == 0.6691
sup = ["FCNN", "SimpleCNN", "ComplexCNN", "RNN_LSTM", "NovelCNN"]
assert sum(eog_o["CondDiff"]["cc"] > eog_o[s]["cc"] for s in sup) == 2  # EOG 2/5
assert sum(emg_o["CondDiff"]["cc"] > emg_o[s]["cc"] for s in sup) == 0  # EMG 0/5
assert all(o["CondDiff"]["cc"] > max(o["SDEdit"]["cc"], o["Noisy"]["cc"])
           for o in (eog_o, emg_o))

# ---- per-arm draw style: one arm = one color (figstyle.C), tiny markers ----
STYLE = {  # (linestyle, marker, linewidth, zorder)
    "Noisy":      ("--", "",  1.0, 3),
    "SDEdit":     ("-",  "",  1.0, 3),
    "CondDiff":   ("-",  "o", 1.5, 6),
    "FCNN":       ("-",  "s", 0.8, 2),
    "SimpleCNN":  ("-",  "^", 0.8, 2),
    "ComplexCNN": ("-",  "D", 0.8, 2),
    "RNN_LSTM":   ("-",  "v", 0.8, 2),
    "NovelCNN":   ("-",  "P", 0.8, 2),
}
X_ALL = 7.0          # x position of the marginal "all" (overall) strip


def dodge(vals, sep):
    """Least-squares label dodging: min separation `sep`, clusters stay
    centered on their data (isotonic regression / pool-adjacent-violators)."""
    vals = np.asarray(vals, float)
    order = np.argsort(vals)
    t = vals[order] - np.arange(len(order)) * sep
    blocks = []
    for v in t:
        blocks.append([v, 1])
        while len(blocks) > 1 and blocks[-2][0] >= blocks[-1][0]:
            m2, n2 = blocks.pop()
            m1, n1 = blocks.pop()
            blocks.append([(m1 * n1 + m2 * n2) / (n1 + n2), n1 + n2])
    z = np.concatenate([[m] * n for m, n in blocks])
    out = np.empty_like(vals)
    out[order] = z + np.arange(len(order)) * sep
    return out


fig, axes = plt.subplots(2, 3, figsize=(figstyle.FULL, 4.2),
                         gridspec_kw=dict(left=0.075, right=0.895, top=0.94,
                                          bottom=0.10, wspace=0.42, hspace=0.30))

for r, artifact in enumerate(("EOG", "EMG")):
    per, overall = DATA[artifact]
    for c, metric in enumerate(METRICS):
        ax = axes[r, c]
        for m in METHODS:
            ls, mk, lw, zo = STYLE[m]
            y = [per[m][s][metric] for s in SNRS]
            ax.plot(SNRS, y, ls, marker=mk, ms=2.0, color=C[m], lw=lw,
                    zorder=zo, markeredgewidth=0)
            ax.plot([X_ALL], [overall[m][metric]], mk or "o", ms=2.6,
                    color=C[m], zorder=zo, markeredgewidth=0, clip_on=False)
        ax.axvline(6.0, color="0.82", lw=0.5, zorder=1)
        ax.grid(axis="y", lw=0.3, alpha=0.25, zorder=0)
        ax.set_xlim(-5.5, X_ALL + 0.6)
        ax.set_xticks([-5, 0, 5, X_ALL])
        ax.set_xticklabels(["-5", "0", "5", "all"])
        if metric == "cc":
            ax.set_ylim(0.2, 1.0)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_ylabel("CC")
            # decision-relevant overall CCs, quoted in the caption
            best = max(sup, key=lambda s: overall[s]["cc"])
            for m in ("CondDiff", best):
                v = overall[m]["cc"]
                ax.annotate(f"{v:.3f}", (X_ALL, v), xytext=(3.5, 0),
                            textcoords="offset points", fontsize=5,
                            color=C[m], va="center", ha="left",
                            annotation_clip=False)
        else:
            ax.set_yscale("log")
            ax.set_ylabel("RRMSE$_t$" if metric == "rrmse_t" else "RRMSE$_s$")
            lo = min(per[m][s][metric] for m in METHODS for s in SNRS)
            hi = max(per[m][s][metric] for m in METHODS for s in SNRS)
            ax.set_ylim(lo / 1.15, hi * 1.15)
            ticks = [t for t in (0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20)
                     if lo / 1.15 <= t <= hi * 1.15]
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            fmt = ScalarFormatter()
            fmt.set_scientific(False)
            ax.yaxis.set_major_formatter(fmt)
            ax.yaxis.set_minor_locator(FixedLocator([]))
            ax.yaxis.set_minor_formatter(NullFormatter())
        if r == 1:
            ax.set_xlabel("SNR (dB)")
        figstyle.panel(ax, "ABCDEF"[r * 3 + c])
    # row (artifact) tag, left of the row's first panel
    axes[r, 0].text(-0.30, 0.5, artifact, transform=axes[r, 0].transAxes,
                    rotation=90, va="center", ha="center", fontsize=8,
                    fontweight="bold")
    # direct color-matched arm labels, right of the rightmost panel,
    # attached (with dodging) to the overall dots on its log scale
    ax = axes[r, 2]
    ov = np.array([overall[m]["rrmse_s"] for m in METHODS])
    ax_h = ax.get_position().height * fig.get_size_inches()[1]  # inches
    logrange = np.log10(ax.get_ylim()[1] / ax.get_ylim()[0])
    ylab = 10 ** dodge(np.log10(ov), 0.095 / ax_h * logrange)
    for m, y0, y1 in zip(METHODS, ov, ylab):
        ax.annotate(LABEL.get(m, m), (X_ALL + 0.55, y1), fontsize=5.5,
                    color=C[m], va="center", ha="left", annotation_clip=False,
                    fontweight="bold" if m == "CondDiff" else "normal")

figstyle.save(fig, "fig-bench-snr")
