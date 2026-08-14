"""FLAGSHIP-M35 execution CLI. Rules frozen in reports/m35_preregistration.md."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eeg_chart.analytic import canonical_clean
from eeg_chart.geodesic import rho_eb, transport_family
from eeg_chart.run_m0 import _canon_path, _load_panel, _stat, transport_context
from eeg_chart.run_m13 import PANEL_TRANSPORT


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results/flagship_m35"
REPORT = ROOT / "reports"
HARD_GATE_MIN_SECONDS = 60
RHO_GRID = (("0.00", 0.0), ("0.25", 0.25), ("0.50", 0.5), ("0.75", 0.75),
            ("EB", None), ("1.00", 1.0))
SUPPORT_SECONDS = {"klados": 15.0, "bci2b": 300.0}   # nominal; bci2b support >> 60 s


def _anchor_operators(cells):
    """Frozen λ-gated anchor operators per cell (S3c closed form, 2-half within)."""
    pop_by_cell, lam_by_cell = {}, {}
    withins = {cell.cell: float(np.mean(np.square(cell.a_halves[0] - cell.a_halves[1])) / 2.0)
               for cell in cells}
    threshold = float(np.percentile(list(withins.values()), 95))
    for cell in cells:
        others = [c.a_support for c in cells if c.subject != cell.subject]
        pop = np.mean(others, axis=0)
        tau2 = float(np.mean([np.mean(np.square(c.a_support - pop)) for c in cells
                              if c.subject != cell.subject]))
        support_seconds = SUPPORT_SECONDS[cell.panel]
        if support_seconds < HARD_GATE_MIN_SECONDS or withins[cell.cell] > threshold:
            lam = 0.0
        else:
            lam = rho_eb(tau2, withins[cell.cell] * 2.0)   # within of a single fit
        pop_by_cell[cell.cell] = pop
        lam_by_cell[cell.cell] = lam
    return pop_by_cell, lam_by_cell


def u1() -> None:
    from eeg_scad.cli.run_v43 import bootstrap_draws, holm

    canon = np.load(_canon_path())["u_canon"]
    out = {}
    for panel in ("klados", "bci2b"):
        cells, lift = _load_panel(panel)
        context = transport_context(cells, lift, canon, whitening=PANEL_TRANSPORT[panel],
                                    split_half_abstain=True)
        pop_ops, lams = _anchor_operators(cells)
        ordered = sorted(context["per_cell"].items())
        unit_rows: dict[str, dict[str, list[float]]] = {}
        for index, (cell_id, entry) in enumerate(ordered):
            cell = entry["cell"]
            donor = next(other for _, other in ordered[index + 1:] + ordered[:index]
                         if other["cell"].subject != cell.subject)
            transports = {
                "T-POP": transport_family(context["lift"], context["lift_pinv"],
                                          context["sigma_bar"], None, entry["base"],
                                          entry["base"], 0.0),
                "T-MATCH": transport_family(context["lift"], context["lift_pinv"],
                                            context["sigma_bar"], cell.sigma_support,
                                            entry["rotation"], entry["base"], entry["rho"],
                                            whitening=PANEL_TRANSPORT[panel]),
                "T-WRONG": transport_family(context["lift"], context["lift_pinv"],
                                            context["sigma_bar"],
                                            donor["cell"].sigma_support, donor["rotation"],
                                            entry["base"], donor["rho"],
                                            whitening=PANEL_TRANSPORT[panel]),
            }
            c0 = pop_ops[cell.cell]
            c_match = c0 + lams[cell.cell] * (cell.a_support - c0)
            donor_c = pop_ops[cell.cell] + lams[donor["cell"].cell] \
                * (donor["cell"].a_support - pop_ops[cell.cell])
            unit = cell.subject if panel != "klados" else cell.cell
            for episode in cell.episodes:
                anchors = {"A0-none": np.zeros_like(episode["y"]),
                           "A0-POP": c0 @ episode["drive"],
                           "A0-MATCH": c_match @ episode["drive"],
                           "A0-WRONGg": donor_c @ episode["drive"]}
                for t_name, arm in transports.items():
                    for a_name, anchor in anchors.items():
                        if t_name == "T-WRONG" and a_name != "A0-MATCH":
                            continue                       # descriptive row only
                        if a_name == "A0-WRONGg" and t_name != "T-MATCH":
                            continue                       # descriptive row only
                        cleaned = canonical_clean(arm, context["u_canon"],
                                                  context["sigma_bar_inv"],
                                                  episode["y"] - anchor)
                        rrmse = float(np.linalg.norm(cleaned - episode["x"])
                                      / max(np.linalg.norm(episode["x"]), 1e-12))
                        unit_rows.setdefault(f"{t_name}|{a_name}", {}) \
                            .setdefault(unit, []).append(rrmse)
        per = {arm: {u: float(np.mean(v)) for u, v in units.items()}
               for arm, units in unit_rows.items()}
        common = sorted(set.intersection(*(set(v) for v in per.values())))
        def delta(a, b):
            return np.asarray([per[a][u] - per[b][u] for u in common])
        uf1 = delta("T-POP|A0-MATCH", "T-MATCH|A0-MATCH")
        uf2 = delta("T-MATCH|A0-POP", "T-MATCH|A0-MATCH")
        base = np.asarray([per["T-POP|A0-POP"][u] for u in common])
        joint = np.asarray([per["T-MATCH|A0-MATCH"][u] for u in common])
        leg_t = np.asarray([per["T-MATCH|A0-POP"][u] for u in common])
        leg_a = np.asarray([per["T-POP|A0-MATCH"][u] for u in common])
        best_single = np.minimum(leg_t, leg_a)
        uf3 = best_single - joint
        gain_joint = base - joint
        gain_sum = (base - leg_t) + (base - leg_a)
        additivity = float(gain_joint.mean() / gain_sum.mean()) if abs(gain_sum.mean()) > 1e-9 \
            else float("nan")
        rng_draws = {name: bootstrap_draws(values) for name, values in
                     (("UF-1", uf1), ("UF-2", uf2), ("UF-3", uf3))}
        add_draws = []
        rng = np.random.default_rng(420)
        for _ in range(5000):
            pick = rng.integers(0, len(common), len(common))
            gs = gain_sum[pick].mean()
            add_draws.append(gain_joint[pick].mean() / gs if abs(gs) > 1e-9 else np.nan)
        add_draws = np.asarray([v for v in add_draws if np.isfinite(v)])
        out[panel] = {
            "arm_means": {arm: float(np.mean(list(v.values()))) for arm, v in per.items()},
            "UF-1": {**_stat(uf1), "pass": bool(uf1.mean() > 0
                                                and _stat(uf1)["bootstrap_low"] > 0),
                     "p_raw": float(np.mean(rng_draws["UF-1"] <= 0))},
            "UF-2": {**_stat(uf2), "pass": bool(uf2.mean() > 0
                                                and _stat(uf2)["bootstrap_low"] > 0),
                     "p_raw": float(np.mean(rng_draws["UF-2"] <= 0))},
            "UF-3": {**_stat(uf3), "pass": bool(uf3.mean() > 0
                                                and _stat(uf3)["bootstrap_low"] > 0),
                     "p_raw": float(np.mean(rng_draws["UF-3"] <= 0)),
                     "additivity_index": additivity,
                     "additivity_ci": [float(np.quantile(add_draws, .025)),
                                       float(np.quantile(add_draws, .975))]
                     if len(add_draws) else None},
            "descriptive": {"T-WRONG|A0-MATCH_minus_joint":
                            _stat(delta("T-WRONG|A0-MATCH", "T-MATCH|A0-MATCH")),
                            "T-MATCH|A0-WRONGg_minus_joint":
                            _stat(delta("T-MATCH|A0-WRONGg", "T-MATCH|A0-MATCH"))},
            "anchor_lambda_abstained_fraction":
                float(np.mean([lams[c.cell] == 0.0 for c in cells])),
            "units": len(common),
        }
        p_raw = {name: out[panel][name]["p_raw"] for name in ("UF-1", "UF-2", "UF-3")}
        out[panel]["holm"] = {"p_raw": p_raw, "p_adjusted": holm(p_raw)}
    target = RESULT / "u1_factorial"
    target.mkdir(parents=True, exist_ok=True)
    (target / "decision.json").write_text(json.dumps(
        {"preregistration": "reports/m35_preregistration.md", "panels": out,
         "sealed_reads": 0}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({panel: {name: out[panel][name]["pass"]
                              for name in ("UF-1", "UF-2", "UF-3")} for panel in out}))


def p1() -> None:
    """Transport-state linkage at the ρ grid (descriptive; two-leg privacy story)."""
    from eeg_chart.transport import minimal_rotation, ordered_frame, spd_power
    from eeg_scad.evaluation.linkage_diagnostic import linkage

    canon = np.load(_canon_path())["u_canon"]
    rows = []
    for panel in ("klados", "bci2b"):
        cells, lift = _load_panel(panel)
        context = transport_context(cells, lift, canon, whitening=PANEL_TRANSPORT[panel],
                                    split_half_abstain=True)
        bar_root = spd_power(context["sigma_bar"], 0.5)
        bar_inv_root = spd_power(context["sigma_bar"], -0.5)

        def state_feature(cell, half_index: int, rho: float):
            a_half = cell.a_halves[half_index]
            sigma_half = cell.sigma_support   # covariance summary shared; frame varies
            whitened = bar_inv_root @ sigma_half @ bar_inv_root
            sigma_rho = bar_root @ spd_power(whitened, rho) @ bar_root
            eigs = np.sort(np.linalg.eigvalsh(sigma_rho))[-20:]
            rotation = minimal_rotation(ordered_frame(lift @ a_half), canon)
            base = context["per_cell"][cell.cell]["base"]
            frame_feat = rho * ((rotation @ base.T - np.eye(len(base))) @ canon).reshape(-1)
            return np.concatenate((np.log(np.clip(eigs, 1e-12, None)) * rho, frame_feat))

        for label, rho_value in RHO_GRID:
            features = {}
            for cell in cells:
                unit = cell.subject if panel != "klados" else cell.cell
                if unit in features:
                    continue
                rho = context["per_cell"][cell.cell]["rho"] if rho_value is None else rho_value
                features[unit] = (state_feature(cell, 0, rho), state_feature(cell, 1, rho))
            metrics = linkage(features)[0]
            rows.append({"panel": panel, "rho_label": label,
                         "rho_mean": float(np.mean([context["per_cell"][c.cell]["rho"]
                                                    for c in cells]))
                         if rho_value is None else rho_value, **metrics})
    target = RESULT / "p1_privacy"
    target.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(target / "transport_state_privacy.csv", index=False)
    (target / "decision.json").write_text(json.dumps(
        {"descriptive_only": True, "rows": rows, "sealed_reads": 0},
        indent=2, sort_keys=True) + "\n")
    print(frame.groupby(["panel", "rho_label"])[["top1_accuracy", "same_different_auroc"]]
          .mean().round(4).to_string())


# ---------------------------------------------------------------- C-2 BrainID

BRAINID_ROOT = Path("/projects/EEG-foundation-model/Longitudinal_ERP_Brainprint/files")
BRAINID_DAYS = {"day1": "Day_1", "day7": "Day_7", "day80": "Day_80"}  # Day_200 sealed


def _brainid_labels() -> list[str]:
    lines = (BRAINID_ROOT / "ChannelPosition.locs").read_text().strip().splitlines()
    return [line.split()[-1].strip(".") for line in lines]


def _brainid_day(handle, day: str) -> np.ndarray:
    """Concatenate the day's 4 blocks -> (57, T) at 100 Hz, V19-style prep."""
    from scipy.signal import butter, sosfiltfilt

    from eeg_chart.prior_data import _resample_100

    refs = handle[BRAINID_DAYS[day]][:, 0]
    blocks = []
    sos = butter(4, (0.5, 15.0), btype="bandpass", fs=1000.0, output="sos")
    for ref in refs:
        block = np.asarray(handle[ref], np.float64)      # (58, N) or (N, 58)
        if block.shape[0] != 58:
            block = block.T
        eeg = block[:57]
        eeg = eeg - eeg.mean(axis=0, keepdims=True)      # CAR over 57
        eeg = sosfiltfilt(sos, eeg, axis=-1)
        blocks.append(_resample_100(eeg, 1000.0))
    return np.concatenate(blocks, axis=1)


def _blink_assets(day_data: np.ndarray, labels: list[str], rng: np.random.Generator):
    """Pseudo-drive (2 PCs of Fp1/Fp2, 0.5-8 Hz, blink-rich calibration) + operator."""
    from scipy.signal import butter, sosfiltfilt

    from eeg_scad.data.artifact_transfer_v41r import ridge_transfer
    from eeg_scad.data.v24_coordinate_contract import robust_center_scale

    frontal_index = [labels.index("Fp1"), labels.index("Fp2")]
    sos = butter(4, (0.5, 8.0), btype="bandpass", fs=100.0, output="sos")
    frontal = sosfiltfilt(sos, day_data[frontal_index], axis=-1)
    window = 512
    starts = np.arange(0, day_data.shape[1] - window, window)
    rms = np.asarray([float(np.sqrt(np.mean(frontal[:, s:s + window] ** 2))) for s in starts])
    rich = starts[rms >= np.quantile(rms, .9)]
    segments = np.concatenate([frontal[:, s:s + window] for s in rich], axis=1)
    center, scale = robust_center_scale(segments)
    standardized = (segments - center[:, None]) / scale[:, None]
    pca = np.linalg.svd(np.cov(standardized), full_matrices=False)[0][:, :2].T
    _, eeg_scale = robust_center_scale(day_data)

    def drive_of(data: np.ndarray) -> np.ndarray:
        f = sosfiltfilt(sos, data[frontal_index], axis=-1)
        return pca @ ((f - center[:, None]) / scale[:, None])

    rich_eeg = np.concatenate([day_data[:, s:s + window] for s in rich], axis=1)
    operator = ridge_transfer(rich_eeg / eeg_scale[:, None],
                              drive_of(np.concatenate(
                                  [day_data[:, s:s + window] for s in rich], axis=1)), .05)[0]
    halves = []
    mid = rich_eeg.shape[1] // 2
    rich_drive = drive_of(np.concatenate([day_data[:, s:s + window] for s in rich], axis=1))
    for sl in (slice(0, mid), slice(mid, None)):
        halves.append(ridge_transfer(rich_eeg[:, sl] / eeg_scale[:, None],
                                     rich_drive[:, sl], .05)[0])
    return {"drive_of": drive_of, "operator": operator, "halves": halves,
            "eeg_scale": eeg_scale, "rms_threshold": float(np.quantile(rms, .9))}


def c2() -> None:
    import h5py
    from eeg_chart.transport import (airm_frechet_mean, ledoit_wolf_covariance,
                                     minimal_rotation, ordered_frame, sh_lift, spd_power)
    from eeg_chart.positions import _resolve
    from eeg_scad.cli.run_v43 import bootstrap_draws

    labels = _brainid_labels()
    positions, kept, missing = _resolve(labels, allow_missing=True)
    keep_index = [labels.index(name) for name in kept]
    lift = sh_lift(positions)
    lift_pinv = np.linalg.pinv(lift)
    canon = np.load(_canon_path())["u_canon"]
    rng = np.random.default_rng(20269002)
    subjects = [f"S{i}" for i in range(1, 16)]
    per_subject = {}
    for subject in subjects:
        with h5py.File(BRAINID_ROOT / f"{subject}.mat", "r") as handle:
            day1 = _brainid_day(handle, "day1")
            day7 = _brainid_day(handle, "day7")
            day80 = _brainid_day(handle, "day80")
        assets = _blink_assets(day1, labels, rng)
        scaled1 = (day1 / assets["eeg_scale"][:, None])[keep_index]
        sigma = ledoit_wolf_covariance(lift @ scaled1[:, :60000])
        mid = scaled1.shape[1] // 2
        halves_sigma = [ledoit_wolf_covariance(lift @ scaled1[:, :mid]),
                        ledoit_wolf_covariance(lift @ scaled1[:, mid:])]
        per_subject[subject] = {"assets": assets, "sigma": sigma,
                                "sigma_halves": halves_sigma,
                                "queries": {"day7": day7, "day80": day80}}
    sigma_bar = airm_frechet_mean([v["sigma"] for v in per_subject.values()])
    sigma_bar_inv = spd_power(sigma_bar, -1.0)
    a_bar = np.mean([v["assets"]["operator"] for v in per_subject.values()], axis=0)
    base = minimal_rotation(ordered_frame(lift @ a_bar[keep_index]), canon)

    def airm_d(a, b):
        inv_root = spd_power(a, -0.5)
        eigs = np.linalg.eigvalsh(inv_root @ b @ inv_root).clip(1e-12)
        return float(np.linalg.norm(np.log(eigs)))

    between = [airm_d(per_subject[a]["sigma"], per_subject[b]["sigma"])
               for i, a in enumerate(subjects) for b in subjects[i + 1:]]
    between_median = float(np.median(between))
    tau2 = float(np.mean([airm_d(v["sigma"], sigma_bar) ** 2 for v in per_subject.values()]))
    out = {}
    for day in ("day7", "day80"):
        unit_rows: dict[str, dict[str, list[float]]] = {}
        for s_index, subject in enumerate(subjects):
            entry = per_subject[subject]
            v_s = airm_d(entry["sigma_halves"][0], entry["sigma_halves"][1]) ** 2
            abstain = np.sqrt(v_s) > between_median
            rho = 0.0 if abstain else rho_eb(tau2, v_s)
            donor = per_subject[subjects[(s_index + 1) % len(subjects)]]
            arms = {
                "T-POP": transport_family(lift, lift_pinv, sigma_bar, None, base, base, 0.0),
                "T-MATCH": transport_family(lift, lift_pinv, sigma_bar, entry["sigma"],
                                            base, base, rho),
                "T-WRONG": transport_family(lift, lift_pinv, sigma_bar, donor["sigma"],
                                            base, base, rho),
            }
            eigvals, eigvecs = np.linalg.eigh(entry["sigma"])
            perm = rng.permutation(9)
            top = eigvecs[:, -9:]
            gauge_sigma = entry["sigma"] + top @ (np.diag(eigvals[-9:][perm]
                                                          - eigvals[-9:])) @ top.T
            arms["GAUGE-NULL"] = transport_family(lift, lift_pinv, sigma_bar,
                                                  gauge_sigma, base, base, rho)
            query = entry["queries"][day]
            gen_assets = _blink_assets(query, labels, rng)
            drive_full = gen_assets["drive_of"](query)
            scaled_q = (query / entry["assets"]["eeg_scale"][:, None])
            window = 512
            starts = np.arange(0, query.shape[1] - window, window)
            energy = np.asarray([float(np.mean(drive_full[:, s:s + window] ** 2))
                                 for s in starts])
            low = starts[energy <= np.quantile(energy, .3)][:12]
            high = starts[energy >= np.quantile(energy, .7)]
            for i, s in enumerate(low):
                x = scaled_q[keep_index][:, s:s + window]
                donor_start = high[int(rng.integers(len(high)))]
                drive = drive_full[:, donor_start:donor_start + window]
                y = x + gen_assets["operator"][keep_index] @ drive
                for name, arm in arms.items():
                    cleaned = canonical_clean(arm, canon, sigma_bar_inv, y)
                    rrmse = float(np.linalg.norm(cleaned - x) / max(np.linalg.norm(x), 1e-12))
                    unit_rows.setdefault(name, {}).setdefault(subject, []).append(rrmse)
        per = {arm: {u: float(np.mean(v)) for u, v in units.items()}
               for arm, units in unit_rows.items()}
        units = sorted(per["T-POP"])
        gain = np.asarray([per["T-POP"][u] - per["T-MATCH"][u] for u in units])
        wrong = np.asarray([per["T-WRONG"][u] - per["T-POP"][u] for u in units])
        gauge = np.asarray([per["T-POP"][u] - per["GAUGE-NULL"][u] for u in units])
        draws = bootstrap_draws(gain)
        out[day] = {"TG-1": {**_stat(gain), "pass": bool(gain.mean() > 0
                                                         and _stat(gain)["bootstrap_low"] > 0),
                             "p_raw": float(np.mean(draws <= 0))},
                    "wrong_minus_pop": _stat(wrong),
                    "gauge_gain_pop_minus_gauge": _stat(gauge),
                    "arm_means": {arm: float(np.mean(list(v.values())))
                                  for arm, v in per.items()},
                    "units": len(units)}
    out["cross_day_gain_ratio"] = (out["day80"]["TG-1"]["mean"]
                                   / out["day7"]["TG-1"]["mean"]
                                   if abs(out["day7"]["TG-1"]["mean"]) > 1e-9 else None)
    out["montage"] = {"resolved": len(kept), "missing": missing}
    out["day200"] = "SEALED — never dereferenced"
    target = RESULT / "c2_brainid"
    target.mkdir(parents=True, exist_ok=True)
    (target / "decision.json").write_text(json.dumps(out, indent=2, sort_keys=True,
                                                     default=str) + "\n")
    print(json.dumps({day: {"TG1": out[day]["TG-1"]["pass"],
                            "gain": round(out[day]["TG-1"]["mean"], 5)}
                      for day in ("day7", "day80")}))


# --------------------------------------------------------------- D-1 kappa

def d1() -> None:
    """BCI2b MI kappa: CSP(+/-2)-LDA composed per the repo's own pieces;
    [T-MATCH,A0-MATCH]-cleaned vs [T-POP,A0-POP]-cleaned trials; descriptive."""
    import mne
    from scipy.linalg import eigh as scipy_eigh
    from scipy.signal import butter, sosfiltfilt
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import cohen_kappa_score

    from eeg_chart.prior_data import _resample_100
    from eeg_scad.cli.run_v43 import bootstrap_draws
    from eeg_scad.data.artifact_transfer_v41r import ridge_transfer
    from eeg_scad.data.v24_coordinate_contract import robust_center_scale

    canon = np.load(_canon_path())["u_canon"]
    cells, lift = _load_panel("bci2b")
    context = transport_context(cells, lift, canon, whitening=PANEL_TRANSPORT["bci2b"],
                                split_half_abstain=True)
    pop_ops, lams = _anchor_operators(cells)
    root = Path("/projects/EEG-foundation-model/BCI-IV")
    sos_mi = butter(4, (8.0, 30.0), btype="bandpass", fs=100.0, output="sos")
    rows = []
    for cell_id, entry in sorted(context["per_cell"].items()):
        cell = entry["cell"]
        subject = int(cell.subject[1:])
        arms = {"joint": (transport_family(context["lift"], context["lift_pinv"],
                                           context["sigma_bar"], cell.sigma_support,
                                           entry["rotation"], entry["base"], entry["rho"],
                                           whitening=PANEL_TRANSPORT["bci2b"]),
                          pop_ops[cell.cell] + lams[cell.cell]
                          * (cell.a_support - pop_ops[cell.cell])),
                "pop": (transport_family(context["lift"], context["lift_pinv"],
                                         context["sigma_bar"], None, entry["base"],
                                         entry["base"], 0.0), pop_ops[cell.cell])}
        trials, labels_all, sessions = [], [], []
        center = scale = eeg_scale = pca = None
        for run in (1, 2, 3):
            matches = sorted(root.glob(f"B{subject:02d}{run:02d}T.gdf"))
            if not matches:
                continue
            raw = mne.io.read_raw_gdf(matches[0], preload=True, verbose="error")
            picks = np.asarray(raw.get_data(), np.float64) * 1e6
            fs = float(raw.info["sfreq"])
            eeg = _resample_100(np.nan_to_num(picks[:3]), fs)
            eog = _resample_100(np.nan_to_num(picks[3:6]), fs)
            if center is None:
                _, eeg_scale = robust_center_scale(eeg)
                center, scale = robust_center_scale(eog)
                standardized = (eog - center[:, None]) / scale[:, None]
                pca = np.linalg.svd(np.cov(standardized), full_matrices=False)[0][:, :2].T
            drive = pca @ ((eog - center[:, None]) / scale[:, None])
            try:
                events, event_id = mne.events_from_annotations(raw, verbose="error")
            except Exception:
                continue
            codes = {v: k for k, v in event_id.items()}
            for sample, _, code in events:
                name = codes.get(code, "")
                if name not in ("769", "770"):
                    continue
                start = int(sample / fs * 100) + 50           # +0.5 s post cue
                if start + 200 > eeg.shape[1]:
                    continue
                trials.append((eeg[:, start:start + 200] / eeg_scale[:, None],
                               drive[:, start:start + 200]))
                labels_all.append(0 if name == "769" else 1)
                sessions.append(run)
        labels_arr = np.asarray(labels_all)
        sessions_arr = np.asarray(sessions)
        if len(trials) < 40 or len(set(labels_all)) < 2 or 3 not in sessions:
            rows.append({"subject": cell.subject, "status": "insufficient_trials"})
            continue
        for arm_name, (transport, operator) in arms.items():
            cleaned = []
            for y, drive in trials:
                pad = np.pad(y - operator @ drive, ((0, 0), (0, 312)), mode="reflect")
                out = canonical_clean(transport, canon, context["sigma_bar_inv"], pad)
                cleaned.append(sosfiltfilt(sos_mi, out[:, :200], axis=-1))
            cleaned = np.stack(cleaned)
            train = sessions_arr < 3
            test = sessions_arr == 3
            c0 = np.mean([t @ t.T for t in cleaned[train][labels_arr[train] == 0]], axis=0)
            c1 = np.mean([t @ t.T for t in cleaned[train][labels_arr[train] == 1]], axis=0)
            vals, vecs = scipy_eigh(c1, c0 + c1 + 1e-9 * np.eye(3))
            filters = np.concatenate((vecs[:, :1], vecs[:, -1:]), axis=1).T
            def features(stack):
                proj = np.einsum("fc,nct->nft", filters, stack)
                var = proj.var(axis=2)
                return np.log(var / var.sum(axis=1, keepdims=True))
            lda = LinearDiscriminantAnalysis().fit(features(cleaned[train]),
                                                   labels_arr[train])
            kappa = float(cohen_kappa_score(labels_arr[test],
                                            lda.predict(features(cleaned[test]))))
            rows.append({"subject": cell.subject, "arm": arm_name, "kappa": kappa,
                         "train_trials": int(train.sum()), "test_trials": int(test.sum())})
    frame = pd.DataFrame([r for r in rows if "kappa" in r])
    pivot = frame.pivot(index="subject", columns="arm", values="kappa")
    delta = (pivot["joint"] - pivot["pop"]).dropna()
    payload = {"descriptive_only": True,
               "kappa_joint_mean": float(pivot["joint"].mean()),
               "kappa_pop_mean": float(pivot["pop"].mean()),
               "joint_minus_pop": _stat(delta),
               "per_subject": pivot.round(4).to_dict("index"),
               "skipped": [r for r in rows if "status" in r], "sealed_reads": 0}
    target = RESULT / "d1_kappa"
    target.mkdir(parents=True, exist_ok=True)
    (target / "decision.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"joint": payload["kappa_joint_mean"], "pop": payload["kappa_pop_mean"],
                      "delta": round(payload["joint_minus_pop"]["mean"], 4)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="unit", required=True)
    for name in ("u1", "p1", "c2", "d1"):
        sub.add_parser(name)
    args = parser.parse_args()
    if args.unit == "u1":
        u1()
    elif args.unit == "p1":
        p1()
    elif args.unit == "c2":
        c2()
    elif args.unit == "d1":
        d1()


if __name__ == "__main__":
    main()
