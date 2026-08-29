#!/usr/bin/env python3
"""PAPER-FINAL D-wave — downstream utility on the ambulatory panel's native BCI tasks.

Design + frozen interpretation grid: DOWNSTREAM_UTILITY_WAVE_DESIGN.md (committed
before any compute).  Tasks: D1 = SSVEP 3-class CCA decoding (training-free,
closed-form), D2 = ERP target/nontarget shrinkage-LDA (within-participant CV) +
ERP-preservation endpoints.  Trials with onset < 120 s (the calibration prefix)
are excluded.  Diffusion arms use the frozen seed-20261201 fold checkpoints.

Modes:
  probe               CPU gate on one cell (event units, SSVEP peak frequencies,
                      CCA-on-RAW accuracy); freezes frequencies for the wave.
  d1 --fold F         GPU: denoise SSVEP trial windows (arms NO_A0/MATCH/POP).
  d2 --fold F         GPU: denoise non-overlapping 512 tiles from 120 s on.
  cpu-arms --fold F   CPU: ICA/ASR/SGEYESUB cleaned records -> same windows/tiles.
  decode              CPU: CCA + LDA + ERP preservation on all arms; aggregates,
                      contamination-tertile stratification, decision JSON+arrays.
Add --heldout to d1/d2/cpu-arms/decode for the sealed-8 single pass (only after
the dev decision is banked).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pf_common import ARRAYS, OUT, SEED, load_model, stat

DWAVE = OUT / "dwave"
DEN = DWAVE / "denoised"
BIDS = Path("/projects/EEG-foundation-model/mobile_bci")
MERGED_ROOT = Path("/projects/EEG-foundation-model/derived/denoiseNet/flagship_m35/sealed_root")
SEALED = ("sub-01", "sub-04", "sub-08", "sub-10", "sub-13", "sub-16", "sub-20", "sub-22")
CALIB = 12000
WINDOW = 512
RATE = 100.0
D1_NOISE = 820000
D2_NOISE = 830000
DIFF_ARMS = ("NO_A0", "MATCH", "POP")
CPU_ARMS = ("ICA", "ASR", "SGEYESUB")
EPOCH_PRE, EPOCH_POST = 20, 80  # samples around ERP onset (-0.2..+0.8 s)


# ----------------------------------------------------------------- utilities

def load_events(sub: str, ses: str, task: str):
    path = BIDS / sub / ses / "eeg" / f"{sub}_{ses}_task-{task}_events.tsv"
    rows = []
    for line in path.read_text().strip().splitlines()[1:]:
        onset, duration, value = line.split("\t")
        rows.append({"onset": int(float(onset)), "duration": float(duration),
                     "value": int(float(value))})
    return rows


def cca_corr(x: np.ndarray, y: np.ndarray) -> float:
    qx, _ = np.linalg.qr(x - x.mean(0))
    qy, _ = np.linalg.qr(y - y.mean(0))
    return float(np.linalg.svd(qx.T @ qy, compute_uv=False)[0])


def cca_reference(freq: float, n: int = WINDOW) -> np.ndarray:
    t = np.arange(n) / RATE
    return np.stack([np.sin(2 * np.pi * freq * t), np.cos(2 * np.pi * freq * t),
                     np.sin(4 * np.pi * freq * t), np.cos(4 * np.pi * freq * t)], 1)


def cca_predict(window_cxt: np.ndarray, freqs: dict, channels: np.ndarray) -> int:
    x = window_cxt[channels].T  # T x C
    scores = {value: cca_corr(x, cca_reference(freq)) for value, freq in freqs.items()}
    return max(scores, key=scores.get)


def occipital_indices(eeg_names) -> np.ndarray:
    names = [str(n) for n in eeg_names]
    picks = [i for i, n in enumerate(names)
             if n.upper().startswith(("O", "PO", "P")) and not n[0] in "LR"]
    return np.asarray(picks, dtype=int)


def fold_contexts(heldout: bool):
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.cli.run_v44 import _gated_assets
    from eeg_scad.data.artifact_transfer_v41r import TransferRegistry
    from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry

    data, folds, _ = configs()
    if heldout:
        data = dict(data)
        data["v19_derived_root"] = str(MERGED_ROOT)
        folds = [{"fold": 99, "train": list(data["participants"]), "validation": [],
                  "test": list(SEALED)}]
    out = []
    for fold in folds:
        registry30 = TransferRegistry(data, fold, 30, .05)
        eb120 = EBTransferRegistry(data, fold, registry30, 120)
        out.append((data, fold, registry30, eb120, _gated_assets(registry30, eb120)))
    return out


def cell_signals(registry30, key):
    from eeg_scad.data.artifact_transfer_v41r import bipolar_eog
    eeg, eye, names = registry30._load(*key)
    eog = bipolar_eog(eye, names)
    cell = registry30.cells[key]
    length = min(eeg.shape[1], eog.shape[1])
    y = eeg[:, :length] / registry30.eeg_scale[:, None]
    drive = (eog[:, :length] - cell.eog_center[:, None]) / cell.eog_scale[:, None]
    return y, drive


def d1_trials(events, length):
    return [e for e in events
            if e["onset"] >= CALIB and e["onset"] + WINDOW <= length]


def d2_tiles(length):
    return list(range(CALIB, length - WINDOW + 1, WINDOW))


# --------------------------------------------------------------------- probe

def probe() -> None:
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.data.artifact_transfer_v41r import TransferRegistry

    data, folds, _ = configs()
    fold = folds[0]
    registry30 = TransferRegistry(data, fold, 30, .05)
    key = ("sub-02", "ses-02", "SSVEP")
    y, drive = cell_signals(registry30, key)
    eeg_names = np.load(Path(data["v19_derived_root"]) / "prepared" / key[0]
                        / f"{key[1]}_{key[2]}.npz")["eeg_names"]
    occ = occipital_indices(eeg_names)
    events = load_events(*key)
    trials = d1_trials(events, y.shape[1])
    values = sorted({e["value"] for e in events})

    # per-class trial spectrum vs inter-trial baseline (checks onset units too)
    freqs_axis = np.fft.rfftfreq(WINDOW, 1 / RATE)
    band = (freqs_axis >= 4) & (freqs_axis <= 20)
    class_freq = {}
    trial_power, base_power = [], []
    for value in values:
        spectra = []
        for e in [t for t in trials if t["value"] == value]:
            seg = y[occ][:, e["onset"]:e["onset"] + WINDOW]
            spectra.append(np.abs(np.fft.rfft(seg, axis=1)) ** 2)
        mean_spec = np.mean(spectra, axis=(0, 1))
        class_freq[value] = float(freqs_axis[band][np.argmax(mean_spec[band])])
        trial_power.append(float(mean_spec[band].max()))
    for e in trials[:20]:
        start = e["onset"] + WINDOW + 100
        if start + WINDOW <= y.shape[1]:
            seg = y[occ][:, start:start + WINDOW]
            base_power.append(float(np.mean(
                np.abs(np.fft.rfft(seg, axis=1)) ** 2, axis=0)[band].max()))
    contrast = float(np.mean(trial_power) / max(np.mean(base_power), 1e-12))

    correct = sum(cca_predict(y[:, e["onset"]:e["onset"] + WINDOW], class_freq, occ)
                  == e["value"] for e in trials)
    acc = correct / len(trials)

    erp_events = load_events(key[0], key[1], "ERP")
    counts = {v: sum(e["value"] == v for e in erp_events) for v in
              sorted({e["value"] for e in erp_events})}
    distinct = len(set(round(f, 1) for f in class_freq.values())) == len(values)
    verdict = bool(acc >= 0.5 and distinct and contrast > 1.2)
    payload = {"cell": "|".join(key), "n_trials_post_calibration": len(trials),
               "occipital_channels": [str(eeg_names[i]) for i in occ],
               "class_frequencies_hz": class_freq,
               "trial_vs_intertrial_peak_power_ratio": contrast,
               "cca_raw_accuracy": acc, "chance": 1 / 3,
               "erp_class_counts": counts,
               "gate_pass": verdict}
    DWAVE.mkdir(parents=True, exist_ok=True)
    (DWAVE / "probe.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=1, sort_keys=True))
    if not verdict:
        raise SystemExit("D-wave probe gate FAILED — do not launch the fleet")


# ------------------------------------------------------------------ GPU legs

def run_diffusion(task: str, only_fold, heldout: bool) -> None:
    import torch
    from eeg_scad.cli.run_v44 import sample_bank_eog
    from eeg_scad.models.calib_saddpm_cond_v42r import LinearX0Schedule

    device = torch.device("cuda")
    schedule = LinearX0Schedule().to(device)
    DEN.mkdir(parents=True, exist_ok=True)
    for data, fold, registry30, eb120, assets in fold_contexts(heldout):
        fold_id = fold["fold"]
        if only_fold is not None and fold_id != only_fold:
            continue
        if heldout:
            models = [load_model(f, device, SEED) for f in range(5)]
        else:
            models = [load_model(fold_id, device, SEED)]
        for key in sorted(assets):
            if key[0] not in fold["test"] or key[2] != ("SSVEP" if task == "d1" else "ERP"):
                continue
            out_path = DEN / f"{task}_{'|'.join(key)}.npz"
            if out_path.is_file():
                continue
            y, drive = cell_signals(registry30, key)
            if task == "d1":
                events = load_events(*key)
                trials = d1_trials(events, y.shape[1])
                starts = [e["onset"] for e in trials]
                labels = [e["value"] for e in trials]
            else:
                starts = d2_tiles(y.shape[1])
                labels = []
            if not starts:
                continue
            y_stack = np.stack([y[:, s:s + WINDOW] for s in starts]).astype(np.float32)
            d_stack = [drive[:, s:s + WINDOW] for s in starts]
            asset = assets[key]
            hard = bool(eb120.cells[key].hard_gate)
            noise_seed = (D1_NOISE if task == "d1" else D2_NOISE) \
                + fold_id * 100 + SEED % 100
            store = {"starts": np.asarray(starts), "labels": np.asarray(labels),
                     "veog_rms": np.asarray([float(np.sqrt(np.mean(d[0] ** 2)))
                                             for d in d_stack]),
                     "RAW": y_stack,
                     "LINEAR": np.stack([y_stack[i] - (asset["C_gated"] @ d_stack[i])
                                         for i in range(len(starts))]).astype(np.float32),
                     "hard_gate": np.asarray(int(hard))}
            for arm in DIFF_ARMS:
                a0 = []
                for d in d_stack:
                    if arm == "NO_A0" or (arm == "MATCH" and hard):
                        a0.append(np.zeros((46, WINDOW)))
                    elif arm == "MATCH":
                        a0.append(asset["C_gated"] @ d)
                    else:
                        a0.append(asset["C0"] @ d)
                sig = asset["sig_pop"] if arm == "POP" else asset["sig_gated"]
                sig_stack = np.stack([sig] * len(starts))
                output = np.mean([sample_bank_eog(model, schedule, y_stack,
                                                  np.stack(a0), sig_stack, device,
                                                  noise_seed)
                                  for model in models], axis=0)
                if not np.isfinite(output).all():
                    raise FloatingPointError(f"nonfinite D-wave output {key} {arm}")
                store[arm] = output.astype(np.float32)
            np.savez_compressed(out_path, **store)
            print(json.dumps({"task": task, "cell": "|".join(key),
                              "windows": len(starts)}), flush=True)


# ------------------------------------------------------------- CPU arms leg

def cpu_arms(only_fold, heldout: bool) -> None:
    import mne
    mne.set_log_level("ERROR")
    try:
        import asrpy
    except ImportError:
        asrpy = None
    DEN.mkdir(parents=True, exist_ok=True)
    eeg_names46 = [f"E{i:02d}" for i in range(46)]
    info = mne.create_info(eeg_names46 + ["VEOG", "HEOG"], RATE,
                           ["eeg"] * 46 + ["eog", "eog"])
    info_eeg = mne.create_info(eeg_names46, RATE, ["eeg"] * 46)
    for data, fold, registry30, eb120, assets in fold_contexts(heldout):
        fold_id = fold["fold"]
        if only_fold is not None and fold_id != only_fold:
            continue
        for key in sorted(assets):
            if key[0] not in fold["test"]:
                continue
            out_path = DEN / f"cpuarms_{'|'.join(key)}.npz"
            if out_path.is_file():
                continue
            y, drive = cell_signals(registry30, key)
            cleaned = {}
            # ICA (fit on 1-Hz highpassed copy of the whole record, EOG-corr excl.)
            raw = mne.io.RawArray(np.vstack([y, drive]), info, verbose="ERROR")
            raw_filt = raw.copy().filter(l_freq=1.0, h_freq=None, verbose="ERROR")
            ica = mne.preprocessing.ICA(n_components=0.999999, method="fastica",
                                        random_state=SEED, max_iter=1000)
            ica.fit(raw_filt, picks="eeg")
            bads, _ = ica.find_bads_eog(raw_filt)
            ica.exclude = sorted(set(bads))
            cleaned["ICA"] = ica.apply(raw.copy()).get_data(picks="eeg")
            # ASR (calibrated on the same 120-s prefix). asrpy has a data-dependent
            # off-by-one reshape bug in calibration; guard per cell and record.
            if asrpy is not None:
                try:
                    cleaner = asrpy.ASR(sfreq=RATE)
                    cleaner.fit(mne.io.RawArray(y[:, :CALIB], info_eeg,
                                                verbose="ERROR"))
                    cleaned["ASR"] = cleaner.transform(
                        mne.io.RawArray(y, info_eeg, verbose="ERROR")).get_data()
                except Exception as error:
                    print(json.dumps({"asr_failed": "|".join(key),
                                      "error": str(error)[:120]}), flush=True)
            # SGEYESUB-style rank-2 projection from the 120-s raw ridge fit
            c_full = eb120.operator(*key, "RAW")
            q, _ = np.linalg.qr(c_full)
            cleaned["SGEYESUB"] = y - (q @ q.T) @ y
            store = {}
            if key[2] == "SSVEP":
                trials = d1_trials(load_events(*key), y.shape[1])
                starts = [e["onset"] for e in trials]
            else:
                starts = d2_tiles(y.shape[1])
            for arm, rec in cleaned.items():
                store[arm] = np.stack([rec[:, s:s + WINDOW] for s in starts]) \
                    .astype(np.float32)
            store["starts"] = np.asarray(starts)
            np.savez_compressed(out_path, **store)
            print(json.dumps({"cpuarms": "|".join(key), "windows": len(starts),
                              "ica_excluded": len(ica.exclude),
                              "asr": bool(asrpy)}), flush=True)


# ---------------------------------------------------------------- decode leg

ALL_ARMS = ("RAW", "LINEAR") + DIFF_ARMS + CPU_ARMS


def _lda_auc(epochs: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    base = epochs - epochs[:, :, :EPOCH_PRE].mean(axis=2, keepdims=True)
    feats = base[:, :, ::5].reshape(len(base), -1)
    y = (labels == 2).astype(int)
    scores = np.zeros(len(y), float)
    for train, test in StratifiedKFold(5).split(feats, y):
        model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        model.fit(feats[train], y[train])
        scores[test] = model.decision_function(feats[test])
    return float(roc_auc_score(y, scores))


def decode(heldout: bool) -> None:
    probe_payload = json.loads((DWAVE / "probe.json").read_text())
    freqs = {int(k): float(v) for k, v in
             probe_payload["class_frequencies_hz"].items()}
    tag = "heldout" if heldout else "dev"

    d1_rows, d2_rows, erp_rows = [], [], []
    for data, fold, registry30, eb120, assets in fold_contexts(heldout):
        eeg_names = np.load(Path(data["v19_derived_root"]) / "prepared"
                            / fold["test"][0]
                            / f"{data['sessions'][0]}_{data['tasks'][0]}.npz")["eeg_names"]
        occ = occipital_indices(eeg_names)
        for key in sorted(assets):
            if key[0] not in fold["test"]:
                continue
            name = "|".join(key)
            cpu_path = DEN / f"cpuarms_{name}.npz"
            if key[2] == "SSVEP":
                diff_path = DEN / f"d1_{name}.npz"
                if not diff_path.is_file():
                    continue
                diff = np.load(diff_path, allow_pickle=False)
                cpu = np.load(cpu_path, allow_pickle=False) if cpu_path.is_file() else None
                labels = diff["labels"]
                veog = diff["veog_rms"]
                for arm in ALL_ARMS:
                    if arm in ("ICA", "ASR", "SGEYESUB"):
                        if cpu is None or arm not in cpu.files:
                            continue
                        windows = cpu[arm]
                    else:
                        windows = diff[arm]
                    correct = np.asarray(
                        [cca_predict(windows[i], freqs, occ) == labels[i]
                         for i in range(len(labels))])
                    for i in range(len(labels)):
                        d1_rows.append({"participant": key[0], "cell": name,
                                        "arm": arm, "correct": int(correct[i]),
                                        "veog_rms": float(veog[i])})
            else:
                diff_path = DEN / f"d2_{name}.npz"
                if not diff_path.is_file():
                    continue
                diff = np.load(diff_path, allow_pickle=False)
                cpu = np.load(cpu_path, allow_pickle=False) if cpu_path.is_file() else None
                tile_starts = list(diff["starts"])
                tile_of = {s: i for i, s in enumerate(tile_starts)}
                events = load_events(*key)
                epochs_meta = []
                for e in events:
                    lo, hi = e["onset"] - EPOCH_PRE, e["onset"] + EPOCH_POST
                    if lo < CALIB:
                        continue
                    tlo = (lo - CALIB) // WINDOW
                    if tlo != (hi - 1 - CALIB) // WINDOW:
                        continue  # epoch straddles a tile boundary — dropped
                    start = CALIB + tlo * WINDOW
                    if start not in tile_of:
                        continue
                    epochs_meta.append((tile_of[start], lo - start, e["value"],
                                        e["onset"]))
                if len(epochs_meta) < 40:
                    continue
                labels = np.asarray([m[2] for m in epochs_meta])
                # trial contamination from the RAW record drive
                _, drive = cell_signals(registry30, key)
                veog = np.asarray([float(np.sqrt(np.mean(
                    drive[0, m[3] - EPOCH_PRE:m[3] + EPOCH_POST] ** 2)))
                    for m in epochs_meta])
                low_mask = veog <= np.quantile(veog, 1 / 3)
                raw_epochs = None
                for arm in ALL_ARMS:
                    if arm in ("ICA", "ASR", "SGEYESUB"):
                        if cpu is None or arm not in cpu.files:
                            continue
                        tiles = cpu[arm]
                    else:
                        tiles = diff[arm]
                    epochs = np.stack([tiles[t][:, o:o + EPOCH_PRE + EPOCH_POST]
                                       for t, o, _, _ in epochs_meta])
                    if arm == "RAW":
                        raw_epochs = epochs
                    auc = _lda_auc(epochs, labels)
                    d2_rows.append({"participant": key[0], "cell": name,
                                    "arm": arm, "auc": auc,
                                    "n_epochs": len(labels)})
                    # ERP preservation vs RAW low-contamination average
                    pres = []
                    for value in np.unique(labels):
                        sel = (labels == value) & low_mask
                        if sel.sum() < 5 or raw_epochs is None:
                            continue
                        ref = raw_epochs[sel].mean(0)
                        avg = epochs[sel].mean(0)
                        num = float(np.sum((ref - ref.mean()) * (avg - avg.mean())))
                        den = float(np.linalg.norm(ref - ref.mean())
                                    * np.linalg.norm(avg - avg.mean()))
                        pres.append(num / max(den, 1e-12))
                    if pres:
                        erp_rows.append({"participant": key[0], "cell": name,
                                         "arm": arm,
                                         "erp_preservation_corr": float(np.mean(pres))})
            print(json.dumps({"decoded": name}), flush=True)

    def participant_table(rows, metric, arms):
        table = {}
        for arm in arms:
            per: dict[str, list[float]] = {}
            for row in rows:
                if row["arm"] == arm:
                    per.setdefault(row["participant"], []).append(float(row[metric]))
            table[arm] = {p: float(np.mean(v)) for p, v in sorted(per.items())}
        return table

    d1_acc = participant_table(d1_rows, "correct", ALL_ARMS)
    d2_auc = participant_table(d2_rows, "auc", ALL_ARMS)
    erp_pres = participant_table(erp_rows, "erp_preservation_corr", ALL_ARMS)

    def contrasts(table, reference):
        out = {}
        ref = table.get(reference, {})
        for arm, per in table.items():
            if arm == reference:
                continue
            common = [p for p in per if p in ref]
            if common:
                out[f"{arm}_minus_{reference}"] = stat([per[p] - ref[p]
                                                        for p in common])
        return out

    # contamination-tertile stratification for D1 (within-participant tertiles on RAW)
    strata = {}
    tertile_edges = {}
    for row in d1_rows:
        if row["arm"] == "RAW":
            tertile_edges.setdefault(row["participant"], []).append(row["veog_rms"])
    tertile_edges = {p: (np.quantile(v, 1 / 3), np.quantile(v, 2 / 3))
                     for p, v in tertile_edges.items()}
    for level, test in (("low", lambda v, e: v <= e[0]),
                        ("mid", lambda v, e: e[0] < v <= e[1]),
                        ("high", lambda v, e: v > e[1])):
        sub = [r for r in d1_rows if test(r["veog_rms"],
                                          tertile_edges[r["participant"]])]
        tab = participant_table(sub, "correct", ("RAW", "MATCH"))
        common = [p for p in tab["MATCH"] if p in tab["RAW"]]
        strata[level] = stat([tab["MATCH"][p] - tab["RAW"][p] for p in common])

    decision = {
        "design": "DOWNSTREAM_UTILITY_WAVE_DESIGN.md (frozen pre-compute)",
        "frozen_frequencies_hz": freqs,
        "d1_ssvep_accuracy": {arm: float(np.mean(list(per.values())))
                              for arm, per in d1_acc.items() if per},
        "d1_contrasts": {"vs_RAW": contrasts(d1_acc, "RAW"),
                         "vs_NO_A0": contrasts(d1_acc, "NO_A0")},
        "d1_match_minus_raw_by_contamination_tertile": strata,
        "d2_erp_auc": {arm: float(np.mean(list(per.values())))
                       for arm, per in d2_auc.items() if per},
        "d2_contrasts": {"vs_RAW": contrasts(d2_auc, "RAW"),
                         "vs_NO_A0": contrasts(d2_auc, "NO_A0")},
        "erp_preservation": {arm: float(np.mean(list(per.values())))
                             for arm, per in erp_pres.items() if per},
    }
    (DWAVE / f"dwave_decision_{tag}.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(ARRAYS / f"dwave_{tag}.npz",
                        decision=np.asarray(json.dumps(decision)))
    print(json.dumps({"d1": decision["d1_ssvep_accuracy"],
                      "d2": decision["d2_erp_auc"],
                      "strata": {k: v["mean"] for k, v in strata.items()}},
                     indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["probe", "d1", "d2", "cpu-arms", "decode"])
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--heldout", action="store_true")
    args = parser.parse_args()
    if args.mode == "probe":
        probe()
    elif args.mode in ("d1", "d2"):
        run_diffusion(args.mode, args.fold, args.heldout)
    elif args.mode == "cpu-arms":
        cpu_arms(args.fold, args.heldout)
    else:
        decode(args.heldout)


if __name__ == "__main__":
    main()
