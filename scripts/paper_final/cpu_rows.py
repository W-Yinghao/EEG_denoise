#!/usr/bin/env python3
"""PAPER-FINAL CPU reference rows — literature anchor rows on the same protocol.

sgeyesub : calibrated eye-subspace subtraction (Kobler 2020 SGEYESUB style):
           the rank-2 eye subspace is the column space of the same 120-s
           calibration ridge fit (raw, unshrunk); correction projects that
           subspace out of the observation (no runtime EOG needed).
           Both endpoints: paired RRMSE on the standard episodes (closed form,
           same folds/episodes) and the natural windows.
ica      : mne ICA (fastica) fit on each continuous test record (1 Hz high-pass
           copy), components removed by EOG correlation (find_bads_eog on the
           bipolar VEOG/HEOG), applied to the unfiltered record; evaluated on the
           standard natural windows.  The 5.12-s paired episodes are too short for
           a stable 46-channel decomposition, so ICA is reported on continuous
           data only (that constraint is itself the reportable result).
asr      : ASR (asrpy) calibrated per cell on the same 120-s calibration prefix,
           applied to the paired episodes and the natural windows.  If asrpy is
           not importable the mode reports that in one sentence and exits.
aggregate: collects all rows into cpu_reference_rows.json.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from pf_common import ARRAYS, OUT, SEED, participant_means, stat

UNIT_DIR = OUT / "cpu_rows_units"


def _fold_context(fold, data):
    from eeg_scad.cli.run_v44 import _gated_assets
    from eeg_scad.data.artifact_transfer_v41r import (TransferEpisodeSampler,
                                                      TransferRegistry)
    from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry

    registry30 = TransferRegistry(data, fold, 30, .05)
    eb120 = EBTransferRegistry(data, fold, registry30, 120)
    assets = _gated_assets(registry30, eb120)
    sampler = TransferEpisodeSampler(data, fold, "test", SEED + 3, registry30)
    return registry30, eb120, assets, sampler


def sgeyesub() -> None:
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.cli.run_v44 import _bank_drives, _natural_metrics, _natural_windows
    from eeg_scad.evaluation.paired_metrics import paired_metrics

    data, folds, _ = configs()
    UNIT_DIR.mkdir(parents=True, exist_ok=True)
    rows, natural_rows = [], []
    for fold in folds:
        registry30, eb120, assets, sampler = _fold_context(fold, data)
        bank = sampler.sample_balanced(8)
        projectors = {}
        for key in assets:
            c_full = eb120.operator(*key, "RAW")
            q, _ = np.linalg.qr(c_full)
            projectors[key] = q @ q.T
        for clean, observed, artifact, meta in zip(bank["x"], bank["y"],
                                                   bank["artifact"], bank["meta"]):
            key = (meta["participant"], meta["session"], meta["task"])
            predicted = projectors[key] @ np.asarray(observed, np.float64)
            rows.append({"fold": fold["fold"], "participant": key[0],
                         "condition": "SGEYESUB_style",
                         "zero_artifact": meta["zero_artifact"],
                         **paired_metrics(clean, observed, artifact, predicted)})
        for key in assets:
            if key[0] not in fold["test"]:
                continue
            for start, y, drive in _natural_windows(registry30, data, key):
                estimate = projectors[key] @ y
                natural_rows.append({"fold": fold["fold"], "participant": key[0],
                                     "condition": "SGEYESUB_style", "start": start,
                                     **_natural_metrics(y, drive, estimate)})
        print(json.dumps({"fold": fold["fold"], "done": "sgeyesub"}), flush=True)
    (UNIT_DIR / "sgeyesub.json").write_text(json.dumps(
        {"rows": rows, "natural_rows": natural_rows}, indent=1, sort_keys=True) + "\n")


def ica() -> None:
    import mne
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.cli.run_v44 import _natural_metrics, _natural_windows, WINDOW
    from eeg_scad.data.artifact_transfer_v41r import bipolar_eog

    mne.set_log_level("ERROR")
    data, folds, _ = configs()
    UNIT_DIR.mkdir(parents=True, exist_ok=True)
    natural_rows = []
    for fold in folds:
        registry30, eb120, assets, _ = _fold_context(fold, data)
        for key in assets:
            if key[0] not in fold["test"]:
                continue
            eeg, eye, names = registry30._load(*key)
            eog = bipolar_eog(eye, names)
            length = min(eeg.shape[1], eog.shape[1])
            stacked = np.vstack([eeg[:, :length], eog[:, :length]])
            eeg_names = [f"E{i:02d}" for i in range(46)]
            info = mne.create_info(eeg_names + ["VEOG", "HEOG"], 100.0,
                                   ["eeg"] * 46 + ["eog", "eog"])
            raw = mne.io.RawArray(stacked, info, verbose="ERROR")
            raw_filt = raw.copy().filter(l_freq=1.0, h_freq=None, verbose="ERROR")
            decomposition = mne.preprocessing.ICA(n_components=0.999999,
                                                  method="fastica",
                                                  random_state=SEED, max_iter=1000)
            decomposition.fit(raw_filt, picks="eeg")
            bads, _ = decomposition.find_bads_eog(raw_filt)
            decomposition.exclude = sorted(set(bads))
            cleaned = decomposition.apply(raw.copy())
            cleaned_eeg = cleaned.get_data(picks="eeg")
            scale = registry30.eeg_scale[:, None]
            cell = registry30.cells[key]
            for start, y, drive in _natural_windows(registry30, data, key):
                out = cleaned_eeg[:, start:start + WINDOW] / scale
                estimate = y - out
                natural_rows.append({"fold": fold["fold"], "participant": key[0],
                                     "condition": "ICA_eog_corr", "start": start,
                                     "n_excluded": len(decomposition.exclude),
                                     **_natural_metrics(y, drive, estimate)})
            print(json.dumps({"cell": "|".join(key),
                              "excluded": len(decomposition.exclude)}), flush=True)
    (UNIT_DIR / "ica.json").write_text(json.dumps(
        {"rows": [], "natural_rows": natural_rows,
         "paired_note": "5.12-s episodes are too short for a stable 46-channel ICA; "
                        "ICA is evaluated on continuous natural records only"},
        indent=1, sort_keys=True) + "\n")


def asr() -> None:
    try:
        import asrpy
    except ImportError:
        (UNIT_DIR / "asr.json").write_text(json.dumps(
            {"rows": [], "natural_rows": [],
             "note": "asrpy is not installed in the icml environment and could not "
                     "be added; the ASR reference row is omitted"}) + "\n")
        print(json.dumps({"asr": "unavailable"}))
        return
    import mne
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.cli.run_v44 import _natural_metrics, _natural_windows, WINDOW
    from eeg_scad.evaluation.paired_metrics import paired_metrics

    mne.set_log_level("ERROR")
    data, folds, _ = configs()
    UNIT_DIR.mkdir(parents=True, exist_ok=True)
    rate = 100.0
    eeg_names = [f"E{i:02d}" for i in range(46)]
    info = mne.create_info(eeg_names, rate, ["eeg"] * 46)

    def to_raw(array):
        return mne.io.RawArray(np.asarray(array, np.float64), info, verbose="ERROR")

    rows, natural_rows = [], []
    for fold in folds:
        registry30, eb120, assets, sampler = _fold_context(fold, data)
        bank = sampler.sample_balanced(8)
        scale = registry30.eeg_scale[:, None]
        cleaners = {}
        for key in assets:
            if key[0] not in fold["test"]:
                continue
            eeg, _, _ = registry30._load(*key)
            calib = eeg[:, :12000] / scale
            try:  # asrpy has a data-dependent reshape bug in calibration
                cleaner = asrpy.ASR(sfreq=rate)
                cleaner.fit(to_raw(calib))
                cleaners[key] = cleaner
            except Exception as error:
                print(json.dumps({"asr_calibration_failed": "|".join(key),
                                  "error": str(error)[:120]}), flush=True)
        for clean, observed, artifact, meta in zip(bank["x"], bank["y"],
                                                   bank["artifact"], bank["meta"]):
            key = (meta["participant"], meta["session"], meta["task"])
            if key not in cleaners:
                continue
            out = cleaners[key].transform(to_raw(observed)).get_data()
            predicted = np.asarray(observed, np.float64) - out
            rows.append({"fold": fold["fold"], "participant": key[0],
                         "condition": "ASR", "zero_artifact": meta["zero_artifact"],
                         **paired_metrics(clean, observed, artifact, predicted)})
        for key in assets:
            if key[0] not in fold["test"] or key not in cleaners:
                continue
            for start, y, drive in _natural_windows(registry30, data, key):
                out = cleaners[key].transform(to_raw(y)).get_data()
                natural_rows.append({"fold": fold["fold"], "participant": key[0],
                                     "condition": "ASR", "start": start,
                                     **_natural_metrics(y, drive, y - out)})
        print(json.dumps({"fold": fold["fold"], "done": "asr"}), flush=True)
    (UNIT_DIR / "asr.json").write_text(json.dumps(
        {"rows": rows, "natural_rows": natural_rows}, indent=1, sort_keys=True) + "\n")


def aggregate() -> None:
    payloads = {}
    for name in ("sgeyesub", "ica", "asr"):
        path = UNIT_DIR / f"{name}.json"
        payloads[name] = json.loads(path.read_text()) if path.is_file() else None
    report = {}
    for name, payload in payloads.items():
        if payload is None:
            report[name] = {"status": "not run"}
            continue
        entry = {}
        for extra in ("note", "paired_note"):
            if payload.get(extra):
                entry[extra] = payload[extra]
        if payload["rows"]:
            condition = payload["rows"][0]["condition"]
            per = participant_means(payload["rows"], condition)
            entry["paired_rrmse"] = stat(list(per.values()))
        if payload["natural_rows"]:
            condition = payload["natural_rows"][0]["condition"]
            entry["natural"] = {}
            for metric in ("attenuation_db", "low_eog_observation_retention",
                           "coherence_reduction"):
                per = participant_means(payload["natural_rows"], condition, metric)
                entry["natural"][metric] = stat(list(per.values()))
        report[name] = entry
    (OUT / "cpu_reference_rows.json").write_text(json.dumps(report, indent=2,
                                                            sort_keys=True) + "\n")
    np.savez_compressed(ARRAYS / "cpu_reference_rows.npz",
                        report=np.asarray(json.dumps(report)))
    print(json.dumps({k: {kk: (vv["mean"] if isinstance(vv, dict) and "mean" in vv
                               else "...") for kk, vv in v.items()}
                      for k, v in report.items() if isinstance(v, dict)}, indent=1,
                     default=str))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["sgeyesub", "ica", "asr", "aggregate"])
    args = parser.parse_args()
    {"sgeyesub": sgeyesub, "ica": ica, "asr": asr, "aggregate": aggregate}[args.mode]()


if __name__ == "__main__":
    main()
