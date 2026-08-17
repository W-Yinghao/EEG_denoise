#!/usr/bin/env python3
"""Integrity + scope verification for the published eegeyenet_min tree.

`publish` only checked the first 8 bytes of each .mat. This opens every file and
answers three questions that a header check cannot:

  1. Is the file complete and loadable? (a truncated v7 MAT still starts "MATLAB")
  2. Is it the MINIMALLY preprocessed pipeline? Automagic records the ocular-artifact
     removal steps as strings; min must read EOGRegression/iclabel/mara/rpca = "no".
     A max-pipeline file that landed here by mistake reads "yes" and is flagged.
  3. Does it carry the eye-tracking event layer the scope was chosen for?
     (L_saccade / L_fixation / L_blink with sac_* and fix_* fields)

The two paradigms use different MAT versions -- antisaccade is v7.3 (HDF5, h5py) and
dots is v5/v7 (scipy) -- so both loaders are needed. Sharded for Slurm; one JSON per
shard, aggregated by `summarize`.
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np

ROOT = Path("/projects/EEG-foundation-model/eegeyenet/eegeyenet_min")
REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "reports/eegeyenet_verify"
# Automagic fields that must read "no" for the minimally-preprocessed pipeline.
# These are exactly the steps that would delete the ocular artifact under study.
MIN_MUST_BE_NO = ("EOGRegression", "iclabel", "mara", "rpca")
EYE_EVENTS = ("L_saccade", "L_fixation", "L_blink", "R_saccade", "R_fixation", "R_blink")
EYE_FIELDS = ("sac_amplitude", "sac_vmax", "sac_startpos_x", "sac_endpos_x",
              "fix_avgpos_x", "fix_avgpupilsize")
# The two paradigms name the top-level struct differently: antisaccade exports `EEG`
# (plus an `automagic` provenance struct), dots exports `sEEG` with no automagic record.
EEG_VARS = ("EEG", "sEEG")
# Continuous gaze/pupil channels appended after Cz, when the export carries them.
GAZE_LABELS = ("TIME", "L-GAZE-X", "L-GAZE-Y", "L-AREA",
               "R-GAZE-X", "R-GAZE-Y", "R-AREA")


def _h5_text(handle, node) -> str:
    import h5py
    if isinstance(node, h5py.Reference):
        node = handle[node]
    values = np.asarray(node[()]).ravel()
    return "".join(chr(int(c)) for c in values if int(c) > 0)


def _probe_v73(path: Path) -> dict:
    import h5py
    with h5py.File(path, "r") as handle:
        name = next((v for v in EEG_VARS if v in handle), None)
        if name is None:
            raise KeyError(f"no {EEG_VARS} variable; found {[k for k in handle]}")
        eeg = handle[name]
        data = eeg["data"]
        # v7.3 stores EEGLAB data transposed: (samples, channels)
        samples, channels = data.shape
        info = {"mat_version": "7.3", "eeg_var": name, "channels": int(channels),
                "samples": int(samples), "srate": float(eeg["srate"][0, 0]),
                "seconds": float(eeg["xmax"][0, 0])}
        labels = [_h5_text(handle, r)
                  for r in np.asarray(eeg["chanlocs"]["labels"]).ravel()]
        info["chanlocs"] = len(labels)
        info["chanlocs_last"] = labels[-1] if labels else ""
        info["gaze_channels"] = [lab for lab in labels if lab in GAZE_LABELS]
        event = eeg["event"]
        info["event_fields"] = sorted(event.keys())
        types = [_h5_text(handle, r) for r in np.asarray(event["type"]).ravel()]
        info["n_events"] = len(types)
        info["eye_events"] = {name: types.count(name) for name in EYE_EVENTS
                              if types.count(name)}
        automagic = handle.get("automagic")
        info["automagic"] = ({key: _h5_text(handle, automagic[key]["performed"])
                              for key in MIN_MUST_BE_NO
                              if key in automagic and "performed" in automagic[key]}
                             if automagic is not None else {})
        # touch both ends of the data so a truncated HDF5 payload cannot pass silently
        info["finite_head"] = bool(np.isfinite(data[:64, :]).all())
        info["finite_tail"] = bool(np.isfinite(data[-64:, :]).all())
    return info


def _scipy_text(value) -> str:
    while isinstance(value, np.ndarray) and value.size >= 1 and value.dtype == object:
        value = value.ravel()[0]
    if isinstance(value, np.ndarray):
        value = value.ravel()
        return "".join(str(v) for v in value)
    return str(value)


def _probe_v5(path: Path) -> dict:
    from scipy.io import loadmat
    mat = loadmat(path, squeeze_me=False, struct_as_record=False)
    name = next((v for v in EEG_VARS if v in mat), None)
    if name is None:
        raise KeyError(f"no {EEG_VARS} variable; found "
                       f"{[k for k in mat if not k.startswith('__')]}")
    eeg = mat[name]
    while isinstance(eeg, np.ndarray):
        eeg = eeg.ravel()[0]
    data = np.asarray(eeg.data)
    # v5 stores EEGLAB data as (channels, samples)
    channels, samples = data.shape
    info = {"mat_version": "5", "eeg_var": name, "channels": int(channels),
            "samples": int(samples), "srate": float(np.asarray(eeg.srate).ravel()[0]),
            "seconds": float(np.asarray(eeg.xmax).ravel()[0])}
    chanlocs = np.asarray(eeg.chanlocs).ravel()
    labels = [_scipy_text(c.labels) for c in chanlocs]
    info["chanlocs"] = len(labels)
    info["chanlocs_last"] = labels[-1] if labels else ""
    info["gaze_channels"] = [lab for lab in labels if lab in GAZE_LABELS]
    events = np.asarray(eeg.event).ravel()
    info["n_events"] = int(events.size)
    info["event_fields"] = sorted(events[0]._fieldnames) if events.size else []
    types = [_scipy_text(e.type) for e in events]
    info["eye_events"] = {name: types.count(name) for name in EYE_EVENTS
                          if types.count(name)}
    automagic = mat.get("automagic")
    found = {}
    if automagic is not None:
        while isinstance(automagic, np.ndarray):
            automagic = automagic.ravel()[0]
        for key in MIN_MUST_BE_NO:
            step = getattr(automagic, key, None)
            if step is not None:
                while isinstance(step, np.ndarray) and step.size == 1:
                    step = step.ravel()[0]
                performed = getattr(step, "performed", None)
                if performed is not None:
                    found[key] = _scipy_text(performed)
    info["automagic"] = found
    info["finite_head"] = bool(np.isfinite(data[:, :64]).all())
    info["finite_tail"] = bool(np.isfinite(data[:, -64:]).all())
    return info


def probe(path: Path) -> dict:
    row = {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size}
    try:
        row.update(_probe_v73(path) if _is_v73(path) else _probe_v5(path))
    except Exception as error:                          # noqa: BLE001 - reason-coded
        row["error"] = f"{type(error).__name__}: {error}"
        row["traceback"] = traceback.format_exc(limit=3)
        return row
    # --- verdicts
    bad_steps = {k: v for k, v in row["automagic"].items() if v.strip().lower() != "no"}
    row["max_pipeline_steps"] = bad_steps
    # dots exports carry no automagic struct, so absence is "unrecorded", not a failure;
    # only an explicit "yes" on an artifact-removal step is contamination.
    row["pipeline"] = ("contaminated" if bad_steps else
                       "min_confirmed" if row["automagic"] else "no_record")
    row["has_eye_layer"] = bool(row["eye_events"]) and all(
        field in row["event_fields"] for field in EYE_FIELDS)
    row["loadable"] = row["finite_head"] and row["finite_tail"]
    return row


def _is_v73(path: Path) -> bool:
    """MATLAB v7.3 is HDF5 behind a 512-byte userblock, so the magic sits at 512."""
    with path.open("rb") as handle:
        head = handle.read(600)
    return b"HDF5 schema" in head[:128] or head[512:516] == b"\x89HDF"


def run(shard: int, shards: int) -> None:
    files = sorted(p for p in ROOT.rglob("*.mat"))
    mine = [p for index, p in enumerate(files) if index % shards == shard]
    rows = []
    for path in mine:
        row = probe(path)
        rows.append(row)
        flag = ("ERROR" if "error" in row else
                "MAXPIPE" if row["pipeline"] == "contaminated" else
                "NO-EYE" if not row["has_eye_layer"] else
                "TRUNC" if not row["loadable"] else "ok")
        print(f"  {flag:8s} {row['path']:52s} "
              f"{row.get('channels','?')}ch x {row.get('samples','?')} "
              f"@{row.get('srate','?')}Hz  events={row.get('n_events','?')}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"shard_{shard}.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"shard": shard, "files": len(rows),
                      "errors": sum(1 for r in rows if "error" in r)}))


def summarize() -> None:
    rows = [row for path in sorted(OUT.glob("shard_*.json"))
            for row in json.loads(path.read_text())]
    errors = [r for r in rows if "error" in r]
    clean = [r for r in rows if "error" not in r]
    maxpipe = [r for r in clean if r["pipeline"] == "contaminated"]
    no_eye = [r for r in clean if not r["has_eye_layer"]]
    truncated = [r for r in clean if not r["loadable"]]
    by_group: dict[str, dict] = {}
    for row in clean:
        group = row["path"].split("/")[0]
        entry = by_group.setdefault(group, {
            "files": 0, "seconds": 0.0, "channels": set(), "srate": set(), "mat": set(),
            "eye": 0, "gaze": 0, "pipeline": set(), "blinks": 0, "saccades": 0})
        entry["files"] += 1
        entry["seconds"] += row["seconds"]
        entry["channels"].add(row["channels"])
        entry["srate"].add(row["srate"])
        entry["mat"].add(f"{row['mat_version']}/{row['eeg_var']}")
        entry["eye"] += int(row["has_eye_layer"])
        entry["gaze"] += int(bool(row["gaze_channels"]))
        entry["pipeline"].add(row["pipeline"])
        entry["blinks"] += sum(v for k, v in row["eye_events"].items() if "blink" in k)
        entry["saccades"] += sum(v for k, v in row["eye_events"].items()
                                 if "saccade" in k)
    summary = {
        "files_checked": len(rows), "load_errors": len(errors),
        "max_pipeline_contamination": len(maxpipe), "missing_eye_layer": len(no_eye),
        "truncated": len(truncated),
        "verdict": ("PASS" if not errors and not maxpipe and not truncated else "FAIL"),
        "groups": {g: {"files": e["files"], "hours": round(e["seconds"] / 3600, 2),
                       "channels": sorted(e["channels"]), "srate": sorted(e["srate"]),
                       "mat_versions": sorted(e["mat"]),
                       "pipeline": sorted(e["pipeline"]),
                       "files_with_eye_layer": e["eye"],
                       "files_with_continuous_gaze": e["gaze"],
                       "total_blink_events": e["blinks"],
                       "total_saccade_events": e["saccades"]}
                   for g, e in sorted(by_group.items())},
        "error_files": [{"path": r["path"], "error": r["error"]} for r in errors],
        "max_pipeline_files": [{"path": r["path"], "steps": r["max_pipeline_steps"]}
                               for r in maxpipe],
        "no_eye_layer_files": [r["path"] for r in no_eye],
        "truncated_files": [r["path"] for r in truncated],
    }
    target = REPO / "reports/eegeyenet_min_qc.json"
    target.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--shard", type=int, required=True)
    runner.add_argument("--shards", type=int, required=True)
    sub.add_parser("summarize")
    args = parser.parse_args()
    if args.mode == "run":
        run(args.shard, args.shards)
    else:
        summarize()


if __name__ == "__main__":
    main()
