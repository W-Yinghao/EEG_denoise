#!/usr/bin/env python3
"""IRIS sealed-block freeze and quarantine fetch (EEGEyeNet antisaccade).

The dev slice is listing positions 1-30 of antisaccade_task_data/synchronised_min
(28 non-empty; AA2/AB8 empty upstream), already published. The SEALED block is
listing positions 31-90 of the same deterministic Drive listing order: enumerated
once, written to results/iris/sealed/sealed_freeze.json, and never re-chosen.
Empty upstream folders are recorded and counted, not replaced.

Freeze rule (charter section 5): no analysis reads any sealed byte until the
single preregistered opening with operator sign-off. `fetch` downloads into
/projects/EEG-foundation-model/eegeyenet/eegeyenet_sealed/, header-verifies the
MAT magic (acquisition integrity, not analysis), then chmod 000 the tree so an
accidental read fails loudly.

Modes: freeze (refuses to overwrite) | fetch | seal (chmod) | status.
"""
from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from eegeyenet_download import FOLDER_ID, SAFE, _subfolder, download_once, listing  # noqa: E402

FREEZE = REPO / "results/iris/sealed/sealed_freeze.json"
SEALED_ROOT = Path("/projects/EEG-foundation-model/eegeyenet/eegeyenet_sealed")
POS_LO, POS_HI = 31, 90                       # 1-indexed listing positions, inclusive
MAT_MAGIC = (b"MATLAB", b"\x89HDF")


def freeze() -> None:
    if FREEZE.exists():
        raise SystemExit(f"REFUSED: freeze already exists at {FREEZE} — the sealed "
                         "block is chosen once and never re-chosen.")
    top = {e["name"]: e for e in listing(FOLDER_ID)}
    anti = _subfolder(top["antisaccade_task_data"]["id"], "synchronised_min")
    folders = [e for e in listing(anti) if e["kind"] == "folder"]
    block = folders[POS_LO - 1:POS_HI]
    subjects = []
    for offset, folder in enumerate(block):
        if not SAFE.fullmatch(folder["name"]):
            raise ValueError(f"unsafe folder name {folder['name']!r}")
        files = [{"name": item["name"], "id": item["id"]}
                 for item in listing(folder["id"])
                 if item["kind"] == "file" and item["name"].lower().endswith(".mat")]
        subjects.append({"position": POS_LO + offset, "subject": folder["name"],
                         "folder_id": folder["id"], "files": files,
                         "empty_upstream": not files})
        time.sleep(0.3)
    non_empty = [s for s in subjects if not s["empty_upstream"]]
    payload = {
        "asset": "EEGEyeNet antisaccade synchronised_min — IRIS sealed block",
        "rule": ("listing positions 31-90 of the deterministic Drive listing order; "
                 "empty upstream folders recorded and counted, never replaced; "
                 "no analysis contact until the single preregistered opening with "
                 "operator sign-off; dev = positions 1-30 (already published)"),
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "positions": [POS_LO, POS_HI],
        "n_folders": len(subjects), "n_subjects_non_empty": len(non_empty),
        "cross_paradigm_caveat": (
            "dots (EP*) and antisaccade (A*/B*) subject IDs are not mappable from "
            "released metadata; sealed-fight support/prior/atlas assets are therefore "
            "built from antisaccade-dev (+ non-EEGEyeNet panels) only"),
        "subjects": subjects,
    }
    FREEZE.parent.mkdir(parents=True, exist_ok=True)
    FREEZE.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"frozen": len(subjects), "non_empty": len(non_empty),
                      "empty": [s["subject"] for s in subjects if s["empty_upstream"]]}))


def fetch() -> None:
    payload = json.loads(FREEZE.read_text())
    done = skipped = failed = 0
    total = 0
    problems = []
    for entry in payload["subjects"]:
        for item in entry["files"]:
            target = SEALED_ROOT / "antisaccade_min" / entry["subject"] / item["name"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and target.stat().st_size > 0:
                skipped += 1
                total += target.stat().st_size
                continue
            url = (f"https://drive.usercontent.google.com/download?id={item['id']}"
                   "&export=download&confirm=t")
            state, size = download_once(url, target)
            if state == "done":
                done += 1
                total += size
            else:
                failed += 1
                problems.append({"subject": entry["subject"], "name": item["name"],
                                 "state": state})
            print(f"  {state:16s} {entry['subject']:>6s}/{item['name'][:48]:48s} "
                  f"{size / 2 ** 20:8.1f} MiB", flush=True)
            time.sleep(1.0)
    bad = []
    for path in SEALED_ROOT.rglob("*.mat"):
        with path.open("rb") as handle:
            if not any(handle.read(8).startswith(m) for m in MAT_MAGIC):
                bad.append(str(path))
    report = {"downloaded": done, "already_present": skipped, "failed": failed,
              "bytes": total, "bad_header": bad, "problems": problems}
    out = REPO / "results/iris/sealed/sealed_fetch_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("downloaded", "already_present", "failed", "bytes")}))
    if not failed and not bad:
        seal()


def seal() -> None:
    os.chmod(SEALED_ROOT, 0)
    print(f"sealed: chmod 000 {SEALED_ROOT} — reads now fail until the "
          "preregistered opening restores permissions")


def status() -> None:
    mode = stat.S_IMODE(SEALED_ROOT.stat().st_mode) if SEALED_ROOT.exists() else None
    print(json.dumps({"freeze_exists": FREEZE.exists(),
                      "sealed_root": str(SEALED_ROOT),
                      "sealed_root_mode": (oct(mode) if mode is not None else "absent")}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["freeze", "fetch", "seal", "status"])
    args = parser.parse_args()
    {"freeze": freeze, "fetch": fetch, "seal": seal, "status": status}[args.mode]()


if __name__ == "__main__":
    main()
