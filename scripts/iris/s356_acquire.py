#!/usr/bin/env python3
"""S356 acquisition — antisaccade synchronised_min listing positions 91-370 (DEV-CLASS).

Preregistered in reports/iris_prereg_s356.md. Sealed positions 31-90 are never
touched (charter §5: positions 91+ can never join the sealed block). Same tooling
class as the sealed fetch: deterministic enumeration, resumable Range downloads,
per-file atomic rename, header verification, ITT counting of empties/failures.

Modes: manifest (refuses to overwrite) | fetch | status.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from eegeyenet_download import FOLDER_ID, SAFE, _subfolder, download_once, listing  # noqa: E402

MANIFEST = REPO / "results/iris/s356/ext_manifest.json"
EXT_ROOT = Path("/projects/EEG-foundation-model/eegeyenet/eegeyenet_ext")
POS_LO, POS_HI = 91, 370                      # 1-indexed listing positions, inclusive
MAT_MAGIC = (b"MATLAB", b"\x89HDF")


def manifest() -> None:
    if MANIFEST.exists():
        raise SystemExit(f"REFUSED: manifest already exists at {MANIFEST}")
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
                         "files": files, "empty_upstream": not files})
        time.sleep(0.3)
    non_empty = sum(1 for s in subjects if not s["empty_upstream"])
    payload = {
        "asset": "EEGEyeNet antisaccade synchronised_min positions 91-370 — S356 ext",
        "classification": "DEV-CLASS (charter §5: never joins the sealed block)",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "positions": [POS_LO, POS_HI], "n_folders": len(subjects),
        "n_subjects_non_empty": non_empty, "subjects": subjects,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"folders": len(subjects), "non_empty": non_empty}))


def fetch() -> None:
    payload = json.loads(MANIFEST.read_text())
    done = skipped = failed = quota = 0
    total = 0
    problems = []
    for entry in payload["subjects"]:
        if entry["subject"].endswith(".mat"):
            continue                        # Drive junk entry (project_state.mat)
        for item in entry["files"]:
            target = EXT_ROOT / "antisaccade_min" / entry["subject"] / item["name"]
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
                quota += int(state == "quota")
                problems.append({"subject": entry["subject"], "name": item["name"],
                                 "state": state})
            print(f"  {state:16s} {entry['subject']:>6s}/{item['name'][:48]:48s} "
                  f"{size / 2 ** 20:8.1f} MiB", flush=True)
            time.sleep(1.0)
    bad = []
    for path in EXT_ROOT.rglob("*.mat"):
        if not path.is_file() or path.parent.name.endswith(".mat"):
            continue                        # rglob also matches the junk directory
        with path.open("rb") as handle:
            if not any(handle.read(8).startswith(m) for m in MAT_MAGIC):
                bad.append(str(path))
    report = {"downloaded": done, "already_present": skipped, "failed": failed,
              "quota_blocked": quota, "bytes": total, "bad_header": bad,
              "problems": problems[:40],
              "complete": bool(failed == 0 and not bad)}
    out = REPO / "results/iris/s356/ext_fetch_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("downloaded", "already_present", "failed", "quota_blocked",
                       "complete")}))


def status() -> None:
    have = len(list(EXT_ROOT.rglob("*.mat"))) if EXT_ROOT.exists() else 0
    want = (sum(len(s["files"]) for s in json.loads(MANIFEST.read_text())["subjects"])
            if MANIFEST.exists() else None)
    print(json.dumps({"manifest": MANIFEST.exists(), "files_on_disk": have,
                      "files_wanted": want}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["manifest", "fetch", "status"])
    args = parser.parse_args()
    {"manifest": manifest, "fetch": fetch, "status": status}[args.mode]()


if __name__ == "__main__":
    main()
