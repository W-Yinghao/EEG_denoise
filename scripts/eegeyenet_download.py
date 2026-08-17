#!/usr/bin/env python3
"""EEGEyeNet acquisition — the minimally-preprocessed synchronized subset.

Scope (operator-specified, by priority):
  P1  antisaccade_task_data/synchronised_min  -- first N subjects (default 30)
  P2  dots_data/synchronised_min ("Large Grid") -- all subjects
  docs README/LICENSE from Drive, plus the two description PDFs from OSF ktv7m
Explicitly OUT of scope: synchronised_max (maximally preprocessed - ICA has removed the
ocular artifact, i.e. our object of study), processing_speed_data (VSS), prepared/
(benchmark ML feature tensors), and raw unsynchronized data.

Per-subject files are ~130-260 MiB and are NOT subject to the Google public-file quota
that blocks the multi-GiB prepared/ tensors, so this runs on the anonymous route.
Resumable (HTTP Range), per-file atomic rename, header-verified at publish.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

FOLDER_ID = "1iHpnEE6kalLGHaw2Hd8EwJMdVE0K7rk7"
OSF_NODE = "ktv7m"
DATA_ROOT = Path("/projects/EEG-foundation-model/eegeyenet")
STAGING = DATA_ROOT / ".eegeyenet_min.partial"
FINAL = DATA_ROOT / "eegeyenet_min"
REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "reports/eegeyenet_min_manifest.json"
ENTRY = re.compile(
    r'<div class="flip-entry" id="entry-([^"]+)".*?'
    r'<a href="https://drive\.google\.com/(drive/folders|file/d)/[^"]*".*?'
    r'<div class="flip-entry-title">([^<]*)</div>', re.S)
SAFE = re.compile(r"^[A-Za-z0-9._-]+$")
UA = "Mozilla/5.0"
N_ANTISACCADE = 30
MAT_MAGIC = (b"MATLAB", b"\x89HDF")


def _get(url: str, timeout: int = 60, tries: int = 3) -> bytes:
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": UA})
            return urllib.request.urlopen(request, timeout=timeout).read()
        except Exception:                              # noqa: BLE001 - reason-coded
            if attempt == tries - 1:
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("unreachable")


def listing(folder_id: str) -> list[dict]:
    html = _get(f"https://drive.google.com/embeddedfolderview?id={folder_id}#list").decode(
        "utf-8", "replace")
    return [{"id": m.group(1),
             "kind": "folder" if "folders" in m.group(2) else "file",
             "name": m.group(3)} for m in ENTRY.finditer(html)]


def _subfolder(parent_id: str, name: str) -> str:
    for entry in listing(parent_id):
        if entry["name"] == name and entry["kind"] == "folder":
            return entry["id"]
    raise RuntimeError(f"subfolder {name!r} not found")


def build_manifest() -> dict:
    top = {e["name"]: e for e in listing(FOLDER_ID)}
    files: list[dict] = []

    # --- P1 antisaccade, first N subjects (deterministic: listing order)
    anti = _subfolder(top["antisaccade_task_data"]["id"], "synchronised_min")
    subjects = [e for e in listing(anti) if e["kind"] == "folder"][:N_ANTISACCADE]
    print(f"P1 antisaccade: taking {len(subjects)} of the full set", flush=True)
    for subject in subjects:
        for item in listing(subject["id"]):
            if item["kind"] == "file" and item["name"].lower().endswith(".mat"):
                files.append({"group": "antisaccade_min", "subject": subject["name"],
                              "name": item["name"], "id": item["id"]})
        time.sleep(0.3)

    # --- P2 Large Grid (dots), all subjects
    dots = _subfolder(top["dots_data"]["id"], "synchronised_min")
    dot_subjects = [e for e in listing(dots) if e["kind"] == "folder"]
    print(f"P2 dots/Large Grid: taking all {len(dot_subjects)} subjects", flush=True)
    for subject in dot_subjects:
        for item in listing(subject["id"]):
            if item["kind"] == "file" and item["name"].lower().endswith(".mat"):
                files.append({"group": "dots_min", "subject": subject["name"],
                              "name": item["name"], "id": item["id"]})
        time.sleep(0.3)

    # --- small docs from the Drive root
    for name in ("README.md", "LICENSE"):
        if name in top and top[name]["kind"] == "file":
            files.append({"group": "docs", "subject": "", "name": name,
                          "id": top[name]["id"]})

    for entry in files:
        for part in (entry["subject"], entry["name"]):
            if part and not SAFE.fullmatch(part):
                raise ValueError(f"unsafe path component: {part!r}")

    # --- OSF description PDFs (quota-free, official index)
    osf = []
    try:
        node = json.loads(_get(
            f"https://api.osf.io/v2/nodes/{OSF_NODE}/files/osfstorage/").decode())
        stack = list(node.get("data", []))
        while stack:
            item = stack.pop()
            attrs = item["attributes"]
            if attrs.get("kind") == "folder":
                sub = json.loads(_get(
                    item["relationships"]["files"]["links"]["related"]["href"]).decode())
                stack.extend(sub.get("data", []))
            elif attrs["name"].lower().endswith(".pdf"):
                osf.append({"group": "docs", "subject": "", "name": attrs["name"],
                            "osf_download": item["links"]["download"],
                            "bytes": attrs.get("size") or 0})
    except Exception as error:                          # noqa: BLE001 - reason-coded
        print(f"  (OSF PDF enumeration failed: {error})", flush=True)

    manifest = {
        "source_drive": f"https://drive.google.com/drive/folders/{FOLDER_ID}",
        "official_index": f"https://osf.io/{OSF_NODE}/",
        "scope_in": ["antisaccade_task_data/synchronised_min (first "
                     f"{N_ANTISACCADE} subjects)", "dots_data/synchronised_min (all)",
                     "README/LICENSE", "OSF description PDFs"],
        "scope_out": ["synchronised_max (maximally preprocessed; ICA removes the ocular "
                      "artifact under study)", "processing_speed_data (VSS)",
                      "prepared/ (benchmark ML feature tensors)", "raw unsynchronized"],
        "drive_files": files, "osf_files": osf,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    groups: dict[str, int] = {}
    for entry in files:
        groups[entry["group"]] = groups.get(entry["group"], 0) + 1
    print(f"manifest -> {MANIFEST}: {groups}, OSF PDFs: {len(osf)}")
    return manifest


def _target(entry: dict) -> Path:
    parts = [entry["group"]] + ([entry["subject"]] if entry["subject"] else [])
    return STAGING.joinpath(*parts, entry["name"])


def download_once(url: str, target: Path) -> tuple[str, int]:
    temporary = target.with_name(target.name + ".partial")
    offset = temporary.stat().st_size if temporary.exists() else 0
    headers = {"User-Agent": UA, "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=180) as response:
            if "text/html" in response.headers.get("Content-Type", ""):
                return "quota", 0
            status = int(getattr(response, "status", response.getcode()))
            resumed = offset > 0 and status == 206
            if offset and not resumed:
                offset = 0
            length = response.headers.get("Content-Length")
            expected = (offset + int(length)) if length else None
            with temporary.open("ab" if resumed else "wb") as stream:
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
    except urllib.error.HTTPError as error:
        return f"error HTTP {error.code}", 0
    except Exception as error:                          # noqa: BLE001 - reason-coded
        return f"error {type(error).__name__}", 0
    size = temporary.stat().st_size
    if expected is not None and size != expected:
        return f"error incomplete {size}/{expected}", size
    os.replace(temporary, target)
    return "done", size


def run(shard: int, shards: int) -> None:
    manifest = json.loads(MANIFEST.read_text())
    items = [("drive", e) for e in manifest["drive_files"]]
    items += [("osf", e) for e in manifest["osf_files"]]
    mine = [it for index, it in enumerate(items) if index % shards == shard]
    done = skipped = failed = 0
    total = 0
    problems = []
    for kind, entry in mine:
        target = _target(entry)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.stat().st_size > 0:
            skipped += 1
            total += target.stat().st_size
            continue
        url = (entry["osf_download"] if kind == "osf" else
               f"https://drive.usercontent.google.com/download?id={entry['id']}"
               "&export=download&confirm=t")
        state, size = download_once(url, target)
        if state == "done":
            done += 1
            total += size
        else:
            failed += 1
            problems.append({"name": entry["name"], "subject": entry.get("subject", ""),
                             "state": state})
        print(f"  {state:20s} {entry.get('subject',''):>8s}/{entry['name'][:48]:48s} "
              f"{size / 2 ** 20:8.1f} MiB", flush=True)
        time.sleep(1.0)
    report = {"shard": shard, "shards": shards, "assigned": len(mine), "downloaded": done,
              "already_present": skipped, "failed": failed, "bytes": total,
              "problems": problems}
    out = REPO / "reports/eegeyenet_min_shards"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"shard_{shard}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("shard", "assigned", "downloaded", "already_present", "failed")}))


def publish() -> None:
    manifest = json.loads(MANIFEST.read_text())
    entries = manifest["drive_files"] + manifest["osf_files"]
    missing, bad = [], []
    total = 0
    for entry in entries:
        path = _target(entry)
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(entry["name"])
            continue
        total += path.stat().st_size
        if path.suffix.lower() == ".mat":
            with path.open("rb") as handle:
                head = handle.read(8)
            if not any(head.startswith(m) for m in MAT_MAGIC):
                bad.append(entry["name"])
    if missing or bad:
        raise RuntimeError(f"publish blocked: {len(missing)} missing, {len(bad)} bad header"
                           f" (first missing: {missing[:3]}, first bad: {bad[:3]})")
    if FINAL.exists():
        raise RuntimeError(f"publish target already exists: {FINAL}")
    os.replace(STAGING, FINAL)
    write_registry()
    print(json.dumps({"published": str(FINAL), "files": len(entries),
                      "gib": round(total / 2 ** 30, 3)}))


def write_registry() -> dict:
    """Write the registry entry from the manifest. Idempotent; no tree is moved.

    Subject counts are counted, not assumed: the antisaccade slice is the first
    N_ANTISACCADE folders in Drive listing order, but two of them (AA2, AB8) are empty
    upstream, so the delivered subject count is lower than the requested slice.
    """
    manifest = json.loads(MANIFEST.read_text())
    entries = manifest["drive_files"] + manifest["osf_files"]
    total = sum(p.stat().st_size for p in (FINAL if FINAL.exists() else STAGING)
                .rglob("*") if p.is_file())
    subjects: dict[str, set[str]] = {}
    for entry in entries:
        if entry.get("subject"):
            subjects.setdefault(entry["group"], set()).add(entry["subject"])
    counts = {group: len(names) for group, names in sorted(subjects.items())}
    registry = {
        "schema_version": 1, "dataset_id": "eegeyenet_min",
        "status": "verified_available", "path": str(FINAL),
        "source": manifest["source_drive"], "official_index": manifest["official_index"],
        "selection": (
            f"minimally preprocessed synchronized: antisaccade "
            f"{counts.get('antisaccade_min', 0)} subjects (first {N_ANTISACCADE} folders "
            f"in Drive listing order; AA2 and AB8 are empty upstream) + dots/Large Grid "
            f"all {counts.get('dots_min', 0)} subjects; {len(entries)} files"),
        "subjects": counts,
        "excluded": manifest["scope_out"],
        "bytes": total, "files": len(entries),
        "access": "Public Google Drive release linked from OSF ktv7m; anonymous route",
        "sample_read": "MAT header magic verified on every .mat at publish; full-file "
                       "load, completeness and pipeline-provenance check in "
                       "reports/eegeyenet_min_qc.json",
        "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (REPO / "datasets/registry/eegeyenet_min.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n")
    return registry


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("manifest")
    r = sub.add_parser("run")
    r.add_argument("--shard", type=int, required=True)
    r.add_argument("--shards", type=int, required=True)
    sub.add_parser("publish")
    sub.add_parser("registry")
    args = parser.parse_args()
    if args.mode == "manifest":
        build_manifest()
    elif args.mode == "run":
        run(args.shard, args.shards)
    elif args.mode == "registry":
        print(json.dumps(write_registry(), indent=2, sort_keys=True))
    else:
        publish()


if __name__ == "__main__":
    main()
