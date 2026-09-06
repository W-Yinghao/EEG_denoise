#!/usr/bin/env python3
"""WAVE4 E0b: sharded, resumable, atomically published Tobii + E-Prime download.

Reuses the verified 919-series REST pattern: signed URL per file handle, HTTP Range
resume, per-file atomic rename, whole-selection atomic publish. Scene-video .mp4 files
are enumerated in the manifest but NOT downloaded (declared scope note: not used by any
of M1-M4).
"""
from __future__ import annotations

import argparse
import configparser
import json
import os
import re
import shutil
import stat
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

SYNAPSE_CONFIG = Path("/home/infres/yinwang/.synapseConfig")
DATA_ROOT = Path("/projects/EEG-foundation-model")
STAGING = DATA_ROOT / "eye_bci" / ".syn64005218-tobii.partial"
FINAL = DATA_ROOT / "eye_bci" / "syn64005218-tobii"
REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "results/wave4_optical/manifest/e0_manifest.json"
CONTENT_RANGE = re.compile(r"^bytes ([0-9]+)-([0-9]+)/([0-9]+)$")
SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
DOWNLOAD_EXTENSIONS = {".csv", ".txt"}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, *args, **kwargs):
        return None


def load_token() -> str:
    st = os.lstat(SYNAPSE_CONFIG)
    if (stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode)
            or st.st_uid != os.getuid() or stat.S_IMODE(st.st_mode) & 0o077):
        raise PermissionError("unsafe ~/.synapseConfig")
    parser = configparser.ConfigParser(interpolation=None)
    with SYNAPSE_CONFIG.open("r", encoding="utf-8") as stream:
        parser.read_file(stream)
    for section in ("default", "authentication"):
        if parser.has_option(section, "authtoken"):
            token = parser.get(section, "authtoken").strip()
            if token:
                return token
    raise ValueError("no authtoken")


def selection() -> list[dict[str, Any]]:
    payload = json.loads(MANIFEST.read_text())
    if payload.get("decision_rule", {}).get("verdict") != "PROCEED":
        raise RuntimeError("E0 decision rule did not return PROCEED")
    files = []
    for entry in payload["files"]:
        name = entry["name"]
        if Path(name).suffix.lower() not in DOWNLOAD_EXTENSIONS:
            continue
        for component in (entry["subject"], entry["session"], entry["modality"], name):
            if not SAFE_NAME.fullmatch(str(component)):
                raise ValueError(f"unsafe path component: {component}")
        if int(entry["bytes"]) <= 0 or not str(entry["entity_id"]).startswith("syn"):
            raise ValueError(f"invalid manifest entry: {entry}")
        files.append(entry)
    files.sort(key=lambda e: (e["subject"], e["session"], e["modality"], e["name"]))
    return files


def signed_url(token: str, item: dict[str, Any]) -> str:
    query = urlencode({"redirect": "false", "fileAssociateType": "FileEntity",
                       "fileAssociateId": str(item["entity_id"])})
    request = urllib.request.Request(
        f"https://repo-prod.prod.sagebase.org/file/v1/file/{item['file_handle_id']}?{query}",
        headers={"Accept": "text/plain", "Authorization": f"Bearer {token}",
                 "User-Agent": "denoiseNet-private-research/1"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=30) as response:
        if int(response.status) != 200:
            raise RuntimeError("no signed URL")
        value = response.read(8193).decode("utf-8").strip()
    parsed = urlsplit(value)
    if len(value) > 8192 or parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("invalid signed URL")
    return value


def download_once(token: str, item: dict[str, Any], target: Path) -> tuple[int, bool]:
    expected = int(item["bytes"])
    temporary = target.with_name(target.name + ".partial")
    if target.exists():
        if not target.is_file() or target.stat().st_size != expected:
            raise ValueError(f"unexpected existing target: {target}")
        return 0, True
    offset = temporary.stat().st_size if temporary.exists() else 0
    if offset > expected:
        temporary.unlink()
        offset = 0
    if offset == expected:
        os.replace(temporary, target)
        return 0, False
    headers = {"Accept-Encoding": "identity", "User-Agent": "denoiseNet-private-research/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(signed_url(token, item), headers=headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=120) as response:
        status = int(getattr(response, "status", response.getcode()))
        resumed = offset > 0 and status == 206
        if offset and status not in (200, 206):
            raise IOError(f"resume status {status}")
        if not offset and status != 200:
            raise IOError(f"initial status {status}")
        if resumed:
            match = CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
            if match is None or int(match.group(1)) != offset or int(match.group(3)) != expected:
                raise IOError("invalid Content-Range")
        elif offset:
            offset = 0
        length = response.headers.get("Content-Length")
        if length is not None and int(length) != expected - offset:
            raise IOError("unexpected Content-Length")
        written = 0
        with temporary.open("ab" if resumed else "wb") as stream:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
                written += len(chunk)
            stream.flush()
            os.fsync(stream.fileno())
    if temporary.stat().st_size != expected:
        raise IOError(f"incomplete: {temporary.stat().st_size} of {expected}")
    os.replace(temporary, target)
    return written, False


def download(shard: int, shards: int) -> None:
    token = load_token()
    files = selection()
    mine = [f for index, f in enumerate(files) if index % shards == shard]
    STAGING.mkdir(parents=True, exist_ok=True)
    downloaded = existing = 0
    written_bytes = 0
    failures = []
    for item in mine:
        target = STAGING / item["subject"] / item["session"] / item["modality"] / item["name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, 4):
            try:
                written, already = download_once(token, item, target)
                written_bytes += written
                existing += int(already)
                downloaded += int(not already)
                break
            except Exception as error:                      # noqa: BLE001 - reason-coded
                if attempt == 3:
                    failures.append({"entity_id": item["entity_id"],
                                     "name": item["name"], "error": str(error)})
                else:
                    time.sleep(3 * attempt)
    report = {"shard": shard, "shards": shards, "assigned": len(mine),
              "downloaded": downloaded, "already_present": existing,
              "written_bytes": written_bytes, "failures": failures}
    out = REPO / "results/wave4_optical/manifest"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"download_shard_{shard}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "failures"} |
                     {"failures": len(failures)}))


def publish() -> None:
    files = selection()
    missing, wrong = [], []
    total = 0
    for item in files:
        path = STAGING / item["subject"] / item["session"] / item["modality"] / item["name"]
        if not path.is_file():
            missing.append(item["name"])
            continue
        size = path.stat().st_size
        total += size
        if size != int(item["bytes"]):
            wrong.append({"name": item["name"], "expected": int(item["bytes"]), "actual": size})
    if missing or wrong:
        raise RuntimeError(f"publish blocked: {len(missing)} missing, {len(wrong)} wrong-size")
    if FINAL.exists():
        raise RuntimeError(f"publish target already exists: {FINAL}")
    os.replace(STAGING, FINAL)
    manifest = json.loads(MANIFEST.read_text())
    registry = {
        "schema_version": 1, "dataset_id": "eye_bci_tobii",
        "status": "verified_available", "path": str(FINAL),
        "source": "https://www.synapse.org/Synapse:syn64005218",
        "selection": (f"{len(files)} files: 315 Tobii CSV + 315 E-Prime TXT across 31 "
                      "subjects. The 10 Tobii scene-video .mp4 files (S14/S16, 0.697 GiB) "
                      "were enumerated in the E0 manifest but NOT downloaded (declared "
                      "scope note: not used by any of M1-M4)."),
        "bytes": total, "files": len(files),
        "subjects": sorted({f["subject"] for f in files}),
        "access": "Synapse PAT; manifest-only query then sharded resumable download",
        "wave": "WAVE4 E0b", "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                             time.gmtime()),
    }
    (REPO / "datasets/registry/eye_bci_tobii.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"published": str(FINAL), "files": len(files),
                      "gib": round(total / (1 << 30), 3)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("download")
    d.add_argument("--shard", type=int, required=True)
    d.add_argument("--shards", type=int, required=True)
    sub.add_parser("publish")
    args = parser.parse_args()
    if args.mode == "download":
        download(args.shard, args.shards)
    else:
        publish()


if __name__ == "__main__":
    main()
