#!/usr/bin/env python3
"""WAVE4 E0: manifest-ONLY Synapse query for the Eye-BCI Tobii + E-Prime modalities.

Reads entity metadata and file-handle sizes only. No file content is requested and
nothing is downloaded. The frozen decision rule in reports/wave4_preregistration.md
is applied to the manifest and the verdict recorded verbatim.
"""
from __future__ import annotations

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

SYNAPSE_CONFIG = Path("/home/infres/yinwang/.synapseConfig")
PROJECT = "syn64005218"
API = "https://repo-prod.prod.sagebase.org/repo/v1"
DATA_ROOT = Path("/projects/EEG-foundation-model")
LOCAL_EEG = DATA_ROOT / "eye_bci/syn64005218-neuroscan"
OUT = Path(__file__).resolve().parents[1] / "results/wave4_optical/manifest"
SUBJECT_RE = re.compile(r"^S(0[1-9]|[12][0-9]|3[01])$")
TOBII_RE = re.compile(r"tobii", re.IGNORECASE)
EPRIME_RE = re.compile(r"e-?prime", re.IGNORECASE)
MAX_DOWNLOAD_GIB = 300.0
MIN_SUBJECTS = 10


def load_token() -> str:
    st = os.lstat(SYNAPSE_CONFIG)
    if (stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode)
            or st.st_uid != os.getuid() or stat.S_IMODE(st.st_mode) & 0o077):
        raise PermissionError("unsafe ~/.synapseConfig ownership or permissions")
    parser = configparser.ConfigParser(interpolation=None)
    with SYNAPSE_CONFIG.open("r", encoding="utf-8") as stream:
        parser.read_file(stream)
    for section in ("default", "authentication"):
        if parser.has_option(section, "authtoken"):
            token = parser.get(section, "authtoken").strip()
            if token:
                return token
    raise ValueError("no authtoken in ~/.synapseConfig")


def api_json(url: str, token: str, payload: dict | None = None,
             retries: int = 4) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body,
        headers={"Accept": "application/json", "Authorization": f"Bearer {token}",
                 "Content-Type": "application/json",
                 "User-Agent": "denoiseNet-private-research/1"},
        method="GET" if payload is None else "POST")
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
            if exc.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Synapse request failed with {last}") from None
        except urllib.error.URLError as exc:
            last = str(exc.reason)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Synapse request failed after retries: {last}")


def children(token: str, parent: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    next_token = None
    for _ in range(50):
        payload = {"parentId": parent, "includeTypes": ["folder", "file"],
                   "sortBy": "NAME", "sortDirection": "ASC"}
        if next_token:
            payload["nextPageToken"] = next_token
        response = api_json(f"{API}/entity/children", token, payload)
        out.extend(response.get("page", []) or [])
        next_token = response.get("nextPageToken")
        if not next_token:
            break
    return out


def file_meta(token: str, entity_id: str) -> dict[str, Any]:
    response = api_json(f"{API}/entity/{entity_id}/filehandles", token)
    handles = [h for h in response.get("list", []) if isinstance(h, dict)
               and h.get("status") == "AVAILABLE"]
    if not handles:
        return {"bytes": 0, "content_type": None, "file_handle_id": None,
                "status": "NO_AVAILABLE_HANDLE"}
    handle = max(handles, key=lambda h: int(h.get("contentSize") or 0))
    return {"bytes": int(handle.get("contentSize") or 0),
            "content_type": handle.get("contentType"),
            "file_name": handle.get("fileName"),
            "file_handle_id": str(handle.get("id")), "status": "AVAILABLE"}


def main() -> None:
    token = load_token()
    profile = api_json(f"{API}/userProfile", token)
    authenticated = bool(profile.get("ownerId"))
    local_subjects = sorted(p.name for p in LOCAL_EEG.iterdir()
                            if SUBJECT_RE.match(p.name)) if LOCAL_EEG.is_dir() else []
    files: list[dict[str, Any]] = []
    tree_notes: list[str] = []
    subjects = [c for c in children(token, PROJECT) if SUBJECT_RE.match(str(c.get("name")))]
    for subject in subjects:
        sname = str(subject["name"])
        for session in children(token, str(subject["id"])):
            if str(session.get("type", "")).endswith("Folder") is False and \
                    "Folder" not in str(session.get("type", "")):
                continue
            sess_name = str(session.get("name"))
            for modality in children(token, str(session["id"])):
                mod_name = str(modality.get("name"))
                is_tobii = bool(TOBII_RE.search(mod_name))
                is_eprime = bool(EPRIME_RE.search(mod_name))
                if not (is_tobii or is_eprime):
                    continue
                kind = "tobii" if is_tobii else "eprime"
                entries = children(token, str(modality["id"]))
                for entry in entries:
                    if "File" not in str(entry.get("type", "")):
                        tree_notes.append(f"nested folder {sname}/{sess_name}/{mod_name}/"
                                          f"{entry.get('name')} (not recursed)")
                        continue
                    meta = file_meta(token, str(entry["id"]))
                    files.append({"subject": sname, "session": sess_name,
                                  "modality": mod_name, "kind": kind,
                                  "name": str(entry.get("name")),
                                  "entity_id": str(entry.get("id")), **meta})
    tobii_files = [f for f in files if f["kind"] == "tobii"]
    eprime_files = [f for f in files if f["kind"] == "eprime"]
    tobii_subjects = sorted({f["subject"] for f in tobii_files})
    covered = sorted(set(tobii_subjects) & set(local_subjects))
    total_bytes = sum(f["bytes"] for f in files)
    total_gib = total_bytes / (1 << 30)
    usage = shutil.disk_usage(DATA_ROOT)
    free_gib = usage.free / (1 << 30)
    rule_a = len(covered) >= MIN_SUBJECTS
    rule_b = total_gib <= MAX_DOWNLOAD_GIB
    rule_c = free_gib >= 2 * total_gib
    verdict = "PROCEED" if (rule_a and rule_b and rule_c) else "STOP"
    extensions: dict[str, int] = {}
    for f in files:
        extensions[Path(f["name"]).suffix.lower() or "<none>"] = \
            extensions.get(Path(f["name"]).suffix.lower() or "<none>", 0) + 1
    payload = {
        "mode": "wave4-e0-manifest-only",
        "content_downloaded": False,
        "authenticated": authenticated,
        "project": PROJECT,
        "local_eeg_subjects": local_subjects,
        "tobii": {"files": len(tobii_files), "subjects": tobii_subjects,
                  "bytes": sum(f["bytes"] for f in tobii_files)},
        "eprime": {"files": len(eprime_files),
                   "subjects": sorted({f["subject"] for f in eprime_files}),
                   "bytes": sum(f["bytes"] for f in eprime_files)},
        "covered_subjects_with_local_eeg": covered,
        "total_download_bytes": total_bytes,
        "total_download_gib": total_gib,
        "free_space_gib": free_gib,
        "extensions": extensions,
        "sampling_rate": ("NOT DETERMINABLE from manifest metadata (no content read); "
                          "recorded at E1 from the downloaded headers"),
        "decision_rule": {
            "a_tobii_covers_ge_10_local_eeg_subjects": rule_a,
            "b_total_le_300_gib": rule_b,
            "c_free_ge_2x_download": rule_c,
            "verdict": verdict},
        "tree_notes": tree_notes[:50],
        "files": files,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "e0_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"authenticated": authenticated, "tobii_files": len(tobii_files),
                      "tobii_subjects": len(tobii_subjects), "eprime_files": len(eprime_files),
                      "covered": len(covered), "total_gib": round(total_gib, 2),
                      "free_gib": round(free_gib, 1), "verdict": verdict,
                      "rules": payload["decision_rule"]}))


if __name__ == "__main__":
    main()
