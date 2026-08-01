#!/usr/bin/env python3
"""Cold-start the registered PyMuPDF renderer and record bounded evidence."""

from __future__ import annotations

import argparse
import faulthandler
import importlib
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path


CODE_ROOT = Path("/home/infres/yinwang/denoiseNet")
ICML_ENV = Path("/home/infres/yinwang/anaconda3/envs/icml")


def publish_json(path: Path, payload: dict[str, object]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + ".partial")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
    )
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("renderer probe JSON write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    temporary.unlink()


def main() -> int:
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("direct", "conda_run"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id.isdigit():
        raise RuntimeError("numeric SLURM_JOB_ID is required")
    expected_output = (
        CODE_ROOT
        / "reports"
        / "environments"
        / "icml"
        / "jobs"
        / job_id
        / f"renderer-import-{args.mode}.json"
    )
    output = Path(os.path.abspath(args.output))
    if output != expected_output:
        raise RuntimeError("renderer probe output is outside its registered job path")
    parent = output.parent
    parent_metadata = parent.lstat()
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
        raise RuntimeError("renderer probe output parent is unsafe")
    if Path(sys.prefix).resolve() != ICML_ENV or ICML_ENV not in Path(sys.executable).resolve().parents:
        raise RuntimeError("renderer probe is outside the registered icml environment")

    fitz = importlib.import_module("fitz")
    version = getattr(fitz, "VersionBind", None)
    if not isinstance(version, str) or not version:
        raise RuntimeError("PyMuPDF version binding is unavailable")
    publish_json(
        output,
        {
            "schema_version": 1,
            "job_id": job_id,
            "mode": args.mode,
            "status": "import_ok",
            "python_executable": sys.executable,
            "python_prefix": sys.prefix,
            "pymupdf_version": version,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
