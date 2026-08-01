#!/usr/bin/env python3
"""Run one fixed, job-scoped PyMuPDF startup probe."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from renderer_startup import load_registered_pymupdf


CODE_ROOT = Path("/home/infres/yinwang/denoiseNet")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicate", type=int, choices=(1, 2), required=True)
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
        / f"renderer-import-conda_run-none-warnings_default-r{args.replicate}.json"
    )
    output = Path(os.path.abspath(args.output))
    if output != expected_output:
        raise RuntimeError("renderer probe output is outside its registered job path")
    load_registered_pymupdf(
        role=f"probe_r{args.replicate}",
        job_id=job_id,
        preimport_path=output.with_suffix(".preimport.json"),
        result_path=output,
        authority={"kind": "self_audit_candidate", "audit_job_id": job_id},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
