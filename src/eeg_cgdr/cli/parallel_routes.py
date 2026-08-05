"""Slurm-only command for the independent parallel route screen."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from eeg_cgdr.experiments.parallel_subject_aware_routes_v1 import run_stage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    value = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("parallel route config must be a mapping")
    raw_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    task = None if raw_task is None else int(raw_task)
    result = run_stage(value, args.run_dir, args.stage, task)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
