"""Slurm-only CLI for literature-guided exploration v3."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from eeg_cgdr.experiments.literature_guided_v3 import run_stage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("v3 config must be a mapping")
    raw_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    result = run_stage(config, args.run_dir, args.stage, None if raw_task is None else int(raw_task))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
