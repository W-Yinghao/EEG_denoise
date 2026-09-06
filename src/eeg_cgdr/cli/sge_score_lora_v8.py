from __future__ import annotations

import argparse
from pathlib import Path

from eeg_cgdr.experiments.sge_score_lora_v8 import run_stage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task-index", type=int, default=0)
    args = parser.parse_args(); args.run_dir.mkdir(parents=True, exist_ok=True)
    run_stage(args.config, args.stage, args.run_dir, task_index=args.task_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
