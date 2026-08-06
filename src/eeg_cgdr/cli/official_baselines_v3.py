from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg_cgdr.experiments.official_baselines_v3 import run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.run_dir)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
