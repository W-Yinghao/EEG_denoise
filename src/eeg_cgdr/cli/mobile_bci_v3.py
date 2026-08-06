from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg_cgdr.experiments.mobile_bci_v3 import run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.run_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
