from argparse import ArgumentParser
from pathlib import Path

from eeg_cgdr.experiments.project_evidence_freeze import generate


def main() -> None:
    parser = ArgumentParser(description="Generate the CPU-only project scientific evidence freeze")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = generate(args.repo)
    print(f"evidence freeze: {result['governance']['status']}; experiments={result['experiments']}; jobs={result['jobs']}")


if __name__ == "__main__":
    main()
