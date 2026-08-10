from argparse import ArgumentParser
from pathlib import Path
from eeg_cgdr.experiments.raw_support_closure import aggregate, write_report


def main() -> None:
    parser = ArgumentParser(); parser.add_argument("--source", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--report", type=Path, required=True); args = parser.parse_args()
    result = aggregate(args.source, args.output); write_report(result, args.report)


if __name__ == "__main__": main()
