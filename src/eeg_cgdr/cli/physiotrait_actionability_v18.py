from argparse import ArgumentParser
from pathlib import Path

from eeg_cgdr.experiments.physiotrait_actionability_v18 import (
    aggregate_actionability,
    aggregate_headroom,
    aggregate_inventory,
    final_gate,
    freeze_protocol,
    load_config,
    prepare_participant,
    run_actionability_fold,
    run_headroom_fold,
    write_source_inventory,
    write_report,
)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", required=True, choices=["freeze", "prepare", "inventory", "source-inventory", "headroom", "aggregate-headroom", "actionability", "aggregate-actionability", "gate", "report"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    run = args.run_dir or Path(config["result_root"]) / "server_runs" / args.stage / f"task_{args.index:02d}"
    run.mkdir(parents=True, exist_ok=True)
    result = {
        "freeze": lambda: freeze_protocol(config),
        "prepare": lambda: prepare_participant(config, args.index + 1),
        "inventory": lambda: aggregate_inventory(config),
        "source-inventory": lambda: write_source_inventory(config),
        "headroom": lambda: run_headroom_fold(config, args.index, run),
        "aggregate-headroom": lambda: aggregate_headroom(config),
        "actionability": lambda: run_actionability_fold(config, args.index, run),
        "aggregate-actionability": lambda: aggregate_actionability(config),
        "gate": lambda: final_gate(config),
        "report": lambda: write_report(config),
    }[args.stage]()
    print(result)


if __name__ == "__main__":
    main()
