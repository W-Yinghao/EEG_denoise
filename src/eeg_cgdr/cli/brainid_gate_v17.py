from argparse import ArgumentParser
from pathlib import Path

from eeg_cgdr.experiments.brainid_gate_v17 import (
    aggregate_inventory,
    aggregate_m0,
    aggregate_m1,
    evaluate_m0_fold,
    evaluate_verifiers,
    fit_verifier_b,
    freeze_protocol,
    gate01,
    load_config,
    prepare_participant,
    select_m0_fold,
    train_verifier_a,
    write_report,
)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", required=True, choices=["freeze","prepare","inventory","train-a","train-b","eval-m1","aggregate-m1","select-m0","eval-m0","aggregate-m0","gate","report"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(); config = load_config(args.config)
    run = args.run_dir or Path(config["result_root"]) / "server_runs" / args.stage / f"task_{args.index:02d}"
    run.mkdir(parents=True, exist_ok=True)
    result = {
        "freeze": lambda: freeze_protocol(config),
        "prepare": lambda: prepare_participant(config, args.index + 1),
        "inventory": lambda: aggregate_inventory(config),
        "train-a": lambda: train_verifier_a(config, args.index, run),
        "train-b": lambda: fit_verifier_b(config, args.index, run),
        "eval-m1": lambda: evaluate_verifiers(config, args.index, run),
        "aggregate-m1": lambda: aggregate_m1(config),
        "select-m0": lambda: select_m0_fold(config, args.index, run),
        "eval-m0": lambda: evaluate_m0_fold(config, args.index, run),
        "aggregate-m0": lambda: aggregate_m0(config),
        "gate": lambda: gate01(config),
        "report": lambda: write_report(config),
    }[args.stage]()
    print(result)


if __name__ == "__main__":
    main()
