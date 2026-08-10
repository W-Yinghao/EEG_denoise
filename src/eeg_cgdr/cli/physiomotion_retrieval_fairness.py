from argparse import ArgumentParser
from pathlib import Path

from eeg_cgdr.experiments.physiomotion_retrieval_fairness import load_config, stage_aggregate, stage_audit, stage_evaluate, stage_report, stage_select


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "audit": stage_audit(config, args.run_dir)
    elif args.stage == "select": stage_select(config, args.task_index, args.run_dir)
    elif args.stage == "evaluate": stage_evaluate(config, args.task_index, args.run_dir)
    elif args.stage == "aggregate": stage_aggregate(config, args.run_dir)
    elif args.stage == "report": stage_report(config, args.run_dir)
    else: raise ValueError(args.stage)


if __name__ == "__main__":
    main()
