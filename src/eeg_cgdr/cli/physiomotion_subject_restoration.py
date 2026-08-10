from argparse import ArgumentParser
from pathlib import Path
from eeg_cgdr.experiments.physiomotion_subject_restoration import load_config, stage_headroom, stage_headroom_aggregate, stage_metadata, stage_prepare


def main() -> None:
    parser = ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--stage", required=True); parser.add_argument("--task-index", type=int, default=0); parser.add_argument("--run-dir", type=Path, required=True); args = parser.parse_args(); c = load_config(args.config); args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "metadata": stage_metadata(c, args.run_dir)
    elif args.stage == "prepare": stage_prepare(c, args.task_index, args.run_dir)
    elif args.stage == "headroom": stage_headroom(c, args.task_index, args.run_dir)
    elif args.stage == "headroom-aggregate": stage_headroom_aggregate(c, args.run_dir)
    else: raise ValueError(args.stage)


if __name__ == "__main__": main()
