from __future__ import annotations
import argparse
from pathlib import Path
from eeg_cgdr.experiments.sge_dynamic_transfer_v6 import run_stage

def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--stage",required=True); parser.add_argument("--run-dir",type=Path,required=True); parser.add_argument("--task-index",type=int,default=0); parser.add_argument("--model-kind",default=""); parser.add_argument("--seed",type=int,default=20260806)
    args=parser.parse_args(); args.run_dir.mkdir(parents=True,exist_ok=True)
    run_stage(args.config,args.stage,args.run_dir,task_index=args.task_index,model_kind=args.model_kind,seed=args.seed); return 0

if __name__ == "__main__": raise SystemExit(main())

