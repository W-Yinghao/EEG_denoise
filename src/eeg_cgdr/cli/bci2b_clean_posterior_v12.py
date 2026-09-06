from __future__ import annotations
import argparse
from pathlib import Path
from eeg_cgdr.experiments.bci2b_clean_posterior_v12 import run_stage

def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path,required=True);parser.add_argument("--stage",required=True);parser.add_argument("--run-dir",type=Path,required=True);parser.add_argument("--task-index",type=int,default=0)
    args=parser.parse_args();run_stage(args.config,args.stage,args.run_dir,task_index=args.task_index)

if __name__=="__main__":main()
