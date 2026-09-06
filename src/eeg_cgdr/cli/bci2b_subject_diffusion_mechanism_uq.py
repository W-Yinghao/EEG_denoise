from __future__ import annotations
import argparse
from pathlib import Path
from eeg_cgdr.experiments.bci2b_subject_diffusion_mechanism_uq import run_stage

def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);p.add_argument("--stage",required=True);p.add_argument("--task-index",type=int,default=0);p.add_argument("--run-dir",type=Path,required=True);a=p.parse_args();run_stage(a.config,a.stage,a.run_dir,task_index=a.task_index)
if __name__=="__main__":main()

