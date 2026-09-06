from __future__ import annotations
import argparse,json,os
from pathlib import Path
from eeg_cgdr.experiments.physiomotion_hybrid_masked import STAGES,load_config

def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path,required=True);parser.add_argument("--stage",choices=STAGES,required=True);parser.add_argument("--fold",type=int);parser.add_argument("--seed",type=int);parser.add_argument("--run-dir",type=Path);args=parser.parse_args();c=load_config(args.config);run_dir=args.run_dir or Path(c["result_root"])/"runs"/(os.environ.get("SLURM_JOB_ID","local"));run_dir.mkdir(parents=True,exist_ok=True)
    kwargs={};
    if args.stage in {"materialize","technical","train","infer","evaluate"}: kwargs["fold"]=args.fold if args.fold is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID","0"))
    if args.stage in {"train","infer","evaluate","aggregate"}: kwargs["seed"]=args.seed if args.seed is not None else int(c["training_seed"])
    result=STAGES[args.stage](c,run_dir=run_dir,**kwargs);print(json.dumps(result,sort_keys=True))
if __name__=="__main__":main()
