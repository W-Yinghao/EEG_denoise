from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml


def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path,required=True);parser.add_argument("--stage",required=True);parser.add_argument("--run-dir",type=Path,required=True);args=parser.parse_args();config=yaml.safe_load(args.config.read_text());task=os.environ.get("SLURM_ARRAY_TASK_ID");index=None if task is None else int(task)
    if args.stage=="metadata":
        from eeg_cgdr.experiments.mobile_bci_v5 import metadata_stage as run;result=run(config,args.run_dir)
    elif args.stage=="preprocess":
        from eeg_cgdr.experiments.mobile_bci_v5 import preprocess_participant as run
        if index is None:raise ValueError("preprocess requires array index")
        result=run(config,args.run_dir,index)
    elif args.stage=="protocol":
        from eeg_cgdr.experiments.mobile_bci_v5 import protocol_stage as run;result=run(config,args.run_dir)
    elif args.stage=="technical":
        from eeg_cgdr.experiments.mobile_temporal_diffusion_v5 import technical_check as run;result=run(config,args.run_dir)
    elif args.stage=="factorial":
        from eeg_cgdr.experiments.mobile_temporal_diffusion_v5 import run_fold as run
        if index is None:raise ValueError("factorial requires fold index")
        result=run(config,args.run_dir,index,int(config["training_seed"]))
    elif args.stage=="additional":
        from eeg_cgdr.experiments.mobile_temporal_diffusion_v5 import run_fold as run
        if index is None:raise ValueError("additional requires array index")
        seeds=list(config["additional_seeds"]);result=run(config,args.run_dir,index%4,int(seeds[index//4]))
    elif args.stage=="aggregate":
        from eeg_cgdr.experiments.mobile_v5_aggregate import aggregate as run;result=run(config,args.run_dir,[int(config["training_seed"])])
    elif args.stage=="aggregate-three-seed":
        from eeg_cgdr.experiments.mobile_v5_aggregate import aggregate as run;result=run(config,args.run_dir,[int(config["training_seed"]),*map(int,config["additional_seeds"])])
    elif args.stage=="pc-selector":
        from eeg_cgdr.experiments.pc_selector_diagnostic_v5 import run;result=run(config,args.run_dir)
    elif args.stage=="report":
        from eeg_cgdr.experiments.mobile_v5_report import run;result=run(config,args.run_dir)
    elif args.stage=="tests":result={"status":"targeted_tests_completed_by_job_wrapper"}
    else:raise ValueError(args.stage)
    args.run_dir.mkdir(parents=True,exist_ok=True);(args.run_dir/"result_summary.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
