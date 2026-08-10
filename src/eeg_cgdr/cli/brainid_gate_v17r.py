from argparse import ArgumentParser
from pathlib import Path

from eeg_cgdr.experiments.brainid_gate_v17r import (
    aggregate_forensic,
    aggregate_inventory,
    aggregate_m1r,
    evaluate_m1r_fold,
    fit_verifier_b_r,
    freeze_protocol,
    gate01r,
    load_config,
    prepare_control,
    prepare_longitudinal,
    run_carrier_forensic,
    train_verifier_a_r,
    write_report,
)


def main() -> None:
    parser=ArgumentParser();parser.add_argument("--config",type=Path,required=True);parser.add_argument("--stage",required=True,choices=["freeze","prepare-long","prepare-control","inventory","forensic","aggregate-forensic","train-a-r","fit-b-r","eval-m1r","aggregate-m1r","gate","report"]);parser.add_argument("--index",type=int,default=0);parser.add_argument("--run-dir",type=Path);args=parser.parse_args();config=load_config(args.config);run=args.run_dir or Path(config["result_root"])/"server_runs"/args.stage/f"task_{args.index:02d}";run.mkdir(parents=True,exist_ok=True)
    result={"freeze":lambda:freeze_protocol(config),"prepare-long":lambda:prepare_longitudinal(config,args.index+1),"prepare-control":lambda:prepare_control(config,args.index+1),"inventory":lambda:aggregate_inventory(config),"forensic":lambda:run_carrier_forensic(config,args.index,run),"aggregate-forensic":lambda:aggregate_forensic(config),"train-a-r":lambda:train_verifier_a_r(config,args.index,run),"fit-b-r":lambda:fit_verifier_b_r(config,args.index,run),"eval-m1r":lambda:evaluate_m1r_fold(config,args.index,run),"aggregate-m1r":lambda:aggregate_m1r(config),"gate":lambda:gate01r(config),"report":lambda:write_report(config)}[args.stage]();print(result)


if __name__=="__main__": main()
