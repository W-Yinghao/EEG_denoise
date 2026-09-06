from pathlib import Path
import argparse
from eeg_cgdr.experiments.mobile_v5_closure import run_closure
def main()->int:
    p=argparse.ArgumentParser();p.add_argument("--run-dir",type=Path,required=True);a=p.parse_args();run_closure(a.run_dir);return 0
if __name__=="__main__":raise SystemExit(main())

