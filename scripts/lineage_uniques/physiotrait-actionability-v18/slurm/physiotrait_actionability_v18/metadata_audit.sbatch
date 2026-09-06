#!/usr/bin/env bash
#SBATCH --job-name=trait18_meta
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/denoiseNet_physiotrait_actionability_v18/.slurm/physiotrait_actionability_v18/%j.out
#SBATCH --error=/home/infres/yinwang/denoiseNet_physiotrait_actionability_v18/.slurm/physiotrait_actionability_v18/%j.err
set -euo pipefail

repo=/home/infres/yinwang/denoiseNet_physiotrait_actionability_v18
data=/projects/EEG-foundation-model/Longitudinal_ERP_Brainprint
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate /home/infres/yinwang/anaconda3/envs/eeg2025
cd "$repo"
mkdir -p "$repo/.slurm/physiotrait_actionability_v18"
python - "$data" <<'PY'
from pathlib import Path
import sys
import h5py
import numpy as np

root=Path(sys.argv[1])
print("DATA_ROOT",root,"EXISTS",root.exists())
for path in sorted(root.rglob("*")):
    if path.is_file() and path.suffix.lower() in {".txt",".md",".pdf",".py",".m"}:
        print("DOC",path.relative_to(root),path.stat().st_size)
sample=root/"files"/"S1.mat"
with h5py.File(sample,"r") as handle:
    print("TOP_LEVEL",sorted(handle.keys()))
    for key in sorted(handle.keys()):
        obj=handle[key]
        print("OBJECT",key,type(obj).__name__,getattr(obj,"shape",None),getattr(obj,"dtype",None))
    # Shapes and reference counts only. Never dereference Day_7/Day_80/Day_200 signals.
    for key in ("Day_1","Day_7","Day_80","Day_200","GroupB"):
        if key in handle:
            refs=np.asarray(handle[key])
            print("REFS",key,refs.shape,refs.size)
PY

for doc in $(find "$data" -type f \( -name '*.txt' -o -name '*.md' -o -name '*.py' -o -name '*.m' \) | head -n 30); do
    printf '\n===== %s =====\n' "$doc"
    sed -n '1,240p' "$doc" || true
done
