#!/bin/bash
#SBATCH --job-name=vprm_dl
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/sghos104/rlpt/outputs/slurm-vprm_dl-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/outputs/slurm-vprm_dl-%j.out

set -uo pipefail
PY=/scratch/sghos104/envs/rlpt-train/bin/python
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "[vprm_dl] host=$(hostname) job=$SLURM_JOB_ID start=$(date)"

$PY - <<'EOF'
from pathlib import Path
from huggingface_hub import hf_hub_download

for filename in ("annotations.zip", "images.zip"):
    p = Path(hf_hub_download("OpenGVLab/VisualPRM400K-v1.1-Raw", filename, repo_type="dataset"))
    print(f"DONE {filename} at: {p} ({p.stat().st_size/1e9:.2f} GB)", flush=True)
EOF

echo "[vprm_dl] verifying archive integrity"
SNAP=/scratch/sghos104/rlpt/data/hf_cache/hub/datasets--OpenGVLab--VisualPRM400K-v1.1-Raw/snapshots/650e4b2f2af9754643f91f3eb44e56c373127c80
$PY - <<EOF
import zipfile
for name in ("annotations.zip", "images.zip"):
    with zipfile.ZipFile("$SNAP/" + name) as z:
        bad = z.testzip()
        print(f"{name}: {len(z.namelist())} members, corrupt={bad}", flush=True)
EOF

echo "[vprm_dl] done=$(date)"
