#!/bin/bash
#SBATCH --job-name=dl-vprm
#SBATCH --partition=htc
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/sghos104/rlpt/granularity/dl_visualprm_%j.log
set -euo pipefail
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
/scratch/sghos104/rlpt/chunker/env/bin/python - << 'EOF'
from huggingface_hub import snapshot_download
p = snapshot_download(
    "OpenGVLab/VisualPRM-8B",
    allow_patterns=["*.json", "*.txt", "*.model", "*.py", "*.safetensors"],
    ignore_patterns=["*.h5", "*.msgpack", "*.ot", "*tf_model*", "*flax*"],
)
print("ok:", p)
import os
tot = sum(os.path.getsize(os.path.join(r,f))
          for r,_,fs in os.walk(p) for f in fs)
print(f"size: {tot/1e9:.2f} GB")
EOF
