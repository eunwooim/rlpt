#!/bin/bash
#SBATCH --job-name=dl-nli
#SBATCH --partition=htc
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/sghos104/rlpt/granularity/dl_nli_%j.log
set -euo pipefail
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
/scratch/sghos104/rlpt/chunker/env/bin/python - << 'EOF'
from huggingface_hub import snapshot_download
p = snapshot_download(
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "pytorch_model.bin"],
    ignore_patterns=["*.h5", "*.msgpack", "*.ot", "*tf_model*", "*flax*"],
)
print("ok:", p)
from transformers import AutoConfig
print("id2label:", AutoConfig.from_pretrained(p).id2label)
EOF
