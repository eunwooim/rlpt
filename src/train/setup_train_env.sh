#!/bin/bash
# Build the rlpt-train conda env (RL stack + reward deps). Run on a GPU node
# (flash-attn compiles against CUDA), NOT the login node. Heavy + version-sensitive.
#
#   srun -p public --gres=gpu:a100.80gb:1 -c 8 --mem=64G -t 02:00:00 --pty bash
#   bash src/train/setup_train_env.sh
#
# VERIFY torch/CUDA/vllm/flash-attn pins against Sol before trusting this.

set -euo pipefail
module load mamba/latest

ENV=/scratch/sghos104/envs/rlpt-train
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
export NLTK_DATA=/scratch/sghos104/rlpt/data/nltk_data

mamba create -y -p "$ENV" python=3.11
source activate "$ENV"

# torch first (cu124 to match existing env), then the stack
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r /scratch/sghos104/rlpt/src/train/requirements-train.txt

# reward-side model assets
python -m spacy download en_core_web_sm
python - <<'PY'
import os, nltk
d = os.environ["NLTK_DATA"]; os.makedirs(d, exist_ok=True)
for p in ("punkt", "punkt_tab"):
    nltk.download(p, download_dir=d, quiet=True)
print("nltk data ready")
PY

# sanity
python - <<'PY'
import importlib.util as u
for m in ["verl","vllm","ray","transformers","sentence_transformers","spacy","scipy"]:
    print(f"{m}: {'OK' if u.find_spec(m) else 'MISSING'}")
import transformers; print("transformers", transformers.__version__)
PY
echo "[setup] rlpt-train ready at $ENV"
