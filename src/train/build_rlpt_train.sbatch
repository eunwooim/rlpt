#!/bin/bash
#SBATCH --job-name=rlpt-build-env
#SBATCH --partition=htc
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/sghos104/rlpt/data/logs/build-env-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/data/logs/build-env-%j.err

# Rebuild rlpt-train with verl-SANCTIONED PINS (the tested Qwen2.5-VL combo).
# The previous "latest-everything" build resolved to torch 2.11/cu130 +
# transformers 5.10 and HUNG verl's Ray workers on start. Anchor on vllm==0.8.5,
# which pins torch 2.6.0 (cu124) + transformers 4.51.x; install verl on top.
set -uo pipefail
mkdir -p /scratch/sghos104/rlpt/data/logs

module load mamba/latest
ENV=/scratch/sghos104/envs/rlpt-train
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
export NLTK_DATA=/scratch/sghos104/rlpt/data/nltk_data
export PIP_NO_INPUT=1 PIP_DISABLE_PIP_VERSION_CHECK=1

echo "[build] $(date -Iseconds) on $(hostname)"; nvidia-smi -L || true

# wipe the broken env and recreate
rm -rf "$ENV"
mamba create -y -p "$ENV" python=3.11
PIP="$ENV/bin/pip"; PY="$ENV/bin/python"

set -x
$PIP install --upgrade pip wheel setuptools
# 1) anchor: vllm 0.10.x has the async `run_headless` API verl 0.8.0 needs (verl 0.8.0
#    removed sync rollout). vllm pulls a matching torch (2.7.x) + transformers (>=4.53).
$PIP install "vllm==0.10.1.1"
# 2) verl on top (range: vllm>=0.8.5,<=0.12.0)
$PIP install "verl"
# pin transformers to 4.5x: vllm only floors it (>=4.55) and pip picks 5.x, whose
# Qwen2.5-VL rope_scaling format breaks vllm's ModelConfig ("rope_type vs type=mrope").
$PIP install "transformers==4.56.2"
$PIP install accelerate ray datasets pyarrow pandas pillow qwen-vl-utils
# 3) NO flash-attn. The torch-2.7 flash_attn wheels (2.8.x) are built against glibc 2.32,
#    but Sol nodes are RHEL8 (glibc 2.28) -> the .so won't load ("GLIBC_2.32 not found").
#    So we run attention via SDPA instead (train sbatch sets
#    +actor_rollout_ref.model.override_config.attn_implementation=sdpa and
#    actor_rollout_ref.model.use_remove_padding=False). vLLM uses its own bundled
#    kernels for rollout, so flash_attn is not needed. (To use flash-attn later, build
#    it from source on a GPU node against the node glibc.)
# 4) reward-side deps (compute_score runs inside verl workers)
$PIP install sentence-transformers "spacy>=3.8" scipy
$PY -m spacy download en_core_web_sm
$PY - <<'PYEOF'
import os, nltk
d = os.environ["NLTK_DATA"]; os.makedirs(d, exist_ok=True)
for p in ("punkt","punkt_tab"): nltk.download(p, download_dir=d, quiet=True)
print("nltk ok")
PYEOF
set +x

echo "===== IMPORT SANITY ====="
$PY - <<'PYEOF'
import importlib.util as u
for m in ["torch","vllm","verl","ray","transformers","sentence_transformers","spacy","scipy","flash_attn"]:
    print(f"  {m}: {'OK' if u.find_spec(m) else 'MISSING'}")
import torch, transformers, vllm
print("  torch", torch.__version__, "| cuda", torch.version.cuda, "| avail", torch.cuda.is_available())
print("  transformers", transformers.__version__, "| vllm", vllm.__version__)
try:
    from vllm.entrypoints.cli.serve import run_headless  # the API verl 0.8.0 async needs
    print("  run_headless: OK (async rollout supported)")
except Exception as e:
    print("  run_headless: MISSING ->", e)
PYEOF
echo "[build] done $(date -Iseconds)"
