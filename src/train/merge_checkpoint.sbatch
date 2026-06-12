#!/bin/bash
#SBATCH --job-name=rlpt-merge-ckpt
#SBATCH --partition=htc
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/sghos104/rlpt/data/logs/merge-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/data/logs/merge-%j.err

# Consolidate the FSDP-sharded actor checkpoint (global_step_176) into a portable
# Hugging Face model (safetensors + config/tokenizer) that can be loaded with
# transformers/vLLM for inference, eval, or sharing. CPU-only.
set -uo pipefail
mkdir -p /scratch/sghos104/rlpt/data/logs /scratch/sghos104/rlpt/models

module load mamba/latest
ENV_PREFIX=/scratch/sghos104/envs/rlpt-train
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
export PYTHONWARNINGS=ignore
export PYTHONPATH=/scratch/sghos104/rlpt/src/train/flash_attn_shim:${PYTHONPATH:-}
unset ROCR_VISIBLE_DEVICES

cd /scratch/sghos104/rlpt
echo "[merge] $(date -Iseconds) on $(hostname)"

SRC=/scratch/sghos104/rlpt/checkpoints/rlpt-scienceqa/grpo-qwen2_5vl-3b/global_step_176/actor
OUT=/scratch/sghos104/rlpt/models/qwen2_5vl-3b-scienceqa-grpo-step176

"$ENV_PREFIX/bin/python" -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$SRC" \
    --target_dir "$OUT" \
    --trust-remote-code \
    --use_cpu_initialization

echo "=== merged model contents ==="
ls -la "$OUT" 2>/dev/null
echo "[merge] done $(date -Iseconds)"
