#!/bin/bash
#SBATCH --job-name=rlpt-merge-all
#SBATCH --partition=htc
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:30:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/sghos104/rlpt/data/logs/merge-all-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/data/logs/merge-all-%j.err

# Merge the remaining FSDP-sharded actor checkpoints (steps 40-160) into portable
# HF models for the validation plan's per-checkpoint generation pass. Steps 20 and
# 176 are already merged. CPU-only; skips any step whose merged output exists.
set -uo pipefail
mkdir -p /scratch/sghos104/rlpt/data/logs /scratch/sghos104/rlpt/models

module load mamba/latest
ENV_PREFIX=/scratch/sghos104/envs/rlpt-train
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONWARNINGS=ignore
export PYTHONPATH=/scratch/sghos104/rlpt/src/train/flash_attn_shim:${PYTHONPATH:-}
unset ROCR_VISIBLE_DEVICES

cd /scratch/sghos104/rlpt
echo "[merge-all] $(date -Iseconds) on $(hostname)"

CKROOT=/scratch/sghos104/rlpt/checkpoints/rlpt-scienceqa/grpo-qwen2_5vl-3b
FAILED=0

for STEP in 40 60 80 100 120 140 160; do
  OUT=/scratch/sghos104/rlpt/models/qwen2_5vl-3b-scienceqa-grpo-step${STEP}
  if [ -f "$OUT/model.safetensors.index.json" ]; then
    echo "=== step $STEP already merged, skipping ==="
    continue
  fi
  echo "=== merging step $STEP $(date -Iseconds) ==="
  "$ENV_PREFIX/bin/python" -m verl.model_merger merge \
      --backend fsdp \
      --local_dir "$CKROOT/global_step_${STEP}/actor" \
      --target_dir "$OUT" \
      --trust-remote-code \
      --use_cpu_initialization \
    || { echo "  !! step $STEP merge FAILED"; FAILED=1; }
done

echo "=== merged models ==="
ls -d /scratch/sghos104/rlpt/models/qwen2_5vl-3b-scienceqa-grpo-step* 2>/dev/null
echo "[merge-all] done $(date -Iseconds) (failed=$FAILED)"
exit $FAILED
