#!/bin/bash
#SBATCH --job-name=rlpt-merge-nomatch
#SBATCH --partition=htc
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:30:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/sghos104/rlpt/data/logs/merge-nomatch-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/data/logs/merge-nomatch-%j.err

# Merge every FSDP checkpoint of the w_match=0 ablation run into HF models, for
# the cross-arm match-curve comparison (validation test 5). Tolerant of a partial
# final checkpoint if the training job hit its time limit mid-save.
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
echo "[merge-nomatch] $(date -Iseconds) on $(hostname)"

CKROOT=/scratch/sghos104/rlpt/checkpoints/rlpt-scienceqa/grpo-qwen2_5vl-3b-nomatch
FAILED=0

for DIR in "$CKROOT"/global_step_*; do
  [ -d "$DIR/actor" ] || continue
  STEP=${DIR##*global_step_}
  OUT=/scratch/sghos104/rlpt/models/qwen2_5vl-3b-scienceqa-nomatch-step${STEP}
  if [ -f "$OUT/model.safetensors.index.json" ]; then
    echo "=== step $STEP already merged, skipping ==="
    continue
  fi
  echo "=== merging nomatch step $STEP $(date -Iseconds) ==="
  "$ENV_PREFIX/bin/python" -m verl.model_merger merge \
      --backend fsdp \
      --local_dir "$DIR/actor" \
      --target_dir "$OUT" \
      --trust-remote-code \
      --use_cpu_initialization \
    || { echo "  !! step $STEP merge FAILED (partial checkpoint?)"; FAILED=1; }
done

echo "=== merged ablation models ==="
ls -d /scratch/sghos104/rlpt/models/qwen2_5vl-3b-scienceqa-nomatch-step* 2>/dev/null
echo "[merge-nomatch] done $(date -Iseconds) (failed=$FAILED)"
