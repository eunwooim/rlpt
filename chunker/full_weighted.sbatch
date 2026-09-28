#!/bin/bash
#SBATCH --job-name=chunker_weighted
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/sghos104/rlpt/chunker/logs/slurm-full_weighted-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/chunker/logs/slurm-full_weighted-%j.out

# ARM B: WEIGHTED cross-entropy (split-class weight 6.85 = the train 1:0 ratio).
# fp32 (TF32) for DeBERTa-v3 stability. Auto-resumes; after training runs the full
# validation summary (no test eval). Identical to full_plain.sbatch except
# --pos_weight.
set -uo pipefail
REPO=/scratch/sghos104/rlpt/chunker
cd "$REPO" || exit 1
mkdir -p "$REPO/logs"
export PY=/scratch/sghos104/rlpt/chunker/env/bin/python
export HF_HOME=/scratch/sghos104/.hf_cache
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
unset ROCR_VISIBLE_DEVICES

echo "[full_weighted] host=$(hostname) job=${SLURM_JOB_ID:-NA} start=$(date)"
nvidia-smi -L

$PY train.py \
  --data_dir "$REPO/data" \
  --output_dir "$REPO/runs/full_weighted" \
  --precision fp32 --pos_weight 6.85 \
  --epochs 3 --lr 3e-5 --warmup_ratio 0.05 --weight_decay 0.01 \
  --per_device_batch 64 --grad_accum 2 \
  --eval_steps 1000 --save_steps 1000 --eval_subset 10000 --patience 5 --seed 42

echo "[full_weighted] training done $(date); running validation summary"
$PY summarize_val.py --run_dir "$REPO/runs/full_weighted"
echo "[full_weighted] done $(date)"
