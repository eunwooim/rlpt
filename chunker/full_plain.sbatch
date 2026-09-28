#!/bin/bash
#SBATCH --job-name=chunker_plain
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
#SBATCH --output=/scratch/sghos104/rlpt/chunker/logs/slurm-full_plain-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/chunker/logs/slurm-full_plain-%j.out

# ARM A: PLAIN cross-entropy. fp32 (TF32) for DeBERTa-v3 stability. Auto-resumes
# from the newest checkpoint in the output dir; after training, runs the full
# validation summary (no test eval). Identical to full_weighted.sbatch except
# --pos_weight (0 here).
set -uo pipefail
REPO=/scratch/sghos104/rlpt/chunker
cd "$REPO" || exit 1
mkdir -p "$REPO/logs"
export PY=/scratch/sghos104/rlpt/chunker/env/bin/python
export HF_HOME=/scratch/sghos104/.hf_cache
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
unset ROCR_VISIBLE_DEVICES

echo "[full_plain] host=$(hostname) job=${SLURM_JOB_ID:-NA} start=$(date)"
nvidia-smi -L

$PY train.py \
  --data_dir "$REPO/data" \
  --output_dir "$REPO/runs/full_plain" \
  --precision fp32 --pos_weight 0 \
  --epochs 3 --lr 3e-5 --warmup_ratio 0.05 --weight_decay 0.01 \
  --per_device_batch 64 --grad_accum 2 \
  --eval_steps 1000 --save_steps 1000 --eval_subset 10000 --patience 5 --seed 42

echo "[full_plain] training done $(date); running validation summary"
$PY summarize_val.py --run_dir "$REPO/runs/full_plain"
echo "[full_plain] done $(date)"
