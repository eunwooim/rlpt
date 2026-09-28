#!/bin/bash
#SBATCH --job-name=chunker_smoke
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/sghos104/rlpt/chunker/logs/slurm-smoke-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/chunker/logs/slurm-smoke-%j.out

# Smoke train: ~60k train windows (~44k trajectories), 1 epoch, frequent eval, so
# we can confirm the loss falls and the split-F1 eval runs end-to-end BEFORE the
# full run. Not for quality — just pipeline validation.
set -uo pipefail
REPO=/scratch/sghos104/rlpt/chunker
cd "$REPO" || exit 1
mkdir -p "$REPO/logs"

export PY=/scratch/sghos104/rlpt/chunker/env/bin/python
export HF_HOME=/scratch/sghos104/.hf_cache
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
unset ROCR_VISIBLE_DEVICES

echo "[smoke] host=$(hostname) job=${SLURM_JOB_ID:-NA} start=$(date)"
nvidia-smi -L
$PY -c "import torch; print('cuda', torch.cuda.is_available(), '| bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)"

$PY train.py \
  --data_dir "$REPO/data" \
  --output_dir "$REPO/runs/smoke" \
  --max_train_samples 60000 \
  --epochs 1 --lr 3e-5 --warmup_ratio 0.05 --weight_decay 0.01 \
  --per_device_batch 64 --grad_accum 1 \
  --eval_steps 200 --save_steps 400 --eval_subset 4000 --patience 20

echo "[smoke] done $(date)"