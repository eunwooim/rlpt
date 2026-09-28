#!/bin/bash
#SBATCH --job-name=deberta_chunker
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
#SBATCH --output=/scratch/sghos104/rlpt/chunker/logs/slurm-chunker-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/chunker/logs/slurm-chunker-%j.out

# DeBERTa-v3-small chunk-boundary classifier — full finetune.
# Auto-resumes from the latest checkpoint in --output_dir (train.py handles it),
# so requeue/timeout is safe: just re-submit this script and it continues.
set -uo pipefail
REPO=/scratch/sghos104/rlpt/chunker
cd "$REPO" || exit 1
mkdir -p "$REPO/logs"

export PY=/scratch/sghos104/rlpt/chunker/env/bin/python
export HF_HOME=/scratch/sghos104/.hf_cache
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
unset ROCR_VISIBLE_DEVICES

echo "[chunker] host=$(hostname) job=${SLURM_JOB_ID:-NA} start=$(date)"
nvidia-smi -L
$PY -c "import torch; print('cuda', torch.cuda.is_available(), '| bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)"

# Full run. Per-device batch 64 + grad-accum 2 = eff ~128 on one 40GB A100
# (drop to 32/4 if OOM). 3 epochs max with early stopping on val best-F1.
$PY train.py \
  --data_dir "$REPO/data" \
  --output_dir "$REPO/runs/deberta_chunker" \
  --epochs 3 --lr 3e-5 --warmup_ratio 0.05 --weight_decay 0.01 \
  --per_device_batch 64 --grad_accum 2 \
  --eval_steps 2000 --save_steps 2000 --eval_subset 20000 --patience 4

echo "[chunker] done $(date)"
