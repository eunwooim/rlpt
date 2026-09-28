#!/bin/bash
#SBATCH --job-name=eval-arms
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/sghos104/rlpt/chunker/logs/eval_arms_%j.log
# VAL ONLY. test_traj.jsonl is not referenced anywhere in this script.
set -euo pipefail
cd /scratch/sghos104/rlpt/chunker
export HF_HOME=/scratch/sghos104/.hf_cache
PY=./env/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for ARM in keep drop clean mask; do
  echo "=== $ARM  $(date)"
  $PY evaluate.py \
      --model_dir runs/arm_$ARM/best \
      --test_traj data/val_traj.jsonl \
      --out runs/arm_$ARM/eval_val.json \
      --min_tokens 8 --max_tokens 220
  $PY threshold_sweep.py \
      --model_dir runs/arm_$ARM/best \
      --val_traj data/val_traj.jsonl \
      --out runs/arm_$ARM/sweep_val.json \
      --segment_stats runs/arm_$ARM/segment_stats.json \
      --min_tokens 8 --max_tokens 220
done
echo "=== all arms done $(date)"
