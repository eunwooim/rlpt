#!/bin/bash
#SBATCH --job-name=eval1
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/sghos104/rlpt/chunker/logs/eval1_%j.log
# Usage: sbatch eval_one.sbatch <arm> <mode>   mode = eval | sweep
# VAL ONLY.
set -euo pipefail
cd /scratch/sghos104/rlpt/chunker
export HF_HOME=/scratch/sghos104/.hf_cache
PY=./env/bin/python
ARM=$1; MODE=$2
echo "=== $ARM / $MODE  $(hostname)  $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader
if [ "$MODE" = "eval" ]; then
  $PY evaluate.py \
      --model_dir runs/arm_$ARM/best \
      --test_traj data/val_traj.jsonl \
      --out runs/arm_$ARM/eval_val.json \
      --min_tokens 8 --max_tokens 220
else
  $PY threshold_sweep.py \
      --model_dir runs/arm_$ARM/best \
      --val_traj data/val_traj.jsonl \
      --out runs/arm_$ARM/sweep_val.json \
      --segment_stats runs/arm_$ARM/segment_stats.json \
      --min_tokens 8 --max_tokens 220
fi
echo "=== $ARM / $MODE done $(date)"
