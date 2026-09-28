#!/bin/bash
#SBATCH --job-name=chunk-arms
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/sghos104/rlpt/chunker/logs/arm_%A_%a.log
# INVARIANTS: val/test symlinked to the frozen v2 split (identical across arms);
# seed 42, fp32, plain CE -- matches the shipped v2 run. Selection on VAL only.
set -euo pipefail
ARMS=(keep drop clean mask)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
ROOT=/scratch/sghos104/rlpt/chunker
PY=$ROOT/env/bin/python
export HF_HOME=/scratch/sghos104/.hf_cache
echo "=== arm=$ARM host=$(hostname) $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY $ROOT/train.py \
    --data_dir $ROOT/data_arms/$ARM \
    --output_dir $ROOT/runs/arm_$ARM \
    --precision fp32 \
    --pos_weight 0.0 \
    --seed 42
echo "=== arm=$ARM done $(date) ==="
