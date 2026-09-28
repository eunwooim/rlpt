#!/bin/bash
#SBATCH --job-name=test-final
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/sghos104/rlpt/chunker/logs/test_final_%j.log
# THE authoritative test run. mask arm, threshold 0.40, frozen v2 test split.
# Run once. Do not repeat.
set -euo pipefail
cd /scratch/sghos104/rlpt/chunker
export HF_HOME=/scratch/sghos104/.hf_cache
./env/bin/python evaluate.py \
    --model_dir release/chunker-deberta-v3-small-v2-mask \
    --test_traj data/test_traj.jsonl \
    --out release/chunker-deberta-v3-small-v2-mask/eval_test.json \
    --min_tokens 8 --max_tokens 220
echo "=== test done $(date)"
