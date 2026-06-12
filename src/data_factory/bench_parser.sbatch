#!/bin/bash
#SBATCH --job-name=rlpt-bench-parser
#SBATCH --partition=public
#SBATCH --gres=gpu:a100.20gb:1
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=/scratch/sghos104/rlpt/data/logs/bench-parser-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/data/logs/bench-parser-%j.err

# Benchmark caption->scene-graph parsers vs VG's human-gold region graphs
# (attributes scoped to the caption = fair). flan-t5 base vs large, on GPU.

set -euo pipefail
mkdir -p /scratch/sghos104/rlpt/data/logs

module load mamba/latest
ENV_PREFIX=/scratch/sghos104/envs/rlpt-data
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
export NLTK_DATA=/scratch/sghos104/rlpt/data/nltk_data

cd /scratch/sghos104/rlpt
echo "[sbatch] $(date -Iseconds) on $(hostname)"
nvidia-smi -L

for MODEL in lizhuang144/flan-t5-base-VG-factual-sg lizhuang144/flan-t5-large-VG-factual-sg; do
  echo; echo "================ $MODEL ================"
  "$ENV_PREFIX/bin/python" tools/eval_parser_on_vg.py \
      --limit 800 --n 2000 --model "$MODEL" --device cuda --beam 5 --batch-size 128 \
      2>&1 | grep -vE "it/s|Loading weights|move_cache|migrat|Warning|warning"
done
echo "[sbatch] done $(date -Iseconds)"
