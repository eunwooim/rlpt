#!/bin/bash
#SBATCH --job-name=rlpt-score-valid
#SBATCH --partition=htc
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/sghos104/rlpt/data/logs/score-valid-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/data/logs/score-valid-%j.err

# Score the validation generation passes (tests 1-4): own-GT vs same-topic-
# shuffled-GT bipartite match, NLI entailment + alternate-encoder re-scoring,
# image-swap deltas, A-OKVQA transfer accuracy. CPU-only.
set -uo pipefail
mkdir -p /scratch/sghos104/rlpt/data/logs

module load mamba/latest
ENV_PREFIX=/scratch/sghos104/envs/rlpt-train
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NLTK_DATA=/scratch/sghos104/rlpt/data/nltk_data
export PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export RLPT_REWARD_DEVICE=cpu
export OMP_NUM_THREADS=16
unset ROCR_VISIBLE_DEVICES

cd /scratch/sghos104/rlpt
echo "[score-valid] $(date -Iseconds) on $(hostname)"

"$ENV_PREFIX/bin/python" tools/validation_scoring.py

echo "=== RESULTS ==="
for t in main swap aokvqa; do
  echo "--- validation_$t.tsv ---"
  cat "data/logs/validation_results/validation_$t.tsv" 2>/dev/null
done
echo "[score-valid] done $(date -Iseconds)"
