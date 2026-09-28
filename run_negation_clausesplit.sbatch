#!/bin/bash
#SBATCH --job-name=neg_clause
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/sghos104/rlpt/outputs/slurm-neg_clause-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/outputs/slurm-neg_clause-%j.out

set -uo pipefail
REPO=/scratch/sghos104/rlpt
PY=/scratch/sghos104/envs/rlpt-train/bin/python
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
export TOKENIZERS_PARALLELISM=false
cd "$REPO" || exit 1

echo "[neg_clause] host=$(hostname) job=$SLURM_JOB_ID start=$(date)"
nvidia-smi -L || echo "no GPU visible"
$PY -c "import torch; print('cuda available:', torch.cuda.is_available())"

$PY score_negation_clausesplit.py \
  --pairs data/visualprm400k/pairs.jsonl \
  --paragraph_scores src/outputs/visualprm_negation/scores.jsonl \
  --outdir src/outputs/visualprm_negation_clausesplit \
  --md docs/negation_clausesplit_results.md \
  --n 10000 \
  --batch_size 256 \
  --device auto

echo "[neg_clause] done=$(date)"
