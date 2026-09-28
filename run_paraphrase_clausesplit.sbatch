#!/bin/bash
#SBATCH --job-name=para_clause
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/sghos104/rlpt/outputs/slurm-para_clause-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/outputs/slurm-para_clause-%j.out

set -uo pipefail
REPO=/scratch/sghos104/rlpt
PY=/scratch/sghos104/envs/rlpt-train/bin/python
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd "$REPO" || exit 1

echo "[para_clause] host=$(hostname) job=$SLURM_JOB_ID start=$(date)"
nvidia-smi -L || echo "no GPU visible"

# Stage 1: generate faithful paraphrases (vLLM, frees GPU on exit)
$PY build_paraphrase_pairs.py \
  --pairs data/visualprm400k/pairs.jsonl \
  --out data/visualprm400k/paraphrase_pairs.jsonl \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --n 10000 || { echo "[para_clause] stage1 FAILED"; exit 1; }

# Stage 2: clause-split + min-pool NLI scoring, FPR (false-alarm) on faithful paraphrases
$PY score_paraphrase_clausesplit.py \
  --paraphrase_pairs data/visualprm400k/paraphrase_pairs.jsonl \
  --outdir src/outputs/visualprm_paraphrase_clausesplit \
  --md docs/paraphrase_clausesplit_results.md \
  --batch_size 256 \
  --device auto || { echo "[para_clause] stage2 FAILED"; exit 1; }

echo "[para_clause] done=$(date)"
