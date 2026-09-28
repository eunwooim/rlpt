#!/bin/bash
#SBATCH --job-name=reliable_nb
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
#SBATCH --output=/scratch/sghos104/rlpt/outputs/slurm-reliable_nb-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/outputs/slurm-reliable_nb-%j.out

set -uo pipefail

REPO=/scratch/sghos104/rlpt
PY=/scratch/sghos104/envs/rlpt-train/bin/python

export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
export TOKENIZERS_PARALLELISM=false

cd "$REPO" || exit 1

echo "[reliable_nb] host=$(hostname) job=$SLURM_JOB_ID start=$(date)"
nvidia-smi -L || echo "no GPU visible"
$PY -c "import torch; print('cuda available:', torch.cuda.is_available())"

# Full run INCLUDING NegBench (CSVs now present under data/negbench/{images,videos}/).
# Uncapped: NO --max_cases_per_subcategory. All three scorers.
$PY src/metrics/reliable/run_experiment.py \
  --datasets sugarcrepe,sugarcrepepp,negbench \
  --scorers sbert,nli,bertscore \
  --output_dir src/outputs/reliable_full_negbench \
  --batch_size 32 \
  --device auto \
  --seed 42

echo "[reliable_nb] done=$(date) exit_collect_below"
echo "=== load_errors ==="; cat src/outputs/reliable_full_negbench/load_errors.jsonl 2>/dev/null | head
echo "=== scorer_errors ==="; cat src/outputs/reliable_full_negbench/scorer_errors.jsonl 2>/dev/null | head
echo "=== subcategory_counts ==="; cat src/outputs/reliable_full_negbench/subcategory_counts.csv 2>/dev/null
echo "=== scorer_overall ==="; cat src/outputs/reliable_full_negbench/table_scorer_overall.csv 2>/dev/null
