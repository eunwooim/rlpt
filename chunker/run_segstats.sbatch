#!/bin/bash
#SBATCH --job-name=chunker_segstats
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/sghos104/rlpt/chunker/logs/slurm-segstats-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/chunker/logs/slurm-segstats-%j.out

# Per-trajectory segmentation health (zero-cut rate) + a demo that samples across
# DISTINCT question groups. Aggregate boundary P/R can look healthy while a slice
# of trajectories comes back completely unsegmented; this measures that slice.
set -uo pipefail
REPO=/scratch/sghos104/rlpt/chunker
cd "$REPO" || exit 1
export PY=$REPO/env/bin/python
export HF_HOME=/scratch/sghos104/.hf_cache
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
unset ROCR_VISIBLE_DEVICES
WINNER=$REPO/runs/full_plain/best

echo "[segstats] host=$(hostname) job=${SLURM_JOB_ID:-NA} start=$(date)"
nvidia-smi -L

$PY segment_stats.py --model_dir "$WINNER" --traj "$REPO/data/test_traj.jsonl" \
  --out "$REPO/runs/full_plain/segment_stats.json"

echo; echo "=========== DEMO (question-diverse sample, CJK included) ==========="
$PY demo_final.py --model_dir "$WINNER" --traj "$REPO/data/test_traj.jsonl" --n 5

echo "[segstats] done $(date)"
