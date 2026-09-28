#!/bin/bash
#SBATCH --job-name=chunker_finaleval
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
#SBATCH --output=/scratch/sghos104/rlpt/chunker/logs/slurm-finaleval-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/chunker/logs/slurm-finaleval-%j.out

# FINAL TEST EVAL on the winning arm (plain / Arm A), plus the held-out chunker
# demo including a space-free CJK text. This is the ONLY test-set eval run.
set -uo pipefail
REPO=/scratch/sghos104/rlpt/chunker
cd "$REPO" || exit 1
mkdir -p "$REPO/logs"
export PY=$REPO/env/bin/python
export HF_HOME=/scratch/sghos104/.hf_cache
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
unset ROCR_VISIBLE_DEVICES

WINNER=$REPO/runs/full_plain/best

echo "[finaleval] host=$(hostname) job=${SLURM_JOB_ID:-NA} start=$(date)"
nvidia-smi -L
echo "[finaleval] winner=$WINNER"
cat "$WINNER/chunker_meta.json" 2>/dev/null || echo "(no chunker_meta.json)"

echo; echo "=========== TEST-SET EVAL (token F1 by language, over/under-split, frac_under_min) ==========="
$PY evaluate.py --model_dir "$WINNER" \
  --test_traj "$REPO/data/test_traj.jsonl" \
  --out "$REPO/runs/full_plain/eval_test.json"

echo; echo "=========== CHUNKER DEMO (5 held-out trajectories, CJK included) ==========="
$PY demo_final.py --model_dir "$WINNER" \
  --traj "$REPO/data/test_traj.jsonl" --n 5

echo "[finaleval] done $(date)"
