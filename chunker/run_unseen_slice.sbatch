#!/bin/bash
#SBATCH --job-name=chunker_unseen
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
#SBATCH --output=/scratch/sghos104/rlpt/chunker/logs/slurm-unseen-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/chunker/logs/slurm-unseen-%j.out

# Reconstruct which train windows the step-2000 best checkpoint actually saw, and
# emit an UNSEEN CJK eval slice + a SEEN control set.
# Runs on an A100 deliberately: the sampler permutation is seeded off the global
# torch RNG, so the replay must match the ORIGINAL run's device/setup path as
# closely as possible (the training run was on a GPU node).
set -uo pipefail
REPO=/scratch/sghos104/rlpt/chunker
cd "$REPO" || exit 1
mkdir -p "$REPO/logs"
export PY=$REPO/env/bin/python
export HF_HOME=/scratch/sghos104/.hf_cache
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
unset ROCR_VISIBLE_DEVICES

echo "[unseen] host=$(hostname) job=${SLURM_JOB_ID:-NA} start=$(date)"
nvidia-smi -L

$PY unseen_slice.py --out_dir "$REPO" --data_dir "$REPO/data" --best_step 2000

echo; echo "=========== CJK EVAL: UNSEEN slice ==========="
if [ -s "$REPO/cjk_unseen_traj.jsonl" ]; then
  $PY evaluate.py --model_dir "$REPO/runs/full_plain/best" \
    --test_traj "$REPO/cjk_unseen_traj.jsonl" \
    --out "$REPO/runs/full_plain/eval_cjk_unseen.json"
else
  echo "(no unseen CJK trajectories -- slice empty)"
fi

echo; echo "=========== CJK EVAL: SEEN control (memorization check) ==========="
if [ -s "$REPO/cjk_seen_traj.jsonl" ]; then
  $PY evaluate.py --model_dir "$REPO/runs/full_plain/best" \
    --test_traj "$REPO/cjk_seen_traj.jsonl" \
    --out "$REPO/runs/full_plain/eval_cjk_seen.json"
else
  echo "(no seen CJK trajectories)"
fi

echo "[unseen] done $(date)"
