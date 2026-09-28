#!/bin/bash
#SBATCH --job-name=arms_preflight
#SBATCH --account=grp_vgupt140
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/sghos104/rlpt/grpo_arms/logs/preflight-%j.log
#SBATCH --error=/scratch/sghos104/rlpt/grpo_arms/logs/preflight-%j.log

set -uo pipefail
REPO=/scratch/sghos104/rlpt
cd "$REPO" || exit 1

export HF_HOME=$REPO/data/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
unset ROCR_VISIBLE_DEVICES

echo "[preflight] host=$(hostname) job=$SLURM_JOB_ID start=$(date)"
nvidia-smi -L

/scratch/sghos104/envs/rlpt-train/bin/python grpo_arms/preflight.py
rc=$?
echo "[preflight] done=$(date) rc=$rc"
exit $rc
