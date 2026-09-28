#!/bin/bash
#SBATCH --job-name=arms_tausweep
#SBATCH --account=grp_vgupt140
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/sghos104/rlpt/grpo_arms/logs/tau_sweep-%j.log
#SBATCH --error=/scratch/sghos104/rlpt/grpo_arms/logs/tau_sweep-%j.log

set -uo pipefail
REPO=/scratch/sghos104/rlpt
cd "$REPO" || exit 1

export HF_HOME=$REPO/data/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
unset ROCR_VISIBLE_DEVICES

echo "[tau_sweep] host=$(hostname) job=$SLURM_JOB_ID start=$(date)"
nvidia-smi -L

# PY / PREFIX overridable via --export; default = the scorer env (chunker/env),
# the env that serves the training-time reward — tau must be calibrated there.
PY="${PY:-/scratch/sghos104/rlpt/chunker/env/bin/python}"
PREFIX="${PREFIX:-tau_sweep_scorerenv}"
"$PY" grpo_arms/tau_sweep.py --batch_size 96 --out_prefix "$PREFIX"
rc=$?
echo "[tau_sweep] done=$(date) rc=$rc"
exit $rc
