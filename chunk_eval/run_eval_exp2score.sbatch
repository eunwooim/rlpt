#!/bin/bash
#SBATCH --job-name=ceval_exp2s
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/sghos104/rlpt/chunk_eval/logs/slurm-exp2s-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/chunk_eval/logs/slurm-exp2s-%j.out
# Exp 2 GPU scoring — submit ONLY after user go on population examples.
set -uo pipefail
cd /scratch/sghos104/rlpt/chunk_eval || exit 1
unset ROCR_VISIBLE_DEVICES
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
echo "[exp2s] host=$(hostname) job=${SLURM_JOB_ID:-NA} start=$(date)"
nvidia-smi -L
/scratch/sghos104/envs/rlpt-train/bin/python exp2_score.py
RC=$?
echo "[exp2s] rc=$RC end=$(date)"
exit "$RC"
