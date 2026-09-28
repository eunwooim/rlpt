#!/bin/bash
#SBATCH --job-name=qwendl
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/sghos104/rlpt/outputs/slurm-qwendl-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/outputs/slurm-qwendl-%j.out
# Acquire Qwen2.5-VL-3B/7B-Instruct + SHA256 verification against Hub LFS oids.
set -uo pipefail
cd /scratch/sghos104/rlpt || exit 1
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
export PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=0
echo "[qwendl] host=$(hostname) job=${SLURM_JOB_ID:-NA} start=$(date)"
df -h /scratch/sghos104 | tail -1
/scratch/sghos104/envs/rlpt-train/bin/python src/scripts/download_qwen_models.py
RC=$?
echo "[qwendl] rc=$RC end=$(date)"
exit "$RC"
