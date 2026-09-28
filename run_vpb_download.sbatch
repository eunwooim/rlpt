#!/bin/bash
#SBATCH --job-name=vpbdl
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/sghos104/rlpt/outputs/slurm-vpbdl-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/outputs/slurm-vpbdl-%j.out
# Acquire OpenGVLab/VisualProcessBench + SHA256 verification against Hub LFS oids.
set -uo pipefail
cd /scratch/sghos104/rlpt || exit 1
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
export PYTHONUNBUFFERED=1
echo "[vpbdl] host=$(hostname) job=${SLURM_JOB_ID:-NA} start=$(date)"
/scratch/sghos104/envs/rlpt-train/bin/python src/process_reward/download_vpb.py
RC=$?
echo "[vpbdl] rc=$RC end=$(date)"
exit "$RC"
