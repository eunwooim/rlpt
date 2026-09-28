#!/bin/bash
#SBATCH --job-name=ceval_exp3
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/sghos104/rlpt/chunk_eval/logs/slurm-exp3-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/chunk_eval/logs/slurm-exp3-%j.out
set -uo pipefail
cd /scratch/sghos104/rlpt/chunk_eval || exit 1
export PYTHONUNBUFFERED=1
echo "[exp3] host=$(hostname) job=${SLURM_JOB_ID:-NA} start=$(date)"
/scratch/sghos104/envs/rlpt-train/bin/python exp3_weld.py --workers 14
RC=$?
echo "[exp3] rc=$RC end=$(date)"
exit "$RC"
