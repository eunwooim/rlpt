#!/bin/bash
#SBATCH --job-name=ceval_exp2b
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/sghos104/rlpt/chunk_eval/logs/slurm-exp2b-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/chunk_eval/logs/slurm-exp2b-%j.out
set -uo pipefail
cd /scratch/sghos104/rlpt/chunk_eval || exit 1
export PYTHONUNBUFFERED=1
echo "[exp2b] host=$(hostname) job=${SLURM_JOB_ID:-NA} start=$(date)"
/scratch/sghos104/envs/rlpt-train/bin/python exp2_build.py
RC=$?
echo "[exp2b] rc=$RC end=$(date)"
exit "$RC"
