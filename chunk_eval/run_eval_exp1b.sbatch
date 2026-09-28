#!/bin/bash
#SBATCH --job-name=ceval_exp1b
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/sghos104/rlpt/chunk_eval/logs/slurm-exp1b-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/chunk_eval/logs/slurm-exp1b-%j.out
set -uo pipefail
cd /scratch/sghos104/rlpt/chunk_eval || exit 1
export PYTHONUNBUFFERED=1
/scratch/sghos104/envs/rlpt-train/bin/python exp1b_delim_reasons.py
RC=$?
echo "[exp1b] rc=$RC end=$(date)"
exit "$RC"
