#!/bin/bash
#SBATCH --job-name=xdl
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/sghos104/rlpt/chunk_eval/xdataset/logs/slurm-xdl-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/chunk_eval/xdataset/logs/slurm-xdl-%j.out
# Acquire PRM800K + ProcessBench (eval-only) into data/xdataset_eval/.
set -uo pipefail
cd /scratch/sghos104/rlpt/chunk_eval/xdataset || exit 1
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
export PYTHONUNBUFFERED=1
echo "[xdl] host=$(hostname) job=${SLURM_JOB_ID:-NA} start=$(date)"
/scratch/sghos104/envs/rlpt-train/bin/python download_xdataset.py
RC=$?
echo "[xdl] rc=$RC end=$(date)"
exit "$RC"
