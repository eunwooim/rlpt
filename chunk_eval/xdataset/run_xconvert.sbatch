#!/bin/bash
#SBATCH --job-name=xconv
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/sghos104/rlpt/chunk_eval/xdataset/logs/slurm-xconv-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/chunk_eval/xdataset/logs/slurm-xconv-%j.out
# Convert PRM800K + ProcessBench to harness format + convention stats.
set -uo pipefail
cd /scratch/sghos104/rlpt/chunk_eval/xdataset || exit 1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
echo "[xconv] host=$(hostname) job=${SLURM_JOB_ID:-NA} start=$(date)"
/scratch/sghos104/envs/rlpt-train/bin/python convert_xdataset.py
RC=$?
echo "[xconv] rc=$RC end=$(date)"
exit "$RC"
