#!/bin/bash
#SBATCH --job-name=v4_monitor
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/sghos104/rlpt/grpo_arms/logs/monitor_v4-%j.log
# CPU-only monitor for the v4 runs: rewrites the "Live monitor" section of grpo_arms/V4_STATUS.md every 30 min.
cd /scratch/sghos104/rlpt || exit 1
echo "[monitor] start=$(date) host=$(hostname)"
/scratch/sghos104/rlpt/chunker/env/bin/python grpo_arms/monitor_v4.py --loop --interval 1800
echo "[monitor] rc=$? end=$(date)"
