#!/bin/bash
#SBATCH --job-name=full_monitor
#SBATCH --account=grp_vgupt140
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/sghos104/rlpt/grpo_arms/logs/monitor_full-%j.log
# CPU-only monitor/orchestrator for the FULL runs (see monitor_full.py). Self-chains: near its 24 h wall it exits and, unless
# grpo_arms/full/ALL_DONE exists, submits an identical follow-up job so eval submission / pruning / the final report never depend on a login session.
cd /scratch/sghos104/rlpt || exit 1
echo "[monitor] start=$(date) host=$(hostname) job=$SLURM_JOB_ID"
/scratch/sghos104/rlpt/chunker/env/bin/python grpo_arms/monitor_full.py --loop --interval 1800 --budget_s 84600; rc=$?
if [[ ! -f grpo_arms/full/ALL_DONE ]]; then
  NXT=$(sbatch --parsable --account=grp_vgupt140 grpo_arms/monitor_full.sbatch) && echo "[monitor] follow-up monitor job $NXT"
fi
echo "[monitor] rc=$rc end=$(date)"
