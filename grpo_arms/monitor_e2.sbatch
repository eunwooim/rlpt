#!/bin/bash
#SBATCH --job-name=e2_monitor
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/sghos104/rlpt/grpo_arms/logs/monitor_e2-%j.log
# CPU-only monitor/orchestrator for the FULL runs (see monitor_e2.py). Self-chains: near its 24 h wall it exits and, unless
# grpo_arms/e2/ALL_DONE exists, submits an identical follow-up job so eval submission / pruning / the final report never depend on a login session.
cd /scratch/sghos104/rlpt || exit 1
echo "[monitor] start=$(date) host=$(hostname) job=$SLURM_JOB_ID"
/scratch/sghos104/rlpt/chunker/env/bin/python grpo_arms/monitor_e2.py --loop --interval 1800 --budget_s 84600; rc=$?
if [[ ! -f grpo_arms/e2/ALL_DONE ]]; then
  NXT=$(sbatch --parsable --account=grp_bshettah grpo_arms/monitor_e2.sbatch) && echo "[monitor] follow-up monitor job $NXT"
fi
echo "[monitor] rc=$rc end=$(date)"
