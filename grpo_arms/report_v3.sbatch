#!/bin/bash
#SBATCH --job-name=v3_report
#SBATCH --account=grp_vgupt140
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/sghos104/rlpt/grpo_arms/logs/report_v3-%j.log
# CPU post-job: after all v3 eval jobs (afterany), write grpo_arms/V3_REPORT.md and refresh the live monitor section.
set -uo pipefail
cd /scratch/sghos104/rlpt || exit 1
PY=/scratch/sghos104/rlpt/chunker/env/bin/python
"$PY" grpo_arms/monitor_v3.py > /dev/null 2>&1
"$PY" grpo_arms/report_v3.py; rc=$?
{ echo; echo "### V3_REPORT.md written by job $SLURM_JOB_ID at $(date) (rc=$rc)"; } >> grpo_arms/V3_STATUS.md
echo "[report_v3] rc=$rc end=$(date)"; exit $rc
