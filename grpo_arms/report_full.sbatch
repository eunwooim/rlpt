#!/bin/bash
#SBATCH --job-name=full_report
#SBATCH --account=grp_vgupt140
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/sghos104/rlpt/grpo_arms/logs/report_full-%j.log
# CPU post-job (afterany on both training chains and every eval): write grpo_arms/FULL_REPORT.md.
set -uo pipefail
cd /scratch/sghos104/rlpt || exit 1
PY=/scratch/sghos104/rlpt/chunker/env/bin/python
"$PY" grpo_arms/monitor_full.py --no_actions > /dev/null 2>&1
"$PY" grpo_arms/report_full.py; rc=$?
{ echo; echo "- $(date '+%Y-%m-%d %H:%M') FULL_REPORT.md written by job $SLURM_JOB_ID (rc=$rc)"; } >> grpo_arms/FULL_STATUS.md
[[ $rc -eq 0 ]] && date > grpo_arms/full/ALL_DONE
echo "[report_full] rc=$rc end=$(date)"; exit $rc
