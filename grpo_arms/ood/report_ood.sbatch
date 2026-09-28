#!/bin/bash
#SBATCH --job-name=ood_report
#SBATCH --account=grp_bshettah
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/sghos104/rlpt/grpo_arms/logs/ood_report-%j.log
cd /scratch/sghos104/rlpt || exit 1
/scratch/sghos104/rlpt/chunker/env/bin/python grpo_arms/ood/report_ood.py; rc=$?
{ echo; echo "- $(date '+%Y-%m-%d %H:%M') OOD_REPORT.md written by job $SLURM_JOB_ID (rc=$rc)"; } >> grpo_arms/E2_STATUS.md
echo "[ood_report] rc=$rc end=$(date)"; exit $rc
