#!/bin/bash
#SBATCH --job-name=vpb_split
#SBATCH --account=grp_vgupt140
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/sghos104/rlpt/grpo_arms/logs/vpb_split-%j.log
set -uo pipefail
cd /scratch/sghos104/rlpt || exit 1
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
/scratch/sghos104/envs/rlpt-train/bin/python grpo_arms/build_vpb_split.py; RC=$?
ls -la grpo_arms/data/vpb_dev* grpo_arms/data/vpb_test* | head; cat grpo_arms/data/vpb_dev.parquet.sha256
echo "[vpb_split] rc=$RC end=$(date)"; exit $RC
