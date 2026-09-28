#!/bin/bash
#SBATCH --job-name=reward_reagg
#SBATCH --account=grp_vgupt140
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/sghos104/rlpt/grpo_arms/logs/reward_reagg-%j.log
set -uo pipefail
cd /scratch/sghos104/rlpt || exit 1
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export REAGG_STEPS="${REAGG_STEPS:-40 80 120 160}"
/scratch/sghos104/rlpt/chunker/env/bin/python grpo_arms/reward_reagg.py; rc=$?
echo "[reagg] rc=$rc end=$(date)"; exit $rc
