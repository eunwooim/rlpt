#!/bin/bash
#SBATCH --job-name=redteam
#SBATCH --account=grp_vgupt140
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/sghos104/rlpt/reward_redesign/redteam-%j.log
set -uo pipefail
cd /scratch/sghos104/rlpt/reward_redesign || exit 1
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
/scratch/sghos104/rlpt/chunker/env/bin/python redteam_reward.py --in probe.jsonl --out probe_scored.jsonl
echo "[redteam] rc=$? end=$(date)"
