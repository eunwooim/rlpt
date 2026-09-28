#!/bin/bash
#SBATCH --job-name=redteam_final
#SBATCH --account=grp_vgupt140
#SBATCH --partition=htc
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
OUT="${OUT:-probe_scored_final.jsonl}"
echo "[redteam] start=$(date) PUN_TOL=$(grep -oE '^PUN_TOL *= *[0-9.]+' reward_v2.py) out=$OUT"
/scratch/sghos104/rlpt/chunker/env/bin/python redteam_reward.py --in probe.jsonl --out "$OUT" --skip_old
RC=$?
echo "[redteam] rc=$RC end=$(date)"
exit $RC
