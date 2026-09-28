#!/bin/bash
#SBATCH --job-name=chunker_cjkscan
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/sghos104/rlpt/chunker/logs/slurm-cjkscan-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/chunker/logs/slurm-cjkscan-%j.out

# CPU-only. Measures the CJK multi-step pool and replays prepare_data.py's exact
# seeded split assignment, so we know how many CJK multi-step trajectories could
# ever form an UNSEEN eval slice for the step-2000 checkpoint.
set -uo pipefail
REPO=/scratch/sghos104/rlpt/chunker
cd "$REPO" || exit 1
mkdir -p "$REPO/logs"
export PY=$REPO/env/bin/python
export HF_HOME=/scratch/sghos104/.hf_cache
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "[cjkscan] host=$(hostname) job=${SLURM_JOB_ID:-NA} start=$(date)"
$PY cjk_scan.py --out_dir "$REPO"
echo "[cjkscan] done $(date)"
