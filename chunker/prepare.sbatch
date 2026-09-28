#!/bin/bash
#SBATCH --job-name=chunker_prep
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/sghos104/rlpt/chunker/logs/slurm-prep-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/chunker/logs/slurm-prep-%j.out

# CPU-only data prep (no GPU). The interactive Claude node is capped at 4 GB and
# OOMs; this job gets 64 GB. Tokenization is single-threaded per trajectory.
set -uo pipefail
REPO=/scratch/sghos104/rlpt/chunker
cd "$REPO" || exit 1
mkdir -p "$REPO/logs"

export PY=/scratch/sghos104/rlpt/chunker/env/bin/python
export HF_HOME=/scratch/sghos104/.hf_cache
export HF_DATASETS_CACHE=/scratch/sghos104/.hf_cache/datasets
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "[prep] host=$(hostname) job=${SLURM_JOB_ID:-NA} start=$(date) mem=${SLURM_MEM_PER_NODE:-NA}"
MODE="${1:-full}"
if [ "$MODE" = "smoke" ]; then
  $PY prepare_data.py --out_dir "$REPO/data_smoke" --smoke
else
  $PY prepare_data.py --out_dir "$REPO/data"
fi
echo "[prep] done $(date)"
