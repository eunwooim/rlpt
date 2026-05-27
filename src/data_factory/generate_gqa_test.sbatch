#!/bin/bash
#SBATCH --job-name=rlpt-gqa-gen-test
#SBATCH --partition=public
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/sghos104/rlpt/data/logs/gqa-gen-test-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/data/logs/gqa-gen-test-%j.err

# Smaller test run of the GQA synthetic dataset generation.
#
# Purpose:
#   - Validate the at-scale pipeline on a compute node before committing to
#     the full 12-hour run.
#   - Populate the HF cache (one-time ~25GB download of lmms-lab/GQA) so the
#     subsequent full run has zero download cost.
#
# Limits this run to the first 2000 images and writes to a separate output
# directory (qcvsr_gqa_v1_test/) so it does not collide with the full run.

set -euo pipefail

mkdir -p /scratch/sghos104/rlpt/data/logs

module load mamba/latest
ENV_PREFIX=/scratch/sghos104/envs/rlpt-data

export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_ENABLE_HF_TRANSFER=1

cd /scratch/sghos104/rlpt

echo "[sbatch] starting $(date -Iseconds) on $(hostname)"
echo "[sbatch] HF_HOME=$HF_HOME"
df -h /scratch/sghos104 | tail -1

"$ENV_PREFIX/bin/python" -u src/data_factory/gqa.py \
    --max-images 2000 \
    --out-dir /scratch/sghos104/rlpt/data/qcvsr_gqa_v1_test/

echo "[sbatch] done $(date -Iseconds)"
ls -lh /scratch/sghos104/rlpt/data/qcvsr_gqa_v1_test/
