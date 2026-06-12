#!/bin/bash
#SBATCH --job-name=rlpt-vg-images
#SBATCH --partition=htc
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=/scratch/sghos104/rlpt/data/logs/vg-images-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/data/logs/vg-images-%j.err

# Download the ~36k Visual Genome images we don't already have, into the shared
# data/images/ pool (VG ids == GQA ids, so no collisions). Idempotent.

set -euo pipefail
mkdir -p /scratch/sghos104/rlpt/data/logs

module load mamba/latest
ENV_PREFIX=/scratch/sghos104/envs/rlpt-data

cd /scratch/sghos104/rlpt

echo "[sbatch] starting $(date -Iseconds) on $(hostname)"
"$ENV_PREFIX/bin/python" -u tools/fetch_vg_images.py \
    --image-data data/visual_genome/raw/image_data.json \
    --out-dir data/images \
    --workers 24

echo "[sbatch] done $(date -Iseconds)"
echo "[sbatch] images now on disk: $(ls /scratch/sghos104/rlpt/data/images | wc -l)"
