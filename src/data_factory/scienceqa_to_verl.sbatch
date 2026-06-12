#!/bin/bash
#SBATCH --job-name=rlpt-sqa-verl
#SBATCH --partition=htc
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/sghos104/rlpt/data/logs/sqa-verl-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/data/logs/sqa-verl-%j.err

# Format ScienceQA -> verl parquet (+ extract images). CPU only; htc = short queue.
set -euo pipefail
mkdir -p /scratch/sghos104/rlpt/data/logs

module load mamba/latest
ENV_PREFIX=/scratch/sghos104/envs/rlpt-data
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
export PYTHONWARNINGS=ignore

cd /scratch/sghos104/rlpt
echo "[sbatch] $(date -Iseconds) on $(hostname)"

"$ENV_PREFIX/bin/python" -u src/data_factory/scienceqa_to_verl.py \
    --splits train validation test \
    --out-dir /scratch/sghos104/rlpt/data/scienceqa_verl \
    --image-out-dir /scratch/sghos104/rlpt/data/scienceqa_images

echo "[sbatch] done $(date -Iseconds)"
