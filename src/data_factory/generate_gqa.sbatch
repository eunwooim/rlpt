#!/bin/bash
#SBATCH --job-name=rlpt-gqa-gen
#SBATCH --partition=public
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/sghos104/rlpt/data/logs/gqa-gen-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/data/logs/gqa-gen-%j.err

# Full GQA synthetic dataset generation.
#
# What this job does (single process, CPU-only):
#   1. Downloads lmms-lab/GQA train_balanced_images (~25GB) into
#      /scratch/sghos104/rlpt/data/hf_cache/ on first run.
#   2. Loads scene graphs from /scratch/sghos104/rlpt/data/scene_graphs/.
#   3. For each image, generates spatial / size / count comparison records
#      until per-split quotas are met.
#   4. Writes train.jsonl, eval_id.jsonl, eval_ood_template.jsonl,
#      eval_ood_operation.jsonl, eval_all.jsonl into
#      /scratch/sghos104/rlpt/data/qcvsr_gqa_v1/.
#
# Tunables:
#   max_images=None  -> use all overlap of scene_graphs & image dataset
#   seed=42          -> deterministic splits + sampling
#
# Resource notes:
#   - The current generator is single-process. CPU count above is for
#     HF datasets parallel downloads (HF_DATASETS_NUM_PROC) and pyarrow.
#   - 64G mem is generous; the scene graph file is ~600MB JSON and
#     dataset rows hold PIL images one at a time. Cut to 32G if needed.
#   - Disk: expect ~28GB for hf_cache after first download +
#     ~6-10GB extracted .jpg into data/images/ once all balanced images
#     are saved.

set -euo pipefail

mkdir -p /scratch/sghos104/rlpt/data/logs

module load mamba/latest
ENV_PREFIX=/scratch/sghos104/envs/rlpt-data

export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
# Faster HF downloads if hf_transfer is installed; harmless otherwise.
export HF_HUB_ENABLE_HF_TRANSFER=1

cd /scratch/sghos104/rlpt

echo "[sbatch] starting $(date -Iseconds) on $(hostname)"
echo "[sbatch] HF_HOME=$HF_HOME"
df -h /scratch/sghos104 | tail -1

"$ENV_PREFIX/bin/python" -u src/data_factory/gqa.py

echo "[sbatch] done $(date -Iseconds)"
ls -lh /scratch/sghos104/rlpt/data/qcvsr_gqa_v1/
