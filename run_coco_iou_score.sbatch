#!/bin/bash
#SBATCH --job-name=coco_iou
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/sghos104/rlpt/outputs/slurm-coco_iou-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/outputs/slurm-coco_iou-%j.out

set -uo pipefail
REPO=/scratch/sghos104/rlpt
PY=/scratch/sghos104/envs/rlpt-train/bin/python
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
export TOKENIZERS_PARALLELISM=false
cd "$REPO" || exit 1

echo "[coco_iou] host=$(hostname) job=$SLURM_JOB_ID start=$(date)"
nvidia-smi -L || echo "no GPU visible"
$PY -c "import torch; print('cuda available:', torch.cuda.is_available())"

$PY score_coco_iou.py \
  --pairs data/coco/coco_iou_pairs.jsonl \
  --out src/outputs/coco_iou/scores.jsonl \
  --batch_size 256 \
  --device auto

echo "[coco_iou] scoring done=$(date)"
$PY aggregate_coco_iou.py
echo "[coco_iou] done=$(date)"
