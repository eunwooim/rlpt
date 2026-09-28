#!/bin/bash
#SBATCH --job-name=xreseg-arm
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/sghos104/rlpt/chunk_eval/xdataset/logs/slurm-xreseg-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/chunk_eval/xdataset/logs/slurm-xreseg-%j.out
# Usage: sbatch --export=ALL,TAG=...,MODEL_DIR=...,THR=... run_xreseg_arm.sbatch
set -uo pipefail
cd /scratch/sghos104/rlpt/chunk_eval/xdataset || exit 1
unset ROCR_VISIBLE_DEVICES
export HF_HOME=/scratch/sghos104/.hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "[xreseg] host=$(hostname) job=${SLURM_JOB_ID:-NA} tag=${TAG} model=${MODEL_DIR:-default} thr=${THR:-0.35} start=$(date)"
nvidia-smi -L
ARGS=(--tag "${TAG}" --threshold "${THR:-0.35}" --dp_workers 12)
[ -n "${MODEL_DIR:-}" ] && ARGS+=(--model_dir "${MODEL_DIR}")
/scratch/sghos104/rlpt/chunker/env/bin/python xdataset_resegment.py "${ARGS[@]}"
RC=$?
echo "[xreseg] rc=$RC end=$(date)"
exit "$RC"
