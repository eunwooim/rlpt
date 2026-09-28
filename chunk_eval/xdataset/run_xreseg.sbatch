#!/bin/bash
#SBATCH --job-name=xreseg
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
# Cross-dataset V2 resegmentation. Parameterized via --export:
#   TAG (default full), PRM_EVERY / PB_EVERY / VP_EVERY (default 1).
# Timing run example:
#   sbatch --export=ALL,TAG=timing2k,PRM_EVERY=40,PB_EVERY=3,VP_EVERY=700 \
#          --time=01:00:00 run_xreseg.sbatch
set -uo pipefail
cd /scratch/sghos104/rlpt/chunk_eval/xdataset || exit 1
unset ROCR_VISIBLE_DEVICES
export HF_HOME=/scratch/sghos104/.hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "[xreseg] host=$(hostname) job=${SLURM_JOB_ID:-NA} start=$(date)"
nvidia-smi -L
/scratch/sghos104/rlpt/chunker/env/bin/python xdataset_resegment.py \
    --tag "${TAG:-full}" \
    --prm_every "${PRM_EVERY:-1}" \
    --pb_every "${PB_EVERY:-1}" \
    --vp_every "${VP_EVERY:-1}" \
    --dp_workers 12
RC=$?
echo "[xreseg] rc=$RC end=$(date)"
exit "$RC"
