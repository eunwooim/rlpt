#!/bin/bash
#SBATCH --job-name=qgen
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/sghos104/rlpt/granularity/logs/slurm-qgen-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/granularity/logs/slurm-qgen-%j.out
# Granularity generation. Parameterized via --export:
#   MODEL      repo id or snapshot path (required)
#   TAG        output suffix -> qwen_gen_${TAG}.jsonl (required)
#   CONDITIONS comma list (default: all four)
#   LIMIT      first N questions (default 0 = all 200)
# Smoke example:
#   sbatch --export=ALL,MODEL=Qwen/Qwen2.5-VL-3B-Instruct,TAG=smoke3b,\
# CONDITIONS=A_unconstrained,LIMIT=8 --time=00:40:00 run_qwen_gen.sbatch
set -uo pipefail
cd /scratch/sghos104/rlpt/granularity || exit 1
unset ROCR_VISIBLE_DEVICES
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
echo "[qgen] host=$(hostname) job=${SLURM_JOB_ID:-NA} start=$(date)"
echo "[qgen] MODEL=${MODEL} TAG=${TAG} CONDITIONS=${CONDITIONS:-all} LIMIT=${LIMIT:-0}"
nvidia-smi -L
/scratch/sghos104/envs/rlpt-train/bin/python qwen_generate.py \
    --model_path "${MODEL}" \
    --tag "${TAG}" \
    ${CONDITIONS:+--conditions "${CONDITIONS}"} \
    --limit "${LIMIT:-0}"
RC=$?
echo "[qgen] rc=$RC end=$(date)"
exit "$RC"
