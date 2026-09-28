#!/bin/bash
#SBATCH --job-name=probe_gen
#SBATCH --account=grp_vgupt140
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/sghos104/rlpt/grpo_arms/logs/probe_gen-%j.log
set -uo pipefail
REPO=/scratch/sghos104/rlpt
cd "$REPO" || exit 1
export HF_HOME=$REPO/data/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
unset ROCR_VISIBLE_DEVICES
PY=/scratch/sghos104/envs/rlpt-train/bin/python
echo "[probe_gen] MODEL=$MODEL TAG=$TAG start=$(date)"
"$PY" grpo_arms/vpb_generate.py --model_path "$MODEL" --tag "$TAG" \
  --eval_set "$REPO/grpo_arms/data/probe_prompts.jsonl" --outdir "$REPO/grpo_arms/evals"
RC=$?; echo "[probe_gen] rc=$RC end=$(date)"; exit $RC
