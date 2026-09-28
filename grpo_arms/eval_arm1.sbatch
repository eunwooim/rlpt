#!/bin/bash
#SBATCH --job-name=arm1_vpb_eval
#SBATCH --account=grp_vgupt140
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/sghos104/rlpt/grpo_arms/logs/eval_arm1-%j.log
#SBATCH --error=/scratch/sghos104/rlpt/grpo_arms/logs/eval_arm1-%j.log

set -uo pipefail
REPO=/scratch/sghos104/rlpt
cd "$REPO" || exit 1
PY=/scratch/sghos104/envs/rlpt-train/bin/python
export HF_HOME=$REPO/data/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
# vLLM V1 forks EngineCore after the merge step touched CUDA (documented fix
# from eval job 59199137 in the 2x2 campaign)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="$REPO/src/train/flash_attn_shim${PYTHONPATH:+:$PYTHONPATH}"
unset ROCR_VISIBLE_DEVICES

RUN=$REPO/grpo_arms/runs/arm1_3b_match_chunker
echo "[eval] start=$(date)"

# 1. merge FSDP ckpt 183 -> HF
if [ ! -f "$RUN/hf_final/config.json" ]; then
  "$PY" -m verl.model_merger merge --backend fsdp \
    --local_dir "$RUN/checkpoints/global_step_183/actor" \
    --target_dir "$RUN/hf_final" || exit 1
fi
echo "[eval] merged"

# 2. generate on frozen VPB set: arm1 final + base 3B
"$PY" grpo_arms/vpb_generate.py --model_path "$RUN/hf_final" --tag arm1_step183 || exit 1
"$PY" grpo_arms/vpb_generate.py --model_path Qwen/Qwen2.5-VL-3B-Instruct --tag base3b || exit 1

# 3. score
"$PY" grpo_arms/score_vpb.py
echo "[eval] done=$(date)"
