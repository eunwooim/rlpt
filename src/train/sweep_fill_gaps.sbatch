#!/bin/bash
#SBATCH --job-name=rlpt-sweep-gaps
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/sghos104/rlpt/data/logs/sweep-gaps-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/data/logs/sweep-gaps-%j.err

# Fill the two gaps left by sweep_checkpoints.sbatch:
#   1. UNTRAINED base model anchor on the same 256-row test subset (match floor).
#   2. global_step_20 point (original attempt died on an HF Hub 429 rate limit).
# HF_HUB_OFFLINE avoids the rate limit entirely — everything is in hf_cache.
set -uo pipefail
mkdir -p /scratch/sghos104/rlpt/data/logs

module load mamba/latest
ENV_PREFIX=/scratch/sghos104/envs/rlpt-train
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NLTK_DATA=/scratch/sghos104/rlpt/data/nltk_data
export PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 RLPT_REWARD_DEVICE=cpu VLLM_USE_V1=1
export PYTHONPATH=/scratch/sghos104/rlpt/src/train/flash_attn_shim:${PYTHONPATH:-}
unset ROCR_VISIBLE_DEVICES

cd /scratch/sghos104/rlpt
echo "[sweep-gaps] $(date -Iseconds) on $(hostname)"; nvidia-smi -L

PY="$ENV_PREFIX/bin/python"
DATA=/scratch/sghos104/rlpt/data/scienceqa_verl
REWARD=/scratch/sghos104/rlpt/tools/graph_match_reward.py
MODEL=Qwen/Qwen2.5-VL-3B-Instruct
CKROOT=/scratch/sghos104/rlpt/checkpoints/rlpt-scienceqa/grpo-qwen2_5vl-3b
RESULTS=/scratch/sghos104/rlpt/data/logs/match_sweep_anchors.tsv
printf "step\tacc\tmatch\tformat\treward\n" > "$RESULTS"

run_point() {  # $1 = label (base|step number), $2.. = extra hydra overrides
  local LABEL=$1; shift
  local LOG="/scratch/sghos104/rlpt/data/logs/sweep_${LABEL}.log"
  echo "===== eval $LABEL $(date -Iseconds) ====="
  "$PY" -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo ray_kwargs.ray_init.num_cpus=16 \
      data.train_files="$DATA/test_sweep256.parquet" data.val_files="$DATA/test_sweep256.parquet" \
      data.image_key=images data.max_prompt_length=1280 data.max_response_length=1024 \
      data.train_batch_size=64 data.filter_overlong_prompts=true \
      actor_rollout_ref.model.path="$MODEL" actor_rollout_ref.model.use_remove_padding=False \
      +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
      actor_rollout_ref.actor.ppo_mini_batch_size=16 \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
      actor_rollout_ref.actor.fsdp_config.param_offload=True \
      actor_rollout_ref.ref.fsdp_config.param_offload=True \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
      actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
      custom_reward_function.path="$REWARD" custom_reward_function.name=compute_score \
      trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.logger=['console'] \
      trainer.project_name=rlpt-scienceqa trainer.experiment_name="sweep-$LABEL" \
      trainer.val_only=true "$@" \
      > "$LOG" 2>&1 || echo "  ($LABEL verl exited nonzero — check $LOG)"

  get() { grep -oE "val-[a-z]+/scienceqa_composite/$1/mean@1:[0-9.eE+-]*" "$LOG" 2>/dev/null | tail -1 | sed 's/.*://'; }
  printf "%s\t%s\t%s\t%s\t%s\n" "$LABEL" "$(get acc)" "$(get match)" "$(get format)" "$(get reward)" >> "$RESULTS"
  echo "  -> $(tail -1 "$RESULTS")"
  "$ENV_PREFIX/bin/ray" stop --force >/dev/null 2>&1 || true
  sleep 8
}

run_point base trainer.resume_mode=disable
run_point 20 trainer.resume_mode=resume_path trainer.resume_from_path="$CKROOT/global_step_20"

echo "===== GAP-FILL RESULTS ====="; cat "$RESULTS"
echo "[sweep-gaps] done $(date -Iseconds)"
