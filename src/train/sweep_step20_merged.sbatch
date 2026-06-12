#!/bin/bash
#SBATCH --job-name=rlpt-step20-eval
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/sghos104/rlpt/data/logs/step20-eval-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/data/logs/step20-eval-%j.err

# The step-20 sweep point, take 3. FSDP resume needs the same world size the
# checkpoint was saved with (2 GPUs), but the cluster's 2-GPU allocations are
# currently launch-failing. Workaround: merge the step-20 shards to a standalone
# HF model (CPU-only), then val_only-eval it on 1 GPU via model.path (no resume).
# --no-requeue: on launch failure, die instead of sitting in REQUEUE_HOLD.
set -uo pipefail
mkdir -p /scratch/sghos104/rlpt/data/logs /scratch/sghos104/rlpt/models

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
echo "[step20-eval] $(date -Iseconds) on $(hostname)"; nvidia-smi -L

PY="$ENV_PREFIX/bin/python"
DATA=/scratch/sghos104/rlpt/data/scienceqa_verl
REWARD=/scratch/sghos104/rlpt/tools/graph_match_reward.py
SRC=/scratch/sghos104/rlpt/checkpoints/rlpt-scienceqa/grpo-qwen2_5vl-3b/global_step_20/actor
MERGED=/scratch/sghos104/rlpt/models/qwen2_5vl-3b-scienceqa-grpo-step20
RESULTS=/scratch/sghos104/rlpt/data/logs/match_sweep_anchors.tsv

if [ ! -f "$MERGED/model.safetensors.index.json" ]; then
  echo "===== merging step-20 shards $(date -Iseconds) ====="
  "$PY" -m verl.model_merger merge \
      --backend fsdp \
      --local_dir "$SRC" \
      --target_dir "$MERGED" \
      --trust-remote-code \
      --use_cpu_initialization
fi
ls -la "$MERGED"

LOG=/scratch/sghos104/rlpt/data/logs/sweep_20_merged.log
echo "===== eval step-20 (merged) $(date -Iseconds) ====="
"$PY" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo ray_kwargs.ray_init.num_cpus=16 \
    data.train_files="$DATA/test_sweep256.parquet" data.val_files="$DATA/test_sweep256.parquet" \
    data.image_key=images data.max_prompt_length=1280 data.max_response_length=1024 \
    data.train_batch_size=64 data.filter_overlong_prompts=true \
    actor_rollout_ref.model.path="$MERGED" actor_rollout_ref.model.use_remove_padding=False \
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
    trainer.project_name=rlpt-scienceqa trainer.experiment_name=sweep-20-merged \
    trainer.val_only=true trainer.resume_mode=disable \
    > "$LOG" 2>&1 || echo "  (verl exited nonzero — check $LOG)"

get() { grep -oE "val-[a-z]+/scienceqa_composite/$1/mean@1:[0-9.eE+-]*" "$LOG" 2>/dev/null | tail -1 | sed 's/.*://'; }
printf "%s\t%s\t%s\t%s\t%s\n" "20" "$(get acc)" "$(get match)" "$(get format)" "$(get reward)" >> "$RESULTS"
echo "===== ANCHORS ====="; cat "$RESULTS"
echo "[step20-eval] done $(date -Iseconds)"
