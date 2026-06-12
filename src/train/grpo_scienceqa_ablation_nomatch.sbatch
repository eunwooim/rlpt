#!/bin/bash
#SBATCH --job-name=rlpt-grpo-nomatch
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/sghos104/rlpt/data/logs/grpo-nomatch-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/data/logs/grpo-nomatch-%j.err

# ABLATION (validation plan test 5, the causal keystone): identical GRPO run to
# grpo_scienceqa_qwen2_5vl.sbatch except the reward drops every bipartite-match-
# derived term: RLPT_W_MATCH=0 (no reasoning-match credit) and RLPT_W_PUN=0 (the
# hallucination penalty is computed FROM the matcher, so it must go too).
# Reward = format + hard answer only. If this run's checkpoints show the same
# reasoning-match climb as the main run, the match term wasn't causal; if they
# don't, it was.
set -euo pipefail
mkdir -p /scratch/sghos104/rlpt/data/logs

module load mamba/latest
ENV_PREFIX=/scratch/sghos104/envs/rlpt-train
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NLTK_DATA=/scratch/sghos104/rlpt/data/nltk_data
export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export RLPT_REWARD_DEVICE=cpu
export VLLM_USE_V1=1
export PYTHONPATH=/scratch/sghos104/rlpt/src/train/flash_attn_shim:${PYTHONPATH:-}
unset ROCR_VISIBLE_DEVICES

# THE ablation: kill both matcher-derived reward terms.
export RLPT_W_MATCH=0
export RLPT_W_PUN=0

cd /scratch/sghos104/rlpt
echo "[ablation-nomatch] $(date -Iseconds) on $(hostname)"
nvidia-smi -L

DATA=/scratch/sghos104/rlpt/data/scienceqa_verl
REWARD=/scratch/sghos104/rlpt/tools/graph_match_reward.py
MODEL=Qwen/Qwen2.5-VL-3B-Instruct
NGPUS=2

"$ENV_PREFIX/bin/python" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    ray_kwargs.ray_init.num_cpus=16 \
    data.train_files="$DATA/train.parquet" \
    data.val_files="$DATA/validation.parquet" \
    data.image_key=images \
    data.max_prompt_length=1280 \
    data.max_response_length=1024 \
    data.train_batch_size=64 \
    data.filter_overlong_prompts=true \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    custom_reward_function.path="$REWARD" \
    custom_reward_function.name=compute_score \
    trainer.n_gpus_per_node=$NGPUS \
    trainer.nnodes=1 \
    trainer.logger=['console'] \
    trainer.project_name=rlpt-scienceqa \
    trainer.experiment_name=grpo-qwen2_5vl-3b-nomatch \
    trainer.save_freq=20 \
    trainer.test_freq=25 \
    trainer.total_epochs=2

echo "[ablation-nomatch] done $(date -Iseconds)"
