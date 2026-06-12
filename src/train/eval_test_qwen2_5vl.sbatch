#!/bin/bash
#SBATCH --job-name=rlpt-eval-test
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/sghos104/rlpt/data/logs/eval-test-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/data/logs/eval-test-%j.err

# Evaluate the trained checkpoint (global_step_176) on the TEST split.
# val_only=true -> no training; reports val-core/acc (hard MCQ exact-match), plus
# match (bipartite reasoning F) and format, now that compute_score returns a dict.
set -uo pipefail
mkdir -p /scratch/sghos104/rlpt/data/logs

module load mamba/latest
ENV_PREFIX=/scratch/sghos104/envs/rlpt-train
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
export NLTK_DATA=/scratch/sghos104/rlpt/data/nltk_data
export PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 RLPT_REWARD_DEVICE=cpu VLLM_USE_V1=1
export PYTHONPATH=/scratch/sghos104/rlpt/src/train/flash_attn_shim:${PYTHONPATH:-}
unset ROCR_VISIBLE_DEVICES

cd /scratch/sghos104/rlpt
echo "[eval] $(date -Iseconds) on $(hostname)"; nvidia-smi -L

DATA=/scratch/sghos104/rlpt/data/scienceqa_verl
REWARD=/scratch/sghos104/rlpt/tools/graph_match_reward.py
MODEL=Qwen/Qwen2.5-VL-3B-Instruct
CKPT=/scratch/sghos104/rlpt/checkpoints/rlpt-scienceqa/grpo-qwen2_5vl-3b/global_step_176

"$ENV_PREFIX/bin/python" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    ray_kwargs.ray_init.num_cpus=16 \
    data.train_files="$DATA/test.parquet" \
    data.val_files="$DATA/test.parquet" \
    data.image_key=images \
    data.max_prompt_length=1280 \
    data.max_response_length=1024 \
    data.train_batch_size=64 \
    data.filter_overlong_prompts=true \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    custom_reward_function.path="$REWARD" \
    custom_reward_function.name=compute_score \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.logger=['console'] \
    trainer.project_name=rlpt-scienceqa \
    trainer.experiment_name=eval-test-step176 \
    trainer.val_only=true \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="$CKPT"

echo "[eval] done $(date -Iseconds)"
