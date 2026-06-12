#!/bin/bash
#SBATCH --job-name=rlpt-grpo-smoke
#SBATCH --partition=htc
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/sghos104/rlpt/data/logs/grpo-smoke-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/data/logs/grpo-smoke-%j.err

# FAST smoke: 64 rows, 1 GPU, tiny batch -> exercises the WHOLE loop (model load,
# vLLM rollout, compute_score reward, 1 optimizer step) and exits in one short epoch.
# Confirms the rebuilt env + reward integration before the full run.
set -uo pipefail
mkdir -p /scratch/sghos104/rlpt/data/logs

module load mamba/latest
ENV_PREFIX=/scratch/sghos104/envs/rlpt-train
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
export NLTK_DATA=/scratch/sghos104/rlpt/data/nltk_data
export PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 RLPT_REWARD_DEVICE=cpu
export VLLM_USE_V1=1         # verl async vLLM server needs the V1 engine
export PYTHONPATH=/scratch/sghos104/rlpt/src/train/flash_attn_shim:${PYTHONPATH:-}  # pure-torch flash_attn.bert_padding shim (no glibc-incompatible CUDA wheel)
unset ROCR_VISIBLE_DEVICES   # Sol sets this AMD var; verl rejects it alongside CUDA_VISIBLE_DEVICES

cd /scratch/sghos104/rlpt
echo "[smoke] $(date -Iseconds) on $(hostname)"; nvidia-smi -L

DATA=/scratch/sghos104/rlpt/data/scienceqa_verl
REWARD=/scratch/sghos104/rlpt/tools/graph_match_reward.py
MODEL=Qwen/Qwen2.5-VL-3B-Instruct

"$ENV_PREFIX/bin/python" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    ray_kwargs.ray_init.num_cpus=12 \
    data.train_files="$DATA/smoke.parquet" \
    data.val_files="$DATA/smoke.parquet" \
    data.image_key=images \
    data.max_prompt_length=1280 \
    data.max_response_length=512 \
    data.train_batch_size=8 \
    data.filter_overlong_prompts=true \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    custom_reward_function.path="$REWARD" \
    custom_reward_function.name=compute_score \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.logger=['console'] \
    trainer.project_name=rlpt-scienceqa \
    trainer.experiment_name=grpo-smoke \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1

echo "[smoke] done $(date -Iseconds)"
