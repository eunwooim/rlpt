#!/bin/bash
#SBATCH --job-name=rlpt-grpo-sqa
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/sghos104/rlpt/data/logs/grpo-sqa-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/data/logs/grpo-sqa-%j.err

# GRPO RL on Qwen2.5-VL over ScienceQA, scored by the composite reward
# (format + bipartite reasoning-match + hard answer-correctness - hallucination pun).
#
# PREREQUISITES (not yet in repo — see CLAUDE.md "src/verl/ was removed"):
#   1. verl installed in the env (pip install verl, or add the submodule back) with
#      vLLM + Qwen2.5-VL support.
#   2. Data built:  sbatch src/data_factory/scienceqa_to_verl.sbatch
#   3. SBERT + spaCy models cached (already done under data/hf_cache, NLTK_DATA).
#   4. Reward deps importable by verl workers: sentence-transformers, spacy, scipy,
#      en_core_web_sm. The reward singleton loads SBERT once per worker.

set -euo pipefail
mkdir -p /scratch/sghos104/rlpt/data/logs

module load mamba/latest
ENV_PREFIX=/scratch/sghos104/envs/rlpt-train   # RL stack + reward deps
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
export NLTK_DATA=/scratch/sghos104/rlpt/data/nltk_data
export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1            # stream verl stdout to the log (was buffered before)
export RLPT_REWARD_DEVICE=cpu        # keep SBERT off the policy GPUs
export VLLM_USE_V1=1                 # verl async vLLM server needs the V1 engine
export PYTHONPATH=/scratch/sghos104/rlpt/src/train/flash_attn_shim:${PYTHONPATH:-}  # pure-torch flash_attn.bert_padding shim (no glibc-incompatible CUDA wheel)
unset ROCR_VISIBLE_DEVICES           # Sol sets this AMD var; verl rejects it alongside CUDA_VISIBLE_DEVICES

cd /scratch/sghos104/rlpt
echo "[sbatch] $(date -Iseconds) on $(hostname)"
nvidia-smi -L

DATA=/scratch/sghos104/rlpt/data/scienceqa_verl
REWARD=/scratch/sghos104/rlpt/tools/graph_match_reward.py
MODEL=Qwen/Qwen2.5-VL-3B-Instruct        # committed model: Qwen2.5-VL (3B for iteration speed)
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
    trainer.experiment_name=grpo-qwen2_5vl-3b \
    trainer.save_freq=20 \
    trainer.test_freq=25 \
    trainer.total_epochs=2

echo "[sbatch] done $(date -Iseconds)"

# --- design notes ---------------------------------------------------------
# adv_estimator=grpo        : group-relative advantage; rollout.n=5 is the GROUP
#                             size = the "attempts" per prompt. GRPO normalizes
#                             reward across the 5 -> wasteful wrong attempts get
#                             negative advantage (our "penalise attempts" intent,
#                             realized the STAR-R1 way, no bespoke counter).
# kl_loss + coef 0.001      : the -beta*KL term from STAR-R1 / standard GRPO.
# custom_reward_function    : verl calls compute_score(data_source, solution_str,
#                             ground_truth, extra_info) per rollout -> our composite.
# image_key=images          : multimodal; prompt carries <image>, paths in `images`.
