#!/usr/bin/env bash
# ScienceQA GRPO RLVR
# Qwen2.5-VL-3B-Instruct
# veRL 0.8.0

set -xeuo pipefail

# ============================================================
# Usage
# ============================================================

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <n_gpus_per_node> <save_path> [extra hydra args]"
    exit 1
fi

NGPUS_PER_NODE=$1
SAVE_PATH=$2
shift 2

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/data1/eunwooim/rlpt/src}

DATA_DIR=${DATA_DIR:-/mnt/data1/eunwooim/rlpt/data/scienceqa_rlvr}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/train.parquet}
VAL_FILE=${VAL_FILE:-${DATA_DIR}/val.parquet}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}

REWARD_PATH=${REWARD_PATH:-${PROJECT_ROOT}/rewards/scienceqa_answer_format_reward.py}
REWARD_NAME=${REWARD_NAME:-compute_score}

HF_HOME=${HF_HOME:-/mnt/shared/shared_hf_home}

export HF_HOME
export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "${SAVE_PATH}"

# ============================================================
# Experiment Snapshot
# ============================================================

SNAPSHOT_DIR="${SAVE_PATH}/code_snapshot"
mkdir -p "${SNAPSHOT_DIR}"

{
    echo "date=$(date)"
    echo "script=$0"
    echo "save_path=${SAVE_PATH}"
    echo "extra_args=$*"
} > "${SAVE_PATH}/run_env.txt"

python - <<'PY' > "${SAVE_PATH}/env_versions.txt"
import sys
print("python:", sys.version)

mods = [
    "torch",
    "transformers",
    "datasets",
    "wandb",
    "verl",
    "accelerate",
    "flash_attn",
    "vllm",
    "ray",
]

for name in mods:
    try:
        m = __import__(name)
        print(name, getattr(m, "__version__", "unknown"))
    except Exception as e:
        print(name, "IMPORT_ERROR", repr(e))
PY

tar \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='outputs' \
    --exclude='data' \
    -czf "${SNAPSHOT_DIR}/src_snapshot.tar.gz" \
    -C "${PROJECT_ROOT}" .

cp "$0" "${SNAPSHOT_DIR}/launch_script.sh"

if git -C "${PROJECT_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "${PROJECT_ROOT}" rev-parse HEAD > "${SNAPSHOT_DIR}/git_commit.txt"
    git -C "${PROJECT_ROOT}" status --short > "${SNAPSHOT_DIR}/git_status.txt"
    git -C "${PROJECT_ROOT}" diff > "${SNAPSHOT_DIR}/git_diff.patch" || true
fi

# ============================================================
# Hyperparameters
# ============================================================

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-128}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}

ROLLOUT_N=${ROLLOUT_N:-5}

PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}

LR=${LR:-2e-6}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}

KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}

# ============================================================
# Config Groups
# ============================================================

DATA=(
    data.train_files=${TRAIN_FILE}
    data.val_files=${VAL_FILE}
    data.prompt_key=prompt

    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.val_batch_size=${VAL_BATCH_SIZE}

    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.norm_adv_by_std_in_grpo=True
)

REWARD=(
    reward.custom_reward_function.path=${REWARD_PATH}
    reward.custom_reward_function.name=${REWARD_NAME}
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.tokenizer_path=${MODEL_PATH}
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=False

    ++actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2
    ++actor_rollout_ref.model.override_config._attn_implementation=flash_attention_2
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${LR}

    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}

    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}

    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384

    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.n=${ROLLOUT_N}

    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.90

    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

TRAINER=(
    trainer.project_name=rlpt-scienceqa-rlvr
    trainer.experiment_name=qwen25vl3b-scienceqa-grpo-answer-format

    trainer.default_local_dir=${SAVE_PATH}/checkpoints

    trainer.nnodes=1
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}

    trainer.total_epochs=${TOTAL_EPOCHS}

    trainer.save_freq=100
    trainer.test_freq=50

    trainer.logger='["console","wandb"]'
    trainer.resume_mode=disable
)

# ============================================================
# Launch
# ============================================================

python -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${REWARD[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "$@"
