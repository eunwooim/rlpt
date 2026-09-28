#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=src/scripts/_train_common.bash
source "${SCRIPT_DIR}/_train_common.bash"

MODE="${MODE:-sft}"
DATASET="${DATASET:-visualprm}"
REWARD_ABLATION="${REWARD_ABLATION:-accuracy_format}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"
PREPARE_DATA_ONLY="${PREPARE_DATA_ONLY:-0}"
BATCH_SIZE="${BATCH_SIZE:-}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-}"
IMAGE_MODE="${IMAGE_MODE:-auto}"
IMAGE_ZIP="${IMAGE_ZIP:-}"

require_choice "MODE" "${MODE}" sft rlvr
require_choice "DATASET" "${DATASET}" visualprm llava_cot

DATASET_NAME="${DATASET_NAME:-$(dataset_name_for_alias "${DATASET}")}"
DATASET_CONFIG="${DATASET_CONFIG:-$(dataset_config_for_alias "${DATASET}")}"

ratio_tag="${DATA_RATIO//./p}"
lr_tag="${LEARNING_RATE//./p}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${DATASET}_${MODE}_ratio_${ratio_tag}_seed_${SEED}_lr_${lr_tag}}"
OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-$(make_output_dir "${EXPERIMENT_NAME}")}"

if [[ "${MODE}" == "sft" ]]; then
  ENTRYPOINT="${SFT_ENTRYPOINT}"
else
  ENTRYPOINT="${RL_ENTRYPOINT}"
fi

ensure_entrypoint_exists_for_execution "${ENTRYPOINT}"

cmd=(
  "${PYTHON_BIN}" -u "${ENTRYPOINT}"
  --base_model "${BASE_MODEL}"
  --dataset_name "${DATASET_NAME}"
  --dataset_split "${DATASET_SPLIT}"
  --data_ratio "${DATA_RATIO}"
  --max_seq_len "${MAX_SEQ_LEN}"
  --seed "${SEED}"
  --token_budget "${TOKEN_BUDGET}"
  --learning_rate "${LEARNING_RATE}"
  --output_dir "${OUTPUT_DIR}"
  --report_to "${REPORT_TO}"
)

if [[ -n "${DATASET_CONFIG}" ]]; then
  cmd+=(--dataset_config "${DATASET_CONFIG}")
fi

if [[ "${FORCE}" == "1" ]]; then
  cmd+=(--force)
fi

if [[ "${CONFIG_ONLY}" == "1" ]]; then
  cmd+=(--config_only)
fi

if [[ "${PREPARE_DATA_ONLY}" == "1" ]]; then
  cmd+=(--prepare_data_only)
fi

if [[ -n "${MAX_TRAIN_SAMPLES:-}" ]]; then
  cmd+=(--max_train_samples "${MAX_TRAIN_SAMPLES}")
fi

if [[ -n "${MAX_STEPS:-}" ]]; then
  cmd+=(--max_steps "${MAX_STEPS}")
fi

if [[ "${REPORT_TO}" == "wandb" ]]; then
  cmd+=(--wandb_project "${WANDB_PROJECT}" --wandb_mode "${WANDB_MODE}")
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    cmd+=(--wandb_entity "${WANDB_ENTITY}")
  fi
  if [[ -n "${WANDB_NAME:-}" ]]; then
    cmd+=(--wandb_run_name "${WANDB_NAME}")
  fi
  if [[ -n "${WANDB_RUN_GROUP:-}" ]]; then
    cmd+=(--wandb_group "${WANDB_RUN_GROUP}")
  fi
  if [[ -n "${WANDB_TAGS:-}" ]]; then
    cmd+=(--wandb_tags "${WANDB_TAGS}")
  fi
fi

if [[ "${MODE}" == "rlvr" ]]; then
  REWARD_COMPONENTS="${REWARD_COMPONENTS:-$(reward_components_for_ablation "${REWARD_ABLATION}")}"
  cmd+=(
    --rl_algorithm grpo
    --process_reward rlvr
    --reward_components "${REWARD_COMPONENTS}"
    --reward_ablation "${REWARD_ABLATION}"
  )
  if [[ -n "${BATCH_SIZE}" ]]; then
    cmd+=(--train_batch_size "${BATCH_SIZE}")
  fi
  if [[ -n "${ROLLOUT_N:-}" ]]; then
    cmd+=(--rollout_n "${ROLLOUT_N}")
  fi
  if [[ -n "${PPO_MINI_BATCH_SIZE:-}" ]]; then
    cmd+=(--ppo_mini_batch_size "${PPO_MINI_BATCH_SIZE}")
  fi
  if [[ -n "${PPO_MICRO_BATCH_SIZE_PER_GPU:-}" ]]; then
    cmd+=(--ppo_micro_batch_size_per_gpu "${PPO_MICRO_BATCH_SIZE_PER_GPU}")
  fi
  if [[ -n "${SAVE_FREQ:-}" ]]; then
    cmd+=(--save_freq "${SAVE_FREQ}")
  fi
  if [[ -n "${TEST_FREQ:-}" ]]; then
    cmd+=(--test_freq "${TEST_FREQ}")
  fi
  if [[ -n "${VERL_EXTRA_OVERRIDES:-}" ]]; then
    # space-separated list of Hydra overrides, each passed as its own flag
    for _ov in ${VERL_EXTRA_OVERRIDES}; do
      cmd+=(--verl_extra_override "${_ov}")
    done
  fi
else
  cmd+=(--image_mode "${IMAGE_MODE}")
  if [[ -n "${IMAGE_ZIP}" ]]; then
    cmd+=(--image_zip "${IMAGE_ZIP}")
  fi
  if [[ -n "${BATCH_SIZE}" ]]; then
    cmd+=(--per_device_train_batch_size "${BATCH_SIZE}")
  fi
  if [[ -n "${GRAD_ACCUM_STEPS}" ]]; then
    cmd+=(--gradient_accumulation_steps "${GRAD_ACCUM_STEPS}")
  fi
  if [[ -n "${SAVE_STEPS:-}" ]]; then
    cmd+=(--save_steps "${SAVE_STEPS}")
  fi
  if [[ -n "${SAVE_TOTAL_LIMIT:-}" ]]; then
    cmd+=(--save_total_limit "${SAVE_TOTAL_LIMIT}")
  fi
fi

run_or_print "${OUTPUT_DIR}" "${cmd[@]}"
