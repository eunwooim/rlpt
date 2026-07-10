#!/usr/bin/env bash

# Shared helpers for baseline training launch templates.

rlpt_repo_root() {
  if git rev-parse --show-toplevel >/dev/null 2>&1; then
    git rev-parse --show-toplevel
  else
    cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
  fi
}

ROOT="${ROOT:-$(rlpt_repo_root)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/src/outputs/train}"
CONDA_ROOT="${CONDA_ROOT:-/mnt/data2/eunwooim/.conda}"
HF_HOME="${HF_HOME:-/mnt/shared/shared_hf_home}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
DATASETS_CACHE="${DATASETS_CACHE:-${HF_HOME}/datasets}"
WANDB_DIR="${WANDB_DIR:-${ROOT}/src/outputs/wandb}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

DRY_RUN="${DRY_RUN:-1}"
FORCE="${FORCE:-0}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
DATA_RATIO="${DATA_RATIO:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
SEED="${SEED:-42}"
TOKEN_BUDGET="${TOKEN_BUDGET:-100000000}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MAX_STEPS="${MAX_STEPS:-}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
REPORT_TO="${REPORT_TO:-none}"
WANDB_PROJECT="${WANDB_PROJECT:-rlpt_baselines}"
WANDB_MODE="${WANDB_MODE:-offline}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SFT_ENTRYPOINT="${SFT_ENTRYPOINT:-src/train/sft.py}"
RL_ENTRYPOINT="${RL_ENTRYPOINT:-src/train/rlvr_grpo.py}"

export ROOT
export CONDA_ROOT
export HF_HOME
export HF_HUB_CACHE
export TRANSFORMERS_CACHE
export DATASETS_CACHE
export WANDB_DIR
export REPORT_TO
export WANDB_PROJECT
export WANDB_MODE
export PYTHON_BIN
export CUDA_VISIBLE_DEVICES
export MAX_STEPS

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_choice() {
  local name="$1"
  local value="$2"
  shift 2
  local option
  for option in "$@"; do
    if [[ "${value}" == "${option}" ]]; then
      return 0
    fi
  done
  die "${name} must be one of: $*; got '${value}'"
}

dataset_name_for_alias() {
  case "$1" in
    visualprm)
      echo "OpenGVLab/VisualPRM400K-v1.1-Raw"
      ;;
    llava_cot)
      echo "Xkev/LLaVA-CoT-100k"
      ;;
    *)
      die "DATASET must be one of: visualprm llava_cot; got '$1'"
      ;;
  esac
}

dataset_config_for_alias() {
  case "$1" in
    visualprm)
      echo "default"
      ;;
    llava_cot)
      echo ""
      ;;
    *)
      die "DATASET must be one of: visualprm llava_cot; got '$1'"
      ;;
  esac
}

reward_components_for_ablation() {
  case "$1" in
    accuracy_only)
      echo "accuracy"
      ;;
    format_only)
      echo "format"
      ;;
    accuracy_format)
      echo "accuracy,format"
      ;;
    *)
      die "REWARD_ABLATION must be one of: accuracy_only format_only accuracy_format; got '$1'"
      ;;
  esac
}

shell_join() {
  local arg
  printf "%q" "$1"
  shift || true
  for arg in "$@"; do
    printf " %q" "${arg}"
  done
}

make_output_dir() {
  local experiment_name="$1"
  local output_dir="${OUTPUT_ROOT}/${experiment_name}/run_${RUN_TIMESTAMP}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "${output_dir}"
    return 0
  fi

  if [[ -e "${output_dir}" && "${FORCE}" != "1" ]]; then
    die "output directory exists: ${output_dir}. Set FORCE=1 to allow reuse."
  fi

  mkdir -p \
    "${output_dir}/configs" \
    "${output_dir}/logs" \
    "${output_dir}/checkpoints" \
    "${output_dir}/metrics"
  echo "${output_dir}"
}

write_run_metadata() {
  local output_dir="$1"
  shift
  local command_text
  command_text="$(shell_join "$@")"

  [[ "${DRY_RUN}" == "1" ]] && return 0

  printf "%s\n" "${command_text}" >"${output_dir}/logs/command.txt"
  cat >"${output_dir}/configs/run_metadata.env" <<EOF
ROOT=${ROOT}
BASE_MODEL=${BASE_MODEL}
DATA_RATIO=${DATA_RATIO}
MAX_SEQ_LEN=${MAX_SEQ_LEN}
SEED=${SEED}
TOKEN_BUDGET=${TOKEN_BUDGET}
DATASET_SPLIT=${DATASET_SPLIT}
LEARNING_RATE=${LEARNING_RATE}
MAX_STEPS=${MAX_STEPS}
REPORT_TO=${REPORT_TO}
WANDB_PROJECT=${WANDB_PROJECT}
WANDB_MODE=${WANDB_MODE}
PYTHON_BIN=${PYTHON_BIN}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
RUN_TIMESTAMP=${RUN_TIMESTAMP}
DRY_RUN=${DRY_RUN}
EOF
}

ensure_entrypoint_exists_for_execution() {
  local entrypoint="$1"
  [[ "${DRY_RUN}" == "1" ]] && return 0

  if [[ "${entrypoint}" == /* ]]; then
    [[ -f "${entrypoint}" ]] || die "entrypoint does not exist: ${entrypoint}"
  else
    [[ -f "${ROOT}/${entrypoint}" ]] || die "entrypoint does not exist: ${ROOT}/${entrypoint}"
  fi
}

run_or_print() {
  local output_dir="$1"
  shift
  local command_text
  command_text="$(shell_join "$@")"

  echo "ROOT=${ROOT}"
  echo "OUTPUT_DIR=${output_dir}"
  echo "COMMAND=${command_text}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1: command printed only."
    return 0
  fi

  write_run_metadata "${output_dir}" "$@"
  (
    cd "${ROOT}"
    "$@"
  ) 2>&1 | tee "${output_dir}/logs/run.log"
}
