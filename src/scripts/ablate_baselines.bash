#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN="${DRY_RUN:-1}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
DATA_RATIO="${DATA_RATIO:-0.01}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
SEED="${SEED:-42}"
TOKEN_BUDGET="${TOKEN_BUDGET:-1000000}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
DATASETS="${DATASETS:-visualprm llava_cot}"
REWARD_ABLATIONS="${REWARD_ABLATIONS:-accuracy_only format_only accuracy_format}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

export DRY_RUN BASE_MODEL DATA_RATIO MAX_SEQ_LEN SEED TOKEN_BUDGET LEARNING_RATE RUN_TIMESTAMP

for dataset in ${DATASETS}; do
  for reward_ablation in ${REWARD_ABLATIONS}; do
    echo
    echo "=== rlvr ablation dataset=${dataset} reward=${reward_ablation} ==="
    MODE=rlvr \
      DATASET="${dataset}" \
      REWARD_ABLATION="${reward_ablation}" \
      EXPERIMENT_NAME="ablate_${dataset}_rlvr_${reward_ablation}" \
      bash "${SCRIPT_DIR}/run_baseline.bash"
  done
done
