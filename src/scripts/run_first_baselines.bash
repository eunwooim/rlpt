#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Keep the first comparison fixed and small unless explicitly overridden.
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
DATA_RATIO="${DATA_RATIO:-0.01}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
SEED="${SEED:-42}"
TOKEN_BUDGET="${TOKEN_BUDGET:-100000000}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
DRY_RUN="${DRY_RUN:-1}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

export BASE_MODEL DATA_RATIO MAX_SEQ_LEN SEED TOKEN_BUDGET LEARNING_RATE DRY_RUN RUN_TIMESTAMP

for dataset in visualprm llava_cot; do
  for mode in sft rlvr; do
    echo
    echo "=== ${dataset} ${mode} first baseline ==="
    MODE="${mode}" \
      DATASET="${dataset}" \
      EXPERIMENT_NAME="${dataset}_${mode}_first_baseline" \
      bash "${SCRIPT_DIR}/run_baseline.bash"
  done
done
