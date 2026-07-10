#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN="${DRY_RUN:-1}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
TOKEN_BUDGET="${TOKEN_BUDGET:-1000000}"
DATA_RATIOS="${DATA_RATIOS:-0.01}"
SEEDS="${SEEDS:-42}"
LEARNING_RATES="${LEARNING_RATES:-1e-5}"
DATASETS="${DATASETS:-visualprm llava_cot}"
MODES="${MODES:-sft rlvr}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

export DRY_RUN BASE_MODEL MAX_SEQ_LEN TOKEN_BUDGET RUN_TIMESTAMP

for data_ratio in ${DATA_RATIOS}; do
  for seed in ${SEEDS}; do
    for learning_rate in ${LEARNING_RATES}; do
      for dataset in ${DATASETS}; do
        for mode in ${MODES}; do
          ratio_tag="${data_ratio//./p}"
          lr_tag="${learning_rate//./p}"
          echo
          echo "=== sweep dataset=${dataset} mode=${mode} ratio=${data_ratio} seed=${seed} lr=${learning_rate} ==="
          MODE="${mode}" \
            DATASET="${dataset}" \
            DATA_RATIO="${data_ratio}" \
            SEED="${seed}" \
            LEARNING_RATE="${learning_rate}" \
            EXPERIMENT_NAME="sweep_${dataset}_${mode}_ratio_${ratio_tag}_seed_${seed}_lr_${lr_tag}" \
            bash "${SCRIPT_DIR}/run_baseline.bash"
        done
      done
    done
  done
done
