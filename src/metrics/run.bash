#!/bin/bash
set -euo pipefail

DATA_RUN_NAME="${DATA_RUN_NAME:-initial_run}"
CALIBRATION_RUN="${CALIBRATION_RUN:-run_20260706_224211}"
CALIBRATION_WEIGHTS="${CALIBRATION_WEIGHTS:-src/ckpts/calibration/v1/nli_calibration_weights.json}"
KAPPA="${KAPPA:-0.0}"

CAPTION_DIR="src/outputs/reliable/caption_negation/${DATA_RUN_NAME}"
NUMERIC_DIR="src/outputs/reliable/visualprm_numeric/${DATA_RUN_NAME}"
CALIBRATION_DIR="src/outputs/reliable/calibration/${CALIBRATION_RUN}"
REWARD_DIR="src/outputs/reliable/reward_tables/${DATA_RUN_NAME}"

# Post-processing only. This script reuses existing JSONL scores and does not rerun scorer inference.
python src/metrics/aggregate_hn_neg.py \
  --output_dir "${CAPTION_DIR}" \
  --kappa "${KAPPA}"

python src/metrics/aggregate_visualprm_numeric.py \
  --output_dir "${NUMERIC_DIR}" \
  --kappa "${KAPPA}"

python src/metrics/reward_tables.py \
  --caption_scores "${CAPTION_DIR}/scores_top3.jsonl" \
  --numeric_scores "${NUMERIC_DIR}/raw/numeric_scores.jsonl" \
  --calibration_predictions "${CALIBRATION_DIR}/raw/nli_calibration_predictions.jsonl" \
  --calibration_weights "${CALIBRATION_WEIGHTS}" \
  --kappa "${KAPPA}" \
  --output_dir "${REWARD_DIR}"
