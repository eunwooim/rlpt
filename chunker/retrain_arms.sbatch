#!/bin/bash
#SBATCH --job-name=chunk-arms
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/sghos104/rlpt/chunker/logs/arm_%A_%a.log

# Chunker retrain: single-step negative policy A/B/C/D.
#
# WHY: audit found 83% of single-step VisualPRM records carry step markers,
# numbered lists, or blank-line breaks. They were used as "do not split here"
# negatives. This sweep measures how much that cost us.
#
# ARMS
#   0 keep   baseline, reproduces the shipped v2 model (sanity anchor)
#   1 drop   no single-step records in training
#   2 clean  only structurally clean singles (~17%) as negatives
#   3 mask   all singles, marker-adjacent candidates forced to -100
#
# INVARIANTS -- if any of these break, the arms are not comparable:
#   * val/test are the frozen v2 splits, byte-identical across arms
#   * seed 42, effective batch 128, fp32, identical LR/schedule/early-stopping
#   * selection on VAL only; test stays untouched until one final run

set -euo pipefail

ARMS=(keep drop clean mask)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}

ROOT=/scratch/sghos104/rlpt/chunker
PY=/scratch/sghos104/rlpt/chunker/env/bin/python
export HF_HOME=/scratch/sghos104/.hf_cache

DATA=$ROOT/data_arms/$ARM
OUT=$ROOT/runs/arm_$ARM
mkdir -p "$DATA" "$OUT" "$ROOT/logs"

echo "=== arm=$ARM  host=$(hostname)  $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---------------------------------------------------------------- 1. data
# Rebuild the TRAINING set under this policy. Val/test are NOT rebuilt --
# they are symlinked from the frozen v2 split so every arm evaluates on
# identical data.
if [ ! -f "$DATA/.done" ]; then
  $PY "$ROOT/prepare_data.py" \
      --single-policy "$ARM" \
      --split-dir "$ROOT/data" \
      --out "$DATA" \
      --seed 42 \
      --train-only
  ln -sfn "$ROOT/data/val"  "$DATA/val"
  ln -sfn "$ROOT/data/test" "$DATA/test"
  touch "$DATA/.done"
fi

# ---------------------------------------------------------------- 2. train
$PY "$ROOT/train.py" \
    --data "$DATA" \
    --output-dir "$OUT" \
    --seed 42 \
    --fp32 \
    --loss plain_ce \
    --resume-if-possible

# ---------------------------------------------------------------- 3. eval
# VAL ONLY. No --test flag anywhere in this script, deliberately.
$PY "$ROOT/evaluate.py" \
    --checkpoint "$OUT" \
    --split val \
    --threshold-sweep 0.20,0.90,0.05 \
    --report-false-splits \
    --out "$OUT/eval_val.json"

echo "=== arm=$ARM done $(date) ==="
