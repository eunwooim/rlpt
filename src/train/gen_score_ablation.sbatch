#!/bin/bash
#SBATCH --job-name=rlpt-gen-nomatch
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/sghos104/rlpt/data/logs/gen-nomatch-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/data/logs/gen-nomatch-%j.err

# Validation test 5 cross-arm comparison: generation pass on every merged
# w_match=0 ablation checkpoint (256-row subset, plus an image-swap and A-OKVQA
# pass on the final checkpoint), then CPU scoring with the same harness as the
# main arm. The key readout: does the ablation's own-GT match climb like the
# main run's (match term not causal) or stay near the base floor (causal)?
set -uo pipefail
mkdir -p /scratch/sghos104/rlpt/data/logs /scratch/sghos104/rlpt/outputs/validation_nomatch

module load mamba/latest
ENV_PREFIX=/scratch/sghos104/envs/rlpt-train
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NLTK_DATA=/scratch/sghos104/rlpt/data/nltk_data
export PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export RLPT_REWARD_DEVICE=cpu
export PYTHONPATH=/scratch/sghos104/rlpt/src/train/flash_attn_shim:${PYTHONPATH:-}
unset ROCR_VISIBLE_DEVICES

cd /scratch/sghos104/rlpt
echo "[gen-nomatch] $(date -Iseconds) on $(hostname)"; nvidia-smi -L

PY="$ENV_PREFIX/bin/python"
GEN=tools/gen_eval_outputs.py
SQA=data/scienceqa_verl/test_sweep256.parquet
AOK=data/aokvqa_eval/aokvqa_val256.parquet
OUT=outputs/validation_nomatch
MDIR=/scratch/sghos104/rlpt/models
FAILED=0

# the shared base-model rows come from the main arm (identical pass)
for F in gen_base.jsonl gen_base_swap.jsonl aokvqa_base.jsonl; do
  [ -f "$OUT/$F" ] || cp "outputs/validation/$F" "$OUT/$F" 2>/dev/null || true
done

LAST=""
for MP in "$MDIR"/qwen2_5vl-3b-scienceqa-nomatch-step*; do
  [ -f "$MP/model.safetensors.index.json" ] || continue
  STEP=${MP##*-step}
  LAST=$STEP
done
for MP in "$MDIR"/qwen2_5vl-3b-scienceqa-nomatch-step*; do
  [ -f "$MP/model.safetensors.index.json" ] || continue
  STEP=${MP##*-step}
  if [ ! -f "$OUT/gen_${STEP}.jsonl" ]; then
    echo "=== gen nomatch-$STEP $(date -Iseconds) ==="
    "$PY" "$GEN" --model-path "$MP" --parquet "$SQA" --out "$OUT/gen_${STEP}.jsonl" \
      || { echo "!! gen nomatch-$STEP FAILED"; FAILED=1; }
  fi
  if [ "$STEP" = "$LAST" ]; then
    [ -f "$OUT/gen_${STEP}_swap.jsonl" ] || "$PY" "$GEN" --model-path "$MP" \
        --parquet "$SQA" --out "$OUT/gen_${STEP}_swap.jsonl" --image-swap \
      || { echo "!! swap gen nomatch-$STEP FAILED"; FAILED=1; }
    [ -f "$OUT/aokvqa_${STEP}.jsonl" ] || "$PY" "$GEN" --model-path "$MP" \
        --parquet "$AOK" --out "$OUT/aokvqa_${STEP}.jsonl" \
      || { echo "!! aokvqa gen nomatch-$STEP FAILED"; FAILED=1; }
  fi
done

echo "=== scoring (CPU portion) ==="
export OMP_NUM_THREADS=16
"$PY" tools/validation_scoring.py --gen-dir "$OUT" \
    --out-dir data/logs/validation_results_nomatch

echo "=== NOMATCH ARM RESULTS ==="
for t in main swap aokvqa; do
  echo "--- validation_$t.tsv ---"
  cat "data/logs/validation_results_nomatch/validation_$t.tsv" 2>/dev/null
done
echo "[gen-nomatch] done $(date -Iseconds) (failed=$FAILED)"
