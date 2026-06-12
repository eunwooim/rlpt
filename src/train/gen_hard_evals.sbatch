#!/bin/bash
#SBATCH --job-name=rlpt-hard-evals
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/sghos104/rlpt/data/logs/hard-evals-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/data/logs/hard-evals-%j.err

# Validation follow-ups:
#  (1) ablation-176 A-OKVQA + image-swap passes (fixes the step-80 lexicographic-
#      glob mismatch in gen_score_ablation.sbatch), then rebuild the nomatch
#      swap/aokvqa TSVs so the cross-arm transfer comparison is step-matched.
#  (2) MMK12-test hard benchmark (1,024 rows, math/physics/chem/bio): base vs
#      main-176 vs ablation-176 — does the match term buy accuracy where
#      reasoning is the bottleneck? (ScienceQA is near-saturated for both arms.)
set -uo pipefail
mkdir -p /scratch/sghos104/rlpt/data/logs /scratch/sghos104/rlpt/outputs/validation_hard

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
echo "[hard-evals] $(date -Iseconds) on $(hostname)"; nvidia-smi -L

PY="$ENV_PREFIX/bin/python"
GEN=tools/gen_eval_outputs.py
SQA=data/scienceqa_verl/test_sweep256.parquet
AOK=data/aokvqa_eval/aokvqa_val256.parquet
MMK=data/mmk12_eval/mmk12_test1024.parquet
MDIR=/scratch/sghos104/rlpt/models
BASE=Qwen/Qwen2.5-VL-3B-Instruct
MAIN176=$MDIR/qwen2_5vl-3b-scienceqa-grpo-step176
NOMATCH176=$MDIR/qwen2_5vl-3b-scienceqa-nomatch-step176
NM_OUT=outputs/validation_nomatch
HARD_OUT=outputs/validation_hard
FAILED=0

run_gen() {  # $1=model-path $2=parquet $3=outfile $4=extra flags
  if [ -f "$3" ]; then echo "=== $3 exists, skipping ==="; return; fi
  echo "=== gen $1 -> $3 $(date -Iseconds) ==="
  "$PY" "$GEN" --model-path "$1" --parquet "$2" --out "$3" $4 \
    || { echo "!! gen ($3) FAILED"; FAILED=1; }
}

# ---- (1) step-matched ablation-176 transfer/swap passes -------------------
run_gen "$NOMATCH176" "$AOK" "$NM_OUT/aokvqa_176.jsonl" ""
run_gen "$NOMATCH176" "$SQA" "$NM_OUT/gen_176_swap.jsonl" "--image-swap"

# ---- (2) MMK12 hard benchmark: base / main-176 / ablation-176 -------------
run_gen "$BASE"      "$MMK" "$HARD_OUT/mmk12_base.jsonl" ""
run_gen "$MAIN176"   "$MMK" "$HARD_OUT/mmk12_main176.jsonl" ""
run_gen "$NOMATCH176" "$MMK" "$HARD_OUT/mmk12_nomatch176.jsonl" ""

# ---- scoring (CPU) ---------------------------------------------------------
export OMP_NUM_THREADS=16

echo "=== rescore nomatch swap/aokvqa tables (step-matched) ==="
# main TSV is unchanged by the new files; score swap/aokvqa without NLI/alt
# into a temp dir and copy only those two tables over the step-80 versions.
TMP=data/logs/validation_results_nomatch/.rescore_tmp
"$PY" tools/validation_scoring.py --gen-dir "$NM_OUT" --out-dir "$TMP" \
    --skip-nli --skip-alt \
  || { echo "!! nomatch rescore FAILED"; FAILED=1; }
for t in swap aokvqa; do
  [ -f "$TMP/validation_$t.tsv" ] && \
    cp "$TMP/validation_$t.tsv" "data/logs/validation_results_nomatch/validation_$t.tsv"
done

echo "=== score MMK12 ==="
"$PY" tools/score_mcq_eval.py --gen-dir "$HARD_OUT" --pattern "mmk12_*.jsonl" \
    --out data/logs/validation_results/validation_mmk12.tsv \
  || { echo "!! mmk12 scoring FAILED"; FAILED=1; }

echo "=== RESULTS ==="
echo "--- nomatch validation_swap.tsv (step-matched) ---"
cat data/logs/validation_results_nomatch/validation_swap.tsv 2>/dev/null
echo "--- nomatch validation_aokvqa.tsv (step-matched) ---"
cat data/logs/validation_results_nomatch/validation_aokvqa.tsv 2>/dev/null
echo "--- validation_mmk12.tsv ---"
cat data/logs/validation_results/validation_mmk12.tsv 2>/dev/null
echo "[hard-evals] done $(date -Iseconds) (failed=$FAILED)"
exit $FAILED
