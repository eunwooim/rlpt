#!/bin/bash
#SBATCH --job-name=rlpt-gen-valid
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/sghos104/rlpt/data/logs/gen-valid-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/data/logs/gen-valid-%j.err

# Shared generation pass for validation tests 1-3 (shuffled-GT, NLI re-scoring,
# image-swap probe): greedy vLLM outputs on the 256-row test subset for the base
# model + every merged checkpoint, plus image-swapped passes for base/20/176,
# plus the A-OKVQA transfer parquet (test 4) for base/176 if it exists.
# Skips any output file that already exists, so it is safe to resubmit.
set -uo pipefail
mkdir -p /scratch/sghos104/rlpt/data/logs /scratch/sghos104/rlpt/outputs/validation

module load mamba/latest
ENV_PREFIX=/scratch/sghos104/envs/rlpt-train
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export VLLM_USE_V1=1
# vLLM V1 forks its EngineCore by default; outside Ray (which forces spawn) the
# fork hits "Cannot re-initialize CUDA in forked subprocess" once the parent has
# touched CUDA. Force spawn.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH=/scratch/sghos104/rlpt/src/train/flash_attn_shim:${PYTHONPATH:-}
unset ROCR_VISIBLE_DEVICES

cd /scratch/sghos104/rlpt
echo "[gen-valid] $(date -Iseconds) on $(hostname)"; nvidia-smi -L

PY="$ENV_PREFIX/bin/python"
GEN=tools/gen_eval_outputs.py
SQA=data/scienceqa_verl/test_sweep256.parquet
AOK=data/aokvqa_eval/aokvqa_val256.parquet
OUT=outputs/validation
BASE=Qwen/Qwen2.5-VL-3B-Instruct
MDIR=/scratch/sghos104/rlpt/models
FAILED=0

model_path() {  # base -> HF id, else merged dir
  if [ "$1" = "base" ]; then echo "$BASE"; else echo "$MDIR/qwen2_5vl-3b-scienceqa-grpo-step$1"; fi
}

run_gen() {  # $1=label $2=parquet $3=outfile $4=extra flags
  local MP; MP=$(model_path "$1")
  if [ -f "$3" ]; then echo "=== $3 exists, skipping ==="; return; fi
  if [ "$1" != "base" ] && [ ! -f "$MP/model.safetensors.index.json" ]; then
    echo "!! merged model for step $1 missing ($MP) — skipping"; FAILED=1; return
  fi
  echo "=== gen $1 -> $3 $(date -Iseconds) ==="
  "$PY" "$GEN" --model-path "$MP" --parquet "$2" --out "$3" $4 \
    || { echo "!! gen $1 ($3) FAILED"; FAILED=1; }
}

# 1) normal pass, every checkpoint (tests 1 + 2 share these generations)
for CK in base 20 40 60 80 100 120 140 160 176; do
  run_gen "$CK" "$SQA" "$OUT/gen_${CK}.jsonl" ""
done

# 2) image-swapped pass (test 3)
for CK in base 20 176; do
  run_gen "$CK" "$SQA" "$OUT/gen_${CK}_swap.jsonl" "--image-swap"
done

# 3) A-OKVQA transfer (test 4)
if [ -f "$AOK" ]; then
  for CK in base 176; do
    run_gen "$CK" "$AOK" "$OUT/aokvqa_${CK}.jsonl" ""
  done
else
  echo "(A-OKVQA parquet not present — transfer gen skipped)"
fi

echo "=== outputs ==="; ls -la "$OUT"
echo "[gen-valid] done $(date -Iseconds) (failed=$FAILED)"
exit $FAILED
