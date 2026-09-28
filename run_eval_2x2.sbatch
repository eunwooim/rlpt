#!/bin/bash
#SBATCH --job-name=vprm_eval
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/sghos104/rlpt/outputs/slurm-vprm_eval-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/outputs/slurm-vprm_eval-%j.out

set -uo pipefail
REPO=/scratch/sghos104/rlpt
cd "$REPO" || exit 1
PY=/scratch/sghos104/envs/rlpt-train/bin/python
export HF_HOME=$REPO/data/hf_cache
export PYTHONPATH="$REPO/src/train/flash_attn_shim${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
unset ROCR_VISIBLE_DEVICES
# vLLM V1 forks its EngineCore proc by default; CUDA gets initialized in the
# parent (torch import chain on this env), and fork-after-CUDA-init is fatal.
export VLLM_WORKER_MULTIPROC_METHOD=spawn

T=$REPO/src/outputs/train
EVAL=$REPO/src/outputs/eval_visualprm_2x2
mkdir -p "$EVAL"
echo "[eval2x2] host=$(hostname) job=$SLURM_JOB_ID start=$(date)"
nvidia-smi -L

# ---- Merge the two RLVR FSDP checkpoints to HF format (skip if already merged)
merge() {  # $1 = actor dir, $2 = target dir
  if [ -f "$2/config.json" ]; then echo "[eval2x2] merged exists: $2"; return 0; fi
  $PY -m verl.model_merger merge --backend fsdp --local_dir "$1" --target_dir "$2" \
    || { echo "[eval2x2] MERGE FAILED: $1"; exit 1; }
  # the merger writes weights+config; tokenizer/processor files come from the
  # actor's huggingface/ snapshot saved at checkpoint time
  cp -n "$1"/huggingface/* "$2"/ 2>/dev/null || true
}
merge "$T/visualprm_rlvr_first_baseline/run_20260712_060302/checkpoints/global_step_4/actor" \
      "$EVAL/merged_rlvr_unfiltered"
merge "$T/visualprm_rlvr_filtered_tau085/run_20260718_001542/checkpoints/global_step_4/actor" \
      "$EVAL/merged_rlvr_filtered_tau085"

# ---- Smoke: 8 rows with the base model before committing to all 10 sweeps
$PY src/eval/generate_eval.py --model_path Qwen/Qwen2.5-VL-3B-Instruct \
  --parquet data/mmk12_eval/mmk12_test1024.parquet \
  --out "$EVAL/.smoke.jsonl" --limit 8 \
  || { echo "[eval2x2] SMOKE FAILED"; exit 1; }
echo "[eval2x2] smoke ok: $(wc -l < "$EVAL/.smoke.jsonl") rows"; rm -f "$EVAL/.smoke.jsonl"

# ---- 5 models x 2 benchmarks (fresh process per sweep for clean GPU teardown)
declare -A MODELS=(
  [base]="Qwen/Qwen2.5-VL-3B-Instruct"
  [sft_unfiltered]="$T/visualprm_sft_first_baseline/run_20260712_045241/checkpoints/final"
  [sft_filtered_tau085]="$T/visualprm_sft_filtered_tau085/run_20260718_001034/checkpoints/final"
  [rlvr_unfiltered]="$EVAL/merged_rlvr_unfiltered"
  [rlvr_filtered_tau085]="$EVAL/merged_rlvr_filtered_tau085"
)
declare -A BENCH=(
  [scienceqa]="data/scienceqa_verl/test.parquet"
  [mmk12]="data/mmk12_eval/mmk12_test1024.parquet"
)
for tag in base sft_unfiltered sft_filtered_tau085 rlvr_unfiltered rlvr_filtered_tau085; do
  for b in scienceqa mmk12; do
    out="$EVAL/${tag}__${b}.jsonl"
    if [ -s "$out" ]; then echo "[eval2x2] exists, skipping: $out"; continue; fi
    echo "[eval2x2] generating: $tag on $b $(date)"
    $PY src/eval/generate_eval.py --model_path "${MODELS[$tag]}" \
      --parquet "${BENCH[$b]}" --out "$out" \
      || { echo "[eval2x2] GENERATION FAILED: $tag $b"; exit 1; }
  done
done

# ---- Score everything into one TSV
$PY src/eval/score_eval.py --gen_dir "$EVAL" --pattern "*__*.jsonl" \
  --out "$EVAL/results_2x2.tsv" || { echo "[eval2x2] SCORING FAILED"; exit 1; }

echo "[eval2x2] ALL DONE $(date)"
