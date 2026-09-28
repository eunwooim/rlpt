#!/bin/bash
#SBATCH --job-name=ood_gen
#SBATCH --account=grp_bshettah
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=scg001,scg002,scg003,scg004,sg048,sg049,sg050
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/sghos104/rlpt/grpo_arms/logs/ood_gen-%j.log
# One (model, benchmark) generation: greedy, 2048 tokens, VPB pixel cap. --export=ALL,MODEL_PATH=..,MODEL_TAG=..,BENCH=..
set -uo pipefail
REPO=/scratch/sghos104/rlpt; cd "$REPO" || exit 1
PY=/scratch/sghos104/envs/rlpt-train/bin/python
export HF_HOME=$REPO/data/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="$REPO/src/train/flash_attn_shim${PYTHONPATH:+:$PYTHONPATH}"
unset ROCR_VISIBLE_DEVICES
TAG="ood_${MODEL_TAG}_${BENCH}"
echo "[ood_gen] job=$SLURM_JOB_ID host=$(hostname) model=$MODEL_PATH bench=$BENCH tag=$TAG start=$(date)"
if [[ -s "grpo_arms/evals/ood/vpb_gen_${TAG}.jsonl" ]]; then echo "[ood_gen] already present, skipping"; exit 0; fi
"$PY" grpo_arms/vpb_generate.py --model_path "$MODEL_PATH" --tag "$TAG" --eval_set "data/ood_eval/sets/${BENCH}.jsonl" --outdir grpo_arms/evals/ood --max_tokens 2048; rc=$?
echo "[ood_gen] rc=$rc end=$(date)"; exit $rc
