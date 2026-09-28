#!/bin/bash
#SBATCH --job-name=vprm_sftF_scale
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/sghos104/rlpt/outputs/slurm-vprm_sftF_scale-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/outputs/slurm-vprm_sftF_scale-%j.out

set -uo pipefail
REPO=/scratch/sghos104/rlpt
cd "$REPO" || exit 1

export PYTHON_BIN=/scratch/sghos104/envs/rlpt-train/bin/python
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
export PYTHONPATH="$REPO/src/train/flash_attn_shim${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
unset ROCR_VISIBLE_DEVICES
# Calibration peak was 76.7/80 GiB at micro-batch 4; expandable segments avoid
# fragmentation-induced OOM at that occupancy.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Filtered pool (tau=0.85, 100,194 traces); records re-read verbatim from the
# raw annotations (FilteredVisualPRMJsonlSource, Edit 6).
export RLPT_FILTERED_JSONL=$REPO/data/visualprm_v11_filtered/sft_train.jsonl

# Resume-idempotent: fixed output dir + RLPT_RESUME=1 (no-op on a fresh run,
# picks up latest checkpoint-* on resubmission after a wall-clock kill).
export RLPT_RESUME=1

echo "[sftF_scale] host=$(hostname) job=$SLURM_JOB_ID start=$(date)"
nvidia-smi -L

# Scale config: DATA_RATIO=1 (full pool, no sample cap), effective batch
# 128 = micro 4 x accum 32 on 1 GPU, lr 2e-5, TOKEN_BUDGET 1e8 nominal
# -> 191 optimizer steps -> 24,448 samples seen of the shuffled pool.
DRY_RUN=0 \
MODE=sft \
DATASET=visualprm \
DATA_RATIO=1 \
MAX_SEQ_LEN=4096 \
SEED=42 \
TOKEN_BUDGET=100000000 \
LEARNING_RATE=2e-5 \
BATCH_SIZE=4 \
GRAD_ACCUM_STEPS=32 \
SAVE_STEPS=25 \
SAVE_TOTAL_LIMIT=3 \
FORCE=1 \
EXPERIMENT_NAME=visualprm_sft_filtered_tau085_scale \
OUTPUT_DIR_OVERRIDE=$REPO/src/outputs/train/visualprm_sft_filtered_tau085_scale/run_scale_v1 \
bash src/scripts/run_baseline.bash
rc=$?
echo "[sftF_scale] done=$(date) rc=$rc"
