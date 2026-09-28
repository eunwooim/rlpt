#!/bin/bash
#SBATCH --job-name=vprm_calib
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/sghos104/rlpt/outputs/slurm-vprm_calib-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/outputs/slurm-vprm_calib-%j.out

set -uo pipefail
REPO=/scratch/sghos104/rlpt
cd "$REPO" || exit 1

export PYTHON_BIN=/scratch/sghos104/envs/rlpt-train/bin/python
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
export PYTHONPATH="$REPO/src/train/flash_attn_shim${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
unset ROCR_VISIBLE_DEVICES

export RLPT_FILTERED_JSONL=$REPO/data/visualprm_v11_filtered/sft_train.jsonl

echo "[calib] host=$(hostname) job=$SLURM_JOB_ID start=$(date)"
nvidia-smi -L

# GPU memory poller: 10 s samples so we can report true peak usage at micro=4.
MEMLOG=$REPO/outputs/calib_gpumem_$SLURM_JOB_ID.log
( while true; do
    echo "$(date +%T) $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits)"
    sleep 10
  done ) > "$MEMLOG" &
POLLER=$!

# Calibration for the real (scale) runs: proposed config = effective batch 128
# (micro 4 x accum 32, 1 GPU), lr 2e-5, TOKEN_BUDGET 1e8. MAX_STEPS=6 caps the
# job at ~768 samples: enough to verify no OOM and measure steady-state s/step;
# MAX_TRAIN_SAMPLES=5651 keeps data prep at smoke-run speed (throughput does
# not depend on pool size).
DRY_RUN=0 \
MODE=sft \
DATASET=visualprm \
DATA_RATIO=1 \
MAX_TRAIN_SAMPLES=5651 \
MAX_SEQ_LEN=4096 \
SEED=42 \
TOKEN_BUDGET=100000000 \
LEARNING_RATE=2e-5 \
BATCH_SIZE=4 \
GRAD_ACCUM_STEPS=32 \
MAX_STEPS=6 \
EXPERIMENT_NAME=visualprm_sft_calib_bs128 \
bash src/scripts/run_baseline.bash
rc=$?

kill $POLLER 2>/dev/null
peak=$(awk '{print $2}' "$MEMLOG" | sort -n | tail -1)
echo "[calib] peak GPU memory used: ${peak} MiB (log: $MEMLOG)"
echo "[calib] done=$(date) rc=$rc"
