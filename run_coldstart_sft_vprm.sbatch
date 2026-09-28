#!/bin/bash
#SBATCH --job-name=coldstart_vprm_3b
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/sghos104/rlpt/outputs/slurm-coldstart_vprm_3b-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/outputs/slurm-coldstart_vprm_3b-%j.out
set -uo pipefail
REPO=/scratch/sghos104/rlpt
cd "$REPO" || exit 1
export PYTHON_BIN=/scratch/sghos104/envs/rlpt-train/bin/python
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
export PYTHONPATH="$REPO/src/train/flash_attn_shim${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
unset ROCR_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RLPT_FILTERED_JSONL=$REPO/data/visualprm_v11_filtered/sft_train.jsonl
export RLPT_RESUME=1
mkdir -p "$REPO/src/outputs/train/coldstart_vprm_3b/run_v1"/{logs,configs,checkpoints,eval}
echo "[coldstart_vprm_3b] host=$(hostname) job=$SLURM_JOB_ID start=$(date)"
nvidia-smi -L
# VisualPRM cold start, advisor-specified dose: 150 optimizer updates,
# checkpoint every 25 (-> 25/50/75/100/125/150) for the grain+accuracy
# curve. tokens/step = 4096*4*32 = 524,288 -> TOKEN_BUDGET 150*524288 =
# 78,643,200. lr 1e-5 (proven MetaMath cold-start dose, halved from the
# 2x2 scale config's 2e-5 for capability safety).
DRY_RUN=0 \
MODE=sft \
DATASET=visualprm \
DATA_RATIO=1 \
MAX_SEQ_LEN=4096 \
SEED=42 \
TOKEN_BUDGET=${TOKEN_BUDGET_OVERRIDE:-78643200} \
LEARNING_RATE=1e-5 \
BATCH_SIZE=4 \
GRAD_ACCUM_STEPS=32 \
SAVE_STEPS=25 \
SAVE_TOTAL_LIMIT=10 \
FORCE=1 \
EXPERIMENT_NAME=coldstart_vprm_3b \
OUTPUT_DIR_OVERRIDE=$REPO/src/outputs/train/coldstart_vprm_3b/run_v1 \
bash src/scripts/run_baseline.bash
rc=$?
echo "[coldstart_vprm_3b] done=$(date) rc=$rc"
