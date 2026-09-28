#!/bin/bash
#SBATCH --job-name=coldstart_sft_3b
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/sghos104/rlpt/outputs/slurm-coldstart_sft_3b-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/outputs/slurm-coldstart_sft_3b-%j.out
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
# Generic HF-datasets path on purpose: llava_cot alias (empty dataset_config),
# DATASET_NAME overridden to a local dir holding ONLY the cold-start jsonl.
# No RLPT_FILTERED_JSONL — must NOT route through the VisualPRM source.
export RLPT_RESUME=1
mkdir -p "$REPO/src/outputs/train/coldstart_metamath_3b/run_v1"/{logs,configs,checkpoints,eval}
echo "[coldstart_sft_3b] host=$(hostname) job=$SLURM_JOB_ID start=$(date)"
nvidia-smi -L
# Cold-start SFT: MetaMathQA 18,179 Step-N rows (sha 023c680d), lr 1e-5.
# 1 epoch = ceil(18179/128) = 142 steps; tokens/step = 2048*4*32 = 262,144
# -> TOKEN_BUDGET = 142*262144 = 37,224,448.
DRY_RUN=0 \
MODE=sft \
DATASET=llava_cot \
DATASET_NAME=$REPO/data/metamath_coldstart/train_dir \
DATA_RATIO=1 \
MAX_SEQ_LEN=2048 \
SEED=42 \
TOKEN_BUDGET=${TOKEN_BUDGET_OVERRIDE:-37224448} \
LEARNING_RATE=1e-5 \
BATCH_SIZE=4 \
GRAD_ACCUM_STEPS=32 \
SAVE_STEPS=50 \
SAVE_TOTAL_LIMIT=3 \
FORCE=1 \
EXPERIMENT_NAME=coldstart_metamath_3b \
OUTPUT_DIR_OVERRIDE=$REPO/src/outputs/train/coldstart_metamath_3b/run_v1 \
bash src/scripts/run_baseline.bash
rc=$?
echo "[coldstart_sft_3b] done=$(date) rc=$rc"
