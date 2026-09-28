#!/bin/bash
#SBATCH --job-name=vprm_sft
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
#SBATCH --output=/scratch/sghos104/rlpt/outputs/slurm-vprm_sft-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/outputs/slurm-vprm_sft-%j.out

set -uo pipefail
REPO=/scratch/sghos104/rlpt
cd "$REPO" || exit 1

# Sol environment (supervisor's templates untouched; machine specifics via env only)
export PYTHON_BIN=/scratch/sghos104/envs/rlpt-train/bin/python
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
export PYTHONPATH="$REPO/src/train/flash_attn_shim${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
unset ROCR_VISIBLE_DEVICES

echo "[vprm_sft] host=$(hostname) job=$SLURM_JOB_ID start=$(date)"
nvidia-smi -L
$PYTHON_BIN -c "import flash_attn, torch; print('shim', flash_attn.__version__, '| cuda', torch.cuda.is_available())"

# Fixed first-run controls per src/train/README.md (passed explicitly, not via defaults)
DRY_RUN=0 \
MODE=sft \
DATASET=visualprm \
DATA_RATIO=0.01 \
MAX_SEQ_LEN=4096 \
SEED=42 \
TOKEN_BUDGET=1000000 \
LEARNING_RATE=1e-5 \
EXPERIMENT_NAME=visualprm_sft_first_baseline \
bash src/scripts/run_baseline.bash

echo "[vprm_sft] done=$(date) rc=$?"
