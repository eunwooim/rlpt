#!/bin/bash
#SBATCH --job-name=vprm_rlvr
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100:2
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/sghos104/rlpt/outputs/slurm-vprm_rlvr-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/outputs/slurm-vprm_rlvr-%j.out

set -uo pipefail
REPO=/scratch/sghos104/rlpt
cd "$REPO" || exit 1

export PYTHON_BIN=/scratch/sghos104/envs/rlpt-train/bin/python
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
export PYTHONPATH="$REPO/src/train/flash_attn_shim${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
unset ROCR_VISIBLE_DEVICES
# verl's async vLLM server creates the engine off the main thread; vllm's V1 oracle
# soft-falls back to V0 unless VLLM_USE_V1 is set explicitly, then verl's V1 AsyncLLM
# class errors. verl's main_generation_server sets this itself; main_ppo does not.
export VLLM_USE_V1=1

echo "[vprm_rlvr] host=$(hostname) job=$SLURM_JOB_ID start=$(date)"
nvidia-smi -L
$PYTHON_BIN -c "import flash_attn, torch; print('shim', flash_attn.__version__, '| cuda', torch.cuda.is_available())"

# Pre-start Ray sized to the SLURM allocation. verl's default ray.init(num_cpus=null)
# grabs all node cores and hangs on Sol/BeeGFS (48-worker import storm); connecting to
# a pre-sized cluster via RAY_ADDRESS avoids that without touching supervisor's code.
RAY=$(dirname "$PYTHON_BIN")/ray
"$RAY" start --head --num-cpus="$SLURM_CPUS_PER_TASK" --num-gpus=2 --disable-usage-stats
export RAY_ADDRESS=auto

DRY_RUN=0 \
MODE=rlvr \
DATASET=visualprm \
DATA_RATIO=0.01 \
MAX_SEQ_LEN=4096 \
SEED=42 \
TOKEN_BUDGET=1000000 \
LEARNING_RATE=1e-5 \
EXPERIMENT_NAME=visualprm_rlvr_first_baseline \
bash src/scripts/run_baseline.bash
rc=$?

"$RAY" stop || true
echo "[vprm_rlvr] done=$(date) rc=$rc"
