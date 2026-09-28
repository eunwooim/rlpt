#!/bin/bash
#SBATCH --job-name=vprm_rlvrU_scale
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100:2
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/sghos104/rlpt/outputs/slurm-vprm_rlvrU_scale-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/outputs/slurm-vprm_rlvrU_scale-%j.out

set -uo pipefail
REPO=/scratch/sghos104/rlpt
cd "$REPO" || exit 1

export PYTHON_BIN=/scratch/sghos104/envs/rlpt-train/bin/python
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache
export PYTHONPATH="$REPO/src/train/flash_attn_shim${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
unset ROCR_VISIBLE_DEVICES
# See Env fix A in docs/local_edits_to_training_code.md
export VLLM_USE_V1=1

# Unfiltered pool: full VisualPRM400K-v1.1-Raw (565,149 rows).

echo "[rlvrU_scale] host=$(hostname) job=$SLURM_JOB_ID start=$(date)"
nvidia-smi -L

# Pre-start Ray sized to the SLURM allocation (see smoke sbatch rationale).
RAY=$(dirname "$PYTHON_BIN")/ray
"$RAY" start --head --num-cpus="$SLURM_CPUS_PER_TASK" --num-gpus=2 --disable-usage-stats
export RAY_ADDRESS=auto

# Scale config: 64 prompts/step x rollout_n 8 = 512 rollouts/step; mini 16
# (4 gradient updates/step; gradient batch = 16x8 = 128 sequences, matching
# the SFT effective batch); lr unchanged at 1e-5 (gradient batch is 2x the
# smoke run, under the 4x adjustment threshold). TOKEN_BUDGET 1e8 nominal
# -> 191 GRPO steps. MAX_TRAIN_SAMPLES=24448 caps the seed-42 pool draw so
# parquet/image prep stays ~30 min (consumed prompts: 191x64 = 12,224 <
# train split 23,959 after eval_ratio 0.02). Rewards: accuracy,format only.
# Fixed output dir + FORCE=1 -> verl resume_mode=auto resumes from
# latest_checkpointed_iteration.txt on resubmission; keep 2 actor ckpts.
DRY_RUN=0 \
MODE=rlvr \
DATASET=visualprm \
DATA_RATIO=1 \
MAX_TRAIN_SAMPLES=24448 \
MAX_SEQ_LEN=4096 \
SEED=42 \
TOKEN_BUDGET=100000000 \
LEARNING_RATE=1e-5 \
BATCH_SIZE=64 \
ROLLOUT_N=8 \
PPO_MINI_BATCH_SIZE=16 \
PPO_MICRO_BATCH_SIZE_PER_GPU=1 \
SAVE_FREQ=20 \
TEST_FREQ=25 \
VERL_EXTRA_OVERRIDES="trainer.max_actor_ckpt_to_keep=2" \
FORCE=1 \
EXPERIMENT_NAME=visualprm_rlvr_unfiltered_scale \
OUTPUT_DIR_OVERRIDE=$REPO/src/outputs/train/visualprm_rlvr_unfiltered_scale/run_scale_v1 \
bash src/scripts/run_baseline.bash
rc=$?

"$RAY" stop || true
echo "[rlvrU_scale] done=$(date) rc=$rc"
