#!/bin/bash
#SBATCH --job-name=ceval_exp4t
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/sghos104/rlpt/chunk_eval/logs/slurm-exp4t-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/chunk_eval/logs/slurm-exp4t-%j.out
# Exp 4 TIMING run (gate step): every 100th gold record (~4,958) through both
# variants; prints the full-run hour estimate. Full run is gated on user go.
set -uo pipefail
cd /scratch/sghos104/rlpt/chunk_eval || exit 1
unset ROCR_VISIBLE_DEVICES
export HF_HOME=/scratch/sghos104/.hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "[exp4t] host=$(hostname) job=${SLURM_JOB_ID:-NA} start=$(date)"
nvidia-smi -L
/scratch/sghos104/rlpt/chunker/env/bin/python exp4_resegment.py \
    --sample_every 100 --tag timing5k --dp_workers 12
RC=$?
echo "[exp4t] rc=$RC end=$(date)"
exit "$RC"
