#!/bin/bash
#SBATCH --job-name=ceval_exp4b
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/sghos104/rlpt/chunk_eval/logs/slurm-exp4b-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/chunk_eval/logs/slurm-exp4b-%j.out
# Exp 4b list-tier ablation on the structured subset. Smoke-gate: 2,000
# records first; abort on failure before the full pass.
set -uo pipefail
cd /scratch/sghos104/rlpt/chunk_eval || exit 1
unset ROCR_VISIBLE_DEVICES
export HF_HOME=/scratch/sghos104/.hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/scratch/sghos104/rlpt/chunker/env/bin/python
echo "[exp4b] host=$(hostname) job=${SLURM_JOB_ID:-NA} start=$(date)"
nvidia-smi -L
echo "=========== smoke: 2,000 structured records ==========="
SMOKE_DIR=/tmp/exp4b_smoke_$$
mkdir -p "$SMOKE_DIR"
$PY exp4b_ablation.py --limit_records 2000 --dp_workers 12 --outdir "$SMOKE_DIR"
RC=$?
if [ "$RC" -ne 0 ]; then echo "[exp4b] SMOKE FAILED rc=$RC"; exit "$RC"; fi
echo "=========== full structured subset ==========="
$PY exp4b_ablation.py --dp_workers 12
RC=$?
echo "[exp4b] rc=$RC end=$(date)"
exit "$RC"
