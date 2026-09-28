#!/bin/bash
#SBATCH --job-name=chunker_oversplit
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/sghos104/rlpt/chunker/logs/slurm-oversplit-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/chunker/logs/slurm-oversplit-%j.out

# CPU-ONLY over-splitting analysis. No GPU, no retraining: one forward pass per
# val trajectory (probs computed once, reused across every metric). Threshold =
# the SHIPPED value (0.25), read from the release model's chunker_meta.json.
set -uo pipefail
REPO=/scratch/sghos104/rlpt/chunker
cd "$REPO" || exit 1
mkdir -p "$REPO/logs"
export PY=$REPO/env/bin/python
export HF_HOME=/scratch/sghos104/.hf_cache
export HF_HUB_OFFLINE=1          # compute nodes have no egress; all files are local
export TRANSFORMERS_OFFLINE=1    # -> avoid the blocking hub HEAD that hung job 59643608
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES   # force CPU

echo "[oversplit] host=$(hostname) job=${SLURM_JOB_ID:-NA} start=$(date) threads=$OMP_NUM_THREADS"

# ---------- 0. smoke gate (20/kind) ----------
# The interactive node is CPU-only/4 GB and cannot load the model, so validate
# the code path here on a tiny slice before the full ~19k-trajectory pass.
echo; echo "=========== 0/1 SMOKE (20 per kind) ==========="
$PY oversplit_analysis.py \
  --out_dir "$REPO/logs" \
  --release_reports "" \
  --readme "" \
  --limit 20
SMOKE=$?
echo "[oversplit] smoke rc=$SMOKE"
if [ "$SMOKE" -ne 0 ]; then
  echo "[oversplit] SMOKE FAILED -- aborting before the full pass."
  exit "$SMOKE"
fi

# ---------- 1. full val pass ----------
echo; echo "=========== 1/1 FULL VAL PASS ==========="
$PY oversplit_analysis.py
echo "[oversplit] full rc=$?  done $(date)"
