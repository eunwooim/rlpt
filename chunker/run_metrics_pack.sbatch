#!/bin/bash
#SBATCH --job-name=chunker_metrics_pack
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/sghos104/rlpt/chunker/logs/slurm-metricspack-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/chunker/logs/slurm-metricspack-%j.out

# CPU-ONLY extended metrics pack. VALIDATION ONLY, shipped threshold from the
# release model's chunker_meta.json. One forward pass per trajectory builds a
# per-trajectory prediction cache; every metric then computes from the cache.
# Session-independent: everything (JSON, plots, METRICS_REPORT.md, the
# MORNING_REPORT appendix) is produced by this job alone.
set -uo pipefail
REPO=/scratch/sghos104/rlpt/chunker
cd "$REPO" || exit 1
mkdir -p "$REPO/logs"
export PY=$REPO/env/bin/python
export HF_HOME=/scratch/sghos104/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MPLCONFIGDIR=$REPO/logs/mplcache
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES   # force CPU

echo "[metricspack] host=$(hostname) job=${SLURM_JOB_ID:-NA} start=$(date)"

# ---------- 0. smoke gate (30 per kind, incl. model path + segeval + plots) ----------
echo; echo "=========== 0/1 SMOKE (30 per kind) ==========="
$PY metrics_pack.py --limit 30
SMOKE=$?
echo "[metricspack] smoke rc=$SMOKE"
if [ "$SMOKE" -ne 0 ]; then
  echo "[metricspack] SMOKE FAILED -- aborting before the full pass."
  exit "$SMOKE"
fi

# ---------- 1. full val pass ----------
echo; echo "=========== 1/1 FULL VAL PASS ==========="
$PY metrics_pack.py
RC=$?
echo "[metricspack] full rc=$RC  done $(date)"

echo; echo "=========== RELEASE ADDITIONS ==========="
ls -la "$REPO/release/chunker-deberta-v3-small-v1/METRICS_REPORT.md" \
      "$REPO/release/chunker-deberta-v3-small-v1/reports/metrics_pack.json" 2>/dev/null
ls -la "$REPO/release/chunker-deberta-v3-small-v1/plots/" 2>/dev/null
exit "$RC"
