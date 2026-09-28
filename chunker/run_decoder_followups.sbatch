#!/bin/bash
#SBATCH --job-name=chunker_decoder_followups
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/sghos104/rlpt/chunker/logs/slurm-followups-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/chunker/logs/slurm-followups-%j.out

# CPU-only decoder follow-ups, VAL only, no retraining:
#   1. seam diagnostic (cache-only; already verified interactively, re-run for
#      a coherent MORNING_REPORT block)
#   2. 2x2 decode grid on >512-token multi (stride 384/256 x mean/max merge);
#      one forward pass per stride
#   3. full metrics pack at threshold 0.35 (tag thr035) -> METRICS_REPORT_thr035.md
# Shipped-threshold (0.25) release artifacts are never overwritten.
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

echo "[followups] host=$(hostname) job=${SLURM_JOB_ID:-NA} start=$(date)"

# ---------- 0a. smoke: grid on 12 long trajectories ----------
echo; echo "=========== 0a SMOKE grid (12 traj) ==========="
$PY longtext_diag.py --part grid --limit 12
RC=$?
echo "[followups] grid smoke rc=$RC"
if [ "$RC" -ne 0 ]; then echo "[followups] GRID SMOKE FAILED -- aborting."; exit "$RC"; fi

# ---------- 0b. smoke: metrics pack @ 0.35 (30 per kind, tagged, logs only) ----------
echo; echo "=========== 0b SMOKE metrics pack thr035 ==========="
$PY metrics_pack.py --limit 30 --threshold 0.35 --tag thr035
RC=$?
echo "[followups] pack smoke rc=$RC"
if [ "$RC" -ne 0 ]; then echo "[followups] PACK SMOKE FAILED -- aborting."; exit "$RC"; fi

# ---------- 1. full: seam + grid ----------
echo; echo "=========== 1/2 FULL seam + grid ==========="
$PY longtext_diag.py --part both
RC1=$?
echo "[followups] longtext_diag rc=$RC1  $(date)"

# ---------- 2. full: metrics pack @ 0.35 ----------
echo; echo "=========== 2/2 FULL metrics pack thr035 ==========="
$PY metrics_pack.py --threshold 0.35 --tag thr035
RC2=$?
echo "[followups] metrics_pack thr035 rc=$RC2  $(date)"

echo; echo "=========== RELEASE ADDITIONS ==========="
ls -la "$REPO/release/chunker-deberta-v3-small-v1/METRICS_REPORT_thr035.md" \
      "$REPO/release/chunker-deberta-v3-small-v1/reports/metrics_pack_thr035.json" \
      "$REPO/release/chunker-deberta-v3-small-v1/reports/longtext_diag.json" 2>/dev/null
[ "$RC1" -ne 0 ] && exit "$RC1"
exit "$RC2"
