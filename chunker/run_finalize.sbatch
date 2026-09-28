#!/bin/bash
#SBATCH --job-name=chunker_finalize
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/sghos104/rlpt/chunker/logs/slurm-finalize-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/chunker/logs/slurm-finalize-%j.out

# SESSION-INDEPENDENT FINALIZATION. Everything needed to produce the release runs
# here, in order, with no dependency on an interactive agent:
#   1. threshold_sweep.py  -> val threshold table incl. zero-cut %, decision rule
#   2. package_release.py  -> release/ dir with the SELECTED threshold baked in
#   3. make_morning_report.py -> MORNING_REPORT.md
#
# NOTE: no `set -e`. A failure in an early step must NOT prevent the report from
# being written -- the report degrades to "MISSING" sections and says what broke,
# which is far more useful overnight than an empty directory.
set -uo pipefail
REPO=/scratch/sghos104/rlpt/chunker
cd "$REPO" || exit 1
mkdir -p "$REPO/logs"
export PY=$REPO/env/bin/python
export HF_HOME=/scratch/sghos104/.hf_cache
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
unset ROCR_VISIBLE_DEVICES

RUN=$REPO/runs/full_plain
WINNER=$RUN/best
VERSION=v1
RELEASE=$REPO/release/chunker-deberta-v3-small-$VERSION
STAMP=$(date -Iseconds)

echo "[finalize] host=$(hostname) job=${SLURM_JOB_ID:-NA} start=$(date)"
nvidia-smi -L || true

# ---------- 0. smoke gate ----------
# The interactive node is CPU-only/4 GB and cannot load the model, so the sweep
# code path cannot be validated before submission. Run it on 20 trajectories
# first: if it is broken, fail fast here instead of burning ~1h, while still
# falling through to packaging + report with the training-time threshold.
echo; echo "=========== 0/3 SWEEP SMOKE (20 trajectories) ==========="
$PY threshold_sweep.py \
  --model_dir "$WINNER" \
  --val_traj "$REPO/data/val_traj.jsonl" \
  --out "$REPO/logs/sweep_smoke.json" \
  --segment_stats "$RUN/segment_stats.json" \
  --limit_multi 20
SMOKE_RC=$?
echo "[finalize] sweep smoke rc=$SMOKE_RC"

# ---------- 1. threshold sweep on VALIDATION ----------
echo; echo "=========== 1/3 THRESHOLD SWEEP (val) ==========="
if [ "$SMOKE_RC" -ne 0 ]; then
  echo "[finalize] SKIPPING full sweep — smoke failed (rc=$SMOKE_RC). Packaging will use"
  echo "[finalize] the training-time threshold and the report will flag the sweep as MISSING."
  SWEEP_RC=$SMOKE_RC
else
  $PY threshold_sweep.py \
    --model_dir "$WINNER" \
    --val_traj "$REPO/data/val_traj.jsonl" \
    --out "$RUN/threshold_sweep.json" \
    --segment_stats "$RUN/segment_stats.json"
  SWEEP_RC=$?
fi
echo "[finalize] threshold_sweep rc=$SWEEP_RC"

# extract the chosen threshold; fall back to the training-time value on any failure
CHOSEN=$($PY - <<'PY'
import json, os
p = "/scratch/sghos104/rlpt/chunker/runs/full_plain/threshold_sweep.json"
try:
    d = json.load(open(p))
    c = d.get("chosen")
    print(c["threshold"] if c else "")
except Exception:
    print("")
PY
)
echo "[finalize] chosen threshold = '${CHOSEN:-<none, using training-time value>}'"

# ---------- 2. package ----------
echo; echo "=========== 2/3 PACKAGE RELEASE ==========="
if [ -n "$CHOSEN" ]; then
  $PY package_release.py --run_dir "$RUN" --version "$VERSION" --threshold "$CHOSEN"
else
  $PY package_release.py --run_dir "$RUN" --version "$VERSION"
fi
echo "[finalize] package rc=$?"

# ---------- 3. morning report ----------
echo; echo "=========== 3/3 MORNING REPORT ==========="
$PY make_morning_report.py \
  --run_dir "$RUN" \
  --weighted_dir "$REPO/runs/full_weighted" \
  --release_dir "$RELEASE" \
  --stamp "$STAMP" \
  --out "$REPO/MORNING_REPORT.md"
echo "[finalize] report rc=$?"

# copy the report into the release dir too, so the artifact is self-describing
cp -f "$REPO/MORNING_REPORT.md" "$RELEASE/MORNING_REPORT.md" 2>/dev/null || true

echo; echo "=========== RELEASE CONTENTS ==========="
ls -laR "$RELEASE" 2>/dev/null | head -40
echo "[finalize] done $(date)"
