#!/bin/bash
#SBATCH --job-name=vprm_s2nli
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
#SBATCH --output=/scratch/sghos104/rlpt/outputs/slurm-vprm_s2nli-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/outputs/slurm-vprm_s2nli-%j.out

set -uo pipefail
REPO=/scratch/sghos104/rlpt
cd "$REPO" || exit 1
PY=/scratch/sghos104/envs/rlpt-train/bin/python
export HF_HOME=$REPO/data/hf_cache
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
unset ROCR_VISIBLE_DEVICES

F=$REPO/data/visualprm_v11_filtered
echo "[s2] host=$(hostname) job=$SLURM_JOB_ID start=$(date)"
nvidia-smi -L

# Smoke: 50 band traces end-to-end before committing to the full run
head -50 "$F/stage1_band_sample.jsonl" > "$F/.smoke_in.jsonl"
rm -f "$F/.smoke_kept.jsonl" "$F/.smoke_rej.jsonl" "$F/.smoke_stats.json"
$PY src/data/filter_visualprm_stage2_nli.py \
  --input "$F/.smoke_in.jsonl" --out_kept "$F/.smoke_kept.jsonl" \
  --out_rejected "$F/.smoke_rej.jsonl" --stats "$F/.smoke_stats.json" \
  || { echo "[s2] SMOKE FAILED"; exit 1; }
echo "[s2] smoke ok: $(wc -l < "$F/.smoke_kept.jsonl") kept / $(wc -l < "$F/.smoke_rej.jsonl") rejected"
rm -f "$F/.smoke_in.jsonl" "$F/.smoke_kept.jsonl" "$F/.smoke_rej.jsonl" "$F/.smoke_stats.json"

# Full tau=0.85 kept set (338,685 traces / ~1.36M pairs); resumable on requeue-free rerun
$PY src/data/filter_visualprm_stage2_nli.py \
  --input "$F/stage1_kept_tau0.85.jsonl" \
  --out_kept "$F/stage2_kept_tau0.85.jsonl" \
  --out_rejected "$F/stage2_rejected_tau0.85.jsonl" \
  --stats "$F/stage2_stats_tau0.85.json" \
  || { echo "[s2] FULL RUN FAILED"; exit 1; }
echo "[s2] tau0.85 done $(date)"

# Stage-3 band sample (15,000 traces / ~89k pairs)
$PY src/data/filter_visualprm_stage2_nli.py \
  --input "$F/stage1_band_sample.jsonl" \
  --out_kept "$F/stage2_band_kept.jsonl" \
  --out_rejected "$F/stage2_band_rejected.jsonl" \
  --stats "$F/stage2_band_stats.json" \
  || { echo "[s2] BAND RUN FAILED"; exit 1; }
echo "[s2] band done $(date)"

# tau=0.90 outputs derived from the 0.85 scores (exact — thresholds nest, and
# per-trace NLI scores depend only on trace content), no second GPU pass
$PY src/data/filter_visualprm_stage2_nli.py \
  --input "$F/stage1_kept_tau0.90.jsonl" \
  --derive_from_kept "$F/stage2_kept_tau0.85.jsonl" \
  --derive_from_rejected "$F/stage2_rejected_tau0.85.jsonl" \
  --out_kept "$F/stage2_kept_tau0.90.jsonl" \
  --out_rejected "$F/stage2_rejected_tau0.90.jsonl" \
  --stats "$F/stage2_stats_tau0.90.json" \
  || { echo "[s2] DERIVE FAILED"; exit 1; }

echo "[s2] ALL DONE $(date)"
