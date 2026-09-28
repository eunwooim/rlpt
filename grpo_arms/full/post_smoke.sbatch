#!/bin/bash
#SBATCH --job-name=full_post_smoke
#SBATCH --account=grp_vgupt140
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/sghos104/rlpt/grpo_arms/logs/post_smoke-%j.log
# CPU post-job (afterany on the smoke): check the smoke; on PASS launch run R (adaptive entropy, 10 h + 2 resubs) on grp_vgupt140.
# On FAIL it launches nothing and records the failure (the fallback -- fixed entropy_coeff=0.002 -- is a human decision, see brief).
set -uo pipefail
cd /scratch/sghos104/rlpt || exit 1
SMOKE=$(cat grpo_arms/full/smoke_jobid.txt)
echo "[post_smoke] job=$SLURM_JOB_ID smoke=$SMOKE start=$(date)"
if /scratch/sghos104/rlpt/chunker/env/bin/python grpo_arms/full/check_smoke.py "$SMOKE"; then
  if [[ -f grpo_arms/runs/arm1_3b_rlvr_entropy/chain.json ]]; then echo "[post_smoke] R already launched"; exit 0; fi
  bash grpo_arms/full/launch_full.sh R_rlvr_entropy 10:00:00 | tee grpo_arms/full/R_launch.txt
  { echo; echo "- $(date '+%Y-%m-%d %H:%M') smoke $SMOKE PASSED -> run R launched: $(cat grpo_arms/runs/arm1_3b_rlvr_entropy/chain.json)"; } >> grpo_arms/FULL_STATUS.md
else
  { echo; echo "- $(date '+%Y-%m-%d %H:%M') smoke $SMOKE FAILED (see the Smoke test section) -> run R NOT launched; fix or fall back to fixed entropy_coeff=0.002"; } >> grpo_arms/FULL_STATUS.md
  exit 1
fi
