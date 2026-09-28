#!/bin/bash
#SBATCH --job-name=sympy_audit
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/sghos104/rlpt/outputs/slurm-sympy_audit-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/outputs/slurm-sympy_audit-%j.out

set -uo pipefail
REPO=/scratch/sghos104/rlpt
cd "$REPO" || exit 1
echo "[sympy_audit] host=$(hostname) job=$SLURM_JOB_ID start=$(date)"

/scratch/sghos104/envs/rlpt-train/bin/python src/data/sympy_extract_audit.py \
  --input data/visualprm_v11_filtered/stage2_kept_tau0.85.jsonl \
  --outdir data/visualprm_v11_filtered/sympy_audit \
  --per-source 150 --seed 0
rc=$?
echo "[sympy_audit] done=$(date) rc=$rc"
exit $rc
