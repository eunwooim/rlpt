#!/bin/bash
#SBATCH --job-name=ood_build
#SBATCH --account=grp_bshettah
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/sghos104/rlpt/grpo_arms/logs/ood_build-%j.log
cd /scratch/sghos104/rlpt || exit 1
echo "[ood_build] start=$(date) host=$(hostname)"
/scratch/sghos104/rlpt/chunker/env/bin/python grpo_arms/ood/build_ood_sets.py; rc=$?
wc -l data/ood_eval/sets/*.jsonl; du -sh data/ood_eval/images
echo "[ood_build] rc=$rc end=$(date)"; exit $rc
