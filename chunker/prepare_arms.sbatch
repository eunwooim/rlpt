#!/bin/bash
#SBATCH --job-name=prep-arms
#SBATCH --partition=public
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/sghos104/rlpt/chunker/logs/prep_arms_%j.log
set -euo pipefail
cd /scratch/sghos104/rlpt/chunker
export HF_HOME=/scratch/sghos104/.hf_cache
./env/bin/python prepare_arms.py \
    --out_dir /scratch/sghos104/rlpt/chunker/data_arms \
    --single_total 120000 --seed 42
