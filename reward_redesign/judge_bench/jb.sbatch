#!/bin/bash
#SBATCH --job-name=jb
#SBATCH --account=grp_bshettah
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --exclude=scg001,scg002,scg003,scg004,sg048,sg049,sg050
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/sghos104/rlpt/reward_redesign/judge_bench/out/logs/jb-%j.log
# --export=ALL,TASK=3a,MODEL=cur,SHARD=0,NSHARDS=10
cd /scratch/sghos104/rlpt/reward_redesign/judge_bench || exit 1
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
echo "[jb] job=$SLURM_JOB_ID host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) TASK=$TASK MODEL=$MODEL SHARD=$SHARD/$NSHARDS start=$(date)"
/scratch/sghos104/rlpt/chunker/env/bin/python jb_run.py --task "$TASK" --model "$MODEL" --shard "${SHARD:-0}" --nshards "${NSHARDS:-1}"; rc=$?
echo "[jb] rc=$rc end=$(date)"; exit $rc
