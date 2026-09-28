#!/bin/bash
#SBATCH --job-name=jb_smoke
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
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/sghos104/rlpt/reward_redesign/judge_bench/out/logs/jb_smoke-%j.log
cd /scratch/sghos104/rlpt/reward_redesign/judge_bench || exit 1
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
PY=/scratch/sghos104/rlpt/chunker/env/bin/python
echo "[smoke] job=$SLURM_JOB_ID host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) start=$(date)"
rc=0
for M in v3 cur; do
  for T in 3a 3b 3c_nb 3c_para 3d; do
    "$PY" jb_run.py --task $T --model $M --limit 8 --nshards 1 --shard 0 2>&1 | grep -v "it/s\]" | tail -2 || rc=1
    mv out/${T}_${M}_0.jsonl out/smoke_${T}_${M}.jsonl 2>/dev/null || { echo "[smoke] NO OUTPUT for $T $M"; rc=1; }
  done
done
"$PY" jb_run.py --task 3e_select --model cur --limit 1 2>&1 | grep -v "it/s\]" | tail -3 || rc=1
mv out/3e_pairs.jsonl out/smoke_3e_pairs.jsonl 2>/dev/null || { echo "[smoke] NO 3e_select OUTPUT"; rc=1; }
echo "[smoke] rc=$rc end=$(date)"; exit $rc
