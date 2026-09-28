#!/bin/bash
#SBATCH --job-name=jb_single
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
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/sghos104/rlpt/reward_redesign/judge_bench/out/logs/jb_single-%j.log
# ONE job, one GPU, htc's 4 h maximum: runs every judge-benchmark shard sequentially in the launcher's order (jb_run.py skips shards whose
# output exists, so a follow-up identical job resumes where this one stopped), then aggregates whatever is present.
set -uo pipefail
cd /scratch/sghos104/rlpt/reward_redesign/judge_bench || exit 1
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
PY=/scratch/sghos104/rlpt/chunker/env/bin/python
echo "[jb_single] job=$SLURM_JOB_ID host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) start=$(date)"
run() { echo "[jb_single] >>> task=$1 model=$2 shard=$3/$4 $(date '+%H:%M:%S')"; "$PY" jb_run.py --task "$1" --model "$2" --shard "$3" --nshards "$4" || echo "[jb_single] FAILED task=$1 model=$2 shard=$3 rc=$?"; }
for M in cur v3; do
  for S in 0 1 2 3 4 5 6 7 8 9; do run 3a $M $S 10; done
  for S in 0 1 2 3 4 5 6 7 8 9; do run 3b $M $S 10; done
  for S in 0 1 2 3 4; do run 3c_nb $M $S 5; done
  for S in 0 1 2 3 4; do run 3c_para $M $S 5; done
  run 3d $M 0 1
done
run 3e_score cur 0 1
run 3e_score v3 0 1
echo "[jb_single] all shards attempted $(date); aggregating"
"$PY" jb_aggregate.py; rc=$?
MISSING=$(ls out/*_cur_*.jsonl out/*_v3_*.jsonl 2>/dev/null | grep -v smoke | wc -l)
{ echo; echo "- $(date '+%Y-%m-%d %H:%M') task 3 single job $SLURM_JOB_ID finished (aggregate rc=$rc; shard files present: $MISSING/64) -> reward_redesign/JUDGE_BENCH.md; fairshare after: $(myfairshare | grep bshettah | awk '{print $3, $6}')"; } >> /scratch/sghos104/rlpt/grpo_arms/E2_STATUS.md
echo "[jb_single] end=$(date) rc=$rc"; exit $rc
