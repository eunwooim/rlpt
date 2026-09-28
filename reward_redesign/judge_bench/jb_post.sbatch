#!/bin/bash
#SBATCH --job-name=jb_aggregate
#SBATCH --account=grp_bshettah
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/sghos104/rlpt/reward_redesign/judge_bench/out/logs/jb_aggregate-%j.log
# CPU post-job (afterany on all shards): aggregate -> reward_redesign/JUDGE_BENCH.md + judge_pairs_300.jsonl/.md
cd /scratch/sghos104/rlpt/reward_redesign/judge_bench || exit 1
/scratch/sghos104/rlpt/chunker/env/bin/python jb_aggregate.py; rc=$?
{ echo; echo "- $(date '+%Y-%m-%d %H:%M') task 3: jb_aggregate.py run by job $SLURM_JOB_ID (rc=$rc) -> reward_redesign/JUDGE_BENCH.md; fairshare after: $(myfairshare | grep bshettah | awk '{print $3, $6}')"; } >> /scratch/sghos104/rlpt/grpo_arms/E2_STATUS.md
echo "[jb_aggregate] rc=$rc end=$(date)"; exit $rc
