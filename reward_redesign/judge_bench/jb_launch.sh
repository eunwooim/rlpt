#!/bin/bash
# Submit every judge-benchmark shard as a 1-h htc gpu:1 job on grp_bshettah. 3e_score jobs depend on 3e_select (cur).
set -uo pipefail
cd /scratch/sghos104/rlpt/reward_redesign/judge_bench || exit 1
SB=jb.sbatch; IDS=""
sub() { local J; J=$(sbatch --parsable --account=grp_bshettah --job-name="jb_$1_$2_$3" ${5:+--dependency=$5} --export=ALL,TASK=$1,MODEL=$2,SHARD=$3,NSHARDS=$4 $SB) || { echo "sbatch failed $*"; exit 1; }; echo "$1 $2 $3/$4 -> $J"; IDS="${IDS:+$IDS:}$J"; echo "$J"; }
for M in cur v3; do
  for S in $(seq 0 9); do sub 3a $M $S 10 "" > /dev/null; done
  for S in $(seq 0 9); do sub 3b $M $S 10 "" > /dev/null; done
  for S in $(seq 0 4); do sub 3c_nb $M $S 5 "" > /dev/null; done
  for S in $(seq 0 4); do sub 3c_para $M $S 5 "" > /dev/null; done
  sub 3d $M 0 1 "" > /dev/null
done
# 3e_select already produced out/3e_pairs.jsonl in the smoke job 63645743 (deterministic, seed 0)
sub 3e_score cur 0 1 "" > /dev/null
sub 3e_score v3 0 1 "" > /dev/null
echo "$IDS" | tr ':' '\n' > out/job_ids.txt
echo "submitted $(wc -l < out/job_ids.txt) jobs; ids in out/job_ids.txt"
