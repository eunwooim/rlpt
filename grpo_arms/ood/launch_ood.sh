#!/bin/bash
# Fan out 4 models x 7 benchmarks as 1-hour htc gpu:1 jobs on grp_bshettah (afterok on the set-builder job), then a CPU post-job that scores + writes OOD_REPORT.md.
set -uo pipefail
REPO=/scratch/sghos104/rlpt
DEP="${1:?builder job id}"
declare -A MODELS=( [base3b]="Qwen/Qwen2.5-VL-3B-Instruct" [cs25]="/scratch/sghos104/rlpt/src/outputs/train/coldstart_vprm_unf_3b/run_v1/checkpoints/checkpoint-25"
                    [M183]="/scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_matchv4_softgate/hf_step_183" [R183]="/scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_rlvr_entropy/hf_step_183" )
IDS=""
for B in mathvista mmk12 mathverse mathvision wemath dynamath mmmu; do
  for M in base3b cs25 M183 R183; do
    J=$(sbatch --parsable --account=grp_bshettah --dependency=afterok:$DEP --job-name=ood_${M}_${B} \
        --export=ALL,MODEL_PATH=${MODELS[$M]},MODEL_TAG=$M,BENCH=$B $REPO/grpo_arms/ood/ood_gen.sbatch) || { echo "sbatch failed for $M $B"; exit 1; }
    echo "$M $B -> $J"; IDS="${IDS:+$IDS:}$J"
  done
done
echo "$IDS" > $REPO/grpo_arms/ood/gen_job_ids.txt
JR=$(sbatch --parsable --account=grp_bshettah --dependency=afterany:$IDS $REPO/grpo_arms/ood/report_ood.sbatch)
echo "report post-job -> $JR (afterany on 28 gen jobs)"; echo "$JR" > $REPO/grpo_arms/ood/report_job_id.txt
