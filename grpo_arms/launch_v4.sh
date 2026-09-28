#!/bin/bash
# v4 variant of launch_v3.sh: same chain, every sbatch billed to $ACCOUNT (default grp_bshettah); eval steps 60 and 30.
# Launch one v4 run as a chain: training job (public, 2xA100) + 2 afternotok resubmits (same OUTDIR, verl resume) +
# vpb_test eval jobs for steps 90 and 60 (afterok on the training chain, 1-hour gpu:1 each, 2048 tokens, greedy).
# Usage: bash grpo_arms/launch_v3.sh <ctrl|v3a|v3b> <walltime e.g. 14:00:00>
set -uo pipefail
REPO=/scratch/sghos104/rlpt
NAME="${1:?ctrl|v3a|v3b}"; WALL="${2:?walltime}"
SB=$REPO/grpo_arms/v4/$NAME.sbatch
ACCOUNT="${ACCOUNT:-grp_bshettah}"
OUT=$(grep -m1 '^export OUTDIR=' "$SB" | cut -d= -f2)
MAIL=(--mail-type=END,FAIL --mail-user=sghos104@asu.edu --account=$ACCOUNT)
mkdir -p "$OUT"
J0=$(sbatch --parsable --partition=public --qos=public --time=$WALL "${MAIL[@]}" "$SB") || { echo "sbatch failed"; exit 1; }
J1=$(sbatch --parsable --partition=public --qos=public --time=$WALL "${MAIL[@]}" --job-name=v4_${NAME}_resub1 --dependency=afternotok:$J0 --export=ALL,RESUB=1 "$SB")
J2=$(sbatch --parsable --partition=public --qos=public --time=$WALL "${MAIL[@]}" --job-name=v4_${NAME}_resub2 --dependency=afternotok:$J1 --export=ALL,RESUB=2 "$SB")
# eval chain on vpb_test (paper numbers): final step 90 + step 60, 2048 tokens; no arm1v2 post-job (report_v3.sbatch aggregates later)
ACCOUNT=$ACCOUNT EVAL_SET=$REPO/grpo_arms/data/vpb_test.jsonl DEP="afterok:$J0?afterok:$J1?afterok:$J2" POST=0 \
  bash $REPO/grpo_arms/launch_vpb_eval.sh "$OUT" "$NAME" "60 30" 2048 _2ktest 0 | tee "$OUT/eval_launch.txt"
EVAL_IDS=$(cat "$OUT/vpb_eval_ids_2ktest.txt")
printf '{"run": "%s", "train": "%s", "resub": ["%s", "%s"], "eval_split_2ktest": "%s", "wall": "%s", "outdir": "%s"}\n' \
  "$NAME" "$J0" "$J1" "$J2" "$EVAL_IDS" "$WALL" "$OUT" > "$OUT/chain.json"
cat "$OUT/chain.json"
