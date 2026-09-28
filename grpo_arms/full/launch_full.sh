#!/bin/bash
# Launch one FULL run (M or R) as a training chain on grp_vgupt140: training job (public, 2xA100) + 2 afternotok resubmits
# (same OUTDIR, verl resume_mode=auto). NO eval jobs here: monitor_full.py submits a 1-h htc eval per checkpoint as it appears.
# Usage: bash grpo_arms/full/launch_full.sh <M_match_full|R_rlvr_entropy> <walltime>   (ACCOUNT env overrides; default grp_vgupt140)
set -uo pipefail
REPO=/scratch/sghos104/rlpt
NAME="${1:?M_match_full|R_rlvr_entropy}"; WALL="${2:?walltime}"
SB=$REPO/grpo_arms/full/$NAME.sbatch
ACCOUNT="${ACCOUNT:-grp_vgupt140}"
OUT=$(grep -m1 '^export OUTDIR=' "$SB" | cut -d= -f2)
COMMON=(--parsable --account=$ACCOUNT --partition=public --qos=public --time=$WALL --mail-type=END,FAIL --mail-user=sghos104@asu.edu)
mkdir -p "$OUT"
if [[ -f "$OUT/DONE" ]]; then echo "refusing: $OUT/DONE exists (rename it first: the sbatch is a no-op while it is there)"; exit 1; fi
J0=$(sbatch "${COMMON[@]}" "$SB") || { echo "sbatch failed"; exit 1; }
J1=$(sbatch "${COMMON[@]}" --job-name=${NAME}_resub1 --dependency=afternotok:$J0 --export=ALL,RESUB=1 "$SB")
J2=$(sbatch "${COMMON[@]}" --job-name=${NAME}_resub2 --dependency=afternotok:$J1 --export=ALL,RESUB=2 "$SB")
PREV="${PREV_TRAIN_LOGS:-}"   # M: the v4 B job ids whose Slurm logs hold steps 1-60
printf '{"run": "%s", "train": "%s", "resub": ["%s", "%s"], "wall": "%s", "outdir": "%s", "account": "%s", "prev_train_logs": [%s], "launched": "%s"}\n' \
  "$NAME" "$J0" "$J1" "$J2" "$WALL" "$OUT" "$ACCOUNT" "$PREV" "$(date '+%Y-%m-%d %H:%M')" > "$OUT/chain.json"
cat "$OUT/chain.json"
