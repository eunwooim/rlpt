#!/bin/bash
# Fan-out VPB evaluation: one 1-hour any-GPU job per checkpoint (+ one for the references), then a CPU scoring/report post-job.
# Usage: bash grpo_arms/launch_vpb_eval.sh <RUN_OUTDIR> <TAGPFX> "<steps, e.g. 183 180 160 140>" [MAXTOK=1024] [TAGSFX=""] [DO_REFS=1]
#   MAXTOK/TAGSFX: e.g. 2048 and _2k for the training-cap protocol; DO_REFS=1 also generates cs25 (and base3b when TAGSFX is set).
# Why: on htc, a 3h45 --gres=gpu:a100:1 request was estimated to start 18 h later (>1,000 pending GPU jobs), while 1-hour
# --gres=gpu:1 jobs backfilled onto H100/A100 nodes within minutes (2026-09-11). Each job: merge FSDP->HF (if needed) + generate.
set -uo pipefail
REPO=/scratch/sghos104/rlpt
RUN="${1:?RUN outdir}"; TAGPFX="${2:?tag prefix}"; STEPS="${3:?steps}"; MAXTOK="${4:-1024}"; TAGSFX="${5:-}"; DO_REFS="${6:-1}"
# env knobs: EVAL_SET=<jsonl> (default vpb_eval.jsonl), DEP=<slurm dependency spec for the gen jobs, e.g. afterok:123>,
#            POST=0 to skip the arm1v2 report post-job (v3 uses report_v3.sbatch instead), DO_BASE=1 to force base3b
EVAL_SET="${EVAL_SET:-$REPO/grpo_arms/data/vpb_eval.jsonl}"; DEP="${DEP:-}"; POST="${POST:-1}"
COMMON=(--partition=htc --qos=public --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=01:00:00 --mail-type=FAIL --mail-user=sghos104@asu.edu)
[[ -n "$DEP" ]] && COMMON+=(--dependency="$DEP")
[[ -n "${ACCOUNT:-}" ]] && COMMON+=(--account="$ACCOUNT")      # v4: bill a different account (header default = grp_vgupt140)
IDS=""
if [[ "$DO_REFS" == "1" ]]; then
  DO_BASE="${DO_BASE:-0}"; [[ -n "$TAGSFX" ]] && DO_BASE=1
  J=$(sbatch --parsable "${COMMON[@]}" --job-name=vpb_refs$TAGSFX \
      --export=ALL,RUN=$RUN,DO_CS25=1,DO_BASE=$DO_BASE,STEP_LIST=none,DO_SCORE=0,MAXTOK=$MAXTOK,TAGSFX=$TAGSFX,TAGPFX=$TAGPFX,EVAL_SET=$EVAL_SET $REPO/grpo_arms/eval_arm1v2.sbatch)
  echo "refs -> $J"; IDS="$J"
fi
for S in $STEPS; do
  J=$(sbatch --parsable "${COMMON[@]}" --job-name=vpb_s${S}$TAGSFX \
      --export=ALL,RUN=$RUN,DO_CS25=0,DO_BASE=0,STEP_LIST=$S,DO_SCORE=0,MAXTOK=$MAXTOK,TAGSFX=$TAGSFX,TAGPFX=$TAGPFX,EVAL_SET=$EVAL_SET $REPO/grpo_arms/eval_arm1v2.sbatch)
  echo "step $S -> $J"; IDS="${IDS:+$IDS:}$J"
done
if [[ "$POST" == "0" ]]; then echo "gen_jobs=$IDS"; echo "$IDS" > "$RUN/vpb_eval_ids${TAGSFX}.txt"; exit 0; fi
JR=$(sbatch --parsable --dependency=afterany:$IDS --mail-type=END,FAIL --mail-user=sghos104@asu.edu \
     --export=ALL,RUN=$RUN,DO_SCORE=1 $REPO/grpo_arms/report_post.sbatch)
echo "scoring/report post-job -> $JR (afterany:$IDS)"
echo "{\"gen_jobs\": \"$IDS\", \"report_post\": \"$JR\", \"maxtok\": $MAXTOK, \"tagsfx\": \"$TAGSFX\", \"steps\": \"$STEPS\"}" >> "$RUN/vpb_eval_chain.jsonl"
