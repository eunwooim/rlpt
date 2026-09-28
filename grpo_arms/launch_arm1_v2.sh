#!/bin/bash
# Arm 1 v2 — cold-start init (VisualPRM unfiltered SFT, checkpoint-25), reward_v2 FINAL spec
# (answer gates, dedupe, no-tau Hungarian credit, offset order penalty, step-count pun), native \n\n
# segmentation (no chunker). Real run: 2xA100, MAXRESP=2048, token caps 8192 (fix b), account grp_vgupt140.
#
# Usage:
#   bash launch_arm1_v2.sh smoke                 # 3 steps x 32 prompts on htc, same overrides as the real run
#   bash launch_arm1_v2.sh real <sec_per_step>   # sizes wall/max_steps from the smoke measurement, submits the
#                                                # training job + 2 afternotok resubmits + afterok VPB eval
set -uo pipefail
REPO=/scratch/sghos104/rlpt
CS25=$REPO/src/outputs/train/coldstart_vprm_unf_3b/run_v1/checkpoints/checkpoint-25
MAIL=(--mail-type=END,FAIL --mail-user=sghos104@asu.edu)
PY=$REPO/chunker/env/bin/python
TOKCAP="--verl_extra_override actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_TOK:-8192} \
--verl_extra_override actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
--verl_extra_override actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192"
COMMON="ARM=1,MODEL=$CS25,REWARD=match_v2,SEG=native,TAU=0.45,MAXRESP=2048,NGPU=2,GPUMEM=${GPUMEM:-0.40}"
MODE="${1:-}"

if [[ "$MODE" == "smoke" ]]; then
  OUT=$REPO/grpo_arms/runs/${SMOKE_NAME:-arm1v2_smoke3}
  EXTRA="--max_steps 3 --max_train_samples 96 --save_freq 1000 --test_freq 1000 $TOKCAP"
  sbatch --job-name=arm1v2_smoke3 --partition=htc --qos=public --time=03:00:00 \
    --export=ALL,$COMMON,OUTDIR=$OUT,WATCHDOG=0,EXTRA="$EXTRA" \
    $REPO/grpo_arms/run_arm.sbatch
  exit $?
fi

if [[ "$MODE" == "real" ]]; then
  SEC="${2:?sec_per_step from the smoke required}"
  N_TRAIN=5880; BS=32; TOTAL=$((N_TRAIN / BS))                      # 183 steps = one epoch
  CAP_S=$((48 * 3600)); STARTUP_S=$((45 * 60))
  # wall = sec/step * steps * 1.5 + 45 min ; cap at 48 h by reducing max_steps
  MAX_STEPS=$TOTAL
  NEED_S=$($PY -c "import math; print(int(math.ceil($SEC * $TOTAL * 1.5 + $STARTUP_S)))")
  if (( NEED_S > CAP_S )); then
    MAX_STEPS=$($PY -c "import math; print(int(math.floor(($CAP_S - $STARTUP_S) / (1.5 * $SEC))))")
    NEED_S=$CAP_S
  fi
  WALL=$($PY -c "s=$NEED_S; print('%d-%02d:%02d:00' % (s//86400, (s%86400)//3600, (s%3600)//60))")
  OUT=$REPO/grpo_arms/runs/${RUN_NAME:-arm1_3b_matchv2_final}
  EXTRA="--max_steps $MAX_STEPS --save_freq 20 --test_freq 20 $TOKCAP --verl_extra_override trainer.max_actor_ckpt_to_keep=4"
  mkdir -p "$OUT"
  echo "[launch] sec/step=$SEC total_steps=$TOTAL max_steps=$MAX_STEPS wall=$WALL (need_s=$NEED_S) OUT=$OUT"
  SUB=(--partition=public --qos=public --time=$WALL "${MAIL[@]}" --export=ALL,$COMMON,OUTDIR=$OUT,WATCHDOG=1,EXTRA="$EXTRA")
  J0=$(sbatch --parsable --job-name=arm1v2_train "${SUB[@]}" $REPO/grpo_arms/run_arm.sbatch) || { echo "sbatch failed"; exit 1; }
  J1=$(sbatch --parsable --job-name=arm1v2_resub1 --dependency=afternotok:$J0 --export=ALL,$COMMON,OUTDIR=$OUT,WATCHDOG=1,RESUB=1,EXTRA="$EXTRA" \
       --partition=public --qos=public --time=$WALL "${MAIL[@]}" $REPO/grpo_arms/run_arm.sbatch)
  J2=$(sbatch --parsable --job-name=arm1v2_resub2 --dependency=afternotok:$J1 --export=ALL,$COMMON,OUTDIR=$OUT,WATCHDOG=1,RESUB=2,EXTRA="$EXTRA" \
       --partition=public --qos=public --time=$WALL "${MAIL[@]}" $REPO/grpo_arms/run_arm.sbatch)
  EV=$(sbatch --parsable --partition=htc --qos=public "${MAIL[@]}" --dependency="afterok:$J0?afterok:$J1?afterok:$J2" \
       --export=ALL,RUN=$OUT,N_EVAL=4,TAGPFX=arm1v2 $REPO/grpo_arms/eval_arm1v2.sbatch)
  printf '{"train": "%s", "resub": ["%s", "%s"], "eval": "%s", "sec_per_step": %s, "max_steps": %s, "wall": "%s", "outdir": "%s"}\n' \
    "$J0" "$J1" "$J2" "$EV" "$SEC" "$MAX_STEPS" "$WALL" "$OUT" > "$OUT/chain.json"
  cat "$OUT/chain.json"
  exit 0
fi
echo "usage: $0 smoke | real <sec_per_step>"; exit 2
