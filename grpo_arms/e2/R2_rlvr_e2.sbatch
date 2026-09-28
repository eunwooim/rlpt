#!/bin/bash
#SBATCH --job-name=e2_R2_rlvr
#SBATCH --account=grp_bshettah
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --gres=gpu:a100:2
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/sghos104/rlpt/grpo_arms/logs/arm-%j.log
#SBATCH --error=/scratch/sghos104/rlpt/grpo_arms/logs/arm-%j.log
#SBATCH --signal=B:TERM@900

# ===== E2 run "R2" (2026-09-19): epoch 2 of R (RLVR answer-only + adaptive entropy), resumed from R global_step_183 into a fresh OUTDIR, account grp_bshettah, config unchanged, dataloader seed 43
# shows exactly what differs from the arm1v2 training sbatch. Everything below this block is run_arm.sbatch verbatim.
export ARM=1 MODEL=/scratch/sghos104/rlpt/src/outputs/train/coldstart_vprm_unf_3b/run_v1/checkpoints/checkpoint-25 REWARD=match_v2 SEG=native TAU=0.45 MAXRESP=2048 NGPU=2 GPUMEM=0.40 WATCHDOG=1
export REWARD_MODE=answer_only
export GATE_MODE=hard
export ENTROPY_MODE=adaptive
export OUTDIR=/scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_rlvr_e2
export STATUS_FILE=grpo_arms/E2_STATUS.md
export EXTRA="--max_steps 366 --total_epochs 2 --save_freq 30 --test_freq 10 --learning_rate 1e-6 --extra_val_files /scratch/sghos104/rlpt/grpo_arms/data/vpb_dev.parquet --verl_extra_override actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 --verl_extra_override actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 --verl_extra_override actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 --rollout_n 5 --verl_extra_override actor_rollout_ref.actor.kl_loss_coef=0.01 --verl_extra_override actor_rollout_ref.actor.clip_ratio_low=0.2 --verl_extra_override actor_rollout_ref.actor.clip_ratio_high=0.28 --entrypoint entropy_main --verl_extra_override actor_rollout_ref.actor.calculate_entropy=True --verl_extra_override trainer.max_actor_ckpt_to_keep=-1 --verl_extra_override data.seed=43"
export RESUB="${RESUB:-0}"
# One GRPO arm. Parameterized via --export=ALL,ARM=,MODEL=,REWARD=,SEG=,TAU=,OUTDIR=[,EXTRA=,GPUMEM=]
#   ARM     1..6 (label)
#   MODEL   Qwen/Qwen2.5-VL-3B-Instruct | Qwen/Qwen2.5-VL-7B-Instruct
#   REWARD  match | vprm
#   SEG     chunker | marker | vprm
#   TAU     match threshold (match arms; ignored for vprm)
#   OUTDIR  fixed run dir (resubmit with same OUTDIR -> verl resume_mode=auto resumes)
#   EXTRA   extra train_arm.py args, e.g. "--max_train_samples 64 --max_steps 1"
#   GPUMEM  rollout gpu_memory_utilization (default 0.40)
#   NGPU    GPUs (default 2); MAXRESP response cap (default 1024)
#   WATCHDOG=1  in-job monitor every 15 min (grpo_arms/monitor.py --check): on an unmistakable
#               degeneracy signal it writes $OUTDIR/STOP_WATCHDOG, stops training, exits rc=0 so the
#               afterok eval still runs on the saved checkpoints and the afternotok resubmits are cancelled
#   Chain guards: $OUTDIR/DONE -> job is a no-op (rc=0); $OUTDIR/chain.json (written by launch_arm1_v2.sh)
#               lists the resubmit job ids that a successful run cancels. TERM 15 min before the wall
#               limit -> training is stopped, monitor/status written, exit 124 (-> afternotok resume).

set -uo pipefail
REPO=/scratch/sghos104/rlpt
cd "$REPO" || exit 1

PYTHON_BIN=/scratch/sghos104/envs/rlpt-train/bin/python
VPRM_PY=$REPO/envs/vprm-judge/bin/python
# scorer env = chunker/env (transformers 5.14): the release chunker tokenizer
# cannot load under rlpt-train's transformers 4.56 (documented trap). ALL match
# arms score there so arm 1 vs arm 2 share an identical NLI stack.
SCORER_PY=$REPO/chunker/env/bin/python
export HF_HOME=$REPO/data/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONPATH="$REPO/src/train/flash_attn_shim:$REPO/grpo_arms${PYTHONPATH:+:$PYTHONPATH}"   # grpo_arms: v3_trainer import in Ray workers
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
unset ROCR_VISIBLE_DEVICES
export VLLM_USE_V1=1   # Env fix A: verl builds the V1 AsyncLLM engine off-thread
# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments — vLLM's cumem
# allocator (sleep/wake pool) asserts against it (job 62400185). GPU headroom
# for the tight arms comes from GPUMEM=0.35 instead (arm3 62366932 OOMed at
# 79.1/79.25G with util 0.40).

GPUMEM="${GPUMEM:-0.40}"
EXTRA="${EXTRA:-}"
SOCK=/tmp/arm${ARM}_scorer_${SLURM_JOB_ID}.sock
export RLPT_ARM_REWARD="$REWARD"
export RLPT_SCORER_SOCKET="$SOCK"
export RLPT_MAX_RESPONSE_LENGTH="${MAXRESP:-1024}"
export RLPT_TOKENIZER="$MODEL"
if [[ "$REWARD" == match* ]]; then
  export RLPT_SEGMENTATION="$SEG"
  export RLPT_MATCH_TAU="$TAU"
fi


# ----- epoch-2 prologue (E2): copy the epoch-1 step-183 checkpoint into the fresh OUTDIR once; resume_mode=auto then loads it -----
E1_SRC=/scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_rlvr_entropy
if ! ls -d "$OUTDIR"/checkpoints/global_step_* >/dev/null 2>&1 && [[ ! -f "$OUTDIR/checkpoints/latest_checkpointed_iteration.txt" ]]; then   # FIRST launch only: never on a resubmit (the monitor prunes the copied 183 later; re-running this reset latest->183 and restarted epoch 2, job 63679536)
  mkdir -p "$OUTDIR/checkpoints/global_step_183" "$OUTDIR/logs"
  echo "[e2] copying $E1_SRC/checkpoints/global_step_183/actor -> $OUTDIR (no data.pt: dataloader restarts with data.seed=43) $(date)"
  cp -r "$E1_SRC/checkpoints/global_step_183/actor" "$OUTDIR/checkpoints/global_step_183/" || { echo "[e2] FATAL checkpoint copy failed"; exit 1; }
  echo 183 > "$OUTDIR/checkpoints/latest_checkpointed_iteration.txt"
  cp "$E1_SRC/checkpoints/entropy_coeff_log.jsonl" "$OUTDIR/checkpoints/entropy_coeff_log.jsonl"   # adaptive-entropy controller state (coeff_next at 183 = 0.0)
  [[ -d "$OUTDIR/data" ]] || cp -r "$E1_SRC/data" "$OUTDIR/data"   # same train/val parquet + materialised images (sha bd15b1f1 subset)
  echo "[e2] copy done $(date): $(du -sh $OUTDIR/checkpoints/global_step_183 | cut -f1)"
fi
echo "[arm$ARM] host=$(hostname) job=$SLURM_JOB_ID start=$(date)"
echo "[arm$ARM] MODEL=$MODEL REWARD=$REWARD REWARD_MODE=${REWARD_MODE:-match} SEG=$SEG TAU=${TAU:-} GPUMEM=$GPUMEM MAXRESP=${MAXRESP:-1024} NGPU=${NGPU:-2} OUTDIR=$OUTDIR EXTRA=$EXTRA"
echo "[arm$ARM] subset sha256: $(cat grpo_arms/data/train_subset.sha256)"
STATUS="${STATUS_FILE:-grpo_arms/OVERNIGHT_STATUS.md}"     # v3 runs pass STATUS_FILE=grpo_arms/V3_STATUS.md
export REWARD_MODE="${REWARD_MODE:-match}"                 # v3: answer_only for the control arm (read by score_server + arm_reward)
if [[ -f "$OUTDIR/DONE" ]]; then
  echo "[arm$ARM] DONE marker present in $OUTDIR — chain no-op"
  echo "[arm$ARM] done=$(date) rc=0"; exit 0
fi
echo "### training job $SLURM_JOB_ID started $(date) on $(hostname) (RESUB=${RESUB:-0}, wall=$(squeue -h -j $SLURM_JOB_ID -o %l), existing ckpts: $(ls -d $OUTDIR/checkpoints/global_step_* 2>/dev/null | wc -l))" >> "$STATUS"
nvidia-smi -L

# --- scoring server on GPU 1 (verl shares both GPUs; scorer footprint is small
# for NLI+chunker, ~17 GB for VisualPRM-8B — GPUMEM leaves headroom) ---
if [[ "$REWARD" == match* ]]; then
  SRV_ARGS=(grpo_arms/score_server.py --socket "$SOCK")
  [[ "$SEG" == "chunker" ]] && SRV_ARGS+=(--with_chunker)
  CUDA_VISIBLE_DEVICES=$(( ${NGPU:-2} - 1 )) "$SCORER_PY" "${SRV_ARGS[@]}" \
    > grpo_arms/logs/arm${ARM}_server_${SLURM_JOB_ID}.log 2>&1 &
else
  CUDA_VISIBLE_DEVICES=$(( ${NGPU:-2} - 1 )) "$VPRM_PY" grpo_arms/vprm_server.py --socket "$SOCK" \
    > grpo_arms/logs/arm${ARM}_server_${SLURM_JOB_ID}.log 2>&1 &
fi
SRV_PID=$!

# wait for the server socket (VisualPRM load can take minutes from BeeGFS)
for i in $(seq 1 240); do
  [[ -S "$SOCK" ]] && break
  if ! kill -0 $SRV_PID 2>/dev/null; then
    echo "[arm$ARM] FATAL: scoring server died during startup"; exit 1
  fi
  sleep 5
done
if [[ ! -S "$SOCK" ]]; then
  echo "[arm$ARM] FATAL: scoring server socket never appeared"; kill $SRV_PID; exit 1
fi
"$PYTHON_BIN" - <<EOF || { echo "[arm$ARM] FATAL: server ping failed"; kill $SRV_PID; exit 1; }
import json, socket
c = socket.socket(socket.AF_UNIX); c.settimeout(120); c.connect("$SOCK")
c.sendall(b'{"op": "ping"}\n')
buf = b""
while not buf.endswith(b"\n"):
    buf += c.recv(65536)
print("[arm$ARM] server ping:", json.loads(buf))
EOF

# --- Ray pre-sized to the allocation (BeeGFS import-storm guard) ---
# Job 62044345 died because ray-head came up degraded (dashboard failure + GCS
# deadline-exceeded) and nothing checked: verify with `ray status`, retry once
# from a clean /tmp/ray, else fail loudly before burning the walltime.
RAY=$(dirname "$PYTHON_BIN")/ray
start_ray() {
  # cap the object store: Ray sizes it from NODE memory (154G on job 62046761),
  # ignoring the slurm cgroup — that plus FSDP offload (~208G steady, ~224G at
  # checkpoint save) OOM-killed the job at 200G. 320G req + 30G plasma cap.
  # Isolate this job's Ray from any other cluster on the node (shared /tmp/ray + default
  # port 6379 collided with another user's head on sg009, job 62705704).
  RAY_TMP="/tmp/ray_${SLURM_JOB_ID}"; RAY_PORT=$((20000 + SLURM_JOB_ID % 20000))
  export RAY_ADDRESS="$(hostname -i | awk '{print $1}'):${RAY_PORT}"
  echo "[arm$ARM] ray temp=$RAY_TMP address=$RAY_ADDRESS"
  "$RAY" start --head --num-cpus="$SLURM_CPUS_PER_TASK" --num-gpus="${NGPU:-2}" \
    --temp-dir="$RAY_TMP" --port="$RAY_PORT" \
    --object-store-memory=30000000000 \
    --disable-usage-stats --include-dashboard=false
  for i in $(seq 1 12); do
    if "$RAY" status --address "$RAY_ADDRESS" >/dev/null 2>&1; then return 0; fi
    sleep 5
  done
  return 1
}
if ! start_ray; then
  echo "[arm$ARM] WARN: ray head unhealthy; retrying from clean state"
  "$RAY" stop --force || true
  rm -rf /tmp/ray
  sleep 10
  if ! start_ray; then
    echo "[arm$ARM] FATAL: ray head failed twice"; kill $SRV_PID; exit 1
  fi
fi
export RAY_ADDRESS=auto
"$RAY" status --address "$RAY_ADDRESS" | head -20

TAU_ARG=""
[[ "$REWARD" == match* ]] && TAU_ARG="--tau $TAU"
stop_training() {
  pkill -u "$USER" -f "verl.trainer.main_ppo|entropy_main" 2>/dev/null
  kill "$TRAIN_PID" 2>/dev/null
}
on_term() {
  echo "[arm$ARM] TERM received $(date) (wall limit in ~15 min) — stopping training for a clean resume"
  TIMED_OUT=1
  stop_training
}
TIMED_OUT=0
trap on_term TERM
"$PYTHON_BIN" grpo_arms/train_arm.py \
  --arm "$ARM" \
  --base_model "$MODEL" \
  --reward "$REWARD" \
  --segmentation "$SEG" \
  $TAU_ARG \
  --output_dir "$OUTDIR" \
  --gpu_memory_utilization "$GPUMEM" \
  --max_response_length "${MAXRESP:-1024}" \
  --n_gpus_per_node "${NGPU:-2}" \
  $EXTRA &
TRAIN_PID=$!
WD_PID=""
if [[ "${WATCHDOG:-0}" == "1" ]]; then
  (
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
      sleep 900
      out=$("$SCORER_PY" grpo_arms/monitor.py --run "$OUTDIR" --check 2>&1)
      echo "[watchdog $(date '+%m-%d %H:%M')] $(echo "$out" | tr '\n' '|')"
      if [[ "$(echo "$out" | tail -n 1)" == "STOP" ]]; then
        { echo "STOP by watchdog $(date) job $SLURM_JOB_ID"; echo "$out"; } > "$OUTDIR/STOP_WATCHDOG"
        echo "[watchdog] RED FLAG confirmed -> stopping training (see $OUTDIR/STOP_WATCHDOG)"
        stop_training
        break
      fi
    done
  ) &
  WD_PID=$!
fi
wait "$TRAIN_PID"; rc=$?
[[ -n "$WD_PID" ]] && kill "$WD_PID" 2>/dev/null
trap - TERM

"$RAY" stop || true
kill $SRV_PID 2>/dev/null
rm -f "$SOCK"

# --- post-run: chain bookkeeping, monitor, status ---
if [[ -f "$OUTDIR/STOP_WATCHDOG" ]]; then
  echo "[arm$ARM] stopped by watchdog -> terminal state; rc=0 so the eval chain runs on the saved checkpoints"; rc=0
elif [[ "$TIMED_OUT" == "1" ]]; then
  echo "[arm$ARM] timed out before finishing -> rc=124 (afternotok resubmit resumes from the last checkpoint)"; rc=124
fi
[[ $rc -eq 0 ]] && touch "$OUTDIR/DONE"
if [[ $rc -eq 0 && -f "$OUTDIR/chain.json" ]]; then
  for j in $("$SCORER_PY" -c "import json;print(' '.join(reversed(json.load(open('$OUTDIR/chain.json'))['resub'])))"); do
    [[ "$j" != "$SLURM_JOB_ID" ]] && { scancel "$j" 2>/dev/null && echo "[arm$ARM] cancelled pending resubmit $j"; }
  done
fi
"$SCORER_PY" grpo_arms/monitor.py --run "$OUTDIR" --every 20 > "$OUTDIR/monitor_summary.txt" 2>&1
tail -n 40 "$OUTDIR/monitor_summary.txt"
{
  echo; echo "### training job $SLURM_JOB_ID ended $(date) rc=$rc (RESUB=${RESUB:-0}, timed_out=$TIMED_OUT, watchdog_stop=$([[ -f $OUTDIR/STOP_WATCHDOG ]] && echo yes || echo no), ckpts: $(ls -d $OUTDIR/checkpoints/global_step_* 2>/dev/null | sed 's#.*global_step_##' | sort -n | tr '\n' ' '))"
  echo '```'; grep -vE "^\[monitor\] wrote" "$OUTDIR/monitor_summary.txt" | tail -n 30; echo '```'
} >> "$STATUS"
echo "[arm$ARM] done=$(date) rc=$rc"
exit $rc
