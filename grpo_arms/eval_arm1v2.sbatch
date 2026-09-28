#!/bin/bash
#SBATCH --job-name=arm1v2_vpb_eval
#SBATCH --account=grp_vgupt140
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/sghos104/rlpt/grpo_arms/logs/eval_arm1v2-%j.log
#SBATCH --error=/scratch/sghos104/rlpt/grpo_arms/logs/eval_arm1v2-%j.log

# VPB evaluation for arm 1 v2. --export=ALL,RUN=<OUTDIR>[,N_EVAL=4][,TAGPFX=arm1v2]
# DEFAULTS ARE SIZED FOR ONE CHECKPOINT PER JOB (1 h, any GPU type, 4 cpu, 64 G): on htc a 1-hour --gres=gpu:1 job backfills
# within minutes while a multi-hour single-A100 request waited 18 h (2026-09-11). Use grpo_arms/launch_vpb_eval.sh to fan out.
#   merges the N_EVAL highest saved FSDP checkpoints under $RUN/checkpoints -> $RUN/hf_step_<N>
#   (processor bundle copied from the cold-start checkpoint if the merger did not write it),
#   generates on the frozen 2,856-question VPB set (greedy, 1024 tokens — same as base3b /
#   arm1_step183 references), also the cold-start init (tag cs25) once, then scores everything
#   and appends the results table to OVERNIGHT_STATUS.md and ARM1_V2_REPORT.md.
set -uo pipefail
REPO=/scratch/sghos104/rlpt
cd "$REPO" || exit 1
PY=/scratch/sghos104/envs/rlpt-train/bin/python
CS25=/scratch/sghos104/rlpt/src/outputs/train/coldstart_vprm_unf_3b/run_v1/checkpoints/checkpoint-25
export HF_HOME=$REPO/data/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn      # vLLM V1 forks EngineCore after the merge touched CUDA
export PYTHONPATH="$REPO/src/train/flash_attn_shim${PYTHONPATH:+:$PYTHONPATH}"
unset ROCR_VISIBLE_DEVICES
RUN="${RUN:?RUN (training OUTDIR) required}"
N_EVAL="${N_EVAL:-4}"
TAGPFX="${TAGPFX:-arm1v2}"
DO_CS25="${DO_CS25:-1}"        # 0 = skip the cold-start generation
DO_SCORE="${DO_SCORE:-1}"      # 0 = skip scoring/table (done by a separate CPU job)
STEP_LIST="${STEP_LIST:-}"     # explicit checkpoint steps (space-separated) instead of the top-N_EVAL; "none" = no checkpoints
MAXTOK="${MAXTOK:-1024}"       # generation cap (1024 = frozen protocol of base3b / arm1_step183; 2048 = training cap)
TAGSFX="${TAGSFX:-}"           # tag suffix, e.g. _2k
DO_BASE="${DO_BASE:-0}"        # 1 = also generate Qwen/Qwen2.5-VL-3B-Instruct as base3b${TAGSFX}
EVAL_SET="${EVAL_SET:-$REPO/grpo_arms/data/vpb_eval.jsonl}"   # v3: grpo_arms/data/vpb_test.jsonl (2,456 q) for paper numbers
GEN_ARGS=(--max_tokens "$MAXTOK" --eval_set "$EVAL_SET")
STATUS=$REPO/grpo_arms/OVERNIGHT_STATUS.md
echo "[eval] job=$SLURM_JOB_ID host=$(hostname) start=$(date) RUN=$RUN N_EVAL=$N_EVAL EVAL_SET=$EVAL_SET MAXTOK=$MAXTOK TAGSFX=$TAGSFX"
rc=0

# 1. cold-start init (once)
if [[ "$DO_CS25" == "0" ]]; then
  echo "[eval] DO_CS25=0, skipping cs25"
elif [[ ! -s grpo_arms/evals/vpb_gen_cs25${TAGSFX}.jsonl ]]; then
  "$PY" grpo_arms/vpb_generate.py --model_path "$CS25" --tag "cs25${TAGSFX}" "${GEN_ARGS[@]}" || { echo "[eval] cs25 generate FAILED"; rc=1; }
else
  echo "[eval] cs25 generation already present, skipping"
fi

if [[ "$DO_BASE" == "1" && ! -s grpo_arms/evals/vpb_gen_base3b${TAGSFX}.jsonl ]]; then
  "$PY" grpo_arms/vpb_generate.py --model_path Qwen/Qwen2.5-VL-3B-Instruct --tag "base3b${TAGSFX}" "${GEN_ARGS[@]}" || { echo "[eval] base3b generate FAILED"; rc=1; }
fi

# 2. arm checkpoints: highest N_EVAL global_step_* dirs that contain an actor/ folder
if [[ "$STEP_LIST" == "none" ]]; then STEPS="";
elif [[ -n "$STEP_LIST" ]]; then STEPS="$STEP_LIST";
else STEPS=$(ls -d "$RUN"/checkpoints/global_step_*/actor 2>/dev/null | sed -E 's#.*/global_step_([0-9]+)/actor#\1#' | sort -n | tail -n "$N_EVAL" | sort -rn); fi
echo "[eval] checkpoints to evaluate (desc): $(echo $STEPS | tr '\n' ' ')"
for S in $STEPS; do
  HF="$RUN/hf_step_$S"
  if [[ ! -f "$HF/config.json" ]]; then
    "$PY" -m verl.model_merger merge --backend fsdp \
      --local_dir "$RUN/checkpoints/global_step_$S/actor" --target_dir "$HF" || { echo "[eval] merge step $S FAILED"; rc=1; continue; }
  fi
  for f in added_tokens.json chat_template.jinja merges.txt preprocessor_config.json special_tokens_map.json \
           tokenizer.json tokenizer_config.json video_preprocessor_config.json vocab.json generation_config.json; do
    [[ -f "$HF/$f" ]] || cp "$CS25/$f" "$HF/$f"
  done
  echo "[eval] merged step $S -> $HF ($(ls "$HF" | wc -l) files)"
  if [[ ! -s "grpo_arms/evals/vpb_gen_${TAGPFX}_step${S}${TAGSFX}.jsonl" ]]; then
    "$PY" grpo_arms/vpb_generate.py --model_path "$HF" --tag "${TAGPFX}_step${S}${TAGSFX}" "${GEN_ARGS[@]}" || { echo "[eval] generate step $S FAILED"; rc=1; }
  fi
done

# 3. score + table
if [[ "$DO_SCORE" == "0" ]]; then echo "[eval] DO_SCORE=0; rc=$rc end=$(date)"; exit $rc; fi
"$PY" grpo_arms/score_vpb.py || rc=1
"$PY" - "$TAGPFX" <<'EOF' | tee grpo_arms/evals/vpb_table_arm1v2.md
import json, sys, re
pfx = sys.argv[1]
rows = json.load(open("/scratch/sghos104/rlpt/grpo_arms/evals/vpb_scores.json"))
rows = [r for r in rows if r["n"] == 2856]
by = {r["file"]: r for r in rows}
def get(tag): return by.get(f"vpb_gen_{tag}.jsonl")
base, cs = get("base3b"), get("cs25")
def step_of(tag):
    m = re.search(r"_step(\d+)$", tag); return int(m.group(1)) if m else "-"
tags = ["base3b", "cs25", "arm1_step183"] + sorted([f[8:-6] for f in by if f.startswith(f"vpb_gen_{pfx}_step")], key=step_of)
print("\n| tag | step | n | accuracy | answer-extractable rate | delta vs base | delta vs cs25 | note |")
print("|---|---|---|---|---|---|---|---|")
for t in tags:
    r = get(t)
    if r is None:
        print(f"| {t} | - | - | MISSING | - | - | - | |"); continue
    d_b = f"{r['accuracy']-base['accuracy']:+.4f}" if base else "-"
    d_c = f"{r['accuracy']-cs['accuracy']:+.4f}" if cs else "-"
    note = {"base3b": "Qwen2.5-VL-3B-Instruct (reference)", "cs25": "cold-start init (checkpoint-25)",
            "arm1_step183": "reward-hacked arm-1 v1 reference"}.get(t, "arm 1 v2 (reward_v2 final)")
    print(f"| {t} | {step_of(t) if t != 'cs25' else 25} | {r['n']} | {r['accuracy']:.4f} | {r['format_marker_rate']:.4f} | {d_b} | {d_c} | {note} |")
print("\nper-source accuracy:")
srcs = sorted({s for r in rows for s in r["per_source"]})
print("| tag | " + " | ".join(srcs) + " |"); print("|---|" + "---|" * len(srcs))
for t in tags:
    r = get(t)
    if r: print(f"| {t} | " + " | ".join(f"{r['per_source'][s]['acc']:.4f}" for s in srcs) + " |")
EOF
{
  echo; echo "### VPB evaluation (job $SLURM_JOB_ID, $(date), rc=$rc)"
  echo "generation: greedy, max_tokens 1024, pixel cap 640*28*28, frozen vpb_eval.jsonl sha $(cut -c1-16 grpo_arms/data/vpb_eval.sha256).."
  cat grpo_arms/evals/vpb_table_arm1v2.md
} | tee -a "$STATUS" >> grpo_arms/ARM1_V2_REPORT.md
echo "[eval] rc=$rc end=$(date)"
exit $rc
