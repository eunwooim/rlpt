#!/bin/bash
#SBATCH --job-name=arm1v2_report
#SBATCH --account=grp_vgupt140
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/sghos104/rlpt/grpo_arms/logs/report_post-%j.log
# CPU-only post-job: after the VPB eval ends (afterany), regenerate monitor.csv/png + gate-2 from the full run and
# append the monitor snapshot table, red flags, gate-2 judgment and chain outcome to ARM1_V2_REPORT.md / OVERNIGHT_STATUS.md.
set -uo pipefail
REPO=/scratch/sghos104/rlpt; cd "$REPO" || exit 1
PY=$REPO/chunker/env/bin/python
RUN="${RUN:?}"
# score all VPB generations (stdlib-only scorer) and build the table first, if the eval jobs left it to us
if [[ "${DO_SCORE:-0}" == "1" ]]; then
  "$PY" grpo_arms/score_vpb.py || echo "[report_post] score_vpb FAILED"
  "$PY" - arm1v2 <<'PYT' | tee grpo_arms/evals/vpb_table_arm1v2.md
import json, sys, re, collections
pfx = sys.argv[1]
E = "/scratch/sghos104/rlpt/grpo_arms/evals"
rows = json.load(open(f"{E}/vpb_scores.json"))
rows = [r for r in rows if r["n"] == 2856]
by = {r["file"]: r for r in rows}
def get(tag): return by.get(f"vpb_gen_{tag}.jsonl")
def step_of(tag):
    m = re.search(r"_step(\d+)", tag); return int(m.group(1)) if m else 0
def trunc(tag):
    try:
        fr = collections.Counter(json.loads(l).get("finish_reason") for l in open(f"{E}/vpb_gen_{tag}.jsonl"))
        return f"{fr.get('length', 0) / max(1, sum(fr.values())):.3f}"
    except Exception:
        return "-"
for sfx, cap in (("", 1024), ("_2k", 2048)):
    base, cs = get("base3b" + sfx), get("cs25" + sfx)
    tags = ["base3b" + sfx, "cs25" + sfx] + (["arm1_step183"] if not sfx else []) + \
           sorted([f[8:-6] for f in by if f.startswith(f"vpb_gen_{pfx}_step") and f.endswith(sfx + ".jsonl") and (sfx or "_2k" not in f)], key=step_of)
    if not any(get(t) for t in tags):
        continue
    print(f"\n**max_tokens = {cap}**" + (" (frozen protocol; base3b / arm1_step183 reference numbers)" if not sfx else " (= training response cap)"))
    print("\n| tag | step | n | accuracy | answer-extractable rate | truncated frac | delta vs base | delta vs cs25 | note |")
    print("|---|---|---|---|---|---|---|---|---|")
    for t in tags:
        r = get(t)
        if r is None:
            print(f"| {t} | - | - | MISSING | - | - | - | - | |"); continue
        d_b = f"{r['accuracy']-base['accuracy']:+.4f}" if base else "-"
        d_c = f"{r['accuracy']-cs['accuracy']:+.4f}" if cs else "-"
        note = {"base3b": "Qwen2.5-VL-3B-Instruct (reference)", "cs25": "cold-start init (checkpoint-25)",
                "arm1_step183": "reward-hacked arm-1 v1 reference"}.get(t.replace(sfx, ""), "arm 1 v2 (reward_v2 final)")
        step = 25 if t.startswith("cs25") else (step_of(t) or "-")
        print(f"| {t} | {step} | {r['n']} | {r['accuracy']:.4f} | {r['format_marker_rate']:.4f} | {trunc(t)} | {d_b} | {d_c} | {note} |")
    print("\nper-source accuracy:")
    srcs = sorted({s for r in rows for s in r["per_source"]})
    print("| tag | " + " | ".join(srcs) + " |"); print("|---|" + "---|" * len(srcs))
    for t in tags:
        r = get(t)
        if r: print(f"| {t} | " + " | ".join(f"{r['per_source'][s]['acc']:.4f}" for s in srcs) + " |")
PYT
  { echo; echo "### VPB evaluation (scored by post-job $SLURM_JOB_ID, $(date))"
    echo "generation: greedy, max_tokens 1024, pixel cap 640*28*28, frozen vpb_eval.jsonl sha $(cut -c1-16 grpo_arms/data/vpb_eval.sha256).."
    cat grpo_arms/evals/vpb_table_arm1v2.md; } | tee -a grpo_arms/OVERNIGHT_STATUS.md >> grpo_arms/ARM1_V2_REPORT.md
fi
"$PY" grpo_arms/monitor.py --run "$RUN" --every 20 > "$RUN/monitor_summary.txt" 2>&1; rc=$?
CHAIN=$(cat "$RUN/chain.json" 2>/dev/null)
JOBS=$("$PY" -c "import json,sys; c=json.loads(sys.argv[1]); print(' '.join([c['train']]+c['resub']+[c['eval']]))" "$CHAIN")
OUTCOME=$(sacct -j "$(echo $JOBS | tr ' ' ',')" -o JobID,JobName%16,State,Elapsed,ExitCode,Start,End -P | grep -vE "\.(batch|extern)")
{
  echo; echo "### Monitor + gate-2 (auto post-job $SLURM_JOB_ID, $(date), monitor rc=$rc)"
  echo "files: $RUN/monitor.csv, $RUN/monitor.png, $RUN/monitor_summary.txt"
  echo '```'; grep -vE "^\[monitor\] wrote" "$RUN/monitor_summary.txt"; echo '```'
  echo "chain outcome (sacct):"; echo '```'; echo "$OUTCOME"; echo '```'
  [[ -f "$RUN/STOP_WATCHDOG" ]] && { echo "WATCHDOG STOPPED THE RUN:"; echo '```'; cat "$RUN/STOP_WATCHDOG"; echo '```'; }
  [[ -f "$RUN/DONE" ]] && echo "DONE marker present (training chain finished cleanly)." || echo "NO DONE marker — the training chain did not finish cleanly; see logs/arm-<jobid>.log."
} | tee -a grpo_arms/OVERNIGHT_STATUS.md >> grpo_arms/ARM1_V2_REPORT.md
echo "[report_post] rc=$rc end=$(date)"; exit $rc
