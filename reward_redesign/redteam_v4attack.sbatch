#!/bin/bash
#SBATCH --job-name=redteam_v4atk
#SBATCH --account=grp_bshettah
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/sghos104/rlpt/reward_redesign/redteam-%j.log
# v4 probe gap: adds attack_answer_first (answer first, reference-shaped padding, finish=stop); soft mode then hard mode.
set -uo pipefail
cd /scratch/sghos104/rlpt/reward_redesign || exit 1
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
unset REWARD_MODE
PY=/scratch/sghos104/rlpt/chunker/env/bin/python
for GM in soft hard; do
  echo "[redteam] ===== GATE_MODE=$GM start=$(date)"
  GATE_MODE=$GM "$PY" redteam_reward.py --in probe.jsonl --out probe_scored_v4atk_${GM}.jsonl --skip_old
  echo "[redteam] GATE_MODE=$GM rc=$? end=$(date)"
done
"$PY" - <<'PY'
import json, statistics as st
for gm in ("hard","soft"):
    rows=[json.loads(l) for l in open(f"probe_scored_v4atk_{gm}.jsonl")]
    print(f"[table] GATE_MODE={gm}")
    for pop in ("gold","clean","attack_giant","attack_repeat6","attack_answer_first","degenerate"):
        rs=[r["new"] for r in rows if r["population"]==pop]
        print(f"  {pop:20s} n={len(rs):3d} R_med={st.median(x['R'] for x in rs):.2f} R_mean={st.mean(x['R'] for x in rs):.2f} match={st.median(x['match'] for x in rs):.2f} fmt={st.mean(x['format'] for x in rs):.2f} gated={sum(x['gated'] for x in rs)}")
    cl={r["id"]:r["new"]["R"] for r in rows if r["population"]=="clean"}
    d=[cl[r["id"]]-r["new"]["R"] for r in rows if r["population"]=="attack_answer_first" and r["id"] in cl]
    print(f"  paired median R(clean)-R(attack_answer_first) = {st.median(d):.2f} (n={len(d)}, attack>=clean {st.mean(x<=0 for x in d):.1%})")
PY
echo "[redteam] all done end=$(date)"
