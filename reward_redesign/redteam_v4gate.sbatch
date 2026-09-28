#!/bin/bash
#SBATCH --job-name=redteam_v4gate
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
# v4: probe in SOFT gate mode, then HARD mode again (must be bit-identical to probe_scored_final.jsonl / job 62990551).
set -uo pipefail
cd /scratch/sghos104/rlpt/reward_redesign || exit 1
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
unset REWARD_MODE
PY=/scratch/sghos104/rlpt/chunker/env/bin/python
for GM in soft hard; do
  echo "[redteam] ===== GATE_MODE=$GM start=$(date)"
  GATE_MODE=$GM "$PY" redteam_reward.py --in probe.jsonl --out probe_scored_v4_${GM}.jsonl --skip_old
  echo "[redteam] GATE_MODE=$GM rc=$? end=$(date)"
done
"$PY" - <<'PY'
import json, statistics as st
def load(p): return [json.loads(l) for l in open(p)]
ref, hard, soft = load("probe_scored_final.jsonl"), load("probe_scored_v4_hard.jsonl"), load("probe_scored_v4_soft.jsonl")
kh={(r["population"],r["id"]):r["new"] for r in hard}; kr={(r["population"],r["id"]):r["new"] for r in ref}
keys=["R","answer","match","format","pun","cov","prec","n_segs","n_dup","n_pairs","inv_frac","gated"]
diff=sum(1 for k in kr for f in keys if k in kh and abs(float(kr[k][f])-float(kh[k][f]))>1e-9)
print(f"[compare] hard-mode rerun vs probe_scored_final: mismatching fields = {diff} -> {'BIT-IDENTICAL' if diff==0 else 'DIFFERENT'}")
print("[compare] population medians (R): hard vs soft")
for pop in ("gold","clean","attack_giant","attack_repeat6","degenerate"):
    h=[r["new"]["R"] for r in hard if r["population"]==pop]; s=[r["new"]["R"] for r in soft if r["population"]==pop]
    sg=[r["new"]["gated"] for r in soft if r["population"]==pop]
    print(f"  {pop:15s} hard R_med={st.median(h):.2f} mean={st.mean(h):.2f} | soft R_med={st.median(s):.2f} mean={st.mean(s):.2f} gated={sum(sg)}/{len(sg)}")
cs={r["id"]:r["new"]["R"] for r in soft if r["population"]=="clean"}; dg=[r["new"]["R"] for r in soft if r["population"]=="degenerate"]
print(f"[go] soft: median R(clean)={st.median(cs.values()):.2f}, median R(hacked)={st.median(dg):.2f}, gap={st.median(cs.values())-st.median(dg):.2f} (go if gap >= 1.0 and ordering holds)")
PY
echo "[redteam] all done end=$(date)"
