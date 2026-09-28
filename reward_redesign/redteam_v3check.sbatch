#!/bin/bash
#SBATCH --job-name=redteam_v3chk
#SBATCH --account=grp_vgupt140
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:15:00
#SBATCH --output=/scratch/sghos104/rlpt/reward_redesign/redteam-%j.log
# v3 plumbing check: rerun the probe in match mode (REWARD_MODE unset) and compare per-row with probe_scored_final.jsonl (job 62990551).
set -uo pipefail
cd /scratch/sghos104/rlpt/reward_redesign || exit 1
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
unset REWARD_MODE
echo "[redteam] start=$(date) mode=match out=probe_scored_v3check.jsonl"
/scratch/sghos104/rlpt/chunker/env/bin/python redteam_reward.py --in probe.jsonl --out probe_scored_v3check.jsonl --skip_old
RC=$?
/scratch/sghos104/rlpt/chunker/env/bin/python - <<'PY'
import json
a=[json.loads(l) for l in open("probe_scored_final.jsonl")]; b=[json.loads(l) for l in open("probe_scored_v3check.jsonl")]
print(f"[compare] rows final={len(a)} v3check={len(b)}")
ka={(r["population"],r["id"]):r["new"] for r in a}; kb={(r["population"],r["id"]):r["new"] for r in b}
keys=["R","answer","match","format","pun","cov","prec","n_segs","n_dup","n_pairs","inv_frac","gated"]
diff=0; maxd=0.0
for k in ka:
    if k not in kb: diff+=1; continue
    for f in keys:
        x,y=ka[k][f],kb[k][f]
        if isinstance(x,float) or isinstance(y,float):
            d=abs(float(x)-float(y)); maxd=max(maxd,d)
            if d>1e-9: diff+=1
        elif x!=y: diff+=1
print(f"[compare] mismatching (row,field) count={diff} max_abs_diff={maxd:.3e} -> {'BIT-IDENTICAL' if diff==0 else 'DIFFERENT'}")
PY
echo "[redteam] rc=$RC end=$(date)"; exit $RC
