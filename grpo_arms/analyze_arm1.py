#!/usr/bin/env python
"""Gate-2 analysis for arm 1: component curves + match-vs-answer independence.

Reads the per-step rollout dumps (one jsonl per step, one row per rollout,
fields: score/format/answer/match/n_segments/n_matched/precision/recall).
Outputs grpo_arms/data/arm1_gate2_stats.json and prints the summary.
"""
import glob, json, os, math
from collections import defaultdict

RUN = "/scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_match_chunker"
OUT = "/scratch/sghos104/rlpt/grpo_arms/data/arm1_gate2_stats.json"

def pearson(xs, ys):
    n = len(xs)
    if n < 3: return None
    mx, my = sum(xs)/n, sum(ys)/n
    cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    vx = sum((x-mx)**2 for x in xs); vy = sum((y-my)**2 for y in ys)
    if vx <= 0 or vy <= 0: return None
    return cov / math.sqrt(vx*vy)

steps = {}
for f in sorted(glob.glob(f"{RUN}/rollouts/*.jsonl"), key=lambda p: int(os.path.basename(p).split(".")[0])):
    s = int(os.path.basename(f).split(".")[0])
    rows = [json.loads(l) for l in open(f)]
    m = [float(r["match"]) for r in rows]
    a = [float(r["answer"]) for r in rows]
    fm = [float(r["format"]) for r in rows]
    ns = [float(r["n_segments"]) for r in rows]
    rec = [float(r["match_recall"]) for r in rows]
    prec = [float(r["match_precision"]) for r in rows]
    olen = [len(r["output"].split()) for r in rows]
    m1 = [x for x, y in zip(m, a) if y == 1.0]
    m0 = [x for x, y in zip(m, a) if y == 0.0]
    steps[s] = {
        "n": len(rows),
        "match": sum(m)/len(m), "answer": sum(a)/len(a), "format": sum(fm)/len(fm),
        "score": sum(float(r["score"]) for r in rows)/len(rows),
        "n_segments": sum(ns)/len(ns), "recall": sum(rec)/len(rec), "precision": sum(prec)/len(prec),
        "out_words": sum(olen)/len(olen),
        "corr_match_answer": pearson(m, a),
        "match_given_answer1": sum(m1)/len(m1) if m1 else None,
        "match_given_answer0": sum(m0)/len(m0) if m0 else None,
        "frac_answer1": sum(a)/len(a),
    }

# aggregate into phases
def phase(lo, hi):
    ks = [k for k in steps if lo <= k <= hi]
    def avg(field, cond=lambda v: v is not None):
        vals = [steps[k][field] for k in ks if cond(steps[k][field])]
        return round(sum(vals)/len(vals), 4) if vals else None
    return {f: avg(f) for f in ("match","answer","format","score","n_segments","recall","precision",
                                 "out_words","corr_match_answer","match_given_answer1","match_given_answer0")}

summary = {
    "n_steps": len(steps),
    "phases": {f"{lo}-{hi}": phase(lo, hi) for lo, hi in ((1,20),(21,60),(61,100),(101,140),(141,183))},
    "per_step": {k: steps[k] for k in sorted(steps)},
}
json.dump(summary, open(OUT, "w"), indent=1)
print(f"{'steps':>9} {'match':>6} {'answer':>6} {'format':>6} {'score':>6} {'nseg':>5} {'recall':>6} {'prec':>6} {'words':>6} {'r(m,a)':>7} {'m|a=1':>6} {'m|a=0':>6}")
for k, v in summary["phases"].items():
    print(f"{k:>9} {v['match']:>6} {v['answer']:>6} {v['format']:>6} {v['score']:>6} {v['n_segments']:>5} {v['recall']:>6} {v['precision']:>6} {v['out_words']:>6} {v['corr_match_answer']:>7} {v['match_given_answer1']:>6} {v['match_given_answer0']:>6}")
print(f"-> {OUT}")
