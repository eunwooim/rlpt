#!/usr/bin/env python
"""VPB schema is nested under `response`. Reports granularity overall and
PER POLICY MODEL -- VPB responses come from several policies, so the
per-policy split is direct evidence for how generator choice moves step
granularity, without generating anything.
"""
import json, os, itertools, statistics as st
from collections import Counter, defaultdict

VPB   = "/scratch/sghos104/rlpt/data/visualprocessbench/test.jsonl"
CANON = "/scratch/sghos104/rlpt/canonical.jsonl"

def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs)-1, int(p*len(xs)))] if xs else 0

def stats(name, nsteps, wlens):
    print(f"{name:34s} recs {len(nsteps):>6,}  steps/rec {st.mean(nsteps):5.2f} "
          f"(p50 {pct(nsteps,.5):2.0f})   words/step {st.mean(wlens):6.1f} "
          f"(p50 {pct(wlens,.5):3.0f}  p90 {pct(wlens,.9):4.0f})")

# ---------------- VPB ----------------
rows = [json.loads(l) for l in open(VPB, encoding="utf-8") if l.strip()]
print(f"VisualProcessBench: {len(rows):,} records\n")

by_policy_n, by_policy_w = defaultdict(list), defaultdict(list)
by_source_n = defaultdict(list)
lab = Counter(); vpb_n, vpb_w = [], []
for r in rows:
    steps = (r.get("response") or {}).get("steps") or []
    pol = r.get("policy_model", "?")
    src = r.get("data_source", "?")
    vpb_n.append(len(steps)); by_policy_n[pol].append(len(steps)); by_source_n[src].append(len(steps))
    for s in steps:
        w = len(str(s).split())
        vpb_w.append(w); by_policy_w[pol].append(w)
    for x in (r.get("response") or {}).get("process_correctness") or []:
        lab[str(x)] += 1

print("policy models:", dict(Counter(r.get("policy_model","?") for r in rows)))
print("data sources :", dict(Counter(r.get("data_source","?") for r in rows)))
print("label values :", dict(lab.most_common()))
print()
stats("VPB (all)", vpb_n, vpb_w)
print("\n--- VPB by policy model ---")
for p in sorted(by_policy_n, key=lambda x: -len(by_policy_n[x])):
    if len(by_policy_n[p]) >= 30:
        stats(f"  {p[:30]}", by_policy_n[p], by_policy_w[p])

# ---------------- VisualPRM ----------------
size = os.path.getsize(CANON)
vp_n, vp_w = [], []
with open(CANON, "rb") as f:
    for k in range(40):
        f.seek(int(size*k/40)); f.readline()
        for line in itertools.islice(f, 3000):
            try: r = json.loads(line)
            except Exception: continue
            s = r.get("steps") or []
            if len(s) < 2: continue
            vp_n.append(len(s))
            vp_w += [len(str(x).split()) for x in s]

print("\n" + "="*88)
stats("VisualPRM400K (multi-step)", vp_n, vp_w)
stats("VisualProcessBench (all)", vpb_n, vpb_w)
print(f"\n  ratio VPB/VPRM  words-per-step {st.mean(vpb_w)/st.mean(vp_w):.2f}x   "
      f"steps-per-record {st.mean(vpb_n)/st.mean(vp_n):.2f}x")
print("\nNote: VisualPRM 32.8 words/step ~= 43-46 tokens, consistent with the")
print("46.8 tokens/step recorded in the xdataset REPORT. PRM800K was 31.6 tokens.")
