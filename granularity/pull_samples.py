#!/usr/bin/env python
"""Pull VPB records whose step granularity diverges most from VisualPRM.

Reference: BODY steps from the raw annos/annotations (median 24, mean 30.1).
Body-only because filter_visualprm_stage1.py strips the final-answer line
(17.2% of steps, mean 4.2 words). NOT canonical.jsonl, which pools both arms
of each +1/-1 rollout pair (means 29.5 / 36.3) and gives an inflated 32.8.

Divergence = |log(words_per_step) - log(reference median)|, so a record with
4x-longer steps ranks equally with one 4x-shorter. Plain difference would
return only QvQ (99.1 vs 30.1).

Allocation is proportional across policies with a floor of 5, so Claude and
GPT-4o stay visible instead of the list being 40 QvQ records.

NOTE: VPB steps are the generating policy's own formatting -- its humans
annotated CORRECTNESS, not boundaries. These are granularity-distance
samples, not chunker-failure samples.
"""
import json, math, os, statistics as st
from collections import defaultdict

VPB = "/scratch/sghos104/rlpt/data/visualprocessbench/test.jsonl"
N_FAR = 50
N_NEAR_PER_POLICY = 3


def words(s):
    return len(str(s).split())


def vprm_reference():
    d = json.load(open("vprm_reference.json"))["body"]
    print(f"reference (raw body steps): median {d['p50']:.0f}, "
          f"mean {d['mean']:.1f}, n {d['n']:,}")
    return d["p50"], d["mean"]


ref_med, ref_mean = vprm_reference()

rows = []
with open(VPB, encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        steps = [str(x) for x in ((r.get("response") or {}).get("steps") or [])
                 if str(x).strip()]
        if len(steps) < 2:
            continue
        wl = [words(s) for s in steps]
        wps = st.median(wl)
        if wps <= 0:
            continue
        rows.append({
            "idx": i,
            "policy": r.get("policy_model", "?"),
            "source": r.get("data_source", "?"),
            "n_steps": len(steps),
            "median_words_per_step": wps,
            "mean_words_per_step": round(st.mean(wl), 1),
            "log_dist": abs(math.log(wps) - math.log(ref_med)),
            "ratio_vs_vprm": round(wps / ref_med, 2),
            "short_nonfinal": sum(1 for s in steps[:-1] if words(s) <= 5),
            "question": (r.get("question") or "")[:300],
            "steps": steps,
            "process_correctness": (r.get("response") or {}).get("process_correctness"),
        })

print(f"VPB records with >=2 steps: {len(rows):,}")

by_pol = defaultdict(list)
for r in rows:
    by_pol[r["policy"]].append(r)

total = sum(len(v) for v in by_pol.values())
alloc = {p: max(5, round(N_FAR * len(v) / total)) for p, v in by_pol.items()}
while sum(alloc.values()) > N_FAR:
    alloc[max(alloc, key=lambda p: alloc[p])] -= 1
while sum(alloc.values()) < N_FAR:
    alloc[max(alloc, key=lambda p: len(by_pol[p]))] += 1

far, near = [], []
for p, v in by_pol.items():
    v.sort(key=lambda r: -r["log_dist"])
    far += v[:alloc[p]]
    near += v[-N_NEAR_PER_POLICY:]
far.sort(key=lambda r: -r["log_dist"])

with open("samples_far.jsonl", "w", encoding="utf-8") as f:
    for r in far:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open("samples_near.jsonl", "w", encoding="utf-8") as f:
    for r in near:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\nallocation: {alloc}  (total {sum(alloc.values())})")
print(f"\n{'policy':34s} {'n':>6s} {'med w/step':>11s} {'ratio':>7s}")
print("-" * 62)
for p in sorted(by_pol, key=lambda x: -len(by_pol[x])):
    v = by_pol[p]
    m = st.median([r["median_words_per_step"] for r in v])
    print(f"{p[:34]:34s} {len(v):>6,} {m:>11.0f} {m/ref_med:>7.2f}x")

print(f"\n--- 50 most divergent (top 12) ---")
print(f"{'#':>3s} {'policy':22s} {'src':22s} {'steps':>6s} {'w/step':>7s} {'ratio':>7s}")
for j, r in enumerate(far[:12], 1):
    print(f"{j:>3d} {r['policy'][:22]:22s} {r['source'][:22]:22s} "
          f"{r['n_steps']:>6d} {r['median_words_per_step']:>7.0f} {r['ratio_vs_vprm']:>6.2f}x")

print(f"\n-> samples_far.jsonl ({len(far)} records, full steps + labels)")
print(f"-> samples_near.jsonl ({len(near)} closest, for contrast)")
