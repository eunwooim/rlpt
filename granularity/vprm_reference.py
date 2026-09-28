#!/usr/bin/env python
"""Words-per-step reference from VisualPRM400K RAW annotations.

Schema: steps_with_score = [{"step": str, "score": float,
                             "num_mc_correct": int, "num_mc_total": int}, ...]
The LAST element is the final-answer line ("Final answer: X"), which
filter_visualprm_stage1.py strips from the body (keep rule uses
steps_with_score[-1].score for answer correctness and min over [:-1] for the
body). It is ~12% of steps and skews short, so it is reported separately
rather than pooled.

Subset identity is carried by FILENAME only -- there is no source field in
the records -- so this also gives the per-source breakdown canonical.jsonl
cannot.
"""
import glob, json, os, statistics as st, sys
from collections import defaultdict

ANNOS = "/scratch/sghos104/rlpt/data/visualprm_v11_raw/annos/annotations"
CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 3000   # records per file


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


body_all, final_all, nsteps_all = [], [], []
per_src = {}
files = sorted(glob.glob(os.path.join(ANNOS, "*.jsonl")))
print(f"{len(files)} annotation files, cap {CAP} records each\n")

for path in files:
    name = os.path.basename(path).replace(".jsonl", "")
    body, fin, ns = [], [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= CAP:
                break
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            sws = r.get("steps_with_score") or []
            if len(sws) < 2:
                continue
            steps = [str(x.get("step", "")) for x in sws]
            ns.append(len(steps) - 1)                 # body steps
            body += [len(s.split()) for s in steps[:-1] if s.split()]
            if steps[-1].split():
                fin.append(len(steps[-1].split()))
    if body:
        per_src[name] = (len(ns), st.mean(ns), st.mean(body), q(body, .5))
        body_all += body; final_all += fin; nsteps_all += ns

print(f"{'source':52s} {'recs':>6s} {'body/rec':>9s} {'w/step':>8s} {'p50':>5s}")
print("-" * 84)
for name in sorted(per_src, key=lambda x: -per_src[x][2]):
    n, sr, m, p50 = per_src[name]
    print(f"{name[:52]:52s} {n:>6,} {sr:>9.2f} {m:>8.1f} {p50:>5.0f}")

print("\n" + "=" * 84)
print(f"BODY steps (final-answer line excluded -- the reference for comparison)")
print(f"  n {len(body_all):,}   mean {st.mean(body_all):.1f}   "
      f"p10 {q(body_all,.1):.0f}  p50 {q(body_all,.5):.0f}  p90 {q(body_all,.9):.0f}")
print(f"  body steps per record: mean {st.mean(nsteps_all):.2f}  p50 {q(nsteps_all,.5):.0f}")
print(f"\nFINAL-ANSWER step (stripped downstream, {len(final_all):,} of "
      f"{len(body_all)+len(final_all):,} = "
      f"{100*len(final_all)/(len(body_all)+len(final_all)):.1f}% of all steps)")
print(f"  mean {st.mean(final_all):.1f}   p50 {q(final_all,.5):.0f}")
print(f"\nALL steps pooled (comparable to VPB, which keeps its answer step)")
allw = body_all + final_all
print(f"  mean {st.mean(allw):.1f}   p50 {q(allw,.5):.0f}")

json.dump({"body": {"n": len(body_all), "mean": st.mean(body_all),
                    "p50": q(body_all, .5)},
           "all": {"n": len(allw), "mean": st.mean(allw), "p50": q(allw, .5)},
           "per_source": {k: {"records": v[0], "body_per_rec": v[1],
                              "words_per_step": v[2], "p50": v[3]}
                          for k, v in per_src.items()}},
          open("vprm_reference.json", "w"), indent=2)
print("\n-> vprm_reference.json")
