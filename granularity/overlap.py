#!/usr/bin/env python
"""Distribution overlap, not means.

The 1.67x VPB/VisualPRM gap is a QvQ artifact -- three of four VPB policies
sit at 21-24 words/step. Comparing dataset means is therefore uninformative.
This compares VisualPRM's words-per-step distribution against EACH VPB policy
separately, reporting overlap coefficient (OVL) and two-sample KS.

OVL = sum over bins of min(p_i, q_i) on a shared log-spaced grid. 1.0 = identical,
0.0 = disjoint. KS D is max |CDF difference|; p-value from scipy if available.
"""
import json, os, itertools, math, sys
import statistics as st
from collections import defaultdict

VPB   = "/scratch/sghos104/rlpt/data/visualprocessbench/test.jsonl"
ANNOS = "/scratch/sghos104/rlpt/data/visualprm_v11_raw/annos/annotations"
NBINS = 60


def words(s):
    return len(str(s).split())


def load_vpb():
    by_pol = defaultdict(list)
    allw = []
    with open(VPB, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pol = r.get("policy_model", "?")
            for s in (r.get("response") or {}).get("steps") or []:
                w = words(s)
                if w > 0:
                    by_pol[pol].append(w)
                    allw.append(w)
    return by_pol, allw


def load_vprm():
    """Body-step words from the RAW annotations (steps_with_score[i]["step"]),
    final-answer line excluded. Replaces the earlier canonical.jsonl read,
    which pooled +1/-1 rollout pairs and gave an inflated 32.8 mean."""
    import glob
    out = []
    for path in sorted(glob.glob(os.path.join(ANNOS, "*.jsonl"))):
        with open(path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= 3000:
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
                out += [words(str(x.get("step", ""))) for x in sws[:-1]
                        if str(x.get("step", "")).split()]
    return out


def log_grid(lo=1, hi=600, n=NBINS):
    return [math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * i / n)
            for i in range(n + 1)]


def hist(xs, edges):
    h = [0] * (len(edges) - 1)
    for x in xs:
        for i in range(len(edges) - 1):
            if edges[i] <= x < edges[i + 1]:
                h[i] += 1
                break
        else:
            if x >= edges[-1]:
                h[-1] += 1
    tot = sum(h) or 1
    return [c / tot for c in h]


def ovl(a, b, edges):
    pa, pb = hist(a, edges), hist(b, edges)
    return sum(min(x, y) for x, y in zip(pa, pb))


def ks(a, b):
    a, b = sorted(a), sorted(b)
    i = j = 0
    d = 0.0
    na, nb = len(a), len(b)
    while i < na and j < nb:
        if a[i] <= b[j]:
            i += 1
        else:
            j += 1
        d = max(d, abs(i / na - j / nb))
    return d


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def main():
    by_pol, vpb_all = load_vpb()
    vprm = load_vprm()
    edges = log_grid()

    print(f"VisualPRM steps sampled: {len(vprm):,}")
    print(f"VPB steps total        : {len(vpb_all):,}\n")

    hdr = f"{'distribution':34s} {'n':>8s} {'mean':>7s} {'p10':>5s} {'p50':>5s} " \
          f"{'p90':>5s} {'OVL':>6s} {'KS':>6s}"
    print(hdr); print("-" * len(hdr))
    print(f"{'VisualPRM400K (reference)':34s} {len(vprm):>8,} {st.mean(vprm):>7.1f} "
          f"{q(vprm,.1):>5.0f} {q(vprm,.5):>5.0f} {q(vprm,.9):>5.0f} {'--':>6s} {'--':>6s}")

    rows = []
    for pol in sorted(by_pol, key=lambda p: st.mean(by_pol[p])):
        w = by_pol[pol]
        if len(w) < 200:
            continue
        o, d = ovl(vprm, w, edges), ks(vprm, w)
        rows.append((pol, len(w), st.mean(w), o, d))
        print(f"{pol[:34]:34s} {len(w):>8,} {st.mean(w):>7.1f} "
              f"{q(w,.1):>5.0f} {q(w,.5):>5.0f} {q(w,.9):>5.0f} {o:>6.3f} {d:>6.3f}")

    o, d = ovl(vprm, vpb_all, edges), ks(vprm, vpb_all)
    print(f"{'VPB (all policies pooled)':34s} {len(vpb_all):>8,} {st.mean(vpb_all):>7.1f} "
          f"{q(vpb_all,.1):>5.0f} {q(vpb_all,.5):>5.0f} {q(vpb_all,.9):>5.0f} {o:>6.3f} {d:>6.3f}")

    print("\nOVL: 1.0 = identical distributions, 0.0 = disjoint. KS: max CDF gap.")
    print("If the compact policies show OVL well above the pooled row, the 1.67x")
    print("aggregate gap is confirmed as a QvQ artifact rather than a real")
    print("benchmark-level difference.\n")

    # ASCII densities, shared log bins
    def spark(xs):
        h = hist(xs, edges)
        mx = max(h) or 1
        chars = " ▁▂▃▄▅▆▇█"
        return "".join(chars[min(8, int(9 * v / mx))] for v in h)

    print(f"{'log-spaced 1..600 words/step':34s} density")
    print(f"{'VisualPRM400K':34s} {spark(vprm)}")
    for pol in sorted(by_pol, key=lambda p: st.mean(by_pol[p])):
        if len(by_pol[pol]) >= 200:
            print(f"{pol[:34]:34s} {spark(by_pol[pol])}")

    with open("overlap_stats.json", "w") as f:
        json.dump({"vprm": {"n": len(vprm), "mean": st.mean(vprm)},
                   "by_policy": {p: {"n": n, "mean": m, "ovl": o, "ks": d}
                                 for p, n, m, o, d in rows}}, f, indent=2)
    print("\n-> overlap_stats.json")


if __name__ == "__main__":
    main()
