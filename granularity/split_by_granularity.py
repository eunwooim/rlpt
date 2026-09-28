#!/usr/bin/env python
"""Split VisualPRM records into three granularity bands.

Per-record granularity = MEDIAN words per BODY step (steps_with_score[:-1]);
the trailing final-answer line is excluded because filter_visualprm_stage1.py
strips it and it skews short (mean 4.2 words, 17.2% of all steps).

TWO ANCHORS, because the choice matters:

  --anchor vpb_pooled  centre 50, bands <45 / 45-55 / >55.
      This is Eun Woo's spec. It anchors on VPB's POOLED mean (54.9), which is
      a mixture artifact: three of four VPB policies sit at 21-24 w/step and
      only QvQ (99.1) drags the pooled figure up. Expect a degenerate split --
      VisualPRM's per-source range is 14.2-46.9, so almost everything lands in
      the LOW bucket.

  --anchor vpb_compact centre 23, bands <18 / 18-28 / >28.
      Anchored on the three structured-reasoning policies (Claude 21.0,
      GPT-4o 23.8, InternVL 24.1). Produces three populated buckets.

Report both; the first answers the question as asked, the second gives an
experiment with usable arms.
"""
import argparse, glob, json, os, statistics as st
from collections import Counter, defaultdict

ANNOS = "/scratch/sghos104/rlpt/data/visualprm_v11_raw/annos/annotations"

ANCHORS = {
    "vpb_pooled":  {"centre": 50.0, "lo": 45.0, "hi": 55.0,
                    "note": "VPB pooled mean 54.9 -- a QvQ mixture artifact"},
    "vpb_compact": {"centre": 23.0, "lo": 18.0, "hi": 28.0,
                    "note": "Claude 21.0 / GPT-4o 23.8 / InternVL 24.1"},
}


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="both",
                    choices=["vpb_pooled", "vpb_compact", "both"])
    ap.add_argument("--cap", type=int, default=0,
                    help="records per annotation file, 0 = all")
    ap.add_argument("--out_dir", default="bands")
    ap.add_argument("--min_body_steps", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    recs = []
    files = sorted(glob.glob(os.path.join(ANNOS, "*.jsonl")))
    print(f"reading {len(files)} annotation files"
          f"{f' (cap {a.cap}/file)' if a.cap else ' (full)'}", flush=True)
    for path in files:
        src = os.path.basename(path).replace(".jsonl", "")
        n = 0
        with open(path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if a.cap and i >= a.cap:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                sws = r.get("steps_with_score") or []
                body = [str(x.get("step", "")) for x in sws[:-1]]
                body = [s for s in body if s.split()]
                if len(body) < a.min_body_steps:
                    continue
                wl = [len(s.split()) for s in body]
                recs.append({
                    "source": src,
                    "line_index": i,
                    "image": r.get("image"),
                    "question_orig": r.get("question_orig") or r.get("question"),
                    "answer": r.get("answer"),
                    "n_body_steps": len(body),
                    "median_wps": st.median(wl),
                    "mean_wps": round(st.mean(wl), 2),
                    "min_body_score": min(
                        (x.get("score", 1.0) for x in sws[:-1]), default=None),
                    "final_score": (sws[-1].get("score") if sws else None),
                    "steps": body,
                })
                n += 1
        print(f"  {src[:52]:52s} {n:>7,}", flush=True)

    allw = [r["median_wps"] for r in recs]
    print(f"\n{len(recs):,} records with >={a.min_body_steps} body steps")
    print(f"per-record median words/step: mean {st.mean(allw):.1f}  "
          f"p10 {q(allw,.1):.0f}  p50 {q(allw,.5):.0f}  p90 {q(allw,.9):.0f}")

    anchors = ["vpb_pooled", "vpb_compact"] if a.anchor == "both" else [a.anchor]
    summary = {"n_records": len(recs), "anchors": {}}

    for name in anchors:
        cfg = ANCHORS[name]
        lo, hi = cfg["lo"], cfg["hi"]
        buckets = defaultdict(list)
        for r in recs:
            m = r["median_wps"]
            b = "low" if m < lo else ("high" if m > hi else "match")
            buckets[b].append(r)

        print(f"\n{'='*78}\nANCHOR {name}  (centre {cfg['centre']}, "
              f"match band {lo}-{hi})\n  {cfg['note']}\n{'='*78}")
        print(f"{'band':8s} {'n':>9s} {'%':>7s} {'mean wps':>9s} {'p50':>5s} "
              f"{'body steps/rec':>15s}")
        band_stats = {}
        for b, label in (("low", f"<{lo:g}"), ("match", f"{lo:g}-{hi:g}"),
                         ("high", f">{hi:g}")):
            v = buckets.get(b, [])
            if not v:
                print(f"{label:8s} {0:>9,} {0.0:>6.1f}%  (EMPTY)")
                band_stats[b] = {"n": 0}
                continue
            w = [x["median_wps"] for x in v]
            ns = [x["n_body_steps"] for x in v]
            print(f"{label:8s} {len(v):>9,} {100*len(v)/len(recs):>6.1f}% "
                  f"{st.mean(w):>9.1f} {q(w,.5):>5.0f} {st.mean(ns):>15.2f}")
            band_stats[b] = {"n": len(v), "mean_wps": st.mean(w),
                             "p50_wps": q(w, .5),
                             "mean_body_steps": st.mean(ns),
                             "top_sources": dict(
                                 Counter(x["source"] for x in v).most_common(6))}
            out = os.path.join(a.out_dir, f"{name}_{b}.jsonl")
            with open(out, "w", encoding="utf-8") as f:
                for x in v:
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")

        for b in ("low", "match", "high"):
            if band_stats.get(b, {}).get("n"):
                print(f"  {b:6s} top sources: "
                      f"{list(band_stats[b]['top_sources'].items())[:4]}")
        summary["anchors"][name] = {"config": cfg, "bands": band_stats}

    with open(os.path.join(a.out_dir, "split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n-> {a.out_dir}/{{anchor}}_{{low,match,high}}.jsonl + split_summary.json")


if __name__ == "__main__":
    main()
