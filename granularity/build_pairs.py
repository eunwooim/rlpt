#!/usr/bin/env python
"""Synthetic merge/split perturbation pairs for NLI granularity robustness.

For each gold record with body steps s_1..s_k, build candidate segmentations at
controlled granularity offsets and pair each candidate segment with the gold
step(s) it covers:

  offset  0  identity          candidate == gold step            (positive)
  offset +1  merge 2 adjacent  candidate covers 2 gold steps      (coarser)
  offset +2  merge 3 adjacent  candidate covers 3 gold steps
  offset -1  split in 2        candidate is ~half a gold step     (finer)
  offset -2  split in 3        candidate is ~a third of one

Splits land on SENTENCE boundaries (or clause boundaries when a step is a
single sentence), never mid-token -- a mid-sentence fragment would fail NLI
for reasons unrelated to granularity.

HARD NEGATIVE control: each record also contributes a candidate paired with a
gold step from a DIFFERENT record, same source. Without it, a scorer that
entails everything looks perfect at offset 0 and we could not tell robustness
from insensitivity.

SOURCE STRATIFICATION is mandatory here. The granularity bands are heavily
source-confounded (vpb_compact low is dominated by geoqa+/unigeo_calc, high by
koniq10k/mavis_function_*), so an apparent granularity effect could be domain.
Pairs are drawn evenly across sources and every row carries its source and band
so the analysis can control for both.
"""
import argparse, glob, json, os, random, re, statistics as st
from collections import Counter, defaultdict

ANNOS = "/scratch/sghos104/rlpt/data/visualprm_v11_raw/annos/annotations"

# sentence-ish boundary: terminal punctuation followed by space+capital, or a
# newline. Conservative on purpose -- a bad split is worse than no split.
SENT = re.compile(r'(?<=[.!?])\s+(?=[A-Z"(\\$])|\n+')
CLAUSE = re.compile(r'(?<=[,;:])\s+')


def sent_split(text, n_parts):
    """Split into n_parts on sentence boundaries; fall back to clauses, then
    give up (returns None) rather than cutting mid-sentence."""
    parts = [p for p in SENT.split(text) if p.strip()]
    if len(parts) < n_parts:
        parts = [p for p in CLAUSE.split(text) if p.strip()]
    if len(parts) < n_parts:
        return None
    # greedy balance by word count
    total = sum(len(p.split()) for p in parts)
    target = total / n_parts
    out, cur, acc = [], [], 0
    for p in parts:
        cur.append(p); acc += len(p.split())
        if acc >= target and len(out) < n_parts - 1:
            out.append(" ".join(cur)); cur, acc = [], 0
    if cur:
        out.append(" ".join(cur))
    return out if len(out) == n_parts else None


def band_of(m, lo, hi):
    return "low" if m < lo else ("high" if m > hi else "match")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_source", type=int, default=150,
                    help="gold records sampled per annotation file")
    ap.add_argument("--cap", type=int, default=4000,
                    help="records scanned per file before sampling")
    ap.add_argument("--band_lo", type=float, default=18.0)
    ap.add_argument("--band_hi", type=float, default=28.0)
    ap.add_argument("--min_body_steps", type=int, default=4)
    ap.add_argument("--out", default="nli_pairs.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    # ---- sample gold records, evenly across sources ----
    pool = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(ANNOS, "*.jsonl"))):
        src = os.path.basename(path).replace(".jsonl", "")
        seen = 0
        with open(path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= a.cap:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                sws = r.get("steps_with_score") or []
                body = [str(x.get("step", "")).strip() for x in sws[:-1]]
                body = [s for s in body if s.split()]
                if len(body) < a.min_body_steps:
                    continue
                seen += 1
                item = {"source": src, "line_index": i, "steps": body,
                        "median_wps": st.median([len(s.split()) for s in body])}
                if len(pool[src]) < a.per_source:
                    pool[src].append(item)
                else:
                    j = rng.randrange(seen)
                    if j < a.per_source:
                        pool[src][j] = item
        print(f"  {src[:52]:52s} {len(pool[src]):>5,}", flush=True)

    records = [x for v in pool.values() for x in v]
    rng.shuffle(records)
    print(f"\n{len(records):,} gold records across {len(pool)} sources")

    # ---- build pairs ----
    rows = []
    stat = Counter()
    by_src = defaultdict(list)
    for r in records:
        by_src[r["source"]].append(r)

    for r in records:
        steps = r["steps"]
        k = len(steps)
        band = band_of(r["median_wps"], a.band_lo, a.band_hi)
        base = {"source": r["source"], "line_index": r["line_index"],
                "band": band, "record_median_wps": r["median_wps"],
                "n_gold_steps": k}

        # offset 0 -- identity
        i = rng.randrange(k)
        rows.append({**base, "offset": 0, "kind": "identity",
                     "candidate": steps[i], "gold": steps[i],
                     "gold_span": [i, i]})
        stat[0] += 1

        # offset +1 / +2 -- merge adjacent gold steps
        for width, off in ((2, 1), (3, 2)):
            if k >= width + 1:
                i = rng.randrange(k - width + 1)
                cand = " ".join(steps[i:i + width])
                # pair the merged candidate with EACH covered gold step
                for j in range(i, i + width):
                    rows.append({**base, "offset": off, "kind": f"merge{width}",
                                 "candidate": cand, "gold": steps[j],
                                 "gold_span": [i, i + width - 1]})
                stat[off] += width

        # offset -1 / -2 -- split one gold step
        for n_parts, off in ((2, -1), (3, -2)):
            i = rng.randrange(k)
            parts = sent_split(steps[i], n_parts)
            if parts:
                for p in parts:
                    rows.append({**base, "offset": off, "kind": f"split{n_parts}",
                                 "candidate": p, "gold": steps[i],
                                 "gold_span": [i, i]})
                stat[off] += n_parts
            else:
                stat["split_unavailable"] += 1

        # hard negative -- gold step from a different record, same source
        same = by_src[r["source"]]
        if len(same) > 1:
            for _ in range(6):
                other = same[rng.randrange(len(same))]
                if other["line_index"] != r["line_index"]:
                    rows.append({**base, "offset": None, "kind": "hard_negative",
                                 "candidate": steps[rng.randrange(k)],
                                 "gold": other["steps"][rng.randrange(len(other["steps"]))],
                                 "gold_span": None})
                    stat["hard_negative"] += 1
                    break

    rng.shuffle(rows)
    with open(a.out, "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(f"\n{len(rows):,} pairs -> {a.out}")
    print(f"\n{'offset':18s} {'pairs':>8s}  {'cand words (mean)':>18s}")
    for key in (-2, -1, 0, 1, 2, "hard_negative"):
        sel = [x for x in rows if x["kind"] == "hard_negative"] if key == "hard_negative" \
              else [x for x in rows if x["offset"] == key]
        if sel:
            w = st.mean([len(x["candidate"].split()) for x in sel])
            print(f"{str(key):18s} {len(sel):>8,}  {w:>18.1f}")
    if stat["split_unavailable"]:
        print(f"\nsplit unavailable (step had too few sentence/clause "
              f"boundaries): {stat['split_unavailable']:,} attempts")
    print(f"\nband distribution: "
          f"{dict(Counter(x['band'] for x in rows))}")
    print("\nNOTE: bands are source-confounded -- analyse offset effects WITHIN")
    print("source, or the granularity effect will be confounded with domain.")


if __name__ == "__main__":
    main()
