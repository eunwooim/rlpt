#!/usr/bin/env python
"""EXP 3 — weld detection via sympy anchors (deterministic, CPU).

A "weld" is two reasoning moves fused into one chunk. Signature: the chunk's
extracted claims contain >= 2 DISJOINT anchor assertions — symbol = value
claims for different symbols with no shared claim chaining them.

Operationalization (single place, unit-tested below):
  - claims come verbatim from src/data/sympy_extract_audit.extract_claims
    (v4 normalization rules); each side parsed with parse_side (SIGALRM 2s).
  - anchor assertion: a claim where one side is a SINGLE free symbol and the
    other side is GROUND (parses, no free symbols). -> (symbol, value_str).
  - claim graph: nodes = free symbols of all parsed claims; each claim links
    all symbols appearing in it (lhs+rhs free symbols) into one component.
  - weld_suspect: the chunk has >= 2 anchors whose symbols lie in >= 2
    DIFFERENT connected components of the claim graph.
  Chunks with < 2 anchors can never be weld_suspect (counted as such).

Populations (both restricted to the three verifiable tiers via source_map):
  PRODUCED: chunks of records with method in {marker, model} — reservoir
            100K chunks, seed 0, per-tier reservoirs sized by first-pass
            tier chunk counts (proportional).
  GOLD:     steps of passthrough (multi-step) records from the same tiers —
            reservoir 100K steps, same scheme.
Baseline caveat encoded in the report: some gold steps legitimately assert
two values; the comparison is produced-vs-gold rate per tier, not absolute.

Two passes: pass 1 counts tier populations (cheap, no sympy); pass 2 fills
reservoirs; then multiprocessing sympy analysis of sampled chunks.
"""
import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import orjson

sys.path.insert(0, "/scratch/sghos104/rlpt/src/data")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sympy_extract_audit import extract_claims, parse_side  # noqa: E402
from source_map import TierMap  # noqa: E402

TIERS = ("function", "geometry", "arithmetic")


def analyze_chunk(text):
    """-> dict(n_claims, n_parsed, n_anchors, n_components, weld_suspect,
              anchors=[(sym, val)])"""
    claims, _, _ = extract_claims(text)
    parsed = []
    for lhs_s, rhs_s, _trimmed in claims:
        l, e1 = parse_side(lhs_s)
        if l is None:
            continue
        r, e2 = parse_side(rhs_s)
        if r is None:
            continue
        try:
            lf, rf = l.free_symbols, r.free_symbols
        except Exception:
            continue
        parsed.append((lhs_s, rhs_s, lf, rf))

    # union-find over symbols
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    anchors = []
    for lhs_s, rhs_s, lf, rf in parsed:
        syms = lf | rf
        for s in syms:
            parent.setdefault(s, s)
        syms = list(syms)
        for a, b in zip(syms, syms[1:]):
            union(a, b)
        if len(lf) == 1 and not rf:
            anchors.append((next(iter(lf)), rhs_s))
        elif len(rf) == 1 and not lf:
            anchors.append((next(iter(rf)), lhs_s))

    comp_of_anchor = {find(sym) for sym, _ in anchors}
    weld = len(anchors) >= 2 and len(comp_of_anchor) >= 2
    return {"n_claims": len(claims), "n_parsed": len(parsed),
            "n_anchors": len(anchors),
            "n_anchor_components": len(comp_of_anchor),
            "weld_suspect": weld,
            "anchors": [(str(s), v) for s, v in anchors[:6]]}


def _self_test():
    r = analyze_chunk("First, x = 5. Independently, y = 7.")
    assert r["weld_suspect"], r
    r = analyze_chunk("We have x = 5, so y = x + 2 = 7.")
    assert not r["weld_suspect"], r          # x,y chained via x+2=y claim
    r = analyze_chunk("The total is 3 + 4 = 7.")
    assert not r["weld_suspect"] and r["n_anchors"] == 0, r
    r = analyze_chunk("AB = 6. Also CD = 8.")
    assert r["weld_suspect"], r
    r = analyze_chunk("x = 5.")
    assert not r["weld_suspect"] and r["n_anchors"] == 1, r
    r = analyze_chunk("A = 30, and since B = A, B = 30.")
    assert not r["weld_suspect"], r          # chained through shared claim
    # KNOWN LIMITATION (absorbed by the gold baseline, same extractor both
    # sides): a chain written through prose ("angle B = angle A") loses its
    # linking claim to prose-trimming, so both assignments look disjoint:
    r = analyze_chunk("angle A = 30 degrees, and since angle B = angle A, angle B = 30.")
    assert r["weld_suspect"], r
    print("[exp3] self-test OK", flush=True)


def work(item):
    tier, side, text = item
    try:
        res = analyze_chunk(text)
    except Exception as e:
        return (tier, side, {"error": f"{type(e).__name__}: {e}"}, text)
    return (tier, side, res, text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/scratch/sghos104/rlpt/canonical_chunked_v2.jsonl")
    ap.add_argument("--outdir", default="/scratch/sghos104/rlpt/chunk_eval")
    ap.add_argument("--n_produced", type=int, default=100_000)
    ap.add_argument("--n_gold", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    _self_test()
    tm = TierMap()

    # ---- pass 1: tier chunk counts ----
    prod_counts, gold_counts = Counter(), Counter()
    with open(args.input, "rb") as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            if i % 2_000_000 == 0:
                print(f"[exp3 pass1] {i:,}", flush=True)
            rec = orjson.loads(line)
            prov = rec["metadata"]["chunk_provenance"]
            m = prov["method"]
            if m not in ("marker", "model", "passthrough"):
                continue
            tier, _ = tm.classify(rec.get("images") or [])
            if tier is None:
                continue
            if m == "passthrough":
                if len(rec["steps"]) > 1:
                    gold_counts[tier] += len(rec["steps"])
            else:
                prod_counts[tier] += len(rec["steps"])
    print("[exp3] produced chunk counts per tier:", dict(prod_counts), flush=True)
    print("[exp3] gold step counts per tier:", dict(gold_counts), flush=True)

    def quotas(counts, n_total):
        tot = sum(counts.values())
        q = {t: max(1, round(n_total * counts[t] / tot)) if counts[t] else 0
             for t in TIERS}
        return q

    qp, qg = quotas(prod_counts, args.n_produced), quotas(gold_counts, args.n_gold)
    print("[exp3] quotas produced:", qp, "gold:", qg, flush=True)

    # ---- pass 2: per-tier reservoirs ----
    rng = random.Random(0)
    res_p = {t: [] for t in TIERS}
    res_g = {t: [] for t in TIERS}
    seen_p, seen_g = Counter(), Counter()

    def reservoir(store, seen, key, quota, item):
        seen[key] += 1
        r = store[key]
        if len(r) < quota:
            r.append(item)
        else:
            j = rng.randrange(seen[key])
            if j < quota:
                r[j] = item

    with open(args.input, "rb") as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            if i % 2_000_000 == 0:
                print(f"[exp3 pass2] {i:,}", flush=True)
            rec = orjson.loads(line)
            prov = rec["metadata"]["chunk_provenance"]
            m = prov["method"]
            if m not in ("marker", "model", "passthrough"):
                continue
            tier, _ = tm.classify(rec.get("images") or [])
            if tier is None:
                continue
            if m == "passthrough":
                if len(rec["steps"]) > 1:
                    for ch in rec["steps"]:
                        reservoir(res_g, seen_g, tier, qg[tier], ch)
            else:
                for ch in rec["steps"]:
                    reservoir(res_p, seen_p, tier, qp[tier], ch)

    jobs = ([(t, "produced", ch) for t in TIERS for ch in res_p[t]]
            + [(t, "gold", ch) for t in TIERS for ch in res_g[t]])
    print(f"[exp3] analyzing {len(jobs):,} chunks with sympy "
          f"({args.workers} workers)", flush=True)

    stats = defaultdict(Counter)
    examples = defaultdict(list)
    done = 0
    with Pool(args.workers) as pool:
        for tier, side, res, text in pool.imap_unordered(work, jobs, chunksize=64):
            done += 1
            if done % 10000 == 0:
                print(f"[exp3] {done:,}/{len(jobs):,}", flush=True)
            key = (tier, side)
            stats[key]["chunks"] += 1
            if "error" in res:
                stats[key]["errors"] += 1
                continue
            stats[key]["claims"] += res["n_claims"]
            stats[key]["parsed"] += res["n_parsed"]
            if res["n_anchors"] >= 1:
                stats[key]["with_anchor"] += 1
            if res["n_anchors"] >= 2:
                stats[key]["with_2plus_anchors"] += 1
            if res["weld_suspect"]:
                stats[key]["weld_suspect"] += 1
                if len(examples[key]) < 20:
                    examples[key].append(
                        {"text": text[:1200], "anchors": res["anchors"],
                         "n_anchor_components": res["n_anchor_components"]})

    out = {"tier_chunk_counts": {"produced": dict(prod_counts),
                                 "gold": dict(gold_counts)},
           "quotas": {"produced": qp, "gold": qg},
           "mathv360k_dropped_subsets": tm.mathv_dropped,
           "rates": {}}
    print(f"\n{'tier':12s} {'side':9s} {'chunks':>8s} {'weld':>7s} {'rate':>8s} "
          f"{'anchor%':>8s} {'2+anch%':>8s}")
    for t in TIERS:
        for side in ("gold", "produced"):
            s = stats[(t, side)]
            n = s["chunks"]
            row = {k: s[k] for k in
                   ("chunks", "claims", "parsed", "with_anchor",
                    "with_2plus_anchors", "weld_suspect", "errors")}
            row["weld_rate"] = s["weld_suspect"] / n if n else 0.0
            out["rates"][f"{t}/{side}"] = row
            print(f"{t:12s} {side:9s} {n:>8,d} {s['weld_suspect']:>7,d} "
                  f"{100*row['weld_rate']:7.3f}% "
                  f"{100*s['with_anchor']/max(1,n):7.2f}% "
                  f"{100*s['with_2plus_anchors']/max(1,n):7.2f}%")

    with open(os.path.join(args.outdir, "exp3_stats.json"), "w") as f:
        json.dump(out, f, indent=2)
    exdir = os.path.join(args.outdir, "exp3_examples")
    os.makedirs(exdir, exist_ok=True)
    for (t, side), exs in examples.items():
        with open(os.path.join(exdir, f"weld__{t}__{side}.json"), "w") as f:
            json.dump(exs, f, indent=2)
    print("[exp3] DONE")


if __name__ == "__main__":
    main()
