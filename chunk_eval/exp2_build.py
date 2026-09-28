#!/usr/bin/env python
"""EXP 2 (build stage, CPU) — corruption-calibrated population construction.

Populations (all written to exp2_populations/, one record per line):
  gold        20K passthrough records with >= 3 steps, stratified by n_steps
              buckets 3-4 / 5-7 / 8+ (quota proportional to bucket census).
  merged      derived from the SAME gold sample: adjacent step pairs joined
              (steps 2i and 2i+1 joined with a single "\\n"; odd tail step
              kept as-is). A 5-step record -> 3 chunks.
  fragmented  derived from the SAME gold sample: every step of >= 12
              whitespace tokens split at the word boundary nearest its char
              midpoint whose preceding non-space char is NOT sentence
              punctuation (. ! ? : ;) — a forced mid-thought cut. Steps with
              < 12 tokens (or no admissible boundary) stay whole (counted).
  produced    20K records with method in {marker, model} and n_chunks >= 3,
              sampled with the SAME bucket quotas as gold (bucket = n_chunks)
              so the two populations match on size mix.

Seeds: all sampling via random.Random(0). Mass conservation of merged and
fragmented vs gold is asserted per record (strip-all-whitespace equality).
Writes exp2_build_stats.json + exp2_population_examples.txt (5 records,
gold/merged/fragmented side by side) then STOPS — GPU scoring is gated.
"""
import json
import random
import re
import sys
from collections import Counter

import orjson

INPUT = "/scratch/sghos104/rlpt/canonical_chunked_v2.jsonl"
OUTDIR = "/scratch/sghos104/rlpt/chunk_eval"
N_GOLD = 20_000
N_PROD = 20_000
BUCKETS = (("3-4", 3, 4), ("5-7", 5, 7), ("8+", 8, 10**9))
_WS = re.compile(r"\s+")


def bucket(n):
    for name, lo, hi in BUCKETS:
        if lo <= n <= hi:
            return name
    return None


def merge_pairs(steps):
    out = []
    for i in range(0, len(steps) - 1, 2):
        out.append(steps[i] + "\n" + steps[i + 1])
    if len(steps) % 2:
        out.append(steps[-1])
    return out


SENT_PUNCT = ".!?:;"
# stricter rule (user-approved v2 of the corruption): also reject cuts
# immediately after display-math closers and cuts that would land
# immediately before a line-start list/heading marker — both look like
# clean boundaries rather than forced mid-thought cuts.
MATH_CLOSERS = ("\\]", "]", "\\)")
MARKER_AFTER_RE = re.compile(r"^(?:\d{1,2}[.)]\s|[-*•]\s|#{1,4}\s|\*\*)")


def fragment_step(step):
    """Split at the word boundary nearest the char midpoint that is NOT
    (a) preceded by sentence punctuation .!?:; , (b) preceded by a
    display-math closer \\] ] \\), or (c) followed — across a newline —
    by a line-start list/heading marker. Returns [step] if not splittable."""
    tokens = step.split()
    if len(tokens) < 12:
        return [step], "short"
    # candidate boundaries: start offsets of whitespace runs
    cands = []
    for m in re.finditer(r"\s+", step):
        j = m.start()
        prev = step[:j].rstrip()
        if not prev or prev[-1] in SENT_PUNCT or prev.endswith(MATH_CLOSERS):
            continue
        rest = step[m.end():]
        if not rest.strip():
            continue
        if MARKER_AFTER_RE.match(rest):
            continue
        cands.append(j)
    if not cands:
        return [step], "no_boundary"
    mid = len(step) / 2
    cut = min(cands, key=lambda j: abs(j - mid))
    return [step[:cut], step[cut:]], "split"


def strip_ws(chunks):
    return _WS.sub("", "".join(chunks))


def _self_test():
    s = "The area of the triangle is computed from the base and the height using the standard formula"
    parts, how = fragment_step(s)
    assert how == "split" and len(parts) == 2, (how, parts)
    assert strip_ws(parts) == strip_ws([s])
    assert not parts[0].rstrip().endswith(tuple(SENT_PUNCT))
    parts, how = fragment_step("Too short to split.")
    assert how == "short" and parts == ["Too short to split."]
    m = merge_pairs(["a", "b", "c", "d", "e"])
    assert m == ["a\nb", "c\nd", "e"], m
    print("[exp2-build] self-test OK", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="smoke: only first N rows")
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--input", default=INPUT)
    args = ap.parse_args()
    _self_test()
    rng = random.Random(0)

    # ---- pass 1: bucket census ----
    gold_census, prod_census = Counter(), Counter()
    with open(args.input, "rb") as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            if i % 2_000_000 == 0:
                print(f"[exp2 pass1] {i:,}", flush=True)
            rec = orjson.loads(line)
            prov = rec["metadata"]["chunk_provenance"]
            n = len(rec["steps"])
            b = bucket(n)
            if b is None:
                continue
            if prov["method"] == "passthrough" and n > 1:
                gold_census[b] += 1
            elif prov["method"] in ("marker", "model"):
                prod_census[b] += 1
    print("[exp2] gold census:", dict(gold_census), flush=True)
    print("[exp2] produced census:", dict(prod_census), flush=True)

    tot = sum(gold_census.values())
    if not tot:
        raise SystemExit("[exp2] no gold records with >=3 steps in input slice")
    quota = {b: round(N_GOLD * gold_census[b] / tot) for b, _, _ in BUCKETS}
    # produced uses the SAME quotas (match gold's size mix), capped by census
    quota_p = {b: min(quota[b], prod_census[b]) for b in quota}
    print("[exp2] quotas gold:", quota, "produced:", quota_p, flush=True)

    # ---- pass 2: reservoirs ----
    res_g = {b: [] for b in quota}
    res_p = {b: [] for b in quota}
    seen = Counter()

    def take(store, key, quota_n, item):
        seen[key] += 1
        r = store[key[1]]
        if len(r) < quota_n:
            r.append(item)
        else:
            j = rng.randrange(seen[key])
            if j < quota_n:
                r[j] = item

    with open(args.input, "rb") as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            if i % 2_000_000 == 0:
                print(f"[exp2 pass2] {i:,}", flush=True)
            rec = orjson.loads(line)
            prov = rec["metadata"]["chunk_provenance"]
            n = len(rec["steps"])
            b = bucket(n)
            if b is None:
                continue
            item = {"sid": rec["metadata"]["source_sample_id"],
                    "idx": rec["metadata"]["source_index"],
                    "n": n, "bucket": b, "steps": rec["steps"]}
            if prov["method"] == "passthrough" and n > 1:
                take(res_g, ("g", b), quota[b], item)
            elif prov["method"] in ("marker", "model"):
                item["method"] = prov["method"]
                item["tier"] = prov.get("tier")
                take(res_p, ("p", b), quota_p[b], item)

    # ---- derive corruptions + write populations ----
    import os
    popdir = f"{args.outdir}/exp2_populations"
    os.makedirs(popdir, exist_ok=True)
    stats = {"gold_census": dict(gold_census), "produced_census": dict(prod_census),
             "quota_gold": quota, "quota_produced": quota_p,
             "frag_outcomes": Counter(), "counts": {}}

    files = {p: open(f"{popdir}/{p}.jsonl", "wb") for p in
             ("gold", "merged", "fragmented", "produced")}
    n_written = Counter()
    gold_all = [it for b in res_g for it in res_g[b]]
    prod_all = [it for b in res_p for it in res_p[b]]
    rng.shuffle(gold_all)
    rng.shuffle(prod_all)

    for it in gold_all:
        gold_steps = it["steps"]
        merged = merge_pairs(gold_steps)
        frag, outcomes = [], []
        for st in gold_steps:
            parts, how = fragment_step(st)
            frag.extend(parts)
            outcomes.append(how)
            stats["frag_outcomes"][how] += 1
        assert strip_ws(merged) == strip_ws(gold_steps), it["sid"]
        assert strip_ws(frag) == strip_ws(gold_steps), it["sid"]
        base = {"sid": it["sid"], "idx": it["idx"], "bucket": it["bucket"],
                "n_gold": it["n"]}
        files["gold"].write(orjson.dumps({**base, "chunks": gold_steps}) + b"\n")
        files["merged"].write(orjson.dumps({**base, "chunks": merged}) + b"\n")
        files["fragmented"].write(orjson.dumps(
            {**base, "chunks": frag, "frag_outcomes": outcomes}) + b"\n")
        n_written["gold"] += 1
    for it in prod_all:
        files["produced"].write(orjson.dumps(
            {"sid": it["sid"], "idx": it["idx"], "bucket": it["bucket"],
             "method": it["method"], "tier": it.get("tier"),
             "chunks": it["steps"]}) + b"\n")
        n_written["produced"] += 1
    for f in files.values():
        f.close()

    stats["counts"] = dict(n_written)
    stats["frag_outcomes"] = dict(stats["frag_outcomes"])
    with open(f"{args.outdir}/exp2_build_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ---- 5 side-by-side examples for the gate ----
    ex = rng.sample(gold_all, 5)
    with open(f"{args.outdir}/exp2_population_examples.txt", "w") as f:
        for k, it in enumerate(ex):
            f.write(f"{'='*78}\nEXAMPLE {k+1}  sid={it['sid']} idx={it['idx']} "
                    f"n_steps={it['n']} bucket={it['bucket']}\n{'='*78}\n")
            merged = merge_pairs(it["steps"])
            frag = []
            for st in it["steps"]:
                parts, _ = fragment_step(st)
                frag.extend(parts)
            for name, chunks in (("GOLD", it["steps"]), ("MERGED", merged),
                                 ("FRAGMENTED", frag)):
                f.write(f"\n--- {name} ({len(chunks)} chunks) ---\n")
                for j, c in enumerate(chunks):
                    disp = c if len(c) <= 400 else c[:200] + " [...] " + c[-120:]
                    f.write(f"  [{j}] {disp}\n")
            f.write("\n")
    print("[exp2-build] wrote populations:", dict(n_written))
    print("[exp2-build] frag outcomes:", stats["frag_outcomes"])
    print("[exp2-build] DONE — GATE: show examples before GPU scoring")


if __name__ == "__main__":
    sys.exit(main())
