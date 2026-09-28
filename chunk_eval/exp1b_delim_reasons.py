#!/usr/bin/env python
"""EXP 1 supplement — decompose unbalanced_delims by failing delimiter pair,
full file, per bucket. Motivated by examples showing currency dollar signs
($2.47 trillion) tripping the $-parity check: dollar_parity violations are a
detector artifact on finance/chart text, not a chunking defect. Same bucket
definitions as exp1_completeness.py."""
import json
import sys
from collections import Counter
from multiprocessing import Pool

import orjson

REASONS = ("paren", "bracket", "brace", "latex_paren", "latex_bracket",
           "dollar_parity", "fence_parity")


def why(s):
    out = []
    n_lp, n_rp = s.count("("), s.count(")")
    e_lp, e_rp = s.count("\\("), s.count("\\)")
    n_lb, n_rb = s.count("["), s.count("]")
    e_lb, e_rb = s.count("\\["), s.count("\\]")
    if (n_lp - e_lp) != (n_rp - e_rp):
        out.append("paren")
    if (n_lb - e_lb) != (n_rb - e_rb):
        out.append("bracket")
    if s.count("{") != s.count("}"):
        out.append("brace")
    if e_lp != e_rp:
        out.append("latex_paren")
    if e_lb != e_rb:
        out.append("latex_bracket")
    if (s.count("$") - s.count("\\$")) % 2:
        out.append("dollar_parity")
    if s.count("```") % 2:
        out.append("fence_parity")
    return out


def work(batch):
    c = Counter()
    for line in batch:
        rec = orjson.loads(line)
        prov = rec["metadata"]["chunk_provenance"]
        m = prov["method"]
        if m == "passthrough":
            if len(rec["steps"]) <= 1:
                continue
            b = "gold"
        elif m == "marker":
            b = "marker/" + prov["tier"]
        elif m == "model":
            b = "model"
        elif m == "model_skipped_nonprose":
            b = "model_skipped_nonprose"
        else:
            continue
        for ch in rec["steps"]:
            s = ch.strip()
            if len(s) <= 1:
                continue
            c[(b, "_chunks")] += 1
            w = why(s)
            if w:
                c[(b, "ANY")] += 1
                for k in w:
                    c[(b, k)] += 1
                if "dollar_parity" in w and len(w) == 1:
                    c[(b, "dollar_only")] += 1
    return c


def main():
    path = "/scratch/sghos104/rlpt/canonical_chunked_v2.jsonl"
    tot = Counter()

    def batches(f):
        buf = []
        for line in f:
            buf.append(line)
            if len(buf) >= 20000:
                yield buf
                buf = []
        if buf:
            yield buf

    with open(path, "rb") as f, Pool(14) as pool:
        for i, c in enumerate(pool.imap_unordered(work, batches(f), chunksize=1)):
            tot.update(c)
            if i % 50 == 0:
                print(f"[exp1b] batch {i}", flush=True)

    buckets = ("marker/step", "marker/list", "marker/para", "model",
               "model_skipped_nonprose", "gold")
    out = {}
    print(f"{'bucket':24s} {'chunks':>12s} {'ANY':>7s} {'dollar_only':>11s} "
          + " ".join(f"{r[:11]:>12s}" for r in REASONS))
    for b in buckets:
        n = tot[(b, "_chunks")]
        row = {"chunks": n}
        for k in ("ANY", "dollar_only") + REASONS:
            row[k] = tot[(b, k)]
            row[k + "_rate"] = tot[(b, k)] / n if n else 0.0
        out[b] = row
        print(f"{b:24s} {n:>12,d} {100*row['ANY_rate']:6.2f}% "
              f"{100*row['dollar_only_rate']:10.2f}% "
              + " ".join(f"{100*row[r+'_rate']:11.2f}%" for r in REASONS))
    with open("/scratch/sghos104/rlpt/chunk_eval/exp1b_delim_reasons.json", "w") as f:
        json.dump(out, f, indent=2)
    print("[exp1b] DONE")


if __name__ == "__main__":
    sys.exit(main())
