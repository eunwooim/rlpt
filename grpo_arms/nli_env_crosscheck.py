#!/usr/bin/env python
"""Score a fixed, seeded 100+100 (identity + hard_negative) pair selection from
granularity/nli_pairs.jsonl with the bidirectional s formula, dump s values.

Run under two envs (rlpt-train transformers 4.56 vs chunker/env 5.14) and
compare: if max|delta s| is tiny, the tau sweep (run under 4.56) transfers to
the training-time scorer (5.14).
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nli_match import NLIScorer

PAIRS = "/scratch/sghos104/rlpt/granularity/nli_pairs.jsonl"


def select_pairs(n_per_kind=100, seed=1234):
    by_kind = {"identity": [], "hard_negative": []}
    with open(PAIRS) as f:
        for line in f:
            r = json.loads(line)
            if r["kind"] in by_kind:
                by_kind[r["kind"]].append(r)
    rng = random.Random(seed)
    out = []
    for kind in sorted(by_kind):
        out.extend(rng.sample(by_kind[kind], n_per_kind))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = select_pairs()
    sc = NLIScorer(batch_size=32)
    import numpy as np

    cands = [r["candidate"] for r in rows]
    golds = [r["gold"] for r in rows]
    ef, cf = sc.ec_probs(cands, golds)
    er, cr = sc.ec_probs(golds, cands)
    s = (0.5 * np.array(ef) + 0.5 * np.array(er)
         - np.maximum(np.array(cf), np.array(cr))).clip(0.0, 1.0)
    with open(args.out, "w") as f:
        json.dump({"s": [round(float(v), 6) for v in s],
                   "id2label": sc.id2label}, f)
    print(f"[crosscheck] wrote {len(rows)} scores -> {args.out}")


if __name__ == "__main__":
    main()
