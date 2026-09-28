#!/usr/bin/env python
"""Rebuild ONLY the FRAGMENTED population from the existing gold sample
(exp2_populations/gold.jsonl — same 20K records, same order) using the
tightened fragment rule in exp2_build.fragment_step. Asserts per-record
mass conservation, re-censuses the "clean-looking cut" rate with the SAME
criteria as the original measurement, and hard-fails if residual >= 0.5%."""
import json
import re
import sys

import orjson

sys.path.insert(0, "/scratch/sghos104/rlpt/chunk_eval")
from exp2_build import fragment_step, strip_ws  # noqa: E402

POP = "/scratch/sghos104/rlpt/chunk_eval/exp2_populations"
LISTY = re.compile(r"^\s*(?:\d{1,2}[.)]\s|[-*•]\s|#{1,4}\s|\*\*)")

# tightened-rule self-test (old rule allowed both of these)
parts, how = fragment_step(
    "We compute the value using \\[ x = 2 \\] and then continue on to the "
    "next quantity with more words here")
assert how == "split" and not parts[0].rstrip().endswith("\\]"), parts
parts, how = fragment_step(
    "An intro clause that is long enough to pass the token gate\n"
    "1. first numbered item here\n1. second numbered item text")
assert how in ("split", "no_boundary")
if how == "split":
    assert not ("\n" in parts[0][-2:] and LISTY.match(parts[1].lstrip("\n")))
print("[rebuild] tightened-rule self-test OK")

n = n_split = n_short = n_nb = 0
n_listy = n_afterclose = 0
out = open(POP + "/fragmented.jsonl.tmp", "wb")
with open(POP + "/gold.jsonl", "rb") as f:
    for line in f:
        r = orjson.loads(line)
        gold_steps = r["chunks"]
        frag, outcomes = [], []
        for st in gold_steps:
            parts, how = fragment_step(st)
            frag.extend(parts)
            outcomes.append(how)
            if how == "split":
                n_split += 1
                second = parts[1]
                if LISTY.match(second.lstrip("\n")):
                    n_listy += 1
                if parts[0].rstrip().endswith(("\\]", "]", "\\)")):
                    n_afterclose += 1
            elif how == "short":
                n_short += 1
            else:
                n_nb += 1
        assert strip_ws(frag) == strip_ws(gold_steps), r["sid"]
        out.write(orjson.dumps(
            {"sid": r["sid"], "idx": r["idx"], "bucket": r["bucket"],
             "n_gold": r["n_gold"], "chunks": frag,
             "frag_outcomes": outcomes}) + b"\n")
        n += 1
out.close()

resid_listy = 100 * n_listy / max(1, n_split)
resid_close = 100 * n_afterclose / max(1, n_split)
print(f"[rebuild] records={n:,} split={n_split:,} short={n_short:,} "
      f"no_boundary={n_nb:,}")
print(f"[rebuild] residual clean-looking cuts: before-marker {n_listy} "
      f"({resid_listy:.3f}%), after-math-closer {n_afterclose} "
      f"({resid_close:.3f}%)")
ok = resid_listy < 0.5 and resid_close < 0.5
json.dump({"records": n, "split": n_split, "short": n_short,
           "no_boundary": n_nb, "residual_before_marker_pct": resid_listy,
           "residual_after_closer_pct": resid_close, "pass": ok},
          open("/scratch/sghos104/rlpt/chunk_eval/exp2_fragment_v2_stats.json", "w"),
          indent=2)
if not ok:
    print("[rebuild] FAIL: residual >= 0.5% — fragmented.jsonl NOT replaced")
    sys.exit(1)
import os
os.replace(POP + "/fragmented.jsonl.tmp", POP + "/fragmented.jsonl")
print(f"[rebuild] PASS — fragmented.jsonl replaced ({n:,} records)")
