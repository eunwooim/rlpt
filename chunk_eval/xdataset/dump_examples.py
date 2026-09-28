#!/usr/bin/env python
"""15-example dump per dataset: 5 best, 5 worst, 5 median records by F1,
with predicted-vs-gold boundaries marked inline in the space-joined text.

Markers: ⟦=⟧ boundary predicted AND gold (match), ⟦G⟧ gold-only (missed),
⟦P⟧ predicted-only (spurious). Deterministic: ties broken by sid.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, "/scratch/sghos104/rlpt/chunk_canonical")
from exp4_resegment import gold_boundaries_nonws  # noqa: E402


def render(steps, pred_nonws):
    text = " ".join(steps)
    gold = set(gold_boundaries_nonws(steps))
    pred = set(pred_nonws)
    out = []
    nonws = 0
    for ch in text:
        if not ch.isspace():
            if nonws > 0:
                if nonws in gold and nonws in pred:
                    out.append("⟦=⟧")
                elif nonws in gold:
                    out.append("⟦G⟧")
                elif nonws in pred:
                    out.append("⟦P⟧")
            nonws += 1
        out.append(ch)
    return "".join(out)


def main(tag="full"):
    for ds in ("prm800k", "processbench"):
        rec_path = os.path.join(HERE, f"{tag}_records_{ds}.jsonl")
        steps_by_sid = {}
        with open(os.path.join(HERE, f"{ds}.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                steps_by_sid[r["sid"]] = r["steps"]
        rows = []
        with open(rec_path) as f:
            for line in f:
                rows.append(json.loads(line))
        rows.sort(key=lambda r: (r["f1"], r["sid"]))
        n = len(rows)
        picks = [("WORST", rows[:5]),
                 ("MEDIAN", rows[max(0, n // 2 - 2): n // 2 + 3]),
                 ("BEST", rows[-5:][::-1])]
        out_path = os.path.join(HERE, f"examples_{ds}.txt")
        with open(out_path, "w") as f:
            f.write(f"{ds}: {n} records; markers ⟦=⟧ match ⟦G⟧ gold-only "
                    f"(missed) ⟦P⟧ pred-only (spurious)\n")
            for group, sel in picks:
                for r in sel:
                    f.write(f"\n{'=' * 78}\n[{group}] sid={r['sid']} "
                            f"subset={r['subset']} n_gold={r['n_gold']} "
                            f"n_pred={r['n_pred']} P={r['prec']:.2f} "
                            f"R={r['rec']:.2f} F1={r['f1']:.2f} "
                            f"dens={r['density']:.2f}\n{'=' * 78}\n")
                    txt = render(steps_by_sid[r["sid"]], r["pred_nonws"])
                    if len(txt) > 4000:
                        txt = txt[:2500] + "\n[... truncated ...]\n" + txt[-1200:]
                    f.write(txt + "\n")
        print(f"[dump] wrote {out_path} ({n} records)")


if __name__ == "__main__":
    main(*sys.argv[1:])
