#!/usr/bin/env python
"""Per-trajectory segmentation health on the test set.

Aggregate boundary P/R hides *distribution*: a model averaging 5.28 cuts against
5.88 gold looks fine even if a slice of trajectories is returned completely
unsegmented. A zero-cut multi-step trajectory reaches the downstream reward model
as one undivided blob, so that slice matters more than the mean.

Reports: distribution of predicted cuts vs gold per trajectory, the ZERO-CUT rate
(overall and by language), per-trajectory boundary recall deciles, and how the
zero-cut trajectories differ (length, gold count).
"""
import argparse, json
from collections import Counter
import numpy as np
import chunker as ck
import chunker_common as cc

TOL = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=ck.DEFAULT_MODEL_DIR)
    ap.add_argument("--traj", default="/scratch/sghos104/rlpt/chunker/data/test_traj.jsonl")
    ap.add_argument("--out", default="/scratch/sghos104/rlpt/chunker/runs/full_plain/segment_stats.json")
    ap.add_argument("--min_tokens", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=220)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    thr = ck.load(args.model_dir)
    print(f"[segstats] threshold={thr}", flush=True)
    rows = [json.loads(l) for l in open(args.traj) if json.loads(l)["kind"] == "multi"]
    if args.limit:
        rows = rows[:args.limit]

    recalls, zero_lang, tot_lang = [], Counter(), Counter()
    ncuts, ngolds = [], []
    zero_examples = []
    zero_len, nonzero_len = [], []

    for i, r in enumerate(rows):
        gold, _ = cc.gold_boundary_char_positions(r["steps"])
        out = ck.chunk(r["text"], args.min_tokens, args.max_tokens)
        cuts = out["split_char_offsets"]
        matched, used = 0, set()
        for g in gold:
            for ci, c in enumerate(cuts):
                if ci in used:
                    continue
                if abs(c - (g + 1)) <= TOL:
                    matched += 1; used.add(ci); break
        rec = matched / max(1, len(gold))
        recalls.append(rec); ncuts.append(len(cuts)); ngolds.append(len(gold))
        tot_lang[r["lang"]] += 1
        ntok = sum(out["segment_token_lengths"])
        if len(cuts) == 0:
            zero_lang[r["lang"]] += 1
            zero_len.append(ntok)
            if len(zero_examples) < 5:
                zero_examples.append({"lang": r["lang"], "n_steps": len(r["steps"]),
                                      "chars": len(r["text"]), "tokens": int(ntok),
                                      "n_gold": len(gold), "head": r["text"][:110]})
        else:
            nonzero_len.append(ntok)
        if (i + 1) % 2000 == 0:
            print(f"  ...{i+1:,}/{len(rows):,}  zero-cut so far={sum(zero_lang.values()):,}", flush=True)

    recalls = np.array(recalls); ncuts = np.array(ncuts); ngolds = np.array(ngolds)
    nz = sum(zero_lang.values())
    rep = {
        "threshold": thr,
        "n_multi_traj": len(rows),
        "zero_cut": {
            "count": int(nz), "frac": round(nz / max(1, len(rows)), 4),
            "by_language": {k: {"zero": zero_lang.get(k, 0), "total": tot_lang[k],
                                "frac": round(zero_lang.get(k, 0) / max(1, tot_lang[k]), 4)}
                            for k in tot_lang},
            "mean_tokens_zero_cut": round(float(np.mean(zero_len)), 1) if zero_len else None,
            "mean_tokens_normal": round(float(np.mean(nonzero_len)), 1) if nonzero_len else None,
            "examples": zero_examples,
        },
        "cuts_per_traj": {"mean": round(float(ncuts.mean()), 3), "p10": int(np.percentile(ncuts, 10)),
                          "p50": int(np.percentile(ncuts, 50)), "p90": int(np.percentile(ncuts, 90))},
        "gold_per_traj": {"mean": round(float(ngolds.mean()), 3), "p50": int(np.percentile(ngolds, 50))},
        "per_traj_boundary_recall": {
            "mean": round(float(recalls.mean()), 4),
            "deciles": [round(float(np.percentile(recalls, p)), 3) for p in range(0, 101, 10)],
            "frac_traj_recall_0": round(float((recalls == 0).mean()), 4),
            "frac_traj_recall_ge_0.8": round(float((recalls >= 0.8).mean()), 4),
        },
    }
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))
    print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
