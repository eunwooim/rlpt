#!/usr/bin/env python
"""Gate 1: sweep tau for the bidirectional-NLI pair score on granularity/nli_pairs.jsonl.

Primary question: the threshold that best separates offset-0 identity pairs
(candidate == gold step) from hard negatives (candidate vs a gold step from a
DIFFERENT record, same source). The old 0.15 was calibrated for SBERT cosine;
entailment-probability scores live on a different scale.

Context (not part of the criterion): merge2/merge3/split2/split3 perturbation
pairs, subsampled, so we can see where granularity-mismatched-but-related pairs
land relative to the chosen tau.

Outputs (grpo_arms/data/):
  tau_sweep_scores.jsonl  per-pair {kind, band, source, s, e_fwd, e_rev, c_fwd, c_rev}
  tau_sweep_summary.json  grid metrics, best taus, AUC, per-kind distributions
"""
import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nli_match import NLIScorer, NLI_MODEL

PAIRS = "/scratch/sghos104/rlpt/granularity/nli_pairs.jsonl"
OUT_DIR = "/scratch/sghos104/rlpt/grpo_arms/data"


def pct(a, q):
    return float(np.percentile(a, q))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context_per_kind", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap pairs per kind")
    ap.add_argument("--out_prefix", default="tau_sweep",
                    help="output filenames: {out_prefix}_scores.jsonl / _summary.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    by_kind = {}
    with open(PAIRS) as f:
        for line in f:
            r = json.loads(line)
            by_kind.setdefault(r["kind"], []).append(r)

    rows = []
    for kind, pool in sorted(by_kind.items()):
        if kind in ("identity", "hard_negative"):
            take = pool
        else:
            take = rng.sample(pool, min(args.context_per_kind, len(pool)))
        if args.limit:
            take = take[: args.limit]
        rows.extend(take)
    counts = {k: sum(1 for r in rows if r["kind"] == k) for k in by_kind}
    print(f"[sweep] scoring {len(rows)} pairs: {counts}", flush=True)

    scorer = NLIScorer(batch_size=args.batch_size)
    print(f"[sweep] model={NLI_MODEL} id2label={scorer.id2label} "
          f"ent_idx={scorer.ent_idx} con_idx={scorer.con_idx}", flush=True)

    cands = [r["candidate"] for r in rows]
    golds = [r["gold"] for r in rows]
    e_fwd, c_fwd = scorer.ec_probs(cands, golds)   # candidate -> gold
    print("[sweep] forward direction done", flush=True)
    e_rev, c_rev = scorer.ec_probs(golds, cands)   # gold -> candidate
    print("[sweep] reverse direction done", flush=True)

    s = (0.5 * np.array(e_fwd) + 0.5 * np.array(e_rev)
         - np.maximum(np.array(c_fwd), np.array(c_rev))).clip(0.0, 1.0)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{args.out_prefix}_scores.jsonl"), "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({
                "kind": r["kind"], "band": r["band"], "source": r["source"],
                "offset": r["offset"], "s": round(float(s[i]), 6),
                "e_fwd": round(e_fwd[i], 6), "e_rev": round(e_rev[i], 6),
                "c_fwd": round(c_fwd[i], 6), "c_rev": round(c_rev[i], 6),
            }) + "\n")

    kinds = np.array([r["kind"] for r in rows])
    pos = s[kinds == "identity"]
    neg = s[kinds == "hard_negative"]

    # ROC-AUC (threshold-free): P(identity > hard_negative) + 0.5 P(tie),
    # rank-sum with tie handling
    allv = np.concatenate([pos, neg])
    sorter = np.argsort(allv, kind="mergesort")
    rk = np.empty(len(allv))
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[sorter[j + 1]] == allv[sorter[i]]:
            j += 1
        rk[sorter[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    auc = (rk[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

    grid = np.round(np.arange(0.0, 1.0001, 0.01), 2)
    table = []
    for t in grid:
        tpr = float((pos >= t).mean())
        fpr = float((neg >= t).mean())
        table.append({"tau": float(t), "tpr": round(tpr, 4), "fpr": round(fpr, 4),
                      "youden_j": round(tpr - fpr, 4),
                      "balanced_acc": round((tpr + (1 - fpr)) / 2, 4)})
    best_j = max(table, key=lambda r: r["youden_j"])
    # operating points: tau for FPR caps
    ops = {}
    for cap in (0.01, 0.05, 0.10):
        elig = [r for r in table if r["fpr"] <= cap]
        ops[f"fpr<={cap}"] = min(elig, key=lambda r: r["tau"]) if elig else None

    dists = {}
    for kind in sorted(by_kind):
        v = s[kinds == kind]
        dists[kind] = {"n": int(len(v)), "mean": round(float(v.mean()), 4),
                       "p5": round(pct(v, 5), 4), "p25": round(pct(v, 25), 4),
                       "p50": round(pct(v, 50), 4), "p75": round(pct(v, 75), 4),
                       "p95": round(pct(v, 95), 4),
                       "frac_ge_best_tau": round(float((v >= best_j["tau"]).mean()), 4)}

    summary = {
        "model": NLI_MODEL, "id2label": scorer.id2label,
        "formula": "s = 0.5*E(c->g) + 0.5*E(g->c) - max(C(c->g), C(g->c)), clip [0,1]",
        "n_scored": len(rows), "counts": counts,
        "identity_vs_hard_negative": {
            "auc": round(float(auc), 4),
            "best_tau_youden": best_j,
            "operating_points": ops,
        },
        "grid": table,
        "distributions": dists,
        "context_subsample": {"per_kind": args.context_per_kind, "seed": args.seed},
    }
    with open(os.path.join(OUT_DIR, f"{args.out_prefix}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({k: v for k, v in summary.items() if k != "grid"}, indent=2))
    print("[sweep] top of grid by Youden J:")
    for r in sorted(table, key=lambda r: -r["youden_j"])[:8]:
        print("   ", r)


if __name__ == "__main__":
    main()
