#!/usr/bin/env python
"""Final evaluation of the chunk-boundary classifier on the untouched test split.

Reports, all over CANDIDATE tokens (label != -100) at the trained threshold:
  1. Token-level split P/R/F1 OVERALL and broken out by language bucket
     (en / cjk / other) -> quantifies the deberta-v3-small CJK limitation.
  2. Chunker-level error modes via DP decoding:
       - OVER-split on held-out 1-step trajectories (false splits on text humans
         kept whole): mean predicted splits, fraction with >=1 split.
       - UNDER-split on multi-step trajectories: boundary P/R/F1 vs gold (also by
         language), matching a gold end to a predicted cut within TOL chars.
"""
import argparse, json, os
from collections import defaultdict
import numpy as np

os.environ.setdefault("HF_HOME", "/scratch/sghos104/.hf_cache")
import chunker_common as cc
import chunker as ck

TOL = 2   # chars: a predicted cut (chunk-end, exclusive) matches gold end g when |cut-(g+1)|<=TOL


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=ck.DEFAULT_MODEL_DIR)
    ap.add_argument("--test_traj", default="/scratch/sghos104/rlpt/chunker/data/test_traj.jsonl")
    ap.add_argument("--out", default="/scratch/sghos104/rlpt/chunker/eval_report.json")
    ap.add_argument("--min_tokens", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=220)
    args = ap.parse_args()

    thr = ck.load(args.model_dir)
    print(f"[eval] threshold={thr}")
    rows = [json.loads(l) for l in open(args.test_traj)]

    # ---- 1. token-level split-F1 (overall + by language) ----
    tok_counts = defaultdict(lambda: [0, 0, 0])   # lang -> [tp, fp, fn]
    for r in rows:
        ex = cc.build_token_labels(r["steps"], ck._tok)
        labels = np.array(ex["labels"])
        # reuse the same core tokenization path for probs
        ids, offs, prob = ck._token_probs(ex["text"])
        n = min(len(labels), len(prob))
        labels, prob = labels[:n], prob[:n]
        mask = labels != cc.LABEL_IGNORE
        y = (labels[mask] == cc.LABEL_SPLIT).astype(int)
        pred = (prob[mask] >= thr).astype(int)
        tp = int((pred & y).sum()); fp = int((pred & (1 - y)).sum()); fn = int(((1 - pred) & y).sum())
        for key in ("ALL", r["lang"]):
            tok_counts[key][0] += tp; tok_counts[key][1] += fp; tok_counts[key][2] += fn

    token_f1 = {k: dict(zip(("precision", "recall", "f1"), prf(*v)), tp=v[0], fp=v[1], fn=v[2])
                for k, v in tok_counts.items()}

    # ---- 2. chunker-level over/under split ----
    multi = [r for r in rows if r["kind"] == "multi"]
    single = [r for r in rows if r["kind"] == "single"]

    # over-split on 1-step
    over = defaultdict(lambda: [0, 0, 0])   # lang -> [n_traj, total_false_splits, n_traj_with_split]
    for r in single:
        out = ck.chunk(r["text"], args.min_tokens, args.max_tokens)
        k = len(out["split_char_offsets"])
        for key in ("ALL", r["lang"]):
            over[key][0] += 1; over[key][1] += k; over[key][2] += (1 if k > 0 else 0)
    over_report = {k: {"n_1step_traj": v[0],
                       "mean_false_splits": round(v[1] / max(1, v[0]), 4),
                       "frac_traj_oversplit": round(v[2] / max(1, v[0]), 4)}
                   for k, v in over.items()}

    # under-split on multi: boundary P/R/F1 + short-fragment audit
    bnd = defaultdict(lambda: [0, 0, 0])    # lang -> [tp, fp, fn]
    n_seg = n_seg_short = 0                  # fraction of predicted segments < min_tokens
    for r in multi:
        steps = r["steps"]
        gold, _ = cc.gold_boundary_char_positions(steps)
        out = ck.chunk(r["text"], args.min_tokens, args.max_tokens)
        cuts = out["split_char_offsets"]
        for L in out["segment_token_lengths"]:
            n_seg += 1
            if L < args.min_tokens:
                n_seg_short += 1
        matched_gold = set(); matched_cut = set()
        for gi, g in enumerate(gold):
            for ci, c in enumerate(cuts):
                if ci in matched_cut:
                    continue
                if abs(c - (g + 1)) <= TOL:
                    matched_gold.add(gi); matched_cut.add(ci); break
        tp = len(matched_gold); fn = len(gold) - tp; fp = len(cuts) - len(matched_cut)
        for key in ("ALL", r["lang"]):
            bnd[key][0] += tp; bnd[key][1] += fp; bnd[key][2] += fn
    boundary_f1 = {k: dict(zip(("precision", "recall", "f1"), prf(*v)), tp=v[0], fp=v[1], fn=v[2])
                   for k, v in bnd.items()}

    report = {"threshold": thr, "n_test_traj": len(rows),
              "n_multi": len(multi), "n_single": len(single),
              "token_split_f1": token_f1,
              "boundary_f1_multi": boundary_f1,
              "oversplit_1step": over_report,
              "short_fragments_multi": {
                  "min_tokens": args.min_tokens, "n_segments": n_seg,
                  "n_under_min": n_seg_short,
                  "frac_under_min": round(n_seg_short / max(1, n_seg), 5)}}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
