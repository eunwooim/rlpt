"""Score VisualPRM400K math negation pairs with SBERT, NLI, BERTScore.

Two conditions per example:
  NEG  = sim(pos, neg)       -> correct reasoning vs one-equation-negated copy (hard negative)
  BASE = sim(pos, baseline)  -> correct reasoning vs an UNRELATED solution (random floor)

A faithful reward should score NEG LOW (it changed meaning). A scorer blind to the
edit will score NEG ~1.0 (>> BASE). Writes scores.jsonl, one row per example.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, "/scratch/sghos104/rlpt/src/metrics/reliable")
from scorers.base import make_device
from scorers.sbert import SbertCosineScorer
from scorers.nli import NLIScorer
from scorers.bertscore import BertScoreScorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="/scratch/sghos104/rlpt/data/visualprm400k/pairs.jsonl")
    ap.add_argument("--out", default="/scratch/sghos104/rlpt/src/outputs/visualprm_negation/scores.jsonl")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.pairs)]
    if args.limit:
        rows = rows[: args.limit]
    print(f"loaded {len(rows):,} pairs", flush=True)

    neg_pairs = [(r["pos"], r["neg"]) for r in rows]
    base_pairs = [(r["pos"], r["baseline"]) for r in rows]

    device = make_device(args.device)
    print("device:", device, flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Each scorer writes its own partial file immediately on completion, so a timeout
    # never loses finished scorers and a resubmit RESUMES (skips done scorers).
    for ScorerCls in (SbertCosineScorer, NLIScorer, BertScoreScorer):
        name = ScorerCls.name
        part = out.parent / f"scores_{name}.jsonl"
        if part.exists() and sum(1 for _ in part.open()) == len(rows):
            print(f"[{name}] already complete ({part}); skipping", flush=True)
            continue
        t0 = time.time()
        scorer = ScorerCls(device)
        neg = scorer.score_pairs(neg_pairs, args.batch_size)
        base = scorer.score_pairs(base_pairs, args.batch_size)
        with part.open("w") as w:
            for r, sn, sb in zip(rows, neg, base):
                w.write(json.dumps({"idx": r["idx"], f"{name}_neg": sn, f"{name}_base": sb}) + "\n")
        print(f"[{name}] scored {len(rows):,}x2 in {time.time()-t0:.1f}s -> {part}", flush=True)
        del scorer

    # merge partials into combined scores.jsonl
    base_meta = {r["idx"]: {"idx": r["idx"], "source": r["source"], "edit": r["edit"]} for r in rows}
    for ScorerCls in (SbertCosineScorer, NLIScorer, BertScoreScorer):
        part = out.parent / f"scores_{ScorerCls.name}.jsonl"
        if not part.exists():
            continue
        for line in part.open():
            d = json.loads(line)
            base_meta[d["idx"]].update({k: v for k, v in d.items() if k != "idx"})
    with out.open("w") as w:
        for idx in sorted(base_meta):
            w.write(json.dumps(base_meta[idx]) + "\n")
    print(f"merged -> {out} ({len(base_meta):,} rows)", flush=True)


if __name__ == "__main__":
    main()
