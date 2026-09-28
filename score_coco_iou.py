"""Score COCO IoU ranking triples with SBERT, NLI, BERTScore (text scorers on box strings).

For each triple we form two pairs against the reference (GT box string `pos`):
  HI = sim(pos, neg)   -> candidate with the HIGHER IoU (should score MORE similar)
  LO = sim(pos, NEG)   -> candidate with the LOWER  IoU

A faithful, IoU-aware reward satisfies sim(pos, neg) > sim(pos, NEG), i.e. HI > LO.
Rank accuracy = fraction of triples where HI > LO.  Pure text-similarity scorers have
no reason to track spatial overlap, so we expect rank acc near chance (0.5).

Per-scorer partial files written immediately (resume-safe), then merged to scores.jsonl.
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
    ap.add_argument("--pairs", default="/scratch/sghos104/rlpt/data/coco/coco_iou_pairs.jsonl")
    ap.add_argument("--out", default="/scratch/sghos104/rlpt/src/outputs/coco_iou/scores.jsonl")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.pairs)]
    if args.limit:
        rows = rows[: args.limit]
    print(f"loaded {len(rows):,} triples", flush=True)

    hi_pairs = [(r["pos"], r["neg"]) for r in rows]   # higher IoU
    lo_pairs = [(r["pos"], r["NEG"]) for r in rows]   # lower IoU

    device = make_device(args.device)
    print("device:", device, flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for ScorerCls in (SbertCosineScorer, NLIScorer, BertScoreScorer):
        name = ScorerCls.name
        part = out.parent / f"scores_{name}.jsonl"
        if part.exists() and sum(1 for _ in part.open()) == len(rows):
            print(f"[{name}] already complete; skipping", flush=True)
            continue
        t0 = time.time()
        scorer = ScorerCls(device)
        hi = scorer.score_pairs(hi_pairs, args.batch_size)
        lo = scorer.score_pairs(lo_pairs, args.batch_size)
        with part.open("w") as w:
            for r, sh, sl in zip(rows, hi, lo):
                w.write(json.dumps({"idx": r["idx"], f"{name}_hi": sh, f"{name}_lo": sl}) + "\n")
        print(f"[{name}] scored {len(rows):,}x2 in {time.time()-t0:.1f}s -> {part}", flush=True)
        del scorer

    meta = {r["idx"]: {"idx": r["idx"], "category": r["category"],
                       "iou_neg": r["iou_neg"], "iou_NEG": r["iou_NEG"]} for r in rows}
    for ScorerCls in (SbertCosineScorer, NLIScorer, BertScoreScorer):
        part = out.parent / f"scores_{ScorerCls.name}.jsonl"
        if not part.exists():
            continue
        for line in part.open():
            d = json.loads(line)
            meta[d["idx"]].update({k: v for k, v in d.items() if k != "idx"})
    with out.open("w") as w:
        for idx in sorted(meta):
            w.write(json.dumps(meta[idx]) + "\n")
    print(f"merged -> {out} ({len(meta):,} rows)", flush=True)


if __name__ == "__main__":
    main()
