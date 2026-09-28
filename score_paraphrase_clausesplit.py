"""Stage 2 of the paraphrase precision control: run the SAME clause-split + min-pool NLI
pipeline as the negation test, but on (pos, faithful-paraphrase) pairs, and measure the
FALSE-POSITIVE rate = fraction of correct paraphrases that min-pool WRONGLY scores below
the 0.6 decision threshold (i.e. wrongly flagged as not-equivalent to the correct solution).

Pipeline (identical to score_negation_clausesplit.py):
  split -> SBERT-mpnet cosine -> Hungarian align (tau) -> NLI equiv per matched pair -> MIN.

Lower FPR = better (a faithful paraphrase SHOULD score high; flagging it is the false alarm
min-pool risks because one noisy reworded clause can veto the whole solution).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, "/scratch/sghos104/rlpt/src/metrics/reliable")
from scorers.base import make_device
from scorers.sbert import SbertCosineScorer
from scorers.nli import NLIScorer

# reuse the exact splitter / aligner / pooling from the negation experiment
from score_negation_clausesplit import split_clauses, align, TAU


def score(rows, sbert, nli, batch_size):
    """rows: list of {idx, pos, paraphrase}. Return {idx: (min_equiv, n_matched)}."""
    pos_cl = [split_clauses(r["pos"]) for r in rows]
    cand_cl = [split_clauses(r["paraphrase"]) for r in rows]
    flat, offs = [], []
    for pc, cc in zip(pos_cl, cand_cl):
        offs.append((len(flat), len(pc), len(cc)))
        flat.extend(pc)
        flat.extend(cc)
    print(f"  {len(rows):,} examples, {len(flat):,} clauses -> embedding", flush=True)
    emb = sbert._encode(flat, batch_size)

    nli_pairs, owner = [], []
    for k, (start, npos, ncand) in enumerate(offs):
        ep = emb[start : start + npos]
        ec = emb[start + npos : start + npos + ncand]
        for i, j in align(ep, ec):
            nli_pairs.append((pos_cl[k][i], cand_cl[k][j]))
            owner.append(k)
    print(f"  {len(nli_pairs):,} matched clause-pairs -> NLI", flush=True)
    nli_out = nli.score_pairs(nli_pairs, batch_size)

    per = {k: [] for k in range(len(rows))}
    for eq, o in zip((d["nli_score_equiv"] for d in nli_out), owner):
        per[o].append(eq)
    res = {}
    for k in range(len(rows)):
        vals = per[k]
        res[rows[k]["idx"]] = (min(vals) if vals else 0.0, len(vals))
    return res


def fpr_below(vals, thr):
    """fraction scoring BELOW thr = wrongly flagged (false positive)."""
    return mean(1.0 if v < thr else 0.0 for v in vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paraphrase_pairs", default="/scratch/sghos104/rlpt/data/visualprm400k/paraphrase_pairs.jsonl")
    ap.add_argument("--outdir", default="/scratch/sghos104/rlpt/src/outputs/visualprm_paraphrase_clausesplit")
    ap.add_argument("--md", default="/scratch/sghos104/rlpt/docs/paraphrase_clausesplit_results.md")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    allrows = [json.loads(l) for l in open(args.paraphrase_pairs)]
    rows = [r for r in allrows if r.get("faithful")]
    print(f"{len(allrows):,} paraphrases, {len(rows):,} faithful "
          f"({len(rows)/max(len(allrows),1):.1%}) used for precision", flush=True)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = make_device(args.device)
    print("device:", device, flush=True)
    sbert = SbertCosineScorer(device)
    nli = NLIScorer(device)

    part = outdir / "clausesplit_paraphrase.jsonl"
    if part.exists() and sum(1 for _ in part.open()) == len(rows):
        print("partial complete; loading", flush=True)
        res = {json.loads(l)["idx"]: (json.loads(l)["equiv"], json.loads(l)["n"]) for l in part.open()}
    else:
        t0 = time.time()
        res = score(rows, sbert, nli, args.batch_size)
        with part.open("w") as w:
            for idx, (eq, n) in res.items():
                w.write(json.dumps({"idx": idx, "equiv": eq, "n": n}) + "\n")
        print(f"done in {time.time()-t0:.1f}s -> {part}", flush=True)

    eqs = [res[r["idx"]][0] for r in rows]
    nm = [res[r["idx"]][1] for r in rows]

    L = []
    L.append("# Clause-split + min-pool NLI: paraphrase PRECISION control\n")
    L.append(f"N = {len(eqs):,} faithful paraphrases of correct VisualPRM solutions "
             f"(number-multiset verified identical to pos). Same pipeline as the negation "
             f"test: split -> SBERT-cosine Hungarian align (tau={TAU}) -> NLI **equiv** per "
             f"matched clause-pair -> **min**-pool. Mean matched clauses/example = {mean(nm):.1f}.\n")
    L.append("**FPR here = fraction of CORRECT paraphrases wrongly scored BELOW threshold "
             "(false alarm). Lower = better.** This is the precision side the negation "
             "(recall) test left untested.\n")
    L.append("| metric | clause-split min-equiv |")
    L.append("|---|--:|")
    L.append(f"| FPR@0.6 (paraphrase wrongly flagged) ↓ | **{fpr_below(eqs,0.6):.3f}** |")
    L.append(f"| FPR@0.5 (paraphrase wrongly flagged) ↓ | {fpr_below(eqs,0.5):.3f} |")
    L.append(f"| mean min-equiv on paraphrase (want HIGH) | {mean(eqs):.3f} |")
    L.append(f"| median min-equiv on paraphrase | {median(eqs):.3f} |")
    L.append("")
    L.append("**Reading:** a LOW FPR means min-pool keeps correct rewordings above 0.6 — "
             "the veto is safe to ship as-is. A HIGH FPR means one noisy reworded clause "
             "routinely sinks a correct solution; min-pool is too brittle and needs softening "
             "(soft-min / k-th-lowest quantile) and/or a hard numeric-match term. Compare "
             "against the negation recall FPR@0.6 = 0.008 to see the precision/recall trade.\n")
    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    print(f"\nwrote {args.md}", flush=True)


if __name__ == "__main__":
    main()
