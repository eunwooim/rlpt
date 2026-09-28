"""Clause-split NLI test for VisualPRM numeric negation.

Hypothesis: paragraph-level NLI only half-catches the one-equation edit because the
single contradicting clause is DILUTED in a long, otherwise-identical solution. If we
split each solution into atomic clauses (punctuation/line split), align clauses, run NLI
per aligned clause-pair, and pool with MIN (any contradicted clause vetoes the solution),
the isolated contradiction is no longer outvoted -> the corrupted copy should score LOW
and the fooled-rate (FPR@0.6) should drop vs the paragraph-level 0.53.

Pipeline (mirrors the bipartite-match reward, swapping per-clause scorer to NLI + min-pool):
  1. split pos / candidate into clauses (newlines + sentence punctuation).
  2. embed clauses (SBERT mpnet), cosine matrix, Hungarian one-to-one assignment, tau gate.
  3. NLI equiv on each matched (pos_clause, cand_clause) pair.
  4. solution_equiv = MIN over matched pairs (one bad step sinks it).
Conditions: neg (corrupted copy, want LOW), base (unrelated floor).
Compared against paragraph-level equiv read from the existing scores.jsonl on the SAME idxs.

Per-condition partials are resume-safe.
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
from statistics import mean, median

import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "/scratch/sghos104/rlpt/src/metrics/reliable")
from scorers.base import make_device
from scorers.sbert import SbertCosineScorer
from scorers.nli import NLIScorer

TAU = 0.15          # cosine gate before a clause-pair is considered "aligned"
MAX_CLAUSES = 40


def split_clauses(text, max_clauses=MAX_CLAUSES):
    parts = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for seg in re.split(r"(?<=[.;:])\s+", line):
            seg = seg.strip()
            if seg:
                parts.append(seg)
    if not parts:
        parts = [text.strip()[:512] or "empty"]
    return parts[:max_clauses]


def align(emb_pos, emb_cand):
    """Hungarian one-to-one assignment on cosine; return list of (i,j) with cos>=TAU."""
    sim = (emb_pos @ emb_cand.T)  # both L2-normalized -> cosine
    ri, ci = linear_sum_assignment(-sim.numpy())
    return [(int(i), int(j)) for i, j in zip(ri, ci) if float(sim[i, j]) >= TAU]


def score_condition(rows, cand_field, sbert, nli, batch_size):
    """Return {idx: (solution_equiv, n_matched)} for the given candidate field."""
    # 1. split + embed all clauses for this condition
    pos_cl = [split_clauses(r["pos"]) for r in rows]
    cand_cl = [split_clauses(r[cand_field]) for r in rows]
    flat, offs = [], []
    for pc, cc in zip(pos_cl, cand_cl):
        offs.append((len(flat), len(pc), len(cc)))
        flat.extend(pc)
        flat.extend(cc)
    print(f"  [{cand_field}] {len(rows):,} examples, {len(flat):,} clauses -> embedding", flush=True)
    emb = sbert._encode(flat, batch_size)

    # 2. align per example -> global list of matched clause-text pairs
    nli_pairs, owner = [], []
    for k, (start, npos, ncand) in enumerate(offs):
        ep = emb[start : start + npos]
        ec = emb[start + npos : start + npos + ncand]
        for i, j in align(ep, ec):
            nli_pairs.append((pos_cl[k][i], cand_cl[k][j]))
            owner.append(k)
    print(f"  [{cand_field}] {len(nli_pairs):,} matched clause-pairs -> NLI", flush=True)

    # 3. NLI equiv on every matched pair
    nli_out = nli.score_pairs(nli_pairs, batch_size)

    # 4. min-pool per example
    per = {k: [] for k in range(len(rows))}
    for k, o in zip((d["nli_score_equiv"] for d in nli_out), owner):
        per[o].append(k)
    res = {}
    for k in range(len(rows)):
        vals = per[k]
        res[rows[k]["idx"]] = (min(vals) if vals else 0.0, len(vals))
    return res


def fpr(vals, tau):
    return mean(1.0 if v >= tau else 0.0 for v in vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="/scratch/sghos104/rlpt/data/visualprm400k/pairs.jsonl")
    ap.add_argument("--paragraph_scores", default="/scratch/sghos104/rlpt/src/outputs/visualprm_negation/scores.jsonl")
    ap.add_argument("--outdir", default="/scratch/sghos104/rlpt/src/outputs/visualprm_negation_clausesplit")
    ap.add_argument("--md", default="/scratch/sghos104/rlpt/docs/negation_clausesplit_results.md")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.pairs)][: args.n]
    print(f"loaded {len(rows):,} examples", flush=True)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = make_device(args.device)
    print("device:", device, flush=True)
    sbert = SbertCosineScorer(device)
    nli = NLIScorer(device)

    conditions = {"neg": "neg", "base": "baseline"}
    results = {}
    for cond, field in conditions.items():
        part = outdir / f"clausesplit_{cond}.jsonl"
        if part.exists() and sum(1 for _ in part.open()) == len(rows):
            print(f"[{cond}] partial complete; loading", flush=True)
            results[cond] = {json.loads(l)["idx"]: (json.loads(l)["equiv"], json.loads(l)["n"]) for l in part.open()}
            continue
        t0 = time.time()
        res = score_condition(rows, field, sbert, nli, args.batch_size)
        with part.open("w") as w:
            for idx, (eq, n) in res.items():
                w.write(json.dumps({"idx": idx, "equiv": eq, "n": n}) + "\n")
        print(f"[{cond}] done in {time.time()-t0:.1f}s -> {part}", flush=True)
        results[cond] = res

    # paragraph-level equiv on the same idxs
    idxs = {r["idx"] for r in rows}
    para = {}
    for l in open(args.paragraph_scores):
        d = json.loads(l)
        if d["idx"] in idxs:
            para[d["idx"]] = (d["nli_neg"]["nli_score_equiv"], d["nli_base"]["nli_score_equiv"])

    cs_neg = [results["neg"][i][0] for i in sorted(idxs)]
    cs_base = [results["base"][i][0] for i in sorted(idxs)]
    pa_neg = [para[i][0] for i in sorted(idxs) if i in para]
    pa_base = [para[i][1] for i in sorted(idxs) if i in para]
    n_matched = [results["neg"][i][1] for i in sorted(idxs)]

    L = []
    L.append("# Clause-split + min-pool NLI on VisualPRM numeric negation\n")
    L.append(f"N = {len(cs_neg):,} examples. Each solution split into atomic clauses "
             f"(newline + sentence punctuation), clauses aligned via SBERT-cosine Hungarian "
             f"matching (tau={TAU}), NLI **equiv** scored per matched clause-pair, then pooled "
             f"with **min** (one contradicted clause vetoes the solution). Mean matched "
             f"clauses/example = {mean(n_matched):.1f}.\n")
    L.append("**Lower FPR = better** (corrupted copy correctly scored low / not fooled). "
             "Paragraph-level numbers are the existing whole-solution NLI equiv on the same examples.\n")
    L.append("| metric | paragraph equiv | clause-split min-equiv |")
    L.append("|---|--:|--:|")
    L.append(f"| FPR@0.6 on neg (corrupted) ↓ | {fpr(pa_neg,0.6):.3f} | **{fpr(cs_neg,0.6):.3f}** |")
    L.append(f"| FPR@0.5 on neg (corrupted) ↓ | {fpr(pa_neg,0.5):.3f} | {fpr(cs_neg,0.5):.3f} |")
    L.append(f"| mean equiv on neg (corrupted) | {mean(pa_neg):.3f} | {mean(cs_neg):.3f} |")
    L.append(f"| median equiv on neg | {median(pa_neg):.3f} | {median(cs_neg):.3f} |")
    L.append(f"| mean equiv on base (unrelated floor) | {mean(pa_base):.3f} | {mean(cs_base):.3f} |")
    L.append("")
    L.append("**Reading:** if clause-split min-equiv drops FPR@0.6 on the corrupted copy well "
             "below the paragraph 0.53, splitting removed the dilution and NLI now catches the "
             "isolated numeric flip. Residual FPR is the part NLI still misses on bare numeric "
             "equality (e.g. neutral on `=14` vs `=15`) -- the case for a hard numeric-match term.\n")
    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    print(f"\nwrote {args.md}", flush=True)


if __name__ == "__main__":
    main()
