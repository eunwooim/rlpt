"""Aggregate VisualPRM400K math negation scores into a results markdown.

Per scorer we compare two similarity distributions:
  NEG  = sim(correct, one-equation-negated)   -> want LOW if scorer is math-sensitive
  BASE = sim(correct, unrelated solution)      -> random floor

Key diagnostics:
  * mean/median NEG (closeness to 1.0 = blind to the numeric edit)
  * gap NEG-BASE (how much surface overlap inflates similarity vs an unrelated text)
  * miss-rate(tau) = fraction of NEG pairs scoring >= tau (treated as still-matching)
  * NLI contradiction shift (mean C on NEG vs BASE)
"""
from __future__ import annotations
import json
from pathlib import Path
from statistics import mean, median

SCORES = "/scratch/sghos104/rlpt/src/outputs/visualprm_negation/scores.jsonl"
MD = "/scratch/sghos104/rlpt/docs/visualprm_negation_results.md"
TAUS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

# (label, scorer_key, field)
FIELDS = [
    ("SBERT cosine", "sbert", "raw_score"),
    ("NLI equiv", "nli", "nli_score_equiv"),
    ("NLI coverage", "nli", "nli_score_coverage"),
    ("BERTScore f1", "bertscore", "bertscore_f1"),
]


def quant(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[i]


def auc_gt(a, b):
    """P(a > b) via Mann-Whitney, ties=0.5. a,b are score lists."""
    bs = sorted(b)
    import bisect
    n = len(bs); tot = 0.0
    for x in a:
        lo = bisect.bisect_left(bs, x); hi = bisect.bisect_right(bs, x)
        tot += lo + 0.5 * (hi - lo)
    return tot / (len(a) * n)


def main():
    rows = [json.loads(l) for l in open(SCORES)]
    print(f"loaded {len(rows):,} scored rows")

    lines = []
    lines.append("# VisualPRM400K — Math negation sensitivity (SBERT / NLI / BERTScore)\n")
    lines.append(f"Dataset: VisualPRM400K **math subset** (24 PRM files, geometry/function/visual-math). "
                 f"Hard negative = correct step-by-step solution with the **first equation RHS number changed by +1** "
                 f"(e.g. `∠B = 50°`→`51°`). N = {len(rows):,} pairs.\n")
    lines.append("**NEG** = sim(correct, negated). **BASE** = sim(correct, unrelated solution). "
                 "A math-faithful scorer drives NEG **down** (it noticed the number changed); a scorer blind to the "
                 "edit leaves NEG ≈ 1.0, far above BASE.\n")

    # main table
    lines.append("## 1. Similarity distributions\n")
    lines.append("| scorer (field) | NEG mean | NEG median | NEG p10–p90 | BASE mean | gap (NEG−BASE) |")
    lines.append("|---|--:|--:|--:|--:|--:|")
    dists = {}
    for label, sk, fld in FIELDS:
        neg = [r[f"{sk}_neg"][fld] for r in rows if f"{sk}_neg" in r]
        base = [r[f"{sk}_base"][fld] for r in rows if f"{sk}_base" in r]
        dists[label] = (neg, base)
        lines.append(f"| {label} | {mean(neg):.3f} | {median(neg):.3f} | "
                     f"{quant(neg,0.1):.3f}–{quant(neg,0.9):.3f} | {mean(base):.3f} | {mean(neg)-mean(base):+.3f} |")

    # tau sweep
    lines.append("\n## 2. Missed-corruption rate vs τ  (fraction of NEG pairs scoring ≥ τ — lower = better)\n")
    header = "| scorer (field) | " + " | ".join(f"τ={t}" for t in TAUS) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(TAUS) + 1))
    for label, sk, fld in FIELDS:
        neg, _ = dists[label]
        cells = []
        for t in TAUS:
            cells.append(f"{sum(1 for x in neg if x >= t)/len(neg):.3f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    # NLI contradiction shift
    lines.append("\n## 3. NLI contradiction probability (max of both directions)\n")
    lines.append("| condition | mean C | median C |")
    lines.append("|---|--:|--:|")
    for cond in ("neg", "base"):
        cs = [max(r[f"nli_{cond}"]["C_ab"], r[f"nli_{cond}"]["C_ba"]) for r in rows if f"nli_{cond}" in r]
        lines.append(f"| {cond.upper()} | {mean(cs):.3f} | {median(cs):.3f} |")

    # surface-reliance AUC
    lines.append("\n## 4. Surface-overlap reliance: AUC = P(NEG sim > BASE sim)\n")
    lines.append("High = scorer ranks a *corrupted copy of the same problem* as more similar than an unrelated "
                 "solution, i.e. it keys on surface overlap and is blind to the math error. (1.0 = always; 0.5 = chance.)\n")
    lines.append("| scorer (field) | AUC(NEG>BASE) |")
    lines.append("|---|--:|")
    for label, sk, fld in FIELDS:
        neg, base = dists[label]
        lines.append(f"| {label} | {auc_gt(neg, base):.3f} |")

    # per-source NEG mean (top sources)
    from collections import defaultdict
    bysrc = defaultdict(list)
    for r in rows:
        if "sbert_neg" in r and "nli_neg" in r:
            bysrc[r["source"]].append((r["sbert_neg"]["raw_score"], r["nli_neg"]["nli_score_equiv"]))
    lines.append("\n## 5. NEG mean by source (top by count)\n")
    lines.append("| source | n | SBERT NEG | NLI equiv NEG |")
    lines.append("|---|--:|--:|--:|")
    for src in sorted(bysrc, key=lambda s: -len(bysrc[s]))[:12]:
        v = bysrc[src]
        lines.append(f"| {src} | {len(v)} | {mean(x[0] for x in v):.3f} | {mean(x[1] for x in v):.3f} |")

    Path(MD).parent.mkdir(parents=True, exist_ok=True)
    Path(MD).write_text("\n".join(lines) + "\n")
    print(f"wrote {MD}")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
