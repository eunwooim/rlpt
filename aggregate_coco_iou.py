"""Aggregate COCO IoU ranking scores -> docs/coco_iou_results.md.

Question: do text scorers on box strings respect IoU ordering, i.e. is
sim(pos, neg) > sim(pos, NEG) when neg has the higher IoU?

Metrics per scorer/field:
  * rank acc = P(sim_hi > sim_lo)   (ties 0.5)  -- 1.0 ideal, 0.5 chance
  * mean sim on hi vs lo, and the margin
  * Spearman corr between sim and IoU (pooled over both candidates) -- does sim track overlap?
Plus the IoU "oracle" (rank acc = 1.0 by construction) as the reference any reward should match.
"""
from __future__ import annotations
import json
from pathlib import Path
from statistics import mean

SCORES = "/scratch/sghos104/rlpt/src/outputs/coco_iou/scores.jsonl"
MD = "/scratch/sghos104/rlpt/docs/coco_iou_results.md"

FIELDS = [
    ("SBERT cosine", "sbert", "raw_score"),
    ("NLI equiv", "nli", "nli_score_equiv"),
    ("NLI coverage", "nli", "nli_score_coverage"),
    ("BERTScore f1", "bertscore", "bertscore_f1"),
]


def rank_acc(his, los):
    w = sum(1.0 if h > l else 0.5 if h == l else 0.0 for h, l in zip(his, los))
    return w / len(his)


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def main():
    rows = [json.loads(l) for l in open(SCORES)]
    print(f"loaded {len(rows):,} scored rows")

    L = []
    L.append("# COCO bounding-box IoU ranking — do text scorers respect spatial overlap?\n")
    L.append(f"N = {len(rows):,} triples. Each: a GT box (`pos`, the reference) and two jittered "
             f"candidate boxes — `neg` (HIGHER IoU with GT) and `NEG` (LOWER IoU). Boxes are rendered "
             f"as text `[x1, y1, x2, y2]` in 0-1000 normalized coords (Qwen-VL convention).\n")
    L.append("A faithful, IoU-aware reward must satisfy **sim(pos, neg) > sim(pos, NEG)** — the box that "
             "overlaps the GT more should be judged more similar. **rank acc** = fraction of triples where "
             "that holds (1.0 = always, 0.5 = chance).\n")

    iou_hi = [r["iou_neg"] for r in rows]
    iou_lo = [r["iou_NEG"] for r in rows]
    L.append(f"Mean IoU: neg (high) = {mean(iou_hi):.3f}, NEG (low) = {mean(iou_lo):.3f}, "
             f"margin = {mean(iou_hi) - mean(iou_lo):.3f}.\n")

    L.append("## Rank accuracy & IoU correlation\n")
    L.append("| scorer (field) | rank acc ↑ | mean sim(hi) | mean sim(lo) | margin | Spearman(sim, IoU) |")
    L.append("|---|--:|--:|--:|--:|--:|")
    # IoU oracle row
    L.append(f"| **IoU (oracle)** | **1.000** | {mean(iou_hi):.3f} | {mean(iou_lo):.3f} | "
             f"{mean(iou_hi)-mean(iou_lo):+.3f} | 1.000 |")
    for label, sk, fld in FIELDS:
        hi = [r[f"{sk}_hi"][fld] for r in rows if f"{sk}_hi" in r]
        lo = [r[f"{sk}_lo"][fld] for r in rows if f"{sk}_lo" in r]
        if not hi:
            continue
        ra = rank_acc(hi, lo)
        # pooled sim-vs-IoU correlation across both candidates
        sims = hi + lo
        ious = iou_hi + iou_lo
        sp = spearman(sims, ious)
        L.append(f"| {label} | {ra:.3f} | {mean(hi):.3f} | {mean(lo):.3f} | "
                 f"{mean(hi)-mean(lo):+.3f} | {sp:.3f} |")

    L.append("\n**Reading:** rank acc near 0.5 and Spearman near 0 ⇒ the text scorer's similarity is "
             "**blind to IoU** — it cannot tell which box overlaps the GT more, because it only sees digit "
             "strings. This is the spatial analogue of the numeric-negation blindness, and the quantitative "
             "case for an explicit IoU term in any grounding reward (the supervisor's point).\n")

    Path(MD).parent.mkdir(parents=True, exist_ok=True)
    Path(MD).write_text("\n".join(L) + "\n")
    print(f"wrote {MD}")
    print("\n".join(L))


if __name__ == "__main__":
    main()
