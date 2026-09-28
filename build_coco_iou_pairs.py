"""Build COCO bounding-box IoU ranking triples.

Mirrors the negation experiment, but for SPATIAL grounding instead of numeric reasoning.

For each GT object box we:
  1. Normalize the COCO bbox [x,y,w,h] (absolute px) by image size and scale x1000,
     then convert to integer CORNER coords [x1,y1,x2,y2] in a 0-1000 space
     (the Qwen-VL box convention).  e.g. norm [0.58,0.12,0.22,0.33] -> [580,120,800,450].
  2. Generate TWO random candidate boxes by jittering the GT (random shift + scale),
     so their IoU-with-GT spans a useful range (fully-random boxes would both be ~0 IoU
     and the task would be degenerate).
  3. Compute IoU of each candidate vs the GT box.
  4. Label the HIGHER-IoU candidate `neg` and the LOWER-IoU one `NEG`
     (lowercase neg = closer to GT = should be judged MORE similar).

The ground-truth ordering we want any faithful (IoU-aware) reward to satisfy is
    sim(ref, neg) > sim(ref, NEG)
i.e. the box that overlaps the GT more should score as more similar to it.
Boxes are emitted as TEXT strings "[x1, y1, x2, y2]" so the same SBERT/NLI/BERTScore
text scorers used everywhere else can be tested for whether their text similarity
tracks IoU.  POS = the GT box string (the reference).

Output coco_iou_pairs.jsonl, one row per example.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path


def to_corners_1000(bbox, W, H):
    """COCO [x,y,w,h] px -> [x1,y1,x2,y2] ints in 0-1000 normalized space."""
    x, y, w, h = bbox
    x1 = x / W * 1000.0
    y1 = y / H * 1000.0
    x2 = (x + w) / W * 1000.0
    y2 = (y + h) / H * 1000.0
    return [int(round(v)) for v in (x1, y1, x2, y2)]


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def clip(v, lo=0, hi=1000):
    return max(lo, min(hi, v))


def jitter(box, rng):
    """Random shift + scale of a corner box -> a perturbed corner box (clipped to 0-1000)."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    # random center shift up to +-0.6 box-size, random scale 0.5-1.6
    cx += rng.uniform(-0.6, 0.6) * bw
    cy += rng.uniform(-0.6, 0.6) * bh
    bw *= rng.uniform(0.5, 1.6)
    bh *= rng.uniform(0.5, 1.6)
    nx1, ny1 = clip(cx - bw / 2), clip(cy - bh / 2)
    nx2, ny2 = clip(cx + bw / 2), clip(cy + bh / 2)
    return [int(round(nx1)), int(round(ny1)), int(round(nx2)), int(round(ny2))]


def box_str(b):
    return f"[{b[0]}, {b[1]}, {b[2]}, {b[3]}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default="/scratch/sghos104/rlpt/data/coco/annotations/instances_train2017.json")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_area_px", type=float, default=32 * 32, help="skip tiny boxes")
    ap.add_argument("--out", default="/scratch/sghos104/rlpt/data/coco/coco_iou_pairs.jsonl")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print(f"loading {args.ann} ...", flush=True)
    coco = json.load(open(args.ann))
    imgs = {im["id"]: im for im in coco["images"]}
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    anns = coco["annotations"]
    print(f"images={len(imgs):,} annotations={len(anns):,} categories={len(cats)}", flush=True)

    # eligible: not crowd, reasonable area, valid box
    elig = []
    for a in anns:
        if a.get("iscrowd"):
            continue
        if a.get("area", 0) < args.min_area_px:
            continue
        im = imgs.get(a["image_id"])
        if not im:
            continue
        W, H = im["width"], im["height"]
        x, y, w, h = a["bbox"]
        if w <= 1 or h <= 1:
            continue
        elig.append((a, W, H))
    print(f"eligible boxes: {len(elig):,}", flush=True)

    n = min(args.n, len(elig))
    sample = rng.sample(elig, n)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with out.open("w") as fw:
        for i, (a, W, H) in enumerate(sample):
            gt = to_corners_1000(a["bbox"], W, H)
            if gt[2] <= gt[0] or gt[3] <= gt[1]:
                continue
            c1, c2 = jitter(gt, rng), jitter(gt, rng)
            i1, i2 = iou(gt, c1), iou(gt, c2)
            if i1 == i2:  # tie -> nudge by regenerating one
                c2 = jitter(gt, rng); i2 = iou(gt, c2)
            # neg = higher IoU (more similar), NEG = lower IoU
            if i1 >= i2:
                neg, iou_neg, NEG, iou_NEG = c1, i1, c2, i2
            else:
                neg, iou_neg, NEG, iou_NEG = c2, i2, c1, i1
            fw.write(json.dumps({
                "idx": kept,
                "image_id": a["image_id"],
                "category": cats.get(a["category_id"], "?"),
                "pos": box_str(gt),            # reference (GT box)
                "neg": box_str(neg),           # higher-IoU candidate  (should rank MORE similar)
                "NEG": box_str(NEG),           # lower-IoU candidate
                "iou_neg": round(iou_neg, 4),
                "iou_NEG": round(iou_NEG, 4),
                "gt": gt, "neg_box": neg, "NEG_box": NEG,
            }) + "\n")
            kept += 1
    print(f"wrote {kept:,} triples -> {out}", flush=True)

    # quick sanity
    import statistics as st
    rows = [json.loads(l) for l in out.open()]
    print(f"mean IoU  neg(high)={st.mean(r['iou_neg'] for r in rows):.3f}  "
          f"NEG(low)={st.mean(r['iou_NEG'] for r in rows):.3f}")
    print("\n=== samples ===")
    for r in rows[:5]:
        print(f"[{r['category']}] POS {r['pos']}  | neg {r['neg']} IoU={r['iou_neg']}  "
              f"| NEG {r['NEG']} IoU={r['iou_NEG']}")


if __name__ == "__main__":
    main()
