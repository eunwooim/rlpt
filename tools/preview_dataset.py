"""Render a self-contained HTML preview of a QCVSR GQA JSONL for visual validation.

Samples records per schema, draws the evidence bounding boxes on each image
(colour/labelled by node role), and writes a single standalone HTML file with
the question, gold answer, verifier, and evidence JSON beside each image.

Images are resized and inlined as base64 so the HTML is portable (one file).

Usage:
    python tools/preview_dataset.py \
        --input data/qcvsr_gqa_v1_test/train.jsonl \
        --output data/qcvsr_gqa_v1_test/preview_qcvsr_v1_test.html \
        --per-schema 8
"""

import argparse
import base64
import html
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

MAX_W = 460  # preview image width in px

# Per-role overlay colours (RGB).
ROLE_COLORS = {
    "reference": (255, 140, 0),
    "subject": (60, 130, 255),
    "target": (40, 200, 90),
    "selected": (40, 200, 90),
    "other": (150, 150, 150),
}
PAIR_COLORS = [(255, 60, 60), (60, 130, 255)]


def _font(size):
    for p in ("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _denorm(bbox, w, h):
    x1, y1, x2, y2 = bbox
    return [x1 * w / 1000.0, y1 * h / 1000.0, x2 * w / 1000.0, y2 * h / 1000.0]


def _node_overlays(rec):
    """Yield (bbox_norm, color, label, width) for each drawable evidence node."""
    gt = rec["reward_model"]["ground_truth"]
    schema = gt["target_schema"]
    ev = gt["target_evidence"]
    nodes = ev.get("nodes", [])
    selected_id = ev.get("selected_id")

    for idx, n in enumerate(nodes):
        bbox = n.get("bbox")
        if not bbox:
            continue
        role = n.get("role")
        if schema == "attr_select_spatial":
            if n.get("id") == selected_id:
                color, label, wdt = ROLE_COLORS["selected"], f"{n['name']} ✓", 4
            else:
                color, label, wdt = ROLE_COLORS["other"], n["name"], 2
        elif role in ROLE_COLORS:
            color = ROLE_COLORS[role]
            label = n["name"] if role != "subject" else f"{n['name']} #{idx}"
            wdt = 4 if role in ("reference", "target") else 3
        else:
            color = PAIR_COLORS[idx % 2]
            label, wdt = n.get("name", ""), 3
        yield bbox, color, label, wdt


def render_image(rec):
    """Return base64 PNG of the image with evidence overlays, or None."""
    path = rec["images"][0]
    if not Path(path).exists():
        return None
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return None
    W, H = im.size
    scale = min(1.0, MAX_W / W)
    im = im.resize((max(1, int(W * scale)), max(1, int(H * scale))), Image.LANCZOS)
    w, h = im.size
    d = ImageDraw.Draw(im)
    font = _font(13)
    for bbox, color, label, wdt in _node_overlays(rec):
        x1, y1, x2, y2 = _denorm(bbox, w, h)
        d.rectangle([x1, y1, x2, y2], outline=color, width=wdt)
        if label:
            ty = max(0, y1 - 15)
            d.rectangle([x1, ty, x1 + 8 * len(label) + 6, ty + 15], fill=color)
            d.text((x1 + 3, ty + 1), label, fill=(255, 255, 255), font=font)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def card_html(rec, b64):
    gt = rec["reward_model"]["ground_truth"]
    q = html.escape(rec["extra_info"]["question"])
    ans = html.escape(str(gt["target_answer"]))
    verifier = html.escape(gt["verifier"])
    op = html.escape(str(rec["extra_info"].get("operation", "")))
    ev = html.escape(json.dumps(gt["target_evidence"]))
    img_tag = (f'<img src="data:image/png;base64,{b64}"/>' if b64
               else '<div class="noimg">image not found</div>')
    return f"""
    <div class="card">
      {img_tag}
      <div class="meta">
        <div class="q">{q}</div>
        <div class="a">answer: <b>{ans}</b></div>
        <div class="op">op: {op} &middot; verifier: {verifier}</div>
        <details><summary>evidence</summary><pre>{ev}</pre></details>
      </div>
    </div>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--per-schema", type=int, default=8,
                    help="Default samples per schema.")
    ap.add_argument("--schema-counts", default="",
                    help="Per-schema overrides, e.g. "
                         "'relation_target=20,bbox_pair_spatial_compare=6'. "
                         "Schemas not listed use --per-schema.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    overrides = {}
    for part in filter(None, (p.strip() for p in args.schema_counts.split(","))):
        k, v = part.split("=")
        overrides[k.strip()] = int(v)

    rng = random.Random(args.seed)
    by_schema = defaultdict(list)
    schema_c, op_c = Counter(), Counter()
    with open(args.input) as f:
        for line in f:
            r = json.loads(line)
            s = r["reward_model"]["ground_truth"]["target_schema"]
            by_schema[s].append(r)
            schema_c[s] += 1
            op_c[r["extra_info"].get("operation", "")] += 1

    # Show compositional schemas first, then the spatial/size comparisons.
    DISPLAY_ORDER = [
        "attr_select_spatial", "count_relation", "relation_target",
        "bbox_pair_spatial_compare", "bbox_pair_size_compare", "count_pair_compare",
    ]
    ordered = [s for s in DISPLAY_ORDER if s in by_schema]
    ordered += [s for s in sorted(by_schema) if s not in DISPLAY_ORDER]

    sections = []
    for schema in ordered:
        recs = by_schema[schema]
        n = overrides.get(schema, args.per_schema)
        sample = rng.sample(recs, min(n, len(recs)))
        cards = "".join(card_html(r, render_image(r)) for r in sample)
        sections.append(
            f'<h2>{html.escape(schema)} '
            f'<span class="count">({len(recs)} records)</span></h2>'
            f'<div class="grid">{cards}</div>'
        )

    stats = " &middot; ".join(f"{html.escape(k)}: {v}" for k, v in schema_c.most_common())
    ops = " &middot; ".join(f"{html.escape(k)}: {v}" for k, v in op_c.most_common())

    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>QCVSR preview — {html.escape(Path(args.input).name)}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 24px; background:#f7f7f9; color:#222; }}
  h1 {{ margin-bottom:4px; }}
  .summary {{ color:#555; font-size:13px; margin-bottom:18px; line-height:1.6; }}
  h2 {{ margin-top:30px; border-bottom:2px solid #ddd; padding-bottom:4px; }}
  .count {{ color:#888; font-weight:normal; font-size:14px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:16px; }}
  .card {{ background:#fff; border:1px solid #e2e2e6; border-radius:8px; overflow:hidden; }}
  .card img {{ width:100%; display:block; }}
  .noimg {{ padding:40px; text-align:center; color:#999; background:#eee; }}
  .meta {{ padding:10px 12px; font-size:13px; }}
  .q {{ font-weight:600; margin-bottom:6px; }}
  .a {{ color:#1a7f37; margin-bottom:4px; }}
  .op {{ color:#777; font-size:11px; margin-bottom:6px; }}
  details pre {{ white-space:pre-wrap; word-break:break-all; font-size:11px; background:#f3f3f5;
                 padding:6px; border-radius:4px; }}
  summary {{ cursor:pointer; color:#555; font-size:12px; }}
</style></head><body>
<h1>QCVSR GQA preview</h1>
<div class="summary"><b>{html.escape(args.input)}</b><br>
schemas: {stats}<br>operations: {ops}</div>
{''.join(sections)}
</body></html>"""

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    total = sum(schema_c.values())
    print(f"[preview] {total} records across {len(by_schema)} schemas -> {out}")


if __name__ == "__main__":
    main()
