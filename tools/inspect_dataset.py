"""Inspect a QCVSR GQA dataset jsonl.

Runs a battery of validity checks on each record, writes a per-record CSV,
prints a pass/fail summary, and renders a 5-column sample grid PNG with
bounding boxes drawn on the images.

Usage:
    python tools/inspect_dataset.py
    python tools/inspect_dataset.py --input data/qcvsr_gqa_v1/eval_all.jsonl
"""

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


KNOWN_BAD_PLURALS = {
    "mans": "men",
    "womans": "women",
    "childs": "children",
    "feets": "feet",
    "tooths": "teeth",
    "mouses": "mice",
    "sheeps": "sheep",
    "fishs": "fish",
    "geeses": "geese",
}

REQUIRED_TOP_FIELDS = [
    "question_id", "prompt", "images", "reward_model", "extra_info", "ability"
]


def check_image_exists(rec):
    return Path(rec["images"][0]).exists()


def check_image_loadable(rec):
    try:
        with Image.open(rec["images"][0]) as im:
            im.verify()
        return True
    except Exception:
        return False


def _iter_bboxes(node):
    """Yield bboxes from a node — either single `bbox` or list under `instances`."""
    if "instances" in node:
        for bb in node["instances"]:
            yield bb
    elif "bbox" in node:
        yield node["bbox"]


def _cx(b):
    return (b[0] + b[2]) / 2


def _cy(b):
    return (b[1] + b[3]) / 2


def _area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def check_bbox_in_range(rec):
    ev = rec["reward_model"]["ground_truth"]["target_evidence"]
    for node in ev.get("nodes", []):
        for bbox in _iter_bboxes(node):
            if len(bbox) != 4:
                return False
            if any(not (0 <= v <= 1000) for v in bbox):
                return False
    return True


def check_bbox_valid_order(rec):
    ev = rec["reward_model"]["ground_truth"]["target_evidence"]
    for node in ev.get("nodes", []):
        for bbox in _iter_bboxes(node):
            x1, y1, x2, y2 = bbox
            if not (x1 < x2 and y1 < y2):
                return False
    return True


# Schemas whose evidence is exactly two nodes (one per compared object).
PAIRWISE_SCHEMAS = {
    "bbox_pair_spatial_compare", "bbox_pair_size_compare",
    "count_pair_compare", "attr_select_spatial", "relation_target",
}


def check_evidence_node_count(rec):
    """Pairwise schemas: exactly 2 nodes. count_relation: 1 reference + >=2 subjects."""
    schema = rec["reward_model"]["ground_truth"]["target_schema"]
    ev = rec["reward_model"]["ground_truth"]["target_evidence"]
    nodes = ev.get("nodes", [])
    if schema in PAIRWISE_SCHEMAS:
        return len(nodes) == 2
    if schema == "count_relation":
        refs = [n for n in nodes if n.get("role") == "reference"]
        subs = [n for n in nodes if n.get("role") == "subject"]
        return len(refs) == 1 and len(subs) >= 2 and len(nodes) == len(refs) + len(subs)
    return False


def check_evidence_schema_match(rec):
    schema = rec["reward_model"]["ground_truth"]["target_schema"]
    ev = rec["reward_model"]["ground_truth"]["target_evidence"]
    nodes = ev.get("nodes", [])
    if not nodes:
        return False
    if schema in ("bbox_pair_spatial_compare", "bbox_pair_size_compare"):
        return all("bbox" in n and "name" in n for n in nodes)
    if schema == "count_pair_compare":
        return all("count" in n and "name" in n for n in nodes)
    if schema == "attr_select_spatial":
        if not all("bbox" in n and "name" in n for n in nodes):
            return False
        ids = {n.get("id") for n in nodes}
        return ev.get("selected_id") in ids
    if schema == "count_relation":
        if not all("bbox" in n and "name" in n for n in nodes):
            return False
        subs = [n for n in nodes if n.get("role") == "subject"]
        return "relation" in ev and ev.get("count") == len(subs)
    if schema == "relation_target":
        if not all("bbox" in n and "name" in n for n in nodes):
            return False
        roles = {n.get("role") for n in nodes}
        return {"subject", "target"} <= roles and "relation" in ev
    return False


def check_answer_consistency(rec):
    """Apply the verifier deterministically and check it produces target_answer."""
    gt = rec["reward_model"]["ground_truth"]
    verifier = gt["verifier"]
    ev = gt["target_evidence"]
    nodes = ev.get("nodes", [])
    target = str(gt["target_answer"]).strip().lower()

    try:
        if verifier in ("compare_x_center_min", "compare_x_center_max"):
            a, b = nodes
            if verifier == "compare_x_center_min":
                expected = a["name"] if _cx(a["bbox"]) < _cx(b["bbox"]) else b["name"]
            else:
                expected = a["name"] if _cx(a["bbox"]) > _cx(b["bbox"]) else b["name"]
        elif verifier in ("compare_y_center_min", "compare_y_center_max"):
            a, b = nodes
            if verifier == "compare_y_center_min":
                expected = a["name"] if _cy(a["bbox"]) < _cy(b["bbox"]) else b["name"]
            else:
                expected = a["name"] if _cy(a["bbox"]) > _cy(b["bbox"]) else b["name"]
        elif verifier in ("compare_bbox_area_max", "compare_bbox_area_min"):
            a, b = nodes
            if verifier == "compare_bbox_area_max":
                expected = a["name"] if _area(a["bbox"]) > _area(b["bbox"]) else b["name"]
            else:
                expected = a["name"] if _area(a["bbox"]) < _area(b["bbox"]) else b["name"]
        elif verifier == "compare_count_greater_than":
            a, b = nodes
            expected = "yes" if a["count"] > b["count"] else "no"
        elif verifier == "count_difference":
            a, b = nodes
            expected = str(a["count"] - b["count"])
        elif verifier.startswith("select_") and verifier.endswith("_color"):
            # Consistency = the selected node is the geometric extreme on its axis.
            a, b = nodes
            if "x_center_min" in verifier:
                geo = a if _cx(a["bbox"]) < _cx(b["bbox"]) else b
            elif "x_center_max" in verifier:
                geo = a if _cx(a["bbox"]) > _cx(b["bbox"]) else b
            elif "y_center_min" in verifier:
                geo = a if _cy(a["bbox"]) < _cy(b["bbox"]) else b
            else:  # y_center_max
                geo = a if _cy(a["bbox"]) > _cy(b["bbox"]) else b
            return geo["id"] == ev.get("selected_id")
        elif verifier == "count_related_subjects":
            subs = [n for n in nodes if n.get("role") == "subject"]
            return str(len(subs)) == target and ev.get("count") == len(subs)
        elif verifier == "relation_target_name":
            tgt = [n for n in nodes if n.get("role") == "target"]
            return len(tgt) == 1 and str(tgt[0]["name"]).strip().lower() == target
        else:
            return False
    except (KeyError, IndexError, TypeError, ValueError):
        return False

    return str(expected).strip().lower() == target


def check_pluralization(rec):
    """Question text should not contain naive bad plurals like 'mans', 'feets'."""
    text = rec["extra_info"]["question"].lower()
    words = re.findall(r"\b[a-z]+\b", text)
    return not any(w in KNOWN_BAD_PLURALS for w in words)


def check_required_fields(rec):
    return all(f in rec for f in REQUIRED_TOP_FIELDS)


CHECKS = [
    ("image_exists", check_image_exists),
    ("image_loadable", check_image_loadable),
    ("bbox_in_range", check_bbox_in_range),
    ("bbox_valid_order", check_bbox_valid_order),
    ("evidence_node_count", check_evidence_node_count),
    ("evidence_schema_match", check_evidence_schema_match),
    ("answer_consistency", check_answer_consistency),
    ("pluralization", check_pluralization),
    ("required_fields", check_required_fields),
]


def run_checks(records):
    rows = []
    summary = {name: Counter() for name, _ in CHECKS}
    for r in records:
        row = {"record_id": r.get("question_id", "")}
        for name, fn in CHECKS:
            try:
                ok = bool(fn(r))
            except Exception:
                ok = False
            label = "PASS" if ok else "FAIL"
            row[name] = label
            summary[name][label] += 1
        rows.append(row)
    return rows, summary


def render_sample_grid(records, out_path, n=50, ncols=5, cell_w=320, cell_h=320,
                       caption_h=80, seed=42):
    rng = random.Random(seed)
    samples = rng.sample(records, min(n, len(records)))
    nrows = (len(samples) + ncols - 1) // ncols

    full_h = cell_h + caption_h
    grid = Image.new("RGB", (cell_w * ncols, full_h * nrows), "white")
    draw_grid = ImageDraw.Draw(grid)

    try:
        font_q = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 13)
        font_a = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 14)
        font_op = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font_q = font_a = font_op = ImageFont.load_default()

    bbox_colors = [(255, 60, 60), (60, 130, 255)]

    for i, rec in enumerate(samples):
        row, col = divmod(i, ncols)
        x0, y0 = col * cell_w, row * full_h

        try:
            im = Image.open(rec["images"][0]).convert("RGB")
            W, H = im.size
            d = ImageDraw.Draw(im)
            ev = rec["reward_model"]["ground_truth"]["target_evidence"]
            for j, node in enumerate(ev.get("nodes", [])):
                for bbox in _iter_bboxes(node):
                    x1, y1, x2, y2 = bbox
                    x1 = x1 * W / 1000.0
                    x2 = x2 * W / 1000.0
                    y1 = y1 * H / 1000.0
                    y2 = y2 * H / 1000.0
                    d.rectangle([x1, y1, x2, y2],
                                outline=bbox_colors[j % 2],
                                width=max(3, W // 150))
            scale = min(cell_w / W, cell_h / H)
            new_w, new_h = int(W * scale), int(H * scale)
            im = im.resize((new_w, new_h), Image.LANCZOS)
            paste_x = x0 + (cell_w - new_w) // 2
            paste_y = y0 + (cell_h - new_h) // 2
            grid.paste(im, (paste_x, paste_y))
        except Exception as e:
            draw_grid.text((x0 + 10, y0 + 10),
                           f"load failed:\n{type(e).__name__}",
                           fill="red", font=font_q)

        cy = y0 + cell_h + 4
        q = rec["extra_info"]["question"]
        if len(q) > 54:
            q = q[:52] + "..."
        ans = str(rec["reward_model"]["ground_truth"]["target_answer"])
        op = rec["extra_info"]["operation"]

        draw_grid.text((x0 + 6, cy), f"Q: {q}", fill="black", font=font_q)
        draw_grid.text((x0 + 6, cy + 22), f"A: {ans}",
                       fill=(5, 130, 90), font=font_a)
        draw_grid.text((x0 + 6, cy + 46), f"[{op}]", fill="gray", font=font_op)

        draw_grid.line([(x0, y0 + full_h - 1),
                        (x0 + cell_w, y0 + full_h - 1)],
                       fill=(220, 220, 220))
        if col > 0:
            draw_grid.line([(x0, y0), (x0, y0 + full_h)],
                           fill=(220, 220, 220))

    grid.save(out_path, "PNG", optimize=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",
                   default="data/qcvsr_gqa_v1/train.jsonl",
                   help="Path to the jsonl dataset to inspect.")
    p.add_argument("--out-csv", default="tools/validity_report.csv")
    p.add_argument("--out-grid", default="tools/sample_grid.png")
    p.add_argument("--n-samples", type=int, default=50,
                   help="Number of records in the sample grid.")
    args = p.parse_args()

    in_path = Path(args.input)
    print(f"Loading: {in_path}")
    records = [json.loads(l) for l in in_path.open()]
    print(f"Loaded {len(records)} records")

    print("Running validity checks...")
    rows, summary = run_checks(records)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["record_id"] + [n for n, _ in CHECKS]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV: {out_csv}")

    width = 58
    print("\n" + "=" * width)
    print(f"  {'Check':<26} {'PASS':>8} {'FAIL':>8} {'Pass %':>9}")
    print("-" * width)
    for name, _ in CHECKS:
        s = summary[name]
        total = s["PASS"] + s["FAIL"]
        rate = s["PASS"] / total * 100 if total else 0
        print(f"  {name:<26} {s['PASS']:>8} {s['FAIL']:>8} {rate:>8.1f}%")
    print("=" * width)

    print(f"\nRendering sample grid ({args.n_samples} records, 5 cols)...")
    render_sample_grid(records, args.out_grid, n=args.n_samples)
    print(f"Wrote: {args.out_grid}")


if __name__ == "__main__":
    main()
