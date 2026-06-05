import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont


DEFAULT_JSONL = "/scratch/eunwooim/rlpt/src/data_factory/gqa_v2/eval_all.jsonl"
DEFAULT_OUT_DIR = "/scratch/eunwooim/rlpt/src/visualize/gqa_v2/visualize_boxes/"
DEFAULT_IMAGE_DIR = "/scratch/eunwooim/rlpt/src/visualize/images/"


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def safe_filename(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    name = "".join(out).strip("_")
    return name or "unknown"


def resolve_image_path(record: Dict[str, Any], image_dir: str) -> str:
    candidates = []

    images = record.get("images", [])
    if images:
        candidates.append(images[0])

    extra = record.get("extra_info", {})
    if extra.get("image_path"):
        candidates.append(extra["image_path"])

    image_id = extra.get("image_id")
    if image_id is not None:
        candidates.append(str(Path(image_dir) / f"{image_id}.jpg"))

    for c in candidates:
        p = Path(c)
        if p.exists():
            return str(p)

    raise FileNotFoundError("Image not found. Tried:\n" + "\n".join(candidates))


def get_question_id(record: Dict[str, Any], idx: int) -> str:
    return (
        record.get("question_id")
        or record.get("extra_info", {}).get("question_id")
        or f"idx_{idx}"
    )


def get_gt(record: Dict[str, Any]) -> Dict[str, Any]:
    return record["reward_model"]["ground_truth"]


def get_bbox_nodes(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    gt = get_gt(record)
    evidence = gt.get("target_evidence", {})
    nodes = evidence.get("nodes", [])

    out = []
    if not isinstance(nodes, list):
        return out

    for node in nodes:
        if isinstance(node, dict) and isinstance(node.get("bbox"), list) and len(node["bbox"]) == 4:
            out.append(node)

    return out


def norm1000_to_pixel(box: List[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [
        x1 / 1000.0 * width,
        y1 / 1000.0 * height,
        x2 / 1000.0 * width,
        y2 / 1000.0 * height,
    ]


def clip_box(box: List[float], width: int, height: int) -> Optional[List[float]]:
    x1, y1, x2, y2 = box

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))

    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def draw_record(record: Dict[str, Any], image_path: str, out_path: str) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(14, img.width // 38))
        small_font = ImageFont.truetype("DejaVuSans.ttf", size=max(12, img.width // 48))
    except Exception:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    gt = get_gt(record)
    schema = gt.get("target_schema", record.get("extra_info", {}).get("schema", "unknown_schema"))
    answer = gt.get("target_answer", "")
    question = record.get("extra_info", {}).get("question", "")

    nodes = get_bbox_nodes(record)

    palette = [
        (255, 0, 0),
        (0, 128, 255),
        (0, 180, 0),
        (255, 128, 0),
        (180, 0, 255),
        (255, 0, 180),
    ]

    line_width = max(2, round(min(img.width, img.height) * 0.006))

    for i, node in enumerate(nodes):
        color = palette[i % len(palette)]
        box = norm1000_to_pixel(node["bbox"], img.width, img.height)
        box = clip_box(box, img.width, img.height)
        if box is None:
            continue

        x1, y1, x2, y2 = box

        for off in range(line_width):
            draw.rectangle(
                [x1 - off, y1 - off, x2 + off, y2 + off],
                outline=color,
            )

        label_parts = []
        if node.get("id"):
            label_parts.append(str(node["id"]))
        if node.get("name"):
            label_parts.append(str(node["name"]))
        if node.get("relation"):
            label_parts.append(str(node["relation"]))
        if node.get("attribute"):
            label_parts.append(f"attr={node['attribute']}")
        if node.get("attributes"):
            attrs = ",".join(str(a) for a in node["attributes"][:3])
            label_parts.append(f"[{attrs}]")

        label = " | ".join(label_parts) if label_parts else f"node_{i}"

        tb = draw.textbbox((0, 0), label, font=small_font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad = 4
        tx = max(0, min(x1, img.width - tw - 2 * pad))
        ty = max(0, y1 - th - 2 * pad)

        draw.rectangle(
            [tx, ty, tx + tw + 2 * pad, ty + th + 2 * pad],
            fill=(255, 255, 255),
            outline=color,
        )
        draw.text((tx + pad, ty + pad), label, fill=color, font=small_font)

    title = f"{schema} | answer={answer}"
    subtitle = question[:180]

    y = 0
    for text, used_font in [(title, font), (subtitle, small_font)]:
        if not text:
            continue
        tb = draw.textbbox((0, 0), text, font=used_font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad = 5
        draw.rectangle(
            [0, y, min(img.width, tw + 2 * pad), y + th + 2 * pad],
            fill=(255, 255, 255),
            outline=(0, 0, 0),
        )
        draw.text((pad, y + pad), text, fill=(0, 0, 0), font=used_font)
        y += th + 2 * pad

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of bbox-evidence examples to visualize.")
    parser.add_argument("--jsonl", default=DEFAULT_JSONL)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--image_dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["random", "first"], default="first")
    args = parser.parse_args()

    records = read_jsonl(args.jsonl)

    bbox_records = [
        (idx, r) for idx, r in enumerate(records)
        if len(get_bbox_nodes(r)) > 0
    ]

    if args.mode == "random":
        rng = random.Random(args.seed)
        rng.shuffle(bbox_records)

    selected = bbox_records[: args.n]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"JSONL: {args.jsonl}")
    print(f"Loaded records: {len(records)}")
    print(f"Records with bbox evidence: {len(bbox_records)}")
    print(f"Visualizing: {len(selected)}")
    print(f"Output dir: {out_dir}")

    manifest_path = out_dir / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    for idx, record in selected:
        qid = safe_filename(get_question_id(record, idx))
        out_path = str(out_dir / f"{qid}.jpg")
        image_path = resolve_image_path(record, args.image_dir)

        draw_record(record, image_path, out_path)

        gt = get_gt(record)
        row = {
            "index": idx,
            "question_id": get_question_id(record, idx),
            "schema": gt.get("target_schema"),
            "answer": gt.get("target_answer"),
            "question": record.get("extra_info", {}).get("question"),
            "image": image_path,
            "boxed_image": out_path,
            "nodes": get_bbox_nodes(record),
        }

        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print("=" * 100)
        print(f"QUESTION_ID: {row['question_id']}")
        print(f"SCHEMA: {row['schema']}")
        print(f"QUESTION: {row['question']}")
        print(f"ANSWER: {row['answer']}")
        print(f"SOURCE IMAGE: {image_path}")
        print(f"BOXED IMAGE: {out_path}")

    print(f"\nSaved manifest: {manifest_path}")


if __name__ == "__main__":
    main()