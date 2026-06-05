# Validating if the GQA generation fails from the annotation.
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


SCENE_GRAPH_PATH = "/scratch/eunwooim/train_sceneGraphs.json"
IMAGE_DIR = "/scratch/eunwooim/rlpt/visualize/images"
OUT_DIR = "/scratch/eunwooim/rlpt/visualize/scene_graph_inspection"

STUFF_OBJECTS = {
    "water", "sky", "grass", "road", "street", "sidewalk", "floor", "ground",
    "wall", "ceiling", "field", "snow", "sand", "sea", "ocean", "river",
    "lake", "background", "room", "area", "place", "window", "building",
    "cloud", "clouds", "tree leaves", "leaves"
}


def load_scene_graphs(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): v for k, v in raw.items()}


def get_image_size_from_sg(sg: Dict[str, Any]) -> Tuple[int, int]:
    """
    GQA scene graphs often contain width/height at image level.
    This function is defensive because exact key names may vary.
    """
    width = sg.get("width") or sg.get("image_width") or sg.get("w")
    height = sg.get("height") or sg.get("image_height") or sg.get("h")

    if width is None or height is None:
        return -1, -1

    return int(width), int(height)


def normalize_name(name: Any) -> str:
    return str(name).strip().lower()


def xywh_to_xyxy(x: float, y: float, w: float, h: float) -> List[float]:
    return [x, y, x + w, y + h]


def box_area_xywh(w: float, h: float) -> float:
    return max(0.0, w) * max(0.0, h)


def validate_object_bbox(obj: Dict[str, Any], image_width: int, image_height: int) -> List[str]:
    problems = []

    for k in ["x", "y", "w", "h"]:
        if k not in obj:
            problems.append(f"missing_{k}")

    if problems:
        return problems

    try:
        x = float(obj["x"])
        y = float(obj["y"])
        w = float(obj["w"])
        h = float(obj["h"])
    except Exception:
        return ["non_numeric_bbox"]

    if w <= 0:
        problems.append("non_positive_width")
    if h <= 0:
        problems.append("non_positive_height")

    if image_width > 0 and image_height > 0:
        if x < 0:
            problems.append("x_negative")
        if y < 0:
            problems.append("y_negative")
        if x + w > image_width:
            problems.append("x2_out_of_bounds")
        if y + h > image_height:
            problems.append("y2_out_of_bounds")

        area_ratio = box_area_xywh(w, h) / float(image_width * image_height)
        if area_ratio < 0.005:
            problems.append("too_small_for_generator")
        if area_ratio > 0.50:
            problems.append("too_large_for_generator")

    return problems


def summarize_scene_graphs(scene_graphs: Dict[str, Any]) -> None:
    image_count = len(scene_graphs)

    object_name_counts = Counter()
    attribute_counts = Counter()
    relation_counts = Counter()
    objects_per_image = []
    valid_objects_per_image = []
    bbox_problem_counts = Counter()
    missing_size_count = 0
    empty_object_images = 0
    stuff_object_counts = Counter()

    example_bad_objects = []

    for image_id, sg in scene_graphs.items():
        width, height = get_image_size_from_sg(sg)
        if width <= 0 or height <= 0:
            missing_size_count += 1

        objects = sg.get("objects", {})
        if not isinstance(objects, dict):
            empty_object_images += 1
            continue

        objects_per_image.append(len(objects))

        valid_count = 0

        for object_id, obj in objects.items():
            if not isinstance(obj, dict):
                continue

            name = normalize_name(obj.get("name", ""))
            if not name:
                object_name_counts["<missing_name>"] += 1
            else:
                object_name_counts[name] += 1

            if name in STUFF_OBJECTS:
                stuff_object_counts[name] += 1

            attrs = obj.get("attributes", [])
            if isinstance(attrs, list):
                for attr in attrs:
                    attribute_counts[normalize_name(attr)] += 1

            rels = obj.get("relations", [])
            if isinstance(rels, list):
                for rel in rels:
                    if isinstance(rel, dict):
                        relation_name = normalize_name(rel.get("name", ""))
                        if relation_name:
                            relation_counts[relation_name] += 1

            problems = validate_object_bbox(obj, width, height)
            if problems:
                for p in problems:
                    bbox_problem_counts[p] += 1

                if len(example_bad_objects) < 20:
                    example_bad_objects.append(
                        {
                            "image_id": image_id,
                            "object_id": str(object_id),
                            "name": name,
                            "bbox": {k: obj.get(k) for k in ["x", "y", "w", "h"]},
                            "image_size": [width, height],
                            "problems": problems,
                        }
                    )
            else:
                valid_count += 1

        valid_objects_per_image.append(valid_count)

    print("=" * 80)
    print("SCENE GRAPH SUMMARY")
    print("=" * 80)
    print(f"Images: {image_count}")
    print(f"Images missing width/height: {missing_size_count}")
    print(f"Images with invalid/missing objects dict: {empty_object_images}")

    if objects_per_image:
        print()
        print("Objects per image:")
        print(f"  min: {min(objects_per_image)}")
        print(f"  max: {max(objects_per_image)}")
        print(f"  avg: {sum(objects_per_image) / len(objects_per_image):.2f}")

    if valid_objects_per_image:
        print()
        print("Valid objects per image after bbox checks:")
        print(f"  min: {min(valid_objects_per_image)}")
        print(f"  max: {max(valid_objects_per_image)}")
        print(f"  avg: {sum(valid_objects_per_image) / len(valid_objects_per_image):.2f}")

    print()
    print("Top object names:")
    for name, count in object_name_counts.most_common(30):
        print(f"  {name}: {count}")

    print()
    print("Top STUFF_OBJECTS found:")
    for name, count in stuff_object_counts.most_common(30):
        print(f"  {name}: {count}")

    print()
    print("Top attributes:")
    for name, count in attribute_counts.most_common(30):
        print(f"  {name}: {count}")

    print()
    print("Top relations:")
    for name, count in relation_counts.most_common(30):
        print(f"  {name}: {count}")

    print()
    print("BBox problem counts:")
    for problem, count in bbox_problem_counts.most_common():
        print(f"  {problem}: {count}")

    print()
    print("Example bad objects:")
    for ex in example_bad_objects:
        print(json.dumps(ex, indent=2))


def draw_scene_graph_image(
    image_id: str,
    sg: Dict[str, Any],
    image_dir: Path,
    out_dir: Path,
    max_boxes: int = 30,
) -> bool:
    image_path = image_dir / f"{image_id}.jpg"
    if not image_path.exists():
        return False

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    objects = sg.get("objects", {})
    if not isinstance(objects, dict):
        return False

    draw = ImageDraw.Draw(image)

    items = []
    for object_id, obj in objects.items():
        if not isinstance(obj, dict):
            continue

        name = normalize_name(obj.get("name", ""))
        if not all(k in obj for k in ["x", "y", "w", "h"]):
            continue

        try:
            x = float(obj["x"])
            y = float(obj["y"])
            w = float(obj["w"])
            h = float(obj["h"])
        except Exception:
            continue

        area = max(0.0, w) * max(0.0, h)
        items.append((area, str(object_id), name, x, y, w, h))

    # Draw larger boxes first, so small boxes stay visible.
    items = sorted(items, reverse=True)[:max_boxes]

    for _, object_id, name, x, y, w, h in items:
        x1, y1, x2, y2 = x, y, x + w, y + h

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        label = f"{object_id}: {name}"
        text_x = max(0, x1)
        text_y = max(0, y1 - 12)

        draw.rectangle(
            [text_x, text_y, text_x + 8 * len(label), text_y + 14],
            fill="red",
        )
        draw.text((text_x, text_y), label, fill="white")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_id}_annotated.jpg"
    image.save(out_path, quality=95)

    print(f"Wrote annotated image: {out_path}")
    return True


def write_sample_object_json(
    scene_graphs: Dict[str, Any],
    out_dir: Path,
    num_images: int = 20,
    seed: int = 42,
) -> None:
    rng = random.Random(seed)
    image_ids = list(scene_graphs.keys())
    rng.shuffle(image_ids)
    image_ids = image_ids[:num_images]

    out = {}

    for image_id in image_ids:
        sg = scene_graphs[image_id]
        width, height = get_image_size_from_sg(sg)
        objects = sg.get("objects", {})

        clean_objects = []
        if isinstance(objects, dict):
            for object_id, obj in objects.items():
                if not isinstance(obj, dict):
                    continue

                clean_objects.append(
                    {
                        "object_id": str(object_id),
                        "name": obj.get("name"),
                        "bbox_xywh": [obj.get("x"), obj.get("y"), obj.get("w"), obj.get("h")],
                        "attributes": obj.get("attributes", []),
                        "relations": obj.get("relations", []),
                        "bbox_problems": validate_object_bbox(obj, width, height),
                    }
                )

        out[image_id] = {
            "width": width,
            "height": height,
            "num_objects": len(clean_objects),
            "objects": clean_objects,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sample_scene_graph_objects.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Wrote sample object JSON: {out_path}")


def main() -> None:
    scene_graph_path = Path(SCENE_GRAPH_PATH)
    image_dir = Path(IMAGE_DIR)
    out_dir = Path(OUT_DIR)

    scene_graphs = load_scene_graphs(str(scene_graph_path))

    summarize_scene_graphs(scene_graphs)
    write_sample_object_json(scene_graphs, out_dir, num_images=20, seed=42)

    # Draw annotations only for images that already exist locally.
    rng = random.Random(42)
    image_ids = list(scene_graphs.keys())
    rng.shuffle(image_ids)

    drawn = 0
    for image_id in image_ids:
        if drawn >= 20:
            break

        ok = draw_scene_graph_image(
            image_id=image_id,
            sg=scene_graphs[image_id],
            image_dir=image_dir,
            out_dir=out_dir / "annotated_images",
            max_boxes=30,
        )
        if ok:
            drawn += 1

    print(f"Annotated images written: {drawn}")


if __name__ == "__main__":
    main()
