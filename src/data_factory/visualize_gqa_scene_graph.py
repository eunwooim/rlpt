import os
os.environ["HF_HOME"] = "/scratch/eunwooim/data/rlpt/hf_cache/"

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from PIL import ImageDraw


SCENE_GRAPH_PATH = "/scratch/eunwooim/train_sceneGraphs.json"
OUT_DIR = "/scratch/eunwooim/rlpt/visualize/scene_graph_boxes"

DATASET_REPO = "lmms-lab/GQA"
IMAGE_CONFIG = "train_balanced_images"


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


def box_area_xywh(x: float, y: float, w: float, h: float) -> float:
    return max(0.0, w) * max(0.0, h)


def classify_box_problem(
    x: float,
    y: float,
    w: float,
    h: float,
    image_w: int,
    image_h: int,
) -> List[str]:
    problems = []

    if w <= 0:
        problems.append("non_positive_width")
    if h <= 0:
        problems.append("non_positive_height")

    if x < 0:
        problems.append("x_negative")
    if y < 0:
        problems.append("y_negative")
    if x + w > image_w:
        problems.append("x2_out_of_bounds")
    if y + h > image_h:
        problems.append("y2_out_of_bounds")

    if image_w > 0 and image_h > 0 and w > 0 and h > 0:
        area_ratio = box_area_xywh(x, y, w, h) / float(image_w * image_h)

        if area_ratio < 0.005:
            problems.append("too_small_for_generator")
        if area_ratio > 0.50:
            problems.append("too_large_for_generator")

    return problems


def get_image_dataset():
    images_raw = load_dataset(DATASET_REPO, IMAGE_CONFIG)
    images_ds = images_raw["train"] if hasattr(images_raw, "keys") and "train" in images_raw else images_raw

    id_col = "id" if "id" in images_ds.column_names else "imageId"
    image_id_to_idx = {str(img_id): idx for idx, img_id in enumerate(images_ds[id_col])}

    return images_ds, image_id_to_idx


def draw_scene_graph(
    image_id: str,
    scene_graphs: Dict[str, Any],
    images_ds,
    image_id_to_idx: Dict[str, int],
    out_dir: str,
    draw_filtered_only: bool = False,
    max_boxes: Optional[int] = None,
) -> None:
    if image_id not in scene_graphs:
        raise KeyError(f"image_id not found in scene graph: {image_id}")

    if image_id not in image_id_to_idx:
        raise KeyError(f"image_id not found in HF image dataset: {image_id}")

    row = images_ds[image_id_to_idx[image_id]]
    image = row["image"].convert("RGB")

    image_w, image_h = image.size
    sg = scene_graphs[image_id]
    objects = sg.get("objects", {})

    draw = ImageDraw.Draw(image)

    items = []
    for object_id, obj in objects.items():
        if not isinstance(obj, dict):
            continue

        name = str(obj.get("name", "")).strip().lower()

        if not all(k in obj for k in ("x", "y", "w", "h")):
            continue

        try:
            x = float(obj["x"])
            y = float(obj["y"])
            w = float(obj["w"])
            h = float(obj["h"])
        except Exception:
            continue

        problems = classify_box_problem(x, y, w, h, image_w, image_h)

        generator_keeps = (
            name
            and name not in STUFF_OBJECTS
            and not problems
        )

        if draw_filtered_only and not generator_keeps:
            continue

        area = box_area_xywh(x, y, w, h)
        items.append(
            {
                "area": area,
                "object_id": str(object_id),
                "name": name,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "problems": problems,
                "generator_keeps": generator_keeps,
            }
        )

    # Draw large boxes first so smaller boxes remain visible.
    items = sorted(items, key=lambda d: d["area"], reverse=True)

    if max_boxes is not None:
        items = items[:max_boxes]

    for item in items:
        x = item["x"]
        y = item["y"]
        w = item["w"]
        h = item["h"]

        x1, y1 = x, y
        x2, y2 = x + w, y + h

        if item["generator_keeps"]:
            outline = "lime"
            fill = "lime"
            status = "KEEP"
        else:
            outline = "red"
            fill = "red"
            status = "DROP"

        label = f'{status} {item["object_id"]}: {item["name"]}'

        if item["problems"]:
            label += " | " + ",".join(item["problems"])

        draw.rectangle([x1, y1, x2, y2], outline=outline, width=3)

        text_x = max(0, int(x1))
        text_y = max(0, int(y1) - 16)
        text_w = max(80, 7 * len(label))

        draw.rectangle(
            [text_x, text_y, text_x + text_w, text_y + 15],
            fill=fill,
        )
        draw.text((text_x + 2, text_y + 1), label, fill="black")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    suffix = "filtered" if draw_filtered_only else "all"
    save_path = out_path / f"{image_id}_{suffix}.jpg"
    image.save(save_path, quality=95)

    print(f"Wrote: {save_path}")
    print(f"Image size: {image_w} x {image_h}")
    print(f"Objects drawn: {len(items)}")
    print()
    print("Objects:")
    for item in items:
        print(
            f'{item["object_id"]:>8} '
            f'{item["name"]:<20} '
            f'keep={item["generator_keeps"]} '
            f'bbox=[{item["x"]:.1f},{item["y"]:.1f},{item["w"]:.1f},{item["h"]:.1f}] '
            f'problems={item["problems"]}'
        )


def main():
    scene_graphs = load_scene_graphs(SCENE_GRAPH_PATH)
    images_ds, image_id_to_idx = get_image_dataset()

    # Change this to inspect a specific image.
    image_id = "2414770"

    draw_scene_graph(
        image_id=image_id,
        scene_graphs=scene_graphs,
        images_ds=images_ds,
        image_id_to_idx=image_id_to_idx,
        out_dir=OUT_DIR,
        draw_filtered_only=False,
        max_boxes=None,
    )

    draw_scene_graph(
        image_id=image_id,
        scene_graphs=scene_graphs,
        images_ds=images_ds,
        image_id_to_idx=image_id_to_idx,
        out_dir=OUT_DIR,
        draw_filtered_only=True,
        max_boxes=None,
    )


if __name__ == "__main__":
    main()