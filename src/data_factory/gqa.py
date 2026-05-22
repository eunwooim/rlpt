import os
os.environ["HF_HOME"] = "/mnt/data2/eunwooim/data/rlpt/hf_cache/"

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset


IMAGE_OUTPUT_DIR = Path("/mnt/data2/eunwooim/rlpt/gqa/images/")
LATEST_JSONL_PATH = Path("/mnt/data1/eunwooim/rlpt/archive/gqa_v1_latest.jsonl")
FINAL_JSONL_PATH = Path("/mnt/data1/eunwooim/rlpt/src/data_factory/gqa_v1.jsonl")
DEFAULT_SCENE_GRAPH_PATH = "/mnt/data2/eunwooim/rlpt/gqa/train_sceneGraphs.json" # Downloaded from: https://cs.stanford.edu/people/dorarad/gqa/download.html?utm_source=chatgpt.com


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def xywh_to_xyxy(box: List[float]) -> List[float]:
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def xyxy_pixel_to_norm1000(box: List[float], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = box

    x1 = clamp(x1, 0, width)
    x2 = clamp(x2, 0, width)
    y1 = clamp(y1, 0, height)
    y2 = clamp(y2, 0, height)

    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    return [
        int(round(clamp(x1 / width * 1000, 0, 1000))),
        int(round(clamp(y1 / height * 1000, 0, 1000))),
        int(round(clamp(x2 / width * 1000, 0, 1000))),
        int(round(clamp(y2 / height * 1000, 0, 1000))),
    ]


def is_valid_norm_box(box: List[int]) -> bool:
    if len(box) != 4:
        return False
    x1, y1, x2, y2 = box
    return 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000


def normalize_answer(ans: Any) -> str:
    return str(ans).strip().lower()


def should_keep_answer(raw_answer: Any, max_answer_words: int = 4) -> bool:
    ans = normalize_answer(raw_answer)
    return bool(ans) and len(ans.split()) <= max_answer_words


def load_scene_graphs(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        scene_graphs = json.load(f)
    return {str(k): v for k, v in scene_graphs.items()}


def get_annotation_values(example: Dict[str, Any], key: str) -> List[str]:
    annotations = example.get("annotations", {})
    if not isinstance(annotations, dict):
        return []

    items = annotations.get(key, [])
    if not isinstance(items, list):
        return []

    values = []
    for item in items:
        if isinstance(item, dict) and item.get("value") is not None:
            values.append(str(item["value"]))

    # preserve order, remove duplicates
    seen = set()
    out = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def lookup_scene_object(
    scene_graphs: Dict[str, Any],
    image_id: str,
    object_id: str,
) -> Optional[Dict[str, Any]]:
    sg = scene_graphs.get(str(image_id))
    if not isinstance(sg, dict):
        return None

    objects = sg.get("objects", {})
    if not isinstance(objects, dict):
        return None

    obj = objects.get(str(object_id))
    if not isinstance(obj, dict):
        return None

    if not all(k in obj for k in ("x", "y", "w", "h")):
        return None

    try:
        x, y, w, h = float(obj["x"]), float(obj["y"]), float(obj["w"]), float(obj["h"])
    except Exception:
        return None

    if w <= 0 or h <= 0:
        return None

    return {
        "object_id": str(object_id),
        "name": obj.get("name"),
        "attributes": obj.get("attributes", []),
        "relations": obj.get("relations", []),
        "box_xywh_pixel": [x, y, w, h],
        "box_xyxy_pixel": xywh_to_xyxy([x, y, w, h]),
        "raw_object": obj,
    }


def extract_single_evidence_object(
    example: Dict[str, Any],
    scene_graphs: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Extract exactly one bbox-backed evidence object.

    Priority:
      1. annotations["answer"]
      2. annotations["fullAnswer"]
      3. annotations["question"]

    For v1, we keep only examples where one of these fields maps to exactly
    one valid scene-graph object. Multi-object evidence is skipped for now.
    """
    image_id = example.get("imageId")
    if image_id is None:
        return None, "missing_image_id"

    for key in ("answer", "fullAnswer", "question"):
        object_ids = get_annotation_values(example, key)
        if not object_ids:
            continue

        candidates = []
        for oid in object_ids:
            obj = lookup_scene_object(scene_graphs, str(image_id), oid)
            if obj is not None:
                candidates.append(obj)

        if len(candidates) == 1:
            return candidates[0], f"ok_from_{key}"
        if len(candidates) > 1:
            return None, f"multiple_objects_from_{key}"

    return None, "no_single_evidence_object"


def build_prompt(question: str) -> str:
    return (
        "<image>\n"
        f"Question: {question}\n\n"
        'Output: <evidence>{"bbox":[x1,y1,x2,y2]}</evidence><answer>...</answer>\n'
        "bbox uses normalized 0-1000 xyxy coordinates."
    )


def save_image(image_obj: Any, image_id: str) -> str:
    IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = IMAGE_OUTPUT_DIR / f"{image_id}.jpg"

    if not path.exists():
        image_obj.convert("RGB").save(path, quality=95)

    return str(path)


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tmp_path.replace(path)


def print_sample(record: Dict[str, Any], prefix: str) -> None:
    gt = record["reward_model"]["ground_truth"]

    print(f"\n[{prefix}]")
    print("data_source:", record["data_source"])
    print("image:", record["images"][0])
    print("question:", record["extra_info"]["question"])
    print("target_answer:", gt["target_answer"])
    print("bbox_norm1000:", gt["target_evidence"]["bbox_xyxy_norm1000"])
    print("bbox_pixel:", gt["target_evidence"]["bbox_xyxy_pixel"])
    print("evidence_source:", record["extra_info"]["evidence_source"])
    print("prompt:")
    print(record["prompt"][0]["content"])
    print("-" * 100)


def build_record(
    example: Dict[str, Any],
    idx: int,
    image_obj: Any,
    image_path: str,
    evidence_obj: Dict[str, Any],
    evidence_source: str,
) -> Optional[Dict[str, Any]]:
    question = example.get("question")
    raw_answer = example.get("answer")
    image_id = example.get("imageId")
    types_meta = example.get("types", {})

    if not question or raw_answer is None or image_id is None:
        return None

    width, height = image_obj.size
    bbox_pixel = evidence_obj["box_xyxy_pixel"]
    bbox_norm1000 = xyxy_pixel_to_norm1000(bbox_pixel, width, height)

    if not is_valid_norm_box(bbox_norm1000):
        return None

    structural = types_meta.get("structural", "unknown")
    semantic = types_meta.get("semantic", "unknown")
    detailed = types_meta.get("detailed", "unknown")
    composite_key = f"{structural}_{semantic}_{detailed}"

    target_answer = normalize_answer(raw_answer)

    return {
        "data_source": f"gqa_dual_{composite_key}",
        "prompt": [{"role": "user", "content": build_prompt(question)}],
        "images": [image_path],
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "target_evidence": {
                    "bbox_xyxy_norm1000": bbox_norm1000,
                    "bbox_xyxy_pixel": bbox_pixel,
                    "bbox_format": "xyxy_norm1000",
                    "image_width": width,
                    "image_height": height,
                },
                "target_answer": target_answer,
                "reward_types": {
                    "evidence": "iou",
                    "answer": "exact_match",
                },
            },
        },
        "ability": "dual_evidence_answer",
        "extra_info": {
            "gqa_id": example.get("id", str(idx)),
            "image_id": image_id,
            "image_path": image_path,
            "question": question,
            "raw_answer": str(raw_answer),
            "normalized_answer": target_answer,
            "composite_key": composite_key,
            "structural_type": structural,
            "semantic_type": semantic,
            "detailed_type": detailed,
            "evidence_source": evidence_source,
            "evidence_object_id": evidence_obj["object_id"],
            "evidence_object_name": evidence_obj.get("name"),
            "bbox_xywh_pixel_raw": evidence_obj["box_xywh_pixel"],
        },
    }


def process_gqa_dual_reward(
    dataset_repo: str = "lmms-lab/GQA",
    instruction_config: str = "train_balanced_instructions",
    image_config: str = "train_balanced_images",
    scene_graph_path: str = DEFAULT_SCENE_GRAPH_PATH,
    samples_per_composite_type: int = 200,
    seed: int = 42,
    checkpoint_every: int = 1000,
    preview_every: int = 1000,
    max_answer_words: int = 4,
) -> None:
    IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Image dir:", IMAGE_OUTPUT_DIR)
    print("Latest JSONL:", LATEST_JSONL_PATH)
    print("Final JSONL:", FINAL_JSONL_PATH)

    print(f"\nLoading instructions: {dataset_repo}/{instruction_config}")
    instructions_raw = load_dataset(dataset_repo, instruction_config)
    instructions_ds = instructions_raw["train"] if hasattr(instructions_raw, "keys") and "train" in instructions_raw else instructions_raw
    instructions_ds = instructions_ds.shuffle(seed=seed)

    print(f"Loading images: {dataset_repo}/{image_config}")
    images_raw = load_dataset(dataset_repo, image_config)
    images_ds = images_raw["train"] if hasattr(images_raw, "keys") and "train" in images_raw else images_raw

    id_col = "id" if "id" in images_ds.column_names else "imageId"
    image_id_to_idx = {str(img_id): idx for idx, img_id in enumerate(images_ds[id_col])}
    print(f"Indexed {len(image_id_to_idx)} images")

    print(f"Loading scene graph: {scene_graph_path}")
    scene_graphs = load_scene_graphs(scene_graph_path)
    print(f"Loaded {len(scene_graphs)} scene-graph entries")

    records = []
    skip_stats = Counter()
    composite_counts = defaultdict(int)

    for idx, example in enumerate(instructions_ds):
        types_meta = example.get("types", {})
        if not isinstance(types_meta, dict):
            skip_stats["missing_types"] += 1
            continue

        structural = types_meta.get("structural")
        semantic = types_meta.get("semantic")
        detailed = types_meta.get("detailed")

        if not (structural and semantic and detailed):
            skip_stats["incomplete_types"] += 1
            continue

        if structural not in {"query", "choose", "verify"}:
            skip_stats[f"unsupported_structural:{structural}"] += 1
            continue

        composite_key = f"{structural}_{semantic}_{detailed}"
        if composite_counts.get(composite_key, 0) >= samples_per_composite_type:
            skip_stats["quota_full"] += 1
            continue

        question = example.get("question")
        raw_answer = example.get("answer")
        image_id = example.get("imageId")

        if not question:
            skip_stats["missing_question"] += 1
            continue
        if raw_answer is None:
            skip_stats["missing_answer"] += 1
            continue
        if not should_keep_answer(raw_answer, max_answer_words=max_answer_words):
            skip_stats["answer_too_long_or_empty"] += 1
            continue
        if image_id is None:
            skip_stats["missing_image_id"] += 1
            continue

        image_idx = image_id_to_idx.get(str(image_id))
        if image_idx is None:
            skip_stats["image_not_found"] += 1
            continue

        evidence_obj, evidence_status = extract_single_evidence_object(example, scene_graphs)
        if evidence_obj is None:
            skip_stats[f"evidence_fail:{evidence_status}"] += 1
            continue

        image_row = images_ds[image_idx]
        image_obj = image_row.get("image")
        if image_obj is None:
            skip_stats["missing_image_object"] += 1
            continue

        image_path = save_image(image_obj, str(image_id))
        record = build_record(
            example=example,
            idx=idx,
            image_obj=image_obj,
            image_path=image_path,
            evidence_obj=evidence_obj,
            evidence_source=evidence_status,
        )

        if record is None:
            skip_stats["record_build_failed"] += 1
            continue

        records.append(record)
        composite_counts[composite_key] += 1

        n = len(records)
        if n == 1:
            print_sample(record, "FIRST VALID RECORD")

        if preview_every > 0 and n % preview_every == 0:
            print_sample(record, f"VALID RECORD {n}")

        if checkpoint_every > 0 and n % checkpoint_every == 0:
            write_jsonl(records, LATEST_JSONL_PATH)
            print(f"\n[checkpoint] seen={idx + 1}, written={n}, latest={LATEST_JSONL_PATH}")
            print("[top skip reasons]")
            for k, v in skip_stats.most_common(12):
                print(f"  {k}: {v}")

    records = sorted(records, key=lambda r: r["images"][0])
    for q_idx, record in enumerate(records):
        question_id = f"gqa_dual_v1_{q_idx:08d}"
        record["question_id"] = question_id
        record["extra_info"]["question_id"] = question_id
    write_jsonl(records, LATEST_JSONL_PATH)
    write_jsonl(records, FINAL_JSONL_PATH)

    print("\nDone.")
    print("Written:", len(records))
    print("Latest:", LATEST_JSONL_PATH)
    print("Final:", FINAL_JSONL_PATH)
    print("Images:", IMAGE_OUTPUT_DIR)

    print("\nComposite counts:")
    for k, v in sorted(composite_counts.items()):
        print(f"  {k}: {v}")

    print("\nSkip stats:")
    for k, v in skip_stats.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    process_gqa_dual_reward(
        scene_graph_path=DEFAULT_SCENE_GRAPH_PATH,
        samples_per_composite_type=200,
        checkpoint_every=10000,
        preview_every=1000,
        max_answer_words=4,
    )
