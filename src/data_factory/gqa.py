import os
os.environ["HF_HOME"] = "/scratch/eunwooim/data/rlpt/hf_cache/"

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset


# ============================================================
# Paths
# ============================================================

DEFAULT_SCENE_GRAPH_PATH = "/scratch/eunwooim/train_sceneGraphs.json"
DEFAULT_IMAGE_OUT_DIR = "/scratch/eunwooim/rlpt/visualize/images/"
DEFAULT_OUT_DIR = "/scratch/eunwooim/rlpt/src/data_factory/qcvsr_gqa_v1/"


# ============================================================
# Config
# ============================================================

STUFF_OBJECTS = {
    "water", "sky", "grass", "road", "street", "sidewalk", "floor", "ground",
    "wall", "ceiling", "field", "snow", "sand", "sea", "ocean", "river",
    "lake", "background", "room", "area", "place", "window", "building",
    "cloud", "clouds", "tree leaves", "leaves"
}

TARGET_QUOTAS = {
    "bbox_pair_spatial_compare": {
        "left": 1000,
        "right": 1000,
        "higher": 1000,
        "lower": 1000,
    },
    "bbox_pair_size_compare": {
        "larger": 1000,
        "smaller": 1000,
    },
    "count_pair_compare": {
        "more_than_yesno": 1000,
        "difference": 1000,
    },
}

TRAIN_TEMPLATES = {
    "left": [
        "Which is farther left, the {a} or the {b}?",
        "Which object is more to the left, the {a} or the {b}?",
    ],
    "right": [
        "Which is farther right, the {a} or the {b}?",
        "Which object is more to the right, the {a} or the {b}?",
    ],
    "higher": [
        "Which is higher in the image, the {a} or the {b}?",
        "Which object appears higher, the {a} or the {b}?",
    ],
    "lower": [
        "Which is lower in the image, the {a} or the {b}?",
        "Which object appears lower, the {a} or the {b}?",
    ],
    "larger": [
        "Which object occupies a larger area, the {a} or the {b}?",
        "Which looks larger in the image, the {a} or the {b}?",
    ],
    "smaller": [
        "Which object occupies a smaller area, the {a} or the {b}?",
        "Which looks smaller in the image, the {a} or the {b}?",
    ],
    "more_than_yesno": [
        "Are there more {a}s than {b}s?",
        "Does the image contain more {a}s than {b}s?",
    ],
    "difference": [
        "How many more {a}s are there than {b}s?",
        "What is the difference between the number of {a}s and {b}s?",
    ],
}

OOD_TEMPLATES = {
    "left": [
        "Between the {a} and the {b}, which one is closer to the left side of the image?",
    ],
    "right": [
        "Between the {a} and the {b}, which one is closer to the right side of the image?",
    ],
    "higher": [
        "Between the {a} and the {b}, which one is closer to the top of the image?",
    ],
    "lower": [
        "Between the {a} and the {b}, which one is closer to the bottom of the image?",
    ],
    "larger": [
        "Between the {a} and the {b}, which has the bigger bounding region?",
    ],
    "smaller": [
        "Between the {a} and the {b}, which has the smaller bounding region?",
    ],
    "more_than_yesno": [
        "Is the count of {a}s greater than the count of {b}s?",
    ],
    "difference": [
        "Subtract the number of {b}s from the number of {a}s. What is the result?",
    ],
}


# ============================================================
# Geometry utilities
# ============================================================

def active_quota_ops_for_split(
    split: str,
    quotas: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, int]]:
    active = {}

    for schema, ops in quotas.items():
        active[schema] = {}

        for op, quota in ops.items():
            if split == "eval_ood_operation":
                if allowed_for_ood_operation(schema, op):
                    active[schema][op] = quota
            else:
                if allowed_for_train_or_id(schema, op):
                    active[schema][op] = quota

    return active


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def xywh_to_xyxy(box: List[float]) -> List[float]:
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def box_area(box: List[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_center(box: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def xyxy_to_norm1000(box: List[float], width: int, height: int) -> List[int]:
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


def valid_box(box: List[float], width: int, height: int) -> bool:
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return False

    area_ratio = box_area(box) / float(width * height)
    if area_ratio < 0.005:
        return False
    if area_ratio > 0.50:
        return False

    return True


# ============================================================
# IO utilities
# ============================================================

def load_scene_graphs(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): v for k, v in raw.items()}


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    tmp_path.replace(path)


def save_image(image_obj: Any, image_id: str, image_out_dir: str) -> str:
    out_dir = Path(image_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"{image_id}.jpg"
    if not path.exists():
        image_obj.convert("RGB").save(path, quality=95)

    return str(path)


# ============================================================
# Scene graph object extraction
# ============================================================

def normalize_name(name: Any) -> str:
    return str(name).strip().lower()


def extract_objects(
    scene_graph: Dict[str, Any],
    width: int,
    height: int,
) -> List[Dict[str, Any]]:
    objects = scene_graph.get("objects", {})
    if not isinstance(objects, dict):
        return []

    extracted = []

    for object_id, obj in objects.items():
        if not isinstance(obj, dict):
            continue

        name = normalize_name(obj.get("name"))
        if not name or name in STUFF_OBJECTS:
            continue

        if not all(k in obj for k in ("x", "y", "w", "h")):
            continue

        try:
            x = float(obj["x"])
            y = float(obj["y"])
            w = float(obj["w"])
            h = float(obj["h"])
        except Exception:
            continue

        if w <= 0 or h <= 0:
            continue

        bbox_pixel = xywh_to_xyxy([x, y, w, h])
        if not valid_box(bbox_pixel, width, height):
            continue

        extracted.append(
            {
                "object_id": str(object_id),
                "name": name,
                "attributes": obj.get("attributes", []),
                "relations": obj.get("relations", []),
                "bbox_xywh_pixel": [x, y, w, h],
                "bbox_xyxy_pixel": bbox_pixel,
                "bbox_xyxy_norm1000": xyxy_to_norm1000(bbox_pixel, width, height),
            }
        )

    return extracted


# ============================================================
# Prompt and record construction
# ============================================================

def build_prompt(question: str, schema: str, evidence_template: str) -> str:
    return (
        "<image>\n"
        f"Question: {question}\n"
        "Return exactly:\n"
        f"<schema>{schema}</schema>\n"
        f"<evidence>{evidence_template}</evidence>\n"
        "<answer>...</answer>"
    )


def make_record(
    *,
    question_id: str,
    data_source: str,
    schema: str,
    question: str,
    evidence_template: str,
    image_path: str,
    image_id: str,
    target_evidence: Dict[str, Any],
    target_answer: str,
    verifier: str,
    split: str,
    operation: str,
    extra_info: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "question_id": question_id,
        "data_source": data_source,
        "prompt": [{"role": "user", "content": build_prompt(question, schema, evidence_template)}],
        "images": [image_path],
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "target_schema": schema,
                "target_evidence": target_evidence,
                "target_answer": str(target_answer).strip().lower(),
                "verifier": verifier,
                "reward_types": {
                    "schema": "exact_match",
                    "evidence": "schema_specific",
                    "answer": "exact_match_or_numeric",
                    "consistency": "verifier",
                },
            },
        },
        "ability": "query_conditioned_visual_schema_reasoning",
        "extra_info": {
            "image_id": image_id,
            "question": question,
            "split": split,
            "schema": schema,
            "operation": operation,
            **extra_info,
        },
    }


def node_pair_evidence(a: Dict[str, Any], b: Dict[str, Any], operation: str) -> Dict[str, Any]:
    return {
        "nodes": [
            {
                "id": "object_a",
                "name": a["name"],
                "bbox": a["bbox_xyxy_norm1000"],
            },
            {
                "id": "object_b",
                "name": b["name"],
                "bbox": b["bbox_xyxy_norm1000"],
            },
        ],
        "operation": operation,
        "coordinate_format": "xyxy_norm1000",
    }


def bbox_pair_template() -> str:
    return (
        '{"nodes":[{"id":"object_a","name":"...","bbox":[x1,y1,x2,y2]},'
        '{"id":"object_b","name":"...","bbox":[x1,y1,x2,y2]}],'
        '"operation":"..."}'
    )


def count_pair_template() -> str:
    return (
        '{"nodes":[{"id":"class_a","name":"...","count":0},'
        '{"id":"class_b","name":"...","count":0}],'
        '"operation":"..."}'
    )


# ============================================================
# Candidate generation
# ============================================================

def generate_spatial_candidates(
    image_id: str,
    image_path: str,
    objects: List[Dict[str, Any]],
    rng: random.Random,
    split: str,
    template_bank: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    candidates = []
    schema = "bbox_pair_spatial_compare"
    evidence_template = bbox_pair_template()

    pairs = []
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            a, b = objects[i], objects[j]
            if a["name"] == b["name"]:
                continue

            ax, ay = box_center(a["bbox_xyxy_pixel"])
            bx, by = box_center(b["bbox_xyxy_pixel"])

            if abs(ax - bx) >= 50:
                pairs.append((a, b, "left"))
                pairs.append((a, b, "right"))
            if abs(ay - by) >= 50:
                pairs.append((a, b, "higher"))
                pairs.append((a, b, "lower"))

    rng.shuffle(pairs)

    for a, b, operation in pairs:
        ax, ay = box_center(a["bbox_xyxy_pixel"])
        bx, by = box_center(b["bbox_xyxy_pixel"])

        if operation == "left":
            answer = a["name"] if ax < bx else b["name"]
            verifier = "compare_x_center_min"
        elif operation == "right":
            answer = a["name"] if ax > bx else b["name"]
            verifier = "compare_x_center_max"
        elif operation == "higher":
            answer = a["name"] if ay < by else b["name"]
            verifier = "compare_y_center_min"
        elif operation == "lower":
            answer = a["name"] if ay > by else b["name"]
            verifier = "compare_y_center_max"
        else:
            continue

        question = rng.choice(template_bank[operation]).format(a=a["name"], b=b["name"])

        target_evidence = node_pair_evidence(a, b, verifier)

        candidates.append(
            {
                "data_source": "qcvsr_gqa_bbox_pair_spatial_compare",
                "schema": schema,
                "question": question,
                "evidence_template": evidence_template,
                "image_path": image_path,
                "image_id": image_id,
                "target_evidence": target_evidence,
                "target_answer": answer,
                "verifier": verifier,
                "split": split,
                "operation": operation,
                "extra_info": {
                    "object_a_id": a["object_id"],
                    "object_b_id": b["object_id"],
                    "object_a_bbox_pixel": a["bbox_xyxy_pixel"],
                    "object_b_bbox_pixel": b["bbox_xyxy_pixel"],
                },
            }
        )

    return candidates


def generate_size_candidates(
    image_id: str,
    image_path: str,
    objects: List[Dict[str, Any]],
    rng: random.Random,
    split: str,
    template_bank: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    candidates = []
    schema = "bbox_pair_size_compare"
    evidence_template = bbox_pair_template()

    pairs = []
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            a, b = objects[i], objects[j]
            if a["name"] == b["name"]:
                continue

            area_a = box_area(a["bbox_xyxy_pixel"])
            area_b = box_area(b["bbox_xyxy_pixel"])
            ratio = max(area_a, area_b) / max(min(area_a, area_b), 1.0)

            if ratio < 1.5:
                continue

            pairs.append((a, b, "larger"))
            pairs.append((a, b, "smaller"))

    rng.shuffle(pairs)

    for a, b, operation in pairs:
        area_a = box_area(a["bbox_xyxy_pixel"])
        area_b = box_area(b["bbox_xyxy_pixel"])

        if operation == "larger":
            answer = a["name"] if area_a > area_b else b["name"]
            verifier = "compare_bbox_area_max"
        elif operation == "smaller":
            answer = a["name"] if area_a < area_b else b["name"]
            verifier = "compare_bbox_area_min"
        else:
            continue

        question = rng.choice(template_bank[operation]).format(a=a["name"], b=b["name"])
        target_evidence = node_pair_evidence(a, b, verifier)

        candidates.append(
            {
                "data_source": "qcvsr_gqa_bbox_pair_size_compare",
                "schema": schema,
                "question": question,
                "evidence_template": evidence_template,
                "image_path": image_path,
                "image_id": image_id,
                "target_evidence": target_evidence,
                "target_answer": answer,
                "verifier": verifier,
                "split": split,
                "operation": operation,
                "extra_info": {
                    "object_a_id": a["object_id"],
                    "object_b_id": b["object_id"],
                    "object_a_bbox_pixel": a["bbox_xyxy_pixel"],
                    "object_b_bbox_pixel": b["bbox_xyxy_pixel"],
                },
            }
        )

    return candidates


def generate_count_candidates(
    image_id: str,
    image_path: str,
    objects: List[Dict[str, Any]],
    rng: random.Random,
    split: str,
    template_bank: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    candidates = []
    schema = "count_pair_compare"
    evidence_template = count_pair_template()

    counts = Counter(obj["name"] for obj in objects)
    names = sorted([name for name, count in counts.items() if count >= 1 and name not in STUFF_OBJECTS])

    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if counts[a] == counts[b]:
                continue
            pairs.append((a, b, "more_than_yesno"))
            pairs.append((a, b, "difference"))

    rng.shuffle(pairs)

    for class_a, class_b, operation in pairs:
        count_a = counts[class_a]
        count_b = counts[class_b]

        if operation == "more_than_yesno":
            answer = "yes" if count_a > count_b else "no"
            verifier = "compare_count_greater_than"
        elif operation == "difference":
            answer = str(count_a - count_b)
            verifier = "count_difference"
        else:
            continue

        question = rng.choice(template_bank[operation]).format(a=class_a, b=class_b)

        target_evidence = {
            "nodes": [
                {"id": "class_a", "name": class_a, "count": count_a},
                {"id": "class_b", "name": class_b, "count": count_b},
            ],
            "operation": verifier,
        }

        candidates.append(
            {
                "data_source": "qcvsr_gqa_count_pair_compare",
                "schema": schema,
                "question": question,
                "evidence_template": evidence_template,
                "image_path": image_path,
                "image_id": image_id,
                "target_evidence": target_evidence,
                "target_answer": answer,
                "verifier": verifier,
                "split": split,
                "operation": operation,
                "extra_info": {},
            }
        )

    return candidates


# ============================================================
# Split logic
# ============================================================

def split_image_ids(image_ids: List[str], seed: int) -> Dict[str, List[str]]:
    rng = random.Random(seed)
    ids = list(image_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(0.80 * n)
    n_eval_id = int(0.07 * n)
    n_ood_template = int(0.07 * n)

    return {
        "train": ids[:n_train],
        "eval_id": ids[n_train:n_train + n_eval_id],
        "eval_ood_template": ids[n_train + n_eval_id:n_train + n_eval_id + n_ood_template],
        "eval_ood_operation": ids[n_train + n_eval_id + n_ood_template:],
    }


def get_quota_for_split(split: str, base_quotas: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    if split == "train":
        return base_quotas

    factor = 0.10
    out = {}
    for schema, ops in base_quotas.items():
        out[schema] = {op: max(50, int(v * factor)) for op, v in ops.items()}
    return out


def allowed_for_ood_operation(schema: str, operation: str) -> bool:
    # Example OOD operation split:
    # hold out vertical spatial operations from train-like generation.
    if schema == "bbox_pair_spatial_compare":
        return operation in {"higher", "lower"}
    if schema == "bbox_pair_size_compare":
        return operation in {"smaller"}
    if schema == "count_pair_compare":
        return operation in {"difference"}
    return True


def allowed_for_train_or_id(schema: str, operation: str) -> bool:
    if schema == "bbox_pair_spatial_compare":
        return operation in {"left", "right"}
    if schema == "bbox_pair_size_compare":
        return operation in {"larger"}
    if schema == "count_pair_compare":
        return operation in {"more_than_yesno"}
    return True


# ============================================================
# Main generation
# ============================================================

def generate_dataset(
    scene_graph_path: str = DEFAULT_SCENE_GRAPH_PATH,
    image_out_dir: str = DEFAULT_IMAGE_OUT_DIR,
    out_dir: str = DEFAULT_OUT_DIR,
    dataset_repo: str = "lmms-lab/GQA",
    image_config: str = "train_balanced_images",
    seed: int = 42,
    max_images: Optional[int] = None,
    checkpoint_every: int = 1000,
) -> None:
    rng = random.Random(seed)
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading scene graph: {scene_graph_path}")
    scene_graphs = load_scene_graphs(scene_graph_path)
    print(f"Loaded scene graphs: {len(scene_graphs)}")

    print(f"Loading images: {dataset_repo}/{image_config}")
    images_raw = load_dataset(dataset_repo, image_config)
    images_ds = images_raw["train"] if hasattr(images_raw, "keys") and "train" in images_raw else images_raw

    id_col = "id" if "id" in images_ds.column_names else "imageId"
    image_id_to_idx = {str(img_id): idx for idx, img_id in enumerate(images_ds[id_col])}

    image_ids = sorted(set(scene_graphs.keys()) & set(image_id_to_idx.keys()))
    if max_images is not None:
        rng.shuffle(image_ids)
        image_ids = image_ids[:max_images]

    split_to_ids = split_image_ids(image_ids, seed=seed)

    all_outputs = {}
    skip_stats = Counter()

    for split, ids in split_to_ids.items():
        print(f"\nGenerating split: {split}, images={len(ids)}")

        quotas_raw = get_quota_for_split(split, TARGET_QUOTAS)
        quotas = active_quota_ops_for_split(split, quotas_raw)
        quota_counts = {schema: Counter() for schema in quotas.keys()}
        records = []

        template_bank = OOD_TEMPLATES if split == "eval_ood_template" else TRAIN_TEMPLATES

        for image_id in ids:
            if all(
                quota_counts[schema][op] >= quotas[schema][op]
                for schema in quotas
                for op in quotas[schema]
            ):
                break

            try:
                image_row = images_ds[image_id_to_idx[image_id]]
                image_obj = image_row.get("image")
                if image_obj is None:
                    skip_stats[f"{split}:missing_image"] += 1
                    continue

                width, height = image_obj.size
                image_path = save_image(image_obj, image_id, image_out_dir)

                objects = extract_objects(scene_graphs[image_id], width, height)
                if len(objects) < 2:
                    skip_stats[f"{split}:too_few_objects"] += 1
                    continue

                candidates = []
                candidates.extend(generate_spatial_candidates(image_id, image_path, objects, rng, split, template_bank))
                candidates.extend(generate_size_candidates(image_id, image_path, objects, rng, split, template_bank))
                candidates.extend(generate_count_candidates(image_id, image_path, objects, rng, split, template_bank))

                rng.shuffle(candidates)

                for cand in candidates:
                    schema = cand["schema"]
                    op = cand["operation"]

                    if schema not in quotas or op not in quotas[schema]:
                        continue

                    # if split == "eval_ood_operation":
                    #     if not allowed_for_ood_operation(schema, op):
                    #         continue
                    # else:
                    #     if not allowed_for_train_or_id(schema, op):
                    #         continue

                    if quota_counts[schema][op] >= quotas[schema][op]:
                        continue

                    qid = f"qcvsr_gqa_v1_{split}_{len(records):08d}"

                    record = make_record(
                        question_id=qid,
                        data_source=cand["data_source"],
                        schema=cand["schema"],
                        question=cand["question"],
                        evidence_template=cand["evidence_template"],
                        image_path=cand["image_path"],
                        image_id=cand["image_id"],
                        target_evidence=cand["target_evidence"],
                        target_answer=cand["target_answer"],
                        verifier=cand["verifier"],
                        split=split,
                        operation=cand["operation"],
                        extra_info={
                            "question_id": qid,
                            **cand["extra_info"],
                        },
                    )

                    records.append(record)
                    quota_counts[schema][op] += 1

                    if checkpoint_every > 0 and len(records) % checkpoint_every == 0:
                        latest_path = out_dir_path / f"latest_{split}.jsonl"
                        write_jsonl(records, latest_path)
                        print(f"[{split}] checkpoint records={len(records)} quota_counts={quota_counts}")

            except Exception as e:
                skip_stats[f"{split}:exception:{type(e).__name__}"] += 1
                continue

        records = sorted(
            records,
            key=lambda r: (
                r["data_source"],
                r["extra_info"]["operation"],
                r["images"][0],
                r["question_id"],
            ),
        )

        # Reassign IDs after sorting.
        for idx, record in enumerate(records):
            qid = f"qcvsr_gqa_v1_{split}_{idx:08d}"
            record["question_id"] = qid
            record["extra_info"]["question_id"] = qid

        out_path = out_dir_path / f"{split}.jsonl"
        write_jsonl(records, out_path)
        all_outputs[split] = records

        print(f"[{split}] written={len(records)} path={out_path}")
        print(f"[{split}] quota_counts:")
        for schema, counter in quota_counts.items():
            print(f"  {schema}: {dict(counter)}")

    # Combined train/eval convenience files.
    train_path = out_dir_path / "train.jsonl"
    eval_path = out_dir_path / "eval_all.jsonl"

    write_jsonl(all_outputs.get("train", []), train_path)

    eval_records = []
    for split in ["eval_id", "eval_ood_template", "eval_ood_operation"]:
        eval_records.extend(all_outputs.get(split, []))
    write_jsonl(eval_records, eval_path)

    print("\nDone.")
    print(f"Output dir: {out_dir_path}")
    print(f"Train: {train_path}")
    print(f"Eval all: {eval_path}")
    print("\nSkip stats:")
    for k, v in skip_stats.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    generate_dataset(
        scene_graph_path=DEFAULT_SCENE_GRAPH_PATH,
        image_out_dir=DEFAULT_IMAGE_OUT_DIR,
        out_dir=DEFAULT_OUT_DIR,
        seed=42,
        max_images=None,
        checkpoint_every=1000,
    )
