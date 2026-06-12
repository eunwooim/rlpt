import os
os.environ.setdefault("HF_HOME", "/scratch/sghos104/rlpt/data/hf_cache/")

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset


# ============================================================
# Paths
# ============================================================

DEFAULT_SCENE_GRAPH_PATH = "/scratch/sghos104/rlpt/data/scene_graphs/train_sceneGraph.json"
DEFAULT_IMAGE_OUT_DIR = "/scratch/sghos104/rlpt/data/images/"
DEFAULT_OUT_DIR = "/scratch/sghos104/rlpt/data/qcvsr_gqa_v1/"


# ============================================================
# Config
# ============================================================

STUFF_OBJECTS = {
    "water", "sky", "grass", "road", "street", "sidewalk", "floor", "ground",
    "wall", "ceiling", "field", "snow", "sand", "sea", "ocean", "river",
    "lake", "background", "room", "area", "place", "window", "building",
    "cloud", "clouds", "tree leaves", "leaves"
}

# Irregular singular -> plural map for object names used in count templates.
IRREGULAR_PLURALS = {
    "man": "men", "woman": "women", "child": "children",
    "person": "people", "foot": "feet", "tooth": "teeth",
    "mouse": "mice", "goose": "geese", "ox": "oxen",
}

# Names that should not be pluralized (already plural, invariant, or
# scene-graph annotations that are themselves the plural form).
INVARIANT_PLURALS = {
    "sheep", "deer", "fish", "moose", "salmon", "trout", "aircraft",
    "series", "species", "men", "women", "children", "people",
    "feet", "teeth", "mice", "geese", "oxen",
}


def pluralize(name: str) -> str:
    """Pluralize the last word of an English noun phrase.

    Handles irregular singulars, invariant nouns, and names already in
    plural form (anything ending in 's'). Falls through to standard
    suffix rules otherwise.
    """
    parts = name.split()
    if not parts:
        return name
    last = parts[-1].lower()

    if last in IRREGULAR_PLURALS:
        parts[-1] = IRREGULAR_PLURALS[last]
    elif last in INVARIANT_PLURALS:
        return name
    elif last.endswith(("ss", "x", "z", "ch", "sh")):
        # Sibilant endings take 'es': glass -> glasses, box -> boxes.
        parts[-1] = last + "es"
    elif last.endswith("s"):
        # Treat any other name ending in 's' as already plural. Avoids
        # mangling scene-graph annotations like "pants" or "glasses".
        return name
    elif last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
        parts[-1] = last[:-1] + "ies"
    elif last.endswith("fe"):
        parts[-1] = last[:-2] + "ves"
    elif last.endswith("f") and len(last) > 1 and last[-2] not in "aeiou":
        parts[-1] = last[:-1] + "ves"
    else:
        parts[-1] = last + "s"
    return " ".join(parts)

# Classes that tend to appear many times per image (people, worn clothing,
# small repeated items). GQA scene graphs annotate only a *subset* of instances,
# so for these classes "exactly N annotated" does NOT mean N visible -- e.g. a
# street scene annotates 2 "pants" but a dozen people wear pants. Frame-relative
# instance selection ("the pants on the left") is then ambiguous, so these
# classes are excluded from schemas that localize one instance by position.
CROWDABLE_CLASSES = {
    "person", "people", "man", "men", "woman", "women", "boy", "girl",
    "child", "children", "kid", "kids", "guy", "lady", "gentleman",
    "pedestrian", "crowd", "spectator", "player", "passenger",
    "pants", "jeans", "trousers", "shirt", "t-shirt", "jacket", "coat",
    "shorts", "dress", "skirt", "sweater", "hoodie", "jersey", "uniform",
    "tie", "shoe", "shoes", "sneaker", "sneakers", "boot", "boots",
    "sock", "socks", "glove", "gloves", "sandal", "sandals",
}


def is_crowdable(name: str) -> bool:
    return name in CROWDABLE_CLASSES


# Color attribute vocabulary (for the spatial-selected attribute schema).
COLOR_ATTRIBUTES = {
    "white", "black", "green", "blue", "brown", "red", "gray", "grey",
    "yellow", "orange", "pink", "purple", "silver", "gold", "tan", "beige",
}

# Relation predicates used by the compositional schemas. Each has a SEEN set
# (train / eval_id / eval_ood_template) and an UNSEEN set held out for the
# eval_ood_operation split, so that split tests generalization to new predicates.
COUNT_RELATIONS_SEEN = {"on", "in", "near", "under", "above", "below", "on top of"}
COUNT_RELATIONS_UNSEEN = {"behind", "in front of", "next to", "beside", "inside"}

TARGET_RELATIONS_SEEN = {"wearing", "holding", "on"}
TARGET_RELATIONS_UNSEEN = {"riding", "carrying", "sitting on", "eating"}

TARGET_QUOTAS = {
    "bbox_pair_spatial_compare": {
        "left": 1500,
        "right": 1500,
        "higher": 1500,
        "lower": 1500,
    },
    "bbox_pair_size_compare": {
        "larger": 1500,
        "smaller": 1500,
    },
    # NOTE: count_pair_compare ("more bags than people?") is intentionally
    # excluded -- its evidence is bare counts with no bounding boxes (no visual
    # grounding) and the counts are unreliable under incomplete annotation.
    # Box-grounded counting is covered by count_relation instead.
    "attr_select_spatial": {
        "color_left": 1500,
        "color_right": 1500,
        "color_higher": 1500,
        "color_lower": 1500,
    },
    "count_relation": {
        "count_relation": 1500,
    },
    "relation_target": {
        "relation_target": 1500,
    },
}

# Cap on how many records a single image can contribute to any one (schema, op)
# bucket in a split. Prevents object-dense scenes from dominating the dataset
# and forces the quota loop to pull from many more unique images.
MAX_RECORDS_PER_IMAGE_PER_OP = 2

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
        "Are there more {a_plural} than {b_plural}?",
        "Does the image contain more {a_plural} than {b_plural}?",
    ],
    "difference": [
        "How many more {a_plural} are there than {b_plural}?",
        "What is the difference between the number of {a_plural} and {b_plural}?",
    ],
    "color_left": [
        "What color is the {name} on the left?",
        "What is the color of the leftmost {name}?",
    ],
    "color_right": [
        "What color is the {name} on the right?",
        "What is the color of the rightmost {name}?",
    ],
    "color_higher": [
        "What color is the upper {name}?",
        "What is the color of the {name} nearer the top of the image?",
    ],
    "color_lower": [
        "What color is the lower {name}?",
        "What is the color of the {name} nearer the bottom of the image?",
    ],
    "count_relation": [
        "How many {name_plural} are {rel} the {ref}?",
        "Count the {name_plural} that are {rel} the {ref}.",
    ],
    "relation_target": [
        "What is the {name} {rel}?",
        "What is the {name} {rel}? Answer with the object name.",
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
        "Is the count of {a_plural} greater than the count of {b_plural}?",
    ],
    "difference": [
        "Subtract the number of {b_plural} from the number of {a_plural}. What is the result?",
    ],
    "color_left": [
        "Of the two {name} objects, what color is the one positioned farther to the left?",
    ],
    "color_right": [
        "Of the two {name} objects, what color is the one positioned farther to the right?",
    ],
    "color_higher": [
        "Of the two {name} objects, what color is the one positioned closer to the top?",
    ],
    "color_lower": [
        "Of the two {name} objects, what color is the one positioned closer to the bottom?",
    ],
    "count_relation": [
        "What is the number of {name_plural} located {rel} the {ref}?",
    ],
    "relation_target": [
        "Identify the object that the {name} is {rel}.",
    ],
}


# ============================================================
# Geometry utilities
# ============================================================

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

        bbox_norm = xyxy_to_norm1000(bbox_pixel, width, height)
        # Drop boxes that collapse to zero width/height after rounding to
        # 0-1000 — they break bbox order and the area verifier.
        if bbox_norm[0] >= bbox_norm[2] or bbox_norm[1] >= bbox_norm[3]:
            continue

        extracted.append(
            {
                "object_id": str(object_id),
                "name": name,
                "attributes": obj.get("attributes", []),
                "relations": obj.get("relations", []),
                "bbox_xywh_pixel": [x, y, w, h],
                "bbox_xyxy_pixel": bbox_pixel,
                "bbox_xyxy_norm1000": bbox_norm,
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


def group_objects_by_class(objects: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group object instances by lowercase class name (preserving order)."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for obj in objects:
        groups.setdefault(obj["name"], []).append(obj)
    return groups


def unique_classes(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the single instance of every class that appears exactly once.

    A class with one instance is an *unambiguous referent*: "the cup" points at
    exactly one object, so spatial/size comparisons about it are well posed.
    Crowdable classes are excluded -- a lone annotated "man" may still be one of
    many in the image (incomplete annotation), so "the man" is not reliable.
    """
    groups = group_objects_by_class(objects)
    singletons = [
        insts[0] for name, insts in groups.items()
        if len(insts) == 1 and not is_crowdable(name)
    ]
    return sorted(singletons, key=lambda o: o["name"])


def node_pair_evidence(a: Dict[str, Any], b: Dict[str, Any], operation: str) -> Dict[str, Any]:
    """Pairwise evidence with one bbox per (unique) object."""
    return {
        "nodes": [
            {"id": "object_a", "name": a["name"], "bbox": a["bbox_xyxy_norm1000"]},
            {"id": "object_b", "name": b["name"], "bbox": b["bbox_xyxy_norm1000"]},
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


def attr_select_template() -> str:
    return (
        '{"nodes":[{"id":"inst_a","name":"...","bbox":[x1,y1,x2,y2]},'
        '{"id":"inst_b","name":"...","bbox":[x1,y1,x2,y2]}],'
        '"operation":"...","selected_id":"...","answer_attribute":"color"}'
    )


def count_relation_template() -> str:
    return (
        '{"nodes":[{"id":"reference","name":"...","bbox":[x1,y1,x2,y2],"role":"reference"},'
        '{"id":"subject_0","name":"...","bbox":[x1,y1,x2,y2],"role":"subject"}],'
        '"operation":"count_related_subjects","relation":"...","count":0}'
    )


def relation_target_template() -> str:
    return (
        '{"nodes":[{"id":"subject","name":"...","bbox":[x1,y1,x2,y2],"role":"subject"},'
        '{"id":"target","name":"...","bbox":[x1,y1,x2,y2],"role":"target"}],'
        '"operation":"relation_traverse","relation":"..."}'
    )


def single_color(obj: Dict[str, Any]) -> Optional[str]:
    """Return the object's sole color attribute, or None if zero/ambiguous."""
    colors = [a for a in obj.get("attributes", []) if normalize_name(a) in COLOR_ATTRIBUTES]
    colors = sorted(set(normalize_name(c) for c in colors))
    return colors[0] if len(colors) == 1 else None


def objects_by_id(objects: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {o["object_id"]: o for o in objects}


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
    """Spatial comparison between two *uniquely-referrable* objects.

    Only classes appearing exactly once are eligible, so "the {a}" / "the {b}"
    each denote a single object and the comparison has a determinate answer.
    """
    candidates = []
    schema = "bbox_pair_spatial_compare"
    evidence_template = bbox_pair_template()

    objs = unique_classes(objects)

    pairs = []
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            a, b = objs[i], objs[j]
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
    """Size comparison between two *uniquely-referrable* objects."""
    candidates = []
    schema = "bbox_pair_size_compare"
    evidence_template = bbox_pair_template()

    objs = unique_classes(objects)

    pairs = []
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            a, b = objs[i], objs[j]
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


def generate_attr_select_candidates(
    image_id: str,
    image_path: str,
    objects: List[Dict[str, Any]],
    rng: random.Random,
    split: str,
    template_bank: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """Compositional: spatial selection -> attribute readout.

    Requires a class with exactly two instances that (a) are clearly separated
    on the queried axis and (b) carry *different* single colors -- so the
    spatial clause is what makes the answer unique. Multi-instance is the point.
    """
    candidates = []
    schema = "attr_select_spatial"
    evidence_template = attr_select_template()

    groups = group_objects_by_class(objects)

    for name, insts in sorted(groups.items()):
        if len(insts) != 2:
            continue
        if is_crowdable(name):
            # Crowd-prone classes (clothing, people) are under-annotated, so
            # "the {name} on the left" is ambiguous vs unannotated instances.
            continue
        p, q = insts
        cp, cq = single_color(p), single_color(q)
        if cp is None or cq is None or cp == cq:
            continue

        px, py = box_center(p["bbox_xyxy_pixel"])
        qx, qy = box_center(q["bbox_xyxy_pixel"])

        axis_ops = []
        if abs(px - qx) >= 40:
            axis_ops.append(("color_left", "x", True))    # min center-x
            axis_ops.append(("color_right", "x", False))  # max center-x
        if abs(py - qy) >= 40:
            axis_ops.append(("color_higher", "y", True))
            axis_ops.append(("color_lower", "y", False))

        for operation, axis, want_min in axis_ops:
            if axis == "x":
                pv, qv = px, qx
            else:
                pv, qv = py, qy
            if want_min:
                selected = p if pv < qv else q
            else:
                selected = p if pv > qv else q

            verifier = {
                "color_left": "select_x_center_min_color",
                "color_right": "select_x_center_max_color",
                "color_higher": "select_y_center_min_color",
                "color_lower": "select_y_center_max_color",
            }[operation]

            answer = single_color(selected)
            sel_id = "inst_a" if selected is p else "inst_b"

            question = rng.choice(template_bank[operation]).format(name=name)
            target_evidence = {
                "nodes": [
                    {"id": "inst_a", "name": name, "bbox": p["bbox_xyxy_norm1000"]},
                    {"id": "inst_b", "name": name, "bbox": q["bbox_xyxy_norm1000"]},
                ],
                "operation": verifier,
                "selected_id": sel_id,
                "answer_attribute": "color",
                "coordinate_format": "xyxy_norm1000",
            }

            candidates.append(
                {
                    "data_source": "qcvsr_gqa_attr_select_spatial",
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
                        "class_count": 2,
                        "other_color": cq if selected is p else cp,
                    },
                }
            )

    return candidates


def generate_count_relation_candidates(
    image_id: str,
    image_path: str,
    objects: List[Dict[str, Any]],
    rng: random.Random,
    split: str,
    template_bank: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """Compositional: count subjects bearing a relation to a unique reference.

    "How many {cups} are on the {table}?" -- requires >=2 subjects of one class
    related to a single (uniquely-referrable) reference object.
    """
    candidates = []
    schema = "count_relation"
    evidence_template = count_relation_template()
    operation = "count_relation"

    rel_set = COUNT_RELATIONS_UNSEEN if split == "eval_ood_operation" else COUNT_RELATIONS_SEEN

    groups = group_objects_by_class(objects)
    unique_refs = {name: insts[0] for name, insts in groups.items() if len(insts) == 1}
    id_map = objects_by_id(objects)

    triples = []  # (subject_name, ref_obj, relation, subject_objs)
    for sub_name, insts in groups.items():
        # map ref_id -> {relation -> [subject objs]}
        by_ref_rel: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for o in insts:
            for r in o.get("relations", []):
                rel = normalize_name(r.get("name"))
                ref_id = str(r.get("object"))
                if rel in rel_set and ref_id in id_map:
                    by_ref_rel[(ref_id, rel)].append(o)
        for (ref_id, rel), subs in by_ref_rel.items():
            ref = id_map[ref_id]
            if ref["name"] == sub_name:
                continue
            # reference must be an unambiguous referent
            if ref["name"] not in unique_refs or unique_refs[ref["name"]]["object_id"] != ref_id:
                continue
            if len(subs) >= 2:
                triples.append((sub_name, ref, rel, subs))

    rng.shuffle(triples)

    for sub_name, ref, rel, subs in triples:
        count = len(subs)
        question = rng.choice(template_bank[operation]).format(
            name_plural=pluralize(sub_name), ref=ref["name"], rel=rel,
        )
        nodes = [{
            "id": "reference", "name": ref["name"],
            "bbox": ref["bbox_xyxy_norm1000"], "role": "reference",
        }]
        for k, s in enumerate(subs):
            nodes.append({
                "id": f"subject_{k}", "name": sub_name,
                "bbox": s["bbox_xyxy_norm1000"], "role": "subject",
            })
        target_evidence = {
            "nodes": nodes,
            "operation": "count_related_subjects",
            "relation": rel,
            "count": count,
            "coordinate_format": "xyxy_norm1000",
        }

        candidates.append(
            {
                "data_source": "qcvsr_gqa_count_relation",
                "schema": schema,
                "question": question,
                "evidence_template": evidence_template,
                "image_path": image_path,
                "image_id": image_id,
                "target_evidence": target_evidence,
                "target_answer": str(count),
                "verifier": "count_related_subjects",
                "split": split,
                "operation": operation,
                "extra_info": {"relation": rel, "reference": ref["name"]},
            }
        )

    return candidates


def generate_relation_target_candidates(
    image_id: str,
    image_path: str,
    objects: List[Dict[str, Any]],
    rng: random.Random,
    split: str,
    template_bank: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """Compositional: traverse a relation from a unique subject to its target.

    "What is the {boy} {wearing}?" -- subject must be uniquely referrable and
    have exactly one relation of the queried predicate, so the target is unique.
    """
    candidates = []
    schema = "relation_target"
    evidence_template = relation_target_template()
    operation = "relation_target"

    rel_set = TARGET_RELATIONS_UNSEEN if split == "eval_ood_operation" else TARGET_RELATIONS_SEEN

    groups = group_objects_by_class(objects)
    unique_refs = {name: insts[0] for name, insts in groups.items() if len(insts) == 1}
    id_map = objects_by_id(objects)

    items = []  # (subject_obj, relation, target_obj)
    for name, subj in unique_refs.items():
        by_rel: Dict[str, List[str]] = defaultdict(list)
        for r in subj.get("relations", []):
            rel = normalize_name(r.get("name"))
            ref_id = str(r.get("object"))
            if rel in rel_set and ref_id in id_map:
                by_rel[rel].append(ref_id)
        for rel, target_ids in by_rel.items():
            target_ids = list(dict.fromkeys(target_ids))
            if len(target_ids) != 1:
                continue  # ambiguous: multiple distinct targets
            target = id_map[target_ids[0]]
            if target["name"] == name:
                continue
            items.append((subj, rel, target))

    rng.shuffle(items)

    for subj, rel, target in items:
        question = rng.choice(template_bank[operation]).format(name=subj["name"], rel=rel)
        target_evidence = {
            "nodes": [
                {"id": "subject", "name": subj["name"],
                 "bbox": subj["bbox_xyxy_norm1000"], "role": "subject"},
                {"id": "target", "name": target["name"],
                 "bbox": target["bbox_xyxy_norm1000"], "role": "target"},
            ],
            "operation": "relation_traverse",
            "relation": rel,
            "coordinate_format": "xyxy_norm1000",
        }

        candidates.append(
            {
                "data_source": "qcvsr_gqa_relation_target",
                "schema": schema,
                "question": question,
                "evidence_template": evidence_template,
                "image_path": image_path,
                "image_id": image_id,
                "target_evidence": target_evidence,
                "target_answer": target["name"],
                "verifier": "relation_target_name",
                "split": split,
                "operation": operation,
                "extra_info": {"relation": rel, "subject": subj["name"]},
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

        question = rng.choice(template_bank[operation]).format(
            a=class_a, b=class_b,
            a_plural=pluralize(class_a), b_plural=pluralize(class_b),
        )

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
    # OOD-operation split holds out unseen *operations* (vertical spatial,
    # smaller, difference, vertical attribute-select). The two relation schemas
    # hold out unseen *predicates* instead -- handled inside their generators
    # via the `split` argument -- so their single op is allowed here.
    if schema == "bbox_pair_spatial_compare":
        return operation in {"higher", "lower"}
    if schema == "bbox_pair_size_compare":
        return operation in {"smaller"}
    if schema == "count_pair_compare":
        return operation in {"difference"}
    if schema == "attr_select_spatial":
        return operation in {"color_higher", "color_lower"}
    if schema in ("count_relation", "relation_target"):
        return True
    return True


def allowed_for_train_or_id(schema: str, operation: str) -> bool:
    if schema == "bbox_pair_spatial_compare":
        return operation in {"left", "right"}
    if schema == "bbox_pair_size_compare":
        return operation in {"larger"}
    if schema == "count_pair_compare":
        return operation in {"more_than_yesno"}
    if schema == "attr_select_spatial":
        return operation in {"color_left", "color_right"}
    if schema in ("count_relation", "relation_target"):
        return True
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

        quotas = get_quota_for_split(split, TARGET_QUOTAS)
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
                candidates.extend(generate_attr_select_candidates(image_id, image_path, objects, rng, split, template_bank))
                candidates.extend(generate_count_relation_candidates(image_id, image_path, objects, rng, split, template_bank))
                candidates.extend(generate_relation_target_candidates(image_id, image_path, objects, rng, split, template_bank))

                rng.shuffle(candidates)

                per_image_op_count: Counter = Counter()

                for cand in candidates:
                    schema = cand["schema"]
                    op = cand["operation"]

                    if schema not in quotas or op not in quotas[schema]:
                        continue

                    if split == "eval_ood_operation":
                        if not allowed_for_ood_operation(schema, op):
                            continue
                    else:
                        if not allowed_for_train_or_id(schema, op):
                            continue

                    if quota_counts[schema][op] >= quotas[schema][op]:
                        continue

                    if per_image_op_count[(schema, op)] >= MAX_RECORDS_PER_IMAGE_PER_OP:
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
                    per_image_op_count[(schema, op)] += 1

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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate QCVSR GQA synthetic dataset.")
    p.add_argument("--scene-graph-path", default=DEFAULT_SCENE_GRAPH_PATH)
    p.add_argument("--image-out-dir", default=DEFAULT_IMAGE_OUT_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--dataset-repo", default="lmms-lab/GQA")
    p.add_argument("--image-config", default="train_balanced_images")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-images", type=int, default=None,
                   help="Limit number of images processed (after the dataset is loaded). None = all.")
    p.add_argument("--checkpoint-every", type=int, default=1000)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_dataset(
        scene_graph_path=args.scene_graph_path,
        image_out_dir=args.image_out_dir,
        out_dir=args.out_dir,
        dataset_repo=args.dataset_repo,
        image_config=args.image_config,
        seed=args.seed,
        max_images=args.max_images,
        checkpoint_every=args.checkpoint_every,
    )
