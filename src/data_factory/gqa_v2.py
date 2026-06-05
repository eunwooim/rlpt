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
DEFAULT_IMAGE_OUT_DIR = "/scratch/eunwooim/rlpt/src/visualize/images/"
DEFAULT_OUT_DIR = "/scratch/eunwooim/rlpt/src/data_factory/gqa_v2/"
LABEL_VOCAB_PATH = "/scratch/eunwooim/rlpt/src/data_factory/gqa_label_vocab.json"

# ============================================================
# Global config
# ============================================================

SEED = 42

MIN_AREA_RATIO = 0.005
MAX_AREA_RATIO = 0.45

MIN_SPATIAL_MARGIN_PX = 50
MIN_RELATIVE_AREA_RATIO = 2.0

MAX_PAIRWISE_IOU = 0.0
REQUIRE_ZERO_INTERSECTION_FOR_SPATIAL = True

MAX_RECORDS_PER_IMAGE_TRAIN = 4
MAX_RECORDS_PER_IMAGE_EVAL = 2
MAX_ATTEMPTS_PER_IMAGE = 120
CHECKPOINT_EVERY = 1000

MAX_PER_OBJECT_NAME_PER_SPLIT = 800
MAX_PER_ATTRIBUTE_PER_SPLIT = 400
MAX_PER_RELATION_PER_SPLIT = 300

STUFF_OBJECTS = {
    "water", "sky", "grass", "road", "street", "sidewalk", "floor", "ground",
    "wall", "ceiling", "field", "snow", "sand", "sea", "ocean", "river",
    "lake", "background", "room", "area", "place", "window", "building",
    "cloud", "clouds", "tree leaves", "leaves", "tablecloth", "picnic",
}

PART_OR_TEXTURE_OBJECTS = {
    "spot", "spots", "stripe", "stripes", "pattern", "logo", "text",
    "shadow", "reflection", "mark", "marks", "part", "edge", "corner",
}

PLURAL_MAP = {
    "people": "person",
    "men": "man",
    "women": "woman",
    "children": "child",
    "teeth": "tooth",
    "feet": "foot",
    "bananas": "banana",
    "plantains": "plantain",
    "onions": "onion",
    "spots": "spot",
    "leaves": "leaf",
    "glasses": "glasses",
    "scissors": "scissors",
}


# ============================================================
# Attribute taxonomy
# ============================================================

COLOR_ATTRIBUTES = {
    "black", "white", "red", "blue", "green", "yellow", "orange", "brown",
    "gray", "grey", "pink", "purple", "silver", "gold", "tan", "beige",
    "blond", "blonde", "cream", "cream colored", "dark brown", "light brown",
    "dark", "light", "colorful",
}

MATERIAL_ATTRIBUTES = {
    "wooden", "wood", "metal", "metallic", "plastic", "glass", "ceramic",
    "paper", "cloth", "fabric", "leather", "stone", "brick", "concrete",
    "rubber", "steel", "wicker", "cardboard",
}

PATTERN_ATTRIBUTES = {
    "striped", "spotted", "checkered", "plaid", "dotted", "plain",
    "solid", "patterned", "floral",
}

SHAPE_ATTRIBUTES = {
    "round", "square", "rectangular", "circular", "curved", "straight",
    "pointed", "flat", "oval", "triangular",
}

STATE_ATTRIBUTES = {
    "parked", "standing", "sitting", "lying", "hanging", "mounted",
    "open", "closed", "folded", "stacked", "piled", "cut", "broken",
}

SUBJECTIVE_OR_BAD_ATTRIBUTES = {
    "little", "small", "large", "big", "tiny", "huge", "tall", "short", "long",
    "young", "old", "new", "nice", "good", "bad", "beautiful", "pretty", "ugly",
    "clean", "dirty", "neat", "messy", "normal", "usual", "different", "same",
    "other", "visible", "clear", "unclear", "unknown", "many", "few", "some",
    "several", "various", "delicious", "fresh", "dry", "wet", "empty", "full",
}


def attribute_type(attr: Any) -> Optional[str]:
    attr = str(attr).strip().lower()
    if not attr:
        return None
    if attr in SUBJECTIVE_OR_BAD_ATTRIBUTES:
        return None
    if attr in COLOR_ATTRIBUTES or "colored" in attr or "colour" in attr or "color" in attr:
        return "color"
    if attr in MATERIAL_ATTRIBUTES:
        return "material"
    if attr in PATTERN_ATTRIBUTES:
        return "pattern"
    if attr in SHAPE_ATTRIBUTES:
        return "shape"
    if attr in STATE_ATTRIBUTES:
        return "state"
    return None


def is_concrete_attribute(attr: Any) -> bool:
    return attribute_type(attr) is not None


def typed_attribute_question(object_name: str, attr: str, rng: random.Random) -> str:
    t = attribute_type(attr)

    templates = {
        "color": [
            "What is the color of the {name}?",
            "What color is the {name}?",
        ],
        "material": [
            "What is the material of the {name}?",
            "What material is the {name} made of?",
        ],
        "pattern": [
            "What is the pattern of the {name}?",
            "What pattern does the {name} have?",
        ],
        "shape": [
            "What is the shape of the {name}?",
            "What shape is the {name}?",
        ],
        "state": [
            "What visible state describes the {name}?",
            "What is the visible state of the {name}?",
        ],
    }

    return rng.choice(templates[t]).format(name=object_name)


def typed_attribute_to_object_question(attr: str, rng: random.Random) -> str:
    t = attribute_type(attr)

    templates = {
        "color": [
            "Which object is {attr}?",
            "What object has the color {attr}?",
        ],
        "material": [
            "Which object is made of {attr}?",
            "What object has {attr} material?",
        ],
        "pattern": [
            "Which object has a {attr} pattern?",
            "What object is {attr}?",
        ],
        "shape": [
            "Which object is {attr}?",
            "What object has a {attr} shape?",
        ],
        "state": [
            "Which object is {attr}?",
            "What object is visibly {attr}?",
        ],
    }

    return rng.choice(templates[t]).format(attr=attr)


# ============================================================
# Target quotas
# ============================================================

TARGET_QUOTAS = {
    "attribute_to_object": {
        "attribute_to_object": 1200,
    },
    "object_to_attribute": {
        "object_to_attribute": 1000,
    },
    "relation_to_object": {
        "relation_to_object": 1200,
    },
    "relation_to_subject": {
        "relation_to_subject": 1200,
    },
    "count_object": {
        "count_object": 800,
    },
    "count_attribute": {
        "count_attribute": 800,
    },
    "count_class_attribute": {
        "count_class_attribute": 800,
    },
    "count_pair_compare": {
        "more_than_yesno": 800,
        "difference": 800,
    },
    "bbox_pair_spatial_compare": {
        "left": 600,
        "right": 600,
        "higher": 600,
        "lower": 600,
    },
    "bbox_pair_size_compare": {
        "larger": 300,
        "smaller": 300,
    },
}


# ============================================================
# Templates
# ============================================================

TRAIN_TEMPLATES = {
    "relation_to_object": [
        "What is the {subject_name} {relation}?",
        "Which object is the {subject_name} {relation}?",
    ],
    "relation_to_subject": [
        "What is {relation} the {object_name}?",
        "Which object is {relation} the {object_name}?",
    ],
    "count_object": [
        "How many {name} objects are there?",
        "What is the number of objects labeled {name}?",
    ],
    "count_attribute": [
        "How many {attr} objects are there?",
        "What is the number of objects that are {attr}?",
    ],
    "count_class_attribute": [
        "How many {attr} {name} objects are there?",
        "What is the number of {name} objects that are {attr}?",
    ],
    "more_than_yesno": [
        "Are there more {a} objects than {b} objects?",
        "Does the image contain more objects labeled {a} than objects labeled {b}?",
    ],
    "difference": [
        "How many more {a} objects are there than {b} objects?",
        "What is the difference between the number of {a} objects and {b} objects?",
    ],
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
        "Which object occupies a larger visible area, the {a} or the {b}?",
        "Which has the larger visible bounding region, the {a} or the {b}?",
    ],
    "smaller": [
        "Which object occupies a smaller visible area, the {a} or the {b}?",
        "Which has the smaller visible bounding region, the {a} or the {b}?",
    ],
}

OOD_TEMPLATES = {
    **TRAIN_TEMPLATES,
    "left": [
        "Between the {a} and the {b}, which is closer to the left side?",
    ],
    "right": [
        "Between the {a} and the {b}, which is closer to the right side?",
    ],
    "higher": [
        "Between the {a} and the {b}, which is closer to the top?",
    ],
    "lower": [
        "Between the {a} and the {b}, which is closer to the bottom?",
    ],
}


# ============================================================
# Relation filters
# ============================================================

VAGUE_RELATIONS = {
    "of", "with", "has", "have", "near", "next to", "beside", "by", "around",
    "for", "from", "at", "along", "against", "part of", "belonging to",
}

OPEN_ENDED_SPATIAL_RELATIONS = {
    "to the left of",
    "to the right of",
    "left of",
    "right of",
    "above",
    "under",
    "beneath",
    "over",
    "behind",
    "in front of",
}

RELATION_QA_DENYLIST = VAGUE_RELATIONS | OPEN_ENDED_SPATIAL_RELATIONS

RELATION_FAMILY_HINTS = {
    "interaction": {
        "holding", "wearing", "riding", "carrying", "eating", "drinking",
        "using", "playing", "looking at", "watching", "feeding", "pulling",
        "pushing", "touching", "walking", "walking on",
    },
    "support_contact": {
        "on", "sitting on", "standing on", "lying on", "attached to",
        "hanging from", "mounted on", "parked on", "leaning on",
    },
    "containment": {
        "in", "inside", "containing", "filled with", "covered by",
    },
    "other_semantic": set(),
}


def relation_family(relation: str) -> str:
    relation = str(relation).strip().lower()
    for family, rels in RELATION_FAMILY_HINTS.items():
        if relation in rels:
            return family
    return "other_semantic"


# ============================================================
# IO / utilities
# ============================================================

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_scene_graphs(path: str) -> Dict[str, Any]:
    raw = load_json(path)
    return {str(k): v for k, v in raw.items()}


def load_label_vocab(path: str) -> Dict[str, Any]:
    vocab = load_json(path)
    selected_attributes = set(vocab.get("selected_attributes", {}).keys())
    selected_relations = set(vocab.get("selected_relations", {}).keys())

    if not selected_attributes:
        raise ValueError(f"No selected_attributes found in {path}")
    if not selected_relations:
        raise ValueError(f"No selected_relations found in {path}")

    return {
        "raw": vocab,
        "selected_attributes": selected_attributes,
        "selected_relations": selected_relations,
    }


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def append_jsonl(record: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def save_image(image_obj: Any, image_id: str, image_out_dir: str) -> str:
    out_dir = Path(image_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{image_id}.jpg"
    if not path.exists():
        image_obj.convert("RGB").save(path, quality=95)
    return str(path)


def canonical_name(name: Any) -> str:
    s = str(name).strip().lower()
    if s in PLURAL_MAP:
        return PLURAL_MAP[s]
    if len(s) > 3 and s.endswith("s") and not s.endswith("ss"):
        return s[:-1]
    return s


def is_selected_relation(rel: str, selected_relations: set) -> bool:
    rel = str(rel).strip().lower()
    return bool(rel) and rel in selected_relations


def is_allowed_relation_for_relation_qa(rel: str, selected_relations: set) -> bool:
    rel = str(rel).strip().lower()
    return is_selected_relation(rel, selected_relations) and rel not in RELATION_QA_DENYLIST


def is_allowed_attribute(attr: str, selected_attributes: set) -> bool:
    attr = str(attr).strip().lower()
    return bool(attr) and attr in selected_attributes and is_concrete_attribute(attr)


# ============================================================
# Geometry
# ============================================================

def xywh_to_xyxy(box: List[float]) -> List[float]:
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def box_area(box: List[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_center(box: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def intersection_area(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def boxes_have_zero_intersection(a: List[float], b: List[float], eps: float = 1e-6) -> bool:
    return intersection_area(a, b) <= eps


def box_iou(a: List[float], b: List[float]) -> float:
    inter = intersection_area(a, b)
    union = box_area(a) + box_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def bbox_contains(a: List[float], b: List[float], threshold: float = 0.85) -> bool:
    inter = intersection_area(a, b)
    area_a = max(box_area(a), 1.0)
    area_b = max(box_area(b), 1.0)
    return (inter / area_a >= threshold) or (inter / area_b >= threshold)


def valid_box(box: List[float], width: int, height: int) -> bool:
    x1, y1, x2, y2 = box
    if width <= 0 or height <= 0:
        return False
    if x2 <= x1 or y2 <= y1:
        return False
    ar = box_area(box) / float(width * height)
    return MIN_AREA_RATIO <= ar <= MAX_AREA_RATIO


def xyxy_to_norm1000(box: List[float], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return [
        int(round(x1 / width * 1000)),
        int(round(y1 / height * 1000)),
        int(round(x2 / width * 1000)),
        int(round(y2 / height * 1000)),
    ]


# ============================================================
# Graph normalization
# ============================================================

def normalize_graph(
    image_id: str,
    sg: Dict[str, Any],
    width: int,
    height: int,
    label_vocab: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    raw_objects = sg.get("objects", {})
    if not isinstance(raw_objects, dict):
        return None

    selected_attrs = label_vocab["selected_attributes"]
    selected_rels = label_vocab["selected_relations"]

    nodes = {}
    name_to_ids = defaultdict(list)
    attr_to_ids = defaultdict(list)
    attr_type_to_ids = defaultdict(list)
    class_attr_to_ids = defaultdict(list)
    edges = []

    for object_id, obj in raw_objects.items():
        if not isinstance(obj, dict):
            continue

        raw_name = str(obj.get("name", "")).strip().lower()
        name = canonical_name(raw_name)

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

        box_pixel = xywh_to_xyxy([x, y, w, h])
        if not valid_box(box_pixel, width, height):
            continue

        attrs = []
        raw_attrs = obj.get("attributes", [])
        if isinstance(raw_attrs, list):
            for attr in raw_attrs:
                attr = str(attr).strip().lower()
                if is_allowed_attribute(attr, selected_attrs):
                    attrs.append(attr)
        attrs = sorted(set(attrs))

        node = {
            "id": str(object_id),
            "name": name,
            "raw_name": raw_name,
            "bbox_pixel": box_pixel,
            "bbox": xyxy_to_norm1000(box_pixel, width, height),
            "attributes": attrs,
            "typed_attributes": {a: attribute_type(a) for a in attrs},
            "area": box_area(box_pixel),
            "area_ratio": box_area(box_pixel) / float(width * height),
            "is_part_or_texture": name in PART_OR_TEXTURE_OBJECTS,
        }

        nodes[str(object_id)] = node
        name_to_ids[name].append(str(object_id))

        for attr in attrs:
            attr_to_ids[attr].append(str(object_id))
            attr_type_to_ids[attribute_type(attr)].append(str(object_id))
            class_attr_to_ids[(name, attr)].append(str(object_id))

    if len(nodes) < 2:
        return None

    for object_id, obj in raw_objects.items():
        sid = str(object_id)
        if sid not in nodes:
            continue

        rels = obj.get("relations", [])
        if not isinstance(rels, list):
            continue

        for rel in rels:
            if not isinstance(rel, dict):
                continue

            rel_name = str(rel.get("name", "")).strip().lower()
            oid = str(rel.get("object", ""))

            if oid not in nodes:
                continue
            if sid == oid:
                continue
            if not is_selected_relation(rel_name, selected_rels):
                continue

            edges.append({
                "subject_id": sid,
                "object_id": oid,
                "relation": rel_name,
                "family": relation_family(rel_name),
                "valid_for_relation_qa": is_allowed_relation_for_relation_qa(rel_name, selected_rels),
            })

    return {
        "image_id": image_id,
        "width": width,
        "height": height,
        "nodes": nodes,
        "edges": edges,
        "name_to_ids": dict(name_to_ids),
        "attr_to_ids": dict(attr_to_ids),
        "attr_type_to_ids": dict(attr_type_to_ids),
        "class_attr_to_ids": {f"{k[0]}::{k[1]}": v for k, v in class_attr_to_ids.items()},
    }


def unique_referable_node_ids(graph: Dict[str, Any], allow_parts: bool = True) -> List[str]:
    out = []
    for name, ids in graph["name_to_ids"].items():
        if len(ids) != 1:
            continue
        node = graph["nodes"][ids[0]]
        if node["is_part_or_texture"] and not allow_parts:
            continue
        out.append(ids[0])
    return out


def is_unique_node(graph: Dict[str, Any], node_id: str) -> bool:
    node = graph["nodes"][node_id]
    return len(graph["name_to_ids"].get(node["name"], [])) == 1


def answer_set_relation_object(graph: Dict[str, Any], subject_id: str, relation: str) -> List[str]:
    return [
        e["object_id"]
        for e in graph["edges"]
        if e["subject_id"] == subject_id
        and e["relation"] == relation
        and e.get("valid_for_relation_qa", False)
    ]


def answer_set_relation_subject(graph: Dict[str, Any], object_id: str, relation: str) -> List[str]:
    return [
        e["subject_id"]
        for e in graph["edges"]
        if e["object_id"] == object_id
        and e["relation"] == relation
        and e.get("valid_for_relation_qa", False)
    ]


# ============================================================
# Prompt / schema format
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


def bbox_pair_template() -> str:
    return (
        '{"nodes":[{"id":"object_a","name":"...","bbox":[x1,y1,x2,y2]},'
        '{"id":"object_b","name":"...","bbox":[x1,y1,x2,y2]}],'
        '"operation":"...","coordinate_format":"xyxy_norm1000"}'
    )


def single_object_template() -> str:
    return (
        '{"nodes":[{"id":"object","name":"...","bbox":[x1,y1,x2,y2],'
        '"attribute_type":"...","attributes":["..."]}],'
        '"operation":"...","coordinate_format":"xyxy_norm1000"}'
    )


def attribute_group_template() -> str:
    return (
        '{"nodes":[{"id":"attribute_group","attribute_type":"...","attribute":"...",'
        '"count":0,"members":[{"name":"...","bbox":[x1,y1,x2,y2]}]}],'
        '"operation":"...","coordinate_format":"xyxy_norm1000"}'
    )


def relation_template() -> str:
    return (
        '{"nodes":[{"id":"subject","name":"...","bbox":[x1,y1,x2,y2]},'
        '{"id":"object","name":"...","bbox":[x1,y1,x2,y2]}],'
        '"relation":"...","operation":"...",'
        '"coordinate_format":"xyxy_norm1000"}'
    )


def count_object_template() -> str:
    return (
        '{"nodes":[{"id":"class","name":"...","count":0,'
        '"members":[{"bbox":[x1,y1,x2,y2]}]}],'
        '"operation":"...","coordinate_format":"xyxy_norm1000"}'
    )


def count_pair_template() -> str:
    return (
        '{"nodes":[{"id":"class_a","name":"...","count":0},'
        '{"id":"class_b","name":"...","count":0}],'
        '"operation":"..."}'
    )


def make_record(
    *,
    question_id: str,
    data_source: str,
    image_path: str,
    image_id: str,
    question: str,
    schema: str,
    evidence_template: str,
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


# ============================================================
# Evidence helpers
# ============================================================

def node_for_evidence(node: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    return {"id": node_id, "name": node["name"], "bbox": node["bbox"]}


def member_nodes(graph: Dict[str, Any], ids: List[str]) -> List[Dict[str, Any]]:
    return [
        {
            "name": graph["nodes"][i]["name"],
            "bbox": graph["nodes"][i]["bbox"],
            "attributes": graph["nodes"][i]["attributes"],
        }
        for i in ids
    ]


# ============================================================
# Motif samplers
# ============================================================

def sample_attribute_to_object(
    graph: Dict[str, Any],
    rng: random.Random,
    template_bank: Dict[str, List[str]],
) -> Optional[Dict[str, Any]]:
    attrs = list(graph["attr_to_ids"].keys())
    rng.shuffle(attrs)

    for attr in attrs:
        attr_t = attribute_type(attr)
        if attr_t is None:
            continue

        ids = sorted(set(graph["attr_to_ids"].get(attr, [])))
        if len(ids) != 1:
            continue

        answer_node = graph["nodes"][ids[0]]
        if not is_unique_node(graph, answer_node["id"]):
            continue

        question = typed_attribute_to_object_question(attr, rng)

        evidence = {
            "nodes": [{
                "id": "attribute_group",
                "attribute_type": attr_t,
                "attribute": attr,
                "count": 1,
                "members": member_nodes(graph, ids),
            }],
            "operation": "attribute_to_object",
            "coordinate_format": "xyxy_norm1000",
        }

        return {
            "schema": "attribute_to_object",
            "operation": "attribute_to_object",
            "question": question,
            "target_answer": answer_node["name"],
            "target_evidence": evidence,
            "evidence_template": attribute_group_template(),
            "verifier": "attribute_to_object",
            "data_source": f"gqa_scenegraph/attribute_to_object/{attr_t}",
            "labels": {"objects": [answer_node["name"]], "attributes": [attr], "relation": None},
            "extra_info": {
                "attribute": attr,
                "attribute_type": attr_t,
                "answer_object_id": answer_node["id"],
                "answer_object_name": answer_node["name"],
                "answer_set_size": 1,
            },
        }

    return None


def sample_object_to_attribute(
    graph: Dict[str, Any],
    rng: random.Random,
    template_bank: Dict[str, List[str]],
) -> Optional[Dict[str, Any]]:
    ids = unique_referable_node_ids(graph, allow_parts=True)
    rng.shuffle(ids)

    for node_id in ids:
        node = graph["nodes"][node_id]

        typed_attrs = [(a, attribute_type(a)) for a in node["attributes"] if attribute_type(a) is not None]
        if not typed_attrs:
            continue

        attr, attr_t = rng.choice(typed_attrs)

        question = typed_attribute_question(node["name"], attr, rng)

        evidence = {
            "nodes": [{
                "id": "object",
                "name": node["name"],
                "bbox": node["bbox"],
                "attribute_type": attr_t,
                "attributes": node["attributes"],
            }],
            "operation": "object_to_attribute",
            "coordinate_format": "xyxy_norm1000",
        }

        return {
            "schema": "object_to_attribute",
            "operation": "object_to_attribute",
            "question": question,
            "target_answer": attr,
            "target_evidence": evidence,
            "evidence_template": single_object_template(),
            "verifier": "object_to_attribute",
            "data_source": f"gqa_scenegraph/object_to_attribute/{attr_t}",
            "labels": {"objects": [node["name"]], "attributes": [attr], "relation": None},
            "extra_info": {
                "object_id": node["id"],
                "object_name": node["name"],
                "target_attribute": attr,
                "attribute_type": attr_t,
                "object_attributes": node["attributes"],
            },
        }

    return None


def sample_relation_to_object(
    graph: Dict[str, Any],
    rng: random.Random,
    template_bank: Dict[str, List[str]],
) -> Optional[Dict[str, Any]]:
    edges = [e for e in graph["edges"] if e.get("valid_for_relation_qa", False)]
    rng.shuffle(edges)

    for edge in edges:
        s = graph["nodes"][edge["subject_id"]]
        rel = edge["relation"]

        if not is_unique_node(graph, s["id"]):
            continue

        answer_ids = sorted(set(answer_set_relation_object(graph, s["id"], rel)))
        if len(answer_ids) != 1:
            continue

        answer_node = graph["nodes"][answer_ids[0]]
        if not is_unique_node(graph, answer_node["id"]):
            continue

        question = rng.choice(template_bank["relation_to_object"]).format(
            subject_name=s["name"],
            relation=rel,
        )

        evidence = {
            "nodes": [
                node_for_evidence(s, "subject"),
                node_for_evidence(answer_node, "object"),
            ],
            "relation": rel,
            "operation": "relation_to_object",
            "coordinate_format": "xyxy_norm1000",
        }

        return {
            "schema": "relation_to_object",
            "operation": "relation_to_object",
            "question": question,
            "target_answer": answer_node["name"],
            "target_evidence": evidence,
            "evidence_template": relation_template(),
            "verifier": "relation_to_object",
            "data_source": f"gqa_scenegraph/relation_to_object/{edge['family']}",
            "labels": {"objects": [s["name"], answer_node["name"]], "attributes": [], "relation": rel},
            "extra_info": {
                "subject_id": s["id"],
                "object_id": answer_node["id"],
                "subject_name": s["name"],
                "object_name": answer_node["name"],
                "relation": rel,
                "relation_family": edge["family"],
                "answer_set_size": 1,
            },
        }

    return None


def sample_relation_to_subject(
    graph: Dict[str, Any],
    rng: random.Random,
    template_bank: Dict[str, List[str]],
) -> Optional[Dict[str, Any]]:
    edges = [e for e in graph["edges"] if e.get("valid_for_relation_qa", False)]
    rng.shuffle(edges)

    for edge in edges:
        o = graph["nodes"][edge["object_id"]]
        rel = edge["relation"]

        if not is_unique_node(graph, o["id"]):
            continue

        answer_ids = sorted(set(answer_set_relation_subject(graph, o["id"], rel)))
        if len(answer_ids) != 1:
            continue

        answer_node = graph["nodes"][answer_ids[0]]
        if not is_unique_node(graph, answer_node["id"]):
            continue

        question = rng.choice(template_bank["relation_to_subject"]).format(
            relation=rel,
            object_name=o["name"],
        )

        evidence = {
            "nodes": [
                node_for_evidence(answer_node, "subject"),
                node_for_evidence(o, "object"),
            ],
            "relation": rel,
            "operation": "relation_to_subject",
            "coordinate_format": "xyxy_norm1000",
        }

        return {
            "schema": "relation_to_subject",
            "operation": "relation_to_subject",
            "question": question,
            "target_answer": answer_node["name"],
            "target_evidence": evidence,
            "evidence_template": relation_template(),
            "verifier": "relation_to_subject",
            "data_source": f"gqa_scenegraph/relation_to_subject/{edge['family']}",
            "labels": {"objects": [answer_node["name"], o["name"]], "attributes": [], "relation": rel},
            "extra_info": {
                "subject_id": answer_node["id"],
                "object_id": o["id"],
                "subject_name": answer_node["name"],
                "object_name": o["name"],
                "relation": rel,
                "relation_family": edge["family"],
                "answer_set_size": 1,
            },
        }

    return None


def sample_count_object(graph, rng, template_bank):
    names = list(graph["name_to_ids"].keys())
    rng.shuffle(names)
    for name in names:
        ids = graph["name_to_ids"][name]
        question = rng.choice(template_bank["count_object"]).format(name=name)
        evidence = {
            "nodes": [{
                "id": "class",
                "name": name,
                "count": len(ids),
                "members": member_nodes(graph, ids),
            }],
            "operation": "count_object",
            "coordinate_format": "xyxy_norm1000",
        }
        return {
            "schema": "count_object",
            "operation": "count_object",
            "question": question,
            "target_answer": str(len(ids)),
            "target_evidence": evidence,
            "evidence_template": count_object_template(),
            "verifier": "count_object",
            "data_source": "gqa_scenegraph/count_object",
            "labels": {"objects": [name], "attributes": [], "relation": None},
            "extra_info": {"object_name": name, "count": len(ids), "member_ids": ids},
        }
    return None


def sample_count_attribute(graph, rng, template_bank):
    attrs = list(graph["attr_to_ids"].keys())
    rng.shuffle(attrs)
    for attr in attrs:
        attr_t = attribute_type(attr)
        if attr_t is None:
            continue
        ids = graph["attr_to_ids"][attr]
        question = rng.choice(template_bank["count_attribute"]).format(attr=attr)
        evidence = {
            "nodes": [{
                "id": "attribute_group",
                "attribute_type": attr_t,
                "attribute": attr,
                "count": len(ids),
                "members": member_nodes(graph, ids),
            }],
            "operation": "count_attribute",
            "coordinate_format": "xyxy_norm1000",
        }
        return {
            "schema": "count_attribute",
            "operation": "count_attribute",
            "question": question,
            "target_answer": str(len(ids)),
            "target_evidence": evidence,
            "evidence_template": attribute_group_template(),
            "verifier": "count_attribute",
            "data_source": f"gqa_scenegraph/count_attribute/{attr_t}",
            "labels": {"objects": [], "attributes": [attr], "relation": None},
            "extra_info": {"attribute": attr, "attribute_type": attr_t, "count": len(ids), "member_ids": ids},
        }
    return None


def sample_count_class_attribute(graph, rng, template_bank):
    keys = list(graph["class_attr_to_ids"].keys())
    rng.shuffle(keys)
    for key in keys:
        name, attr = key.split("::", 1)
        attr_t = attribute_type(attr)
        if attr_t is None:
            continue
        ids = graph["class_attr_to_ids"][key]
        question = rng.choice(template_bank["count_class_attribute"]).format(name=name, attr=attr)
        evidence = {
            "nodes": [{
                "id": "class_attribute_group",
                "name": name,
                "attribute_type": attr_t,
                "attribute": attr,
                "count": len(ids),
                "members": member_nodes(graph, ids),
            }],
            "operation": "count_class_attribute",
            "coordinate_format": "xyxy_norm1000",
        }
        return {
            "schema": "count_class_attribute",
            "operation": "count_class_attribute",
            "question": question,
            "target_answer": str(len(ids)),
            "target_evidence": evidence,
            "evidence_template": attribute_group_template(),
            "verifier": "count_class_attribute",
            "data_source": f"gqa_scenegraph/count_class_attribute/{attr_t}",
            "labels": {"objects": [name], "attributes": [attr], "relation": None},
            "extra_info": {"object_name": name, "attribute": attr, "attribute_type": attr_t, "count": len(ids), "member_ids": ids},
        }
    return None


def sample_count_pair_compare(graph, rng, operation, template_bank):
    names = list(graph["name_to_ids"].keys())
    rng.shuffle(names)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ca, cb = len(graph["name_to_ids"][a]), len(graph["name_to_ids"][b])
            if ca != cb:
                pairs.append((a, b, ca, cb))
    rng.shuffle(pairs)

    for a, b, ca, cb in pairs:
        if operation == "more_than_yesno":
            question = rng.choice(template_bank["more_than_yesno"]).format(a=a, b=b)
            answer = "yes" if ca > cb else "no"
            evidence = {
                "nodes": [{"id": "class_a", "name": a, "count": ca}, {"id": "class_b", "name": b, "count": cb}],
                "operation": "compare_count_greater_than",
            }
            verifier = "compare_count_greater_than"
        else:
            if ca >= cb:
                class_a, class_b, count_a, count_b = a, b, ca, cb
            else:
                class_a, class_b, count_a, count_b = b, a, cb, ca
            question = rng.choice(template_bank["difference"]).format(a=class_a, b=class_b)
            answer = str(count_a - count_b)
            evidence = {
                "nodes": [{"id": "class_a", "name": class_a, "count": count_a}, {"id": "class_b", "name": class_b, "count": count_b}],
                "operation": "count_difference",
            }
            verifier = "count_difference"
            a, b, ca, cb = class_a, class_b, count_a, count_b

        return {
            "schema": "count_pair_compare",
            "operation": operation,
            "question": question,
            "target_answer": answer,
            "target_evidence": evidence,
            "evidence_template": count_pair_template(),
            "verifier": verifier,
            "data_source": "gqa_scenegraph/count_pair_compare",
            "labels": {"objects": [a, b], "attributes": [], "relation": None},
            "extra_info": {"class_a_name": a, "class_b_name": b, "class_a_count": ca, "class_b_count": cb},
        }

    return None


def pair_allowed_for_spatial(a, b):
    if box_iou(a["bbox_pixel"], b["bbox_pixel"]) > MAX_PAIRWISE_IOU:
        return False
    if REQUIRE_ZERO_INTERSECTION_FOR_SPATIAL and not boxes_have_zero_intersection(a["bbox_pixel"], b["bbox_pixel"]):
        return False
    return True


def sample_bbox_pair_spatial_compare(graph, rng, operation, template_bank):
    ids = unique_referable_node_ids(graph, allow_parts=True)
    rng.shuffle(ids)
    pairs = []

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a = graph["nodes"][ids[i]]
            b = graph["nodes"][ids[j]]

            if not pair_allowed_for_spatial(a, b):
                continue

            ax, ay = box_center(a["bbox_pixel"])
            bx, by = box_center(b["bbox_pixel"])

            if operation in {"left", "right"} and abs(ax - bx) < MIN_SPATIAL_MARGIN_PX:
                continue
            if operation in {"higher", "lower"} and abs(ay - by) < MIN_SPATIAL_MARGIN_PX:
                continue

            pairs.append((a, b))

    rng.shuffle(pairs)

    for a, b in pairs:
        ax, ay = box_center(a["bbox_pixel"])
        bx, by = box_center(b["bbox_pixel"])

        if operation == "left":
            answer, verifier = (a["name"] if ax < bx else b["name"]), "compare_x_center_min"
        elif operation == "right":
            answer, verifier = (a["name"] if ax > bx else b["name"]), "compare_x_center_max"
        elif operation == "higher":
            answer, verifier = (a["name"] if ay < by else b["name"]), "compare_y_center_min"
        elif operation == "lower":
            answer, verifier = (a["name"] if ay > by else b["name"]), "compare_y_center_max"
        else:
            return None

        question = rng.choice(template_bank[operation]).format(a=a["name"], b=b["name"])
        evidence = {
            "nodes": [node_for_evidence(a, "object_a"), node_for_evidence(b, "object_b")],
            "operation": verifier,
            "coordinate_format": "xyxy_norm1000",
        }

        return {
            "schema": "bbox_pair_spatial_compare",
            "operation": operation,
            "question": question,
            "target_answer": answer,
            "target_evidence": evidence,
            "evidence_template": bbox_pair_template(),
            "verifier": verifier,
            "data_source": "gqa_scenegraph/bbox_pair_spatial_compare",
            "labels": {"objects": [a["name"], b["name"]], "attributes": [], "relation": None},
            "extra_info": {
                "object_a_id": a["id"], "object_b_id": b["id"],
                "object_a_name": a["name"], "object_b_name": b["name"],
                "object_a_center_pixel": [ax, ay], "object_b_center_pixel": [bx, by],
                "explicit_candidate_set": [a["name"], b["name"]],
                "bbox_intersection_area": intersection_area(a["bbox_pixel"], b["bbox_pixel"]),
                "bbox_iou": box_iou(a["bbox_pixel"], b["bbox_pixel"]),
            },
        }

    return None


def sample_bbox_pair_size_compare(graph, rng, operation, template_bank):
    ids = unique_referable_node_ids(graph, allow_parts=True)
    rng.shuffle(ids)
    pairs = []

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a = graph["nodes"][ids[i]]
            b = graph["nodes"][ids[j]]

            if not boxes_have_zero_intersection(a["bbox_pixel"], b["bbox_pixel"]):
                continue
            if bbox_contains(a["bbox_pixel"], b["bbox_pixel"], threshold=0.85):
                continue

            ratio = max(a["area"], b["area"]) / max(min(a["area"], b["area"]), 1.0)
            if ratio < MIN_RELATIVE_AREA_RATIO:
                continue

            pairs.append((a, b, ratio))

    rng.shuffle(pairs)

    for a, b, ratio in pairs:
        if operation == "larger":
            answer = a["name"] if a["area"] > b["area"] else b["name"]
            verifier = "compare_bbox_area_max"
        elif operation == "smaller":
            answer = a["name"] if a["area"] < b["area"] else b["name"]
            verifier = "compare_bbox_area_min"
        else:
            return None

        question = rng.choice(template_bank[operation]).format(a=a["name"], b=b["name"])
        evidence = {
            "nodes": [node_for_evidence(a, "object_a"), node_for_evidence(b, "object_b")],
            "operation": verifier,
            "coordinate_format": "xyxy_norm1000",
        }

        return {
            "schema": "bbox_pair_size_compare",
            "operation": operation,
            "question": question,
            "target_answer": answer,
            "target_evidence": evidence,
            "evidence_template": bbox_pair_template(),
            "verifier": verifier,
            "data_source": "gqa_scenegraph/bbox_pair_size_compare",
            "labels": {"objects": [a["name"], b["name"]], "attributes": [], "relation": None},
            "extra_info": {
                "object_a_id": a["id"], "object_b_id": b["id"],
                "object_a_name": a["name"], "object_b_name": b["name"],
                "object_a_area_pixel": a["area"], "object_b_area_pixel": b["area"],
                "area_ratio": ratio,
                "explicit_candidate_set": [a["name"], b["name"]],
                "bbox_intersection_area": intersection_area(a["bbox_pixel"], b["bbox_pixel"]),
                "bbox_iou": box_iou(a["bbox_pixel"], b["bbox_pixel"]),
            },
        }

    return None


# ============================================================
# Quotas / split logic
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


def get_quota_for_split(split: str) -> Dict[str, Dict[str, int]]:
    if split == "train":
        return TARGET_QUOTAS
    return {schema: {op: max(50, int(v * 0.10)) for op, v in ops.items()} for schema, ops in TARGET_QUOTAS.items()}


def allowed_for_train_or_id(schema: str, operation: str) -> bool:
    if schema == "bbox_pair_spatial_compare":
        return operation in {"left", "right"}
    if schema == "bbox_pair_size_compare":
        return operation in {"larger"}
    if schema == "count_pair_compare":
        return operation in {"more_than_yesno"}
    return True


def allowed_for_ood_operation(schema: str, operation: str) -> bool:
    if schema == "bbox_pair_spatial_compare":
        return operation in {"higher", "lower"}
    if schema == "bbox_pair_size_compare":
        return operation in {"smaller"}
    if schema == "count_pair_compare":
        return operation in {"difference"}
    return True


def active_quotas(split: str) -> Dict[str, Dict[str, int]]:
    raw = get_quota_for_split(split)
    out = {}
    for schema, ops in raw.items():
        keep = {}
        for op, quota in ops.items():
            if split == "eval_ood_operation":
                if allowed_for_ood_operation(schema, op):
                    keep[op] = quota
            else:
                if allowed_for_train_or_id(schema, op):
                    keep[op] = quota
        if keep:
            out[schema] = keep
    return out


def quotas_done(quotas: Dict[str, Dict[str, int]], counts: Dict[str, Counter]) -> bool:
    return all(counts[schema][op] >= quotas[schema][op] for schema in quotas for op in quotas[schema])


def choose_needed_schema_op(quotas, counts, rng):
    needed = []
    for schema, ops in quotas.items():
        for op, target in ops.items():
            remain = target - counts[schema][op]
            if remain > 0:
                needed.append((schema, op, remain))
    if not needed:
        return None

    total = sum(x[2] for x in needed)
    r = rng.uniform(0, total)
    acc = 0.0
    for schema, op, remain in needed:
        acc += remain
        if r <= acc:
            return schema, op
    return needed[-1][0], needed[-1][1]


def labels_allowed(split, motif, label_counts):
    labels = motif["labels"]
    for obj in labels.get("objects", []):
        if label_counts[f"{split}:obj:{obj}"] >= MAX_PER_OBJECT_NAME_PER_SPLIT:
            return False, "object_name_cap"
    for attr in labels.get("attributes", []):
        if label_counts[f"{split}:attr:{attr}"] >= MAX_PER_ATTRIBUTE_PER_SPLIT:
            return False, "attribute_cap"
    rel = labels.get("relation")
    if rel and label_counts[f"{split}:rel:{rel}"] >= MAX_PER_RELATION_PER_SPLIT:
        return False, "relation_cap"
    return True, "ok"


def update_label_counts(split, motif, label_counts):
    labels = motif["labels"]
    for obj in labels.get("objects", []):
        label_counts[f"{split}:obj:{obj}"] += 1
    for attr in labels.get("attributes", []):
        label_counts[f"{split}:attr:{attr}"] += 1
    rel = labels.get("relation")
    if rel:
        label_counts[f"{split}:rel:{rel}"] += 1


def sample_motif_for_schema(graph, schema, operation, rng, template_bank):
    if schema == "attribute_to_object":
        return sample_attribute_to_object(graph, rng, template_bank)
    if schema == "object_to_attribute":
        return sample_object_to_attribute(graph, rng, template_bank)
    if schema == "relation_to_object":
        return sample_relation_to_object(graph, rng, template_bank)
    if schema == "relation_to_subject":
        return sample_relation_to_subject(graph, rng, template_bank)
    if schema == "count_object":
        return sample_count_object(graph, rng, template_bank)
    if schema == "count_attribute":
        return sample_count_attribute(graph, rng, template_bank)
    if schema == "count_class_attribute":
        return sample_count_class_attribute(graph, rng, template_bank)
    if schema == "count_pair_compare":
        return sample_count_pair_compare(graph, rng, operation, template_bank)
    if schema == "bbox_pair_spatial_compare":
        return sample_bbox_pair_spatial_compare(graph, rng, operation, template_bank)
    if schema == "bbox_pair_size_compare":
        return sample_bbox_pair_size_compare(graph, rng, operation, template_bank)
    return None


# ============================================================
# Audit
# ============================================================

def update_audit(audit, record):
    schema = record["extra_info"]["schema"]
    op = record["extra_info"]["operation"]
    answer = record["reward_model"]["ground_truth"]["target_answer"]

    audit["schema_counts"][schema] += 1
    audit["operation_counts"][f"{schema}/{op}"] += 1
    audit["answer_counts"][f"{schema}/{op}/{answer}"] += 1
    audit["data_source_counts"][record["data_source"]] += 1
    audit["image_counts"][record["extra_info"]["image_id"]] += 1

    attr_type = record["extra_info"].get("attribute_type")
    if attr_type:
        audit["attribute_type_counts"][attr_type] += 1

    evidence = record["reward_model"]["ground_truth"]["target_evidence"]
    for node in evidence.get("nodes", []):
        name = node.get("name")
        if name:
            audit["object_name_counts"][name] += 1
        attr = node.get("attribute")
        if attr:
            audit["attribute_counts"][attr] += 1
        attrs = node.get("attributes", [])
        if isinstance(attrs, list):
            for a in attrs:
                audit["attribute_counts"][a] += 1

    rel = evidence.get("relation")
    if rel:
        audit["relation_counts"][rel] += 1

    family = record["extra_info"].get("relation_family")
    if family:
        audit["relation_family_counts"][family] += 1


def serialize_audit(audit):
    out = {}
    for k, v in audit.items():
        if isinstance(v, Counter):
            out[k] = dict(v)
        elif isinstance(v, dict):
            out[k] = {kk: dict(vv) if isinstance(vv, Counter) else vv for kk, vv in v.items()}
        else:
            out[k] = v
    return out


# ============================================================
# Main generator
# ============================================================

def generate_dataset(
    scene_graph_path=DEFAULT_SCENE_GRAPH_PATH,
    image_out_dir=DEFAULT_IMAGE_OUT_DIR,
    out_dir=DEFAULT_OUT_DIR,
    label_vocab_path=LABEL_VOCAB_PATH,
    dataset_repo="lmms-lab/GQA",
    image_config="train_balanced_images",
    seed=SEED,
    max_images: Optional[int] = None,
    checkpoint_every=CHECKPOINT_EVERY,
) -> None:
    rng = random.Random(seed)

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    scene_graphs = load_scene_graphs(scene_graph_path)
    label_vocab = load_label_vocab(label_vocab_path)

    print(f"Loaded scene graphs: {len(scene_graphs)}")
    print(f"Selected attributes: {len(label_vocab['selected_attributes'])}")
    print(f"Selected relations: {len(label_vocab['selected_relations'])}")

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

    global_audit = {
        "skip_stats": Counter(),
        "label_vocab_summary": label_vocab["raw"].get("summary", {}),
    }

    for split, ids in split_to_ids.items():
        quotas = active_quotas(split)
        quota_counts = {schema: Counter() for schema in quotas}
        image_counts = Counter()
        label_counts = Counter()

        max_records_per_image = MAX_RECORDS_PER_IMAGE_TRAIN if split == "train" else MAX_RECORDS_PER_IMAGE_EVAL
        template_bank = OOD_TEMPLATES if split == "eval_ood_template" else TRAIN_TEMPLATES

        out_path = out_dir_path / f"{split}.jsonl"
        latest_path = out_dir_path / f"latest_{split}.jsonl"
        audit_path = out_dir_path / f"audit_{split}.json"

        if out_path.exists():
            out_path.unlink()
        if latest_path.exists():
            latest_path.unlink()

        records_buffer = []
        records_all = []

        audit = {
            "schema_counts": Counter(),
            "operation_counts": Counter(),
            "answer_counts": Counter(),
            "data_source_counts": Counter(),
            "image_counts": Counter(),
            "object_name_counts": Counter(),
            "attribute_counts": Counter(),
            "attribute_type_counts": Counter(),
            "relation_counts": Counter(),
            "relation_family_counts": Counter(),
            "skip_stats": Counter(),
            "active_quotas": quotas,
            "generation_principle": {
                "object_to_attribute": "typed questions such as color/material/pattern/shape/state of object",
                "subjective_attributes_removed": sorted(SUBJECTIVE_OR_BAD_ATTRIBUTES),
                "allowed_attribute_types": ["color", "material", "pattern", "shape", "state"],
                "spatial_pair_requires_zero_intersection": REQUIRE_ZERO_INTERSECTION_FOR_SPATIAL,
            },
        }

        print(f"\nGenerating split={split}, images={len(ids)}")
        print(f"Active quotas: {quotas}")

        for image_id in ids:
            if quotas_done(quotas, quota_counts):
                break

            if image_counts[image_id] >= max_records_per_image:
                continue

            try:
                image_row = images_ds[image_id_to_idx[image_id]]
                image_obj = image_row.get("image")
                if image_obj is None:
                    audit["skip_stats"]["missing_image"] += 1
                    continue

                width, height = image_obj.size
                image_path = save_image(image_obj, image_id, image_out_dir)

                graph = normalize_graph(
                    image_id=image_id,
                    sg=scene_graphs[image_id],
                    width=width,
                    height=height,
                    label_vocab=label_vocab,
                )

                if graph is None:
                    audit["skip_stats"]["invalid_graph"] += 1
                    continue

                attempts = 0
                while image_counts[image_id] < max_records_per_image and attempts < MAX_ATTEMPTS_PER_IMAGE:
                    attempts += 1

                    chosen = choose_needed_schema_op(quotas, quota_counts, rng)
                    if chosen is None:
                        break

                    schema, operation = chosen

                    motif = sample_motif_for_schema(
                        graph=graph,
                        schema=schema,
                        operation=operation,
                        rng=rng,
                        template_bank=template_bank,
                    )

                    if motif is None:
                        audit["skip_stats"][f"no_motif:{schema}/{operation}"] += 1
                        continue

                    ok, reason = labels_allowed(split, motif, label_counts)
                    if not ok:
                        audit["skip_stats"][reason] += 1
                        continue

                    if quota_counts[schema][operation] >= quotas[schema][operation]:
                        continue

                    qid = f"qcvsr_gqa_generalized_v3_{split}_{sum(sum(c.values()) for c in quota_counts.values()):08d}"

                    record = make_record(
                        question_id=qid,
                        data_source=motif["data_source"],
                        image_path=image_path,
                        image_id=image_id,
                        question=motif["question"],
                        schema=motif["schema"],
                        evidence_template=motif["evidence_template"],
                        target_evidence=motif["target_evidence"],
                        target_answer=motif["target_answer"],
                        verifier=motif["verifier"],
                        split=split,
                        operation=motif["operation"],
                        extra_info={
                            "question_id": qid,
                            "generator_version": "generalized_v3_typed_attribute_questions",
                            **motif["extra_info"],
                        },
                    )

                    append_jsonl(record, out_path)
                    records_buffer.append(record)
                    records_all.append(record)

                    quota_counts[schema][operation] += 1
                    image_counts[image_id] += 1
                    update_label_counts(split, motif, label_counts)
                    update_audit(audit, record)

                    if len(records_all) % checkpoint_every == 0:
                        write_jsonl(records_buffer, latest_path)
                        records_buffer = []
                        print(
                            f"[{split}] records={len(records_all)} "
                            f"quota_counts={ {k: dict(v) for k, v in quota_counts.items()} }"
                        )

            except Exception as e:
                audit["skip_stats"][f"exception:{type(e).__name__}"] += 1
                global_audit["skip_stats"][f"{split}:exception:{type(e).__name__}"] += 1
                continue

        if records_buffer:
            write_jsonl(records_buffer, latest_path)

        audit["final_quota_counts"] = {k: dict(v) for k, v in quota_counts.items()}
        audit["num_records"] = len(records_all)
        audit["num_images_used"] = len(audit["image_counts"])

        write_json(serialize_audit(audit), audit_path)
        all_outputs[split] = records_all

        print(f"[{split}] wrote {len(records_all)} records: {out_path}")
        print(f"[{split}] audit: {audit_path}")

    train_path = out_dir_path / "train.jsonl"
    eval_path = out_dir_path / "eval_all.jsonl"

    write_jsonl(all_outputs.get("train", []), train_path)

    eval_records = []
    for split in ["eval_id", "eval_ood_template", "eval_ood_operation"]:
        eval_records.extend(all_outputs.get(split, []))
    write_jsonl(eval_records, eval_path)

    write_json(
        {
            **serialize_audit(global_audit),
            "output_dir": str(out_dir_path),
            "train_path": str(train_path),
            "eval_all_path": str(eval_path),
            "generation_policy": {
                "streaming": True,
                "typed_attribute_questions": True,
                "allowed_attribute_types": ["color", "material", "pattern", "shape", "state"],
                "subjective_attributes_removed": sorted(SUBJECTIVE_OR_BAD_ATTRIBUTES),
                "relation_qa_denylist": sorted(RELATION_QA_DENYLIST),
                "spatial_pair_requires_zero_intersection": REQUIRE_ZERO_INTERSECTION_FOR_SPATIAL,
                "core_rule": "non_count_queries_require_scene_local_singleton_answer_sets; count_queries_allow_non_unique_sets; attribute queries are typed by attribute subject",
            },
        },
        out_dir_path / "audit_global.json",
    )

    print("\nDone.")
    print(f"Output dir: {out_dir_path}")
    print(f"Train: {train_path}")
    print(f"Eval all: {eval_path}")


if __name__ == "__main__":
    generate_dataset(
        scene_graph_path=DEFAULT_SCENE_GRAPH_PATH,
        image_out_dir=DEFAULT_IMAGE_OUT_DIR,
        out_dir=DEFAULT_OUT_DIR,
        label_vocab_path=LABEL_VOCAB_PATH,
        seed=SEED,
        max_images=None,
        checkpoint_every=CHECKPOINT_EVERY,
    )