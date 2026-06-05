import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from PIL import Image, ImageDraw, ImageFont


# ============================================================
# Defaults / conservative v1 filters
# ============================================================

SEED = 42

MIN_BOX_AREA_RATIO = 0.0005
MAX_BOX_AREA_RATIO = 0.65

MIN_CROP_AREA_RATIO = 0.002
MAX_CROP_AREA_RATIO = 0.85
CROP_PADDING_RATIO = 0.12

MAX_EXTRA_OBJECTS_IN_CROP = 8
MAX_MOTIFS_PER_IMAGE = 8
MAX_MOTIFS_PER_IMAGE_PER_FAMILY = 2

MAX_OBJECTS_PER_TARGET = 5
MAX_ATTRIBUTES_PER_TARGET = 5
MAX_RELATIONS_PER_TARGET = 4

MIN_COUNT = 2
MAX_COUNT = 5

STUFF_OR_VAGUE_OBJECTS = {
    "shade", "shadow", "area", "background", "edge", "side", "corner",
    "part", "place", "scene", "view", "photo", "picture", "image",
    "thing", "object", "item",
    "sky", "ground", "floor", "wall", "street", "road", "sidewalk",
    "grass", "field", "water", "snow", "sand",
}

# Keep intentionally strict. The audit file will help revise these.
BAD_ATTRIBUTES = {
    "small", "large", "big", "little", "tiny", "huge", "tall", "short",
    "long", "young", "old", "new", "nice", "good", "bad", "beautiful",
    "pretty", "ugly", "different", "same", "other", "many", "few",
    "several", "some", "visible", "clear", "dirty", "clean",
}

COLOR_ATTRIBUTES = {
    "black", "white", "red", "blue", "green", "yellow", "orange", "brown",
    "gray", "grey", "pink", "purple", "silver", "gold", "tan", "beige",
    "blond", "blonde", "cream",
}

MATERIAL_ATTRIBUTES = {
    "wooden", "wood", "metal", "metallic", "plastic", "glass", "ceramic",
    "paper", "cloth", "fabric", "leather", "stone", "brick", "concrete",
    "rubber", "steel", "wicker",
}

PATTERN_ATTRIBUTES = {
    "striped", "spotted", "checkered", "plaid", "dotted", "plain",
    "patterned", "floral",
}

SHAPE_ATTRIBUTES = {
    "round", "square", "rectangular", "circular", "curved", "straight",
    "flat", "oval", "triangular",
}

STATE_ATTRIBUTES = {
    "open", "closed", "standing", "sitting", "lying", "hanging", "parked",
    "mounted", "folded", "stacked", "piled", "cut", "broken",
}

TARGET_FAMILY_QUOTAS = {
    "attribute": 100,
    "relation": 300,
    "relation_attribute": 350,
    "multi_relation_anchor": 250,
}

COLOR_WORDS = {
    "black", "white", "red", "blue", "green", "yellow", "orange", "brown",
    "gray", "grey", "pink", "purple", "silver", "gold", "tan", "beige",
    "blond", "blonde", "cream",
}

BAD_COUNT_OBJECTS = {
    "pane", "awning", "outlet", "window", "tile", "brick", "leaf", "leaves",
    "line", "stripe", "letter", "number", "sign", "light", "pole"
}

# Strict relation set for initial pilot.
# Avoid "on", "has", "near", "of", "with" in v1 because they dominate VG and are overloaded.
ALLOWED_RELATIONS = {
    "wearing",
    "holding",
    "carrying",
    "riding",
    "eating",
    "drinking",
    "using",
    "sitting on",
    "standing on",
    "walking on",
    "parked on",
    "hanging from",
    "attached to",
    "covering",
    "inside",
    "under",
}

NOISY_RELATIONS = {
    "on", "has", "of", "with", "near", "next to", "along", "at", "by", "in",
}

BAD_RELATION_SYNSETS = {
    "along.r.01",
    "near.r.01",
}

DEFAULT_RELATION_ALIASES = {
    "wears": "wearing",
    "wear": "wearing",
    "wearing a": "wearing",
    "holds": "holding",
    "hold": "holding",
    "holding a": "holding",
    "carried by": "carrying_reverse",
    "carrying": "carrying",
    "riding": "riding",
    "ride": "riding",
    "sits on": "sitting on",
    "sitting on": "sitting on",
    "stands on": "standing on",
    "standing on": "standing on",
    "walks on": "walking on",
    "walking on": "walking on",
    "parked on": "parked on",
    "hanging from": "hanging from",
    "hangs from": "hanging from",
    "hanging on": "hanging from",
    "attached to": "attached to",
    "covering": "covering",
    "inside": "inside",
    "under": "under",
    "beneath": "under",
}

DEFAULT_OBJECT_ALIASES = {
    "people": "person",
    "men": "man",
    "women": "woman",
    "children": "child",
    "kids": "kid",
    "boys": "boy",
    "girls": "girl",
    "bikes": "bike",
    "bicycles": "bicycle",
    "automobile": "car",
    "autos": "car",
    "sneaker": "sneakers",
    "shoe": "shoes",
}

ATTRIBUTE_ALIASES = {
    "grey": "gray",
    "blonde": "blond",
}

COLOR_WORDS = {
    "black", "white", "red", "blue", "green", "yellow", "orange", "brown",
    "gray", "grey", "pink", "purple", "silver", "gold", "tan", "beige",
    "blond", "blonde", "cream",
}

DEFAULT_OBJECT_ALIASES.update({
    "trouser": "pants",
    "trousers": "pants",
    "pant": "pants",
})


# ============================================================
# Streaming JSON array reader: no ijson required
# ============================================================

def object_name_contains_attribute(name: str, attr_value: str) -> bool:
    name_tokens = set(norm_text(name).split())
    attr_tokens = set(norm_text(attr_value).split())
    return bool(name_tokens & attr_tokens)

def object_name_contains_attribute(name: str, attr_value: str) -> bool:
    name_tokens = set(norm_text(name).split())
    attr_tokens = set(norm_text(attr_value).split())
    return bool(name_tokens & attr_tokens)


def is_bad_count_name(name: str) -> bool:
    name = norm_text(name)
    if name in BAD_COUNT_OBJECTS:
        return True
    if name in STUFF_OR_VAGUE_OBJECTS:
        return True
    return False

def iter_json_array(path: str) -> Iterable[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    buffer = ""
    in_array = False

    with open(path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break

            buffer += chunk

            if not in_array:
                buffer = buffer.lstrip()
                if not buffer.startswith("["):
                    raise ValueError(f"Expected top-level JSON array in {path}")
                buffer = buffer[1:]
                in_array = True

            while True:
                buffer = buffer.lstrip()

                if buffer.startswith("]"):
                    return

                if buffer.startswith(","):
                    buffer = buffer[1:].lstrip()

                try:
                    item, idx = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break

                if isinstance(item, dict):
                    yield item

                buffer = buffer[idx:]

            if len(buffer) > 100 * 1024 * 1024:
                raise RuntimeError(
                    f"Parser buffer exceeded 100MB while reading {path}. "
                    "Install ijson if this happens."
                )


# ============================================================
# IO
# ============================================================

def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_alias_file(path: Optional[str]) -> Dict[str, str]:
    """
    VG alias files are commonly comma-separated groups:
      automobile, auto, car
    We canonicalize every item in a row to the first item.
    """
    if not path:
        return {}

    p = Path(path)
    if not p.exists():
        print(f"[WARN] Alias file not found: {path}")
        return {}

    alias = {}
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = [norm_text(x) for x in re.split(r",|\t", line)]
            parts = [x for x in parts if x]
            if len(parts) < 2:
                continue

            canonical = parts[0]
            for item in parts:
                alias[item] = canonical

    return alias


# ============================================================
# Normalization / synsets
# ============================================================

def norm_text(x: Any) -> str:
    x = str(x).strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x.strip(" .,:;!?\"'")


def get_first_synset(d: Dict[str, Any]) -> str:
    synsets = d.get("synsets", [])
    if isinstance(synsets, list) and synsets:
        return norm_text(str(synsets[0]))
    return ""


def get_image_id(item: Dict[str, Any]) -> Optional[str]:
    for key in ["image_id", "imageId", "id"]:
        if key in item:
            return str(item[key])
    return None


def get_object_id(obj: Dict[str, Any]) -> Optional[str]:
    for key in ["object_id", "objectId", "id"]:
        if key in obj:
            return str(obj[key])
    return None


def get_raw_object_names(obj: Dict[str, Any]) -> List[str]:
    names = []

    if isinstance(obj.get("names"), list):
        names.extend(obj["names"])

    if isinstance(obj.get("name"), str):
        names.append(obj["name"])

    if isinstance(obj.get("synsets"), list):
        names.extend([str(x).split(".")[0] for x in obj["synsets"] if isinstance(x, str)])

    names = [norm_text(x) for x in names if x]
    names = [x for x in names if x]
    return names


def canonical_object_name(
    obj: Dict[str, Any],
    object_aliases: Dict[str, str],
) -> Tuple[str, str, str]:
    """
    Returns canonical_name, raw_name, synset.
    """
    names = get_raw_object_names(obj)
    synset = get_first_synset(obj)

    if not names:
        return "", "", synset

    # Prefer shorter names, then aliases.
    raw_name = sorted(set(names), key=lambda x: (len(x.split()), len(x), x))[0]
    canonical = object_aliases.get(raw_name, DEFAULT_OBJECT_ALIASES.get(raw_name, raw_name))

    if canonical in STUFF_OR_VAGUE_OBJECTS:
        return "", raw_name, synset

    if not canonical or canonical in {"object", "thing", "item"}:
        return "", raw_name, synset

    return canonical, raw_name, synset


def canonical_relation(
    raw_predicate: Any,
    rel_synset: str,
    relationship_aliases: Dict[str, str],
) -> str:
    raw = norm_text(raw_predicate)
    if not raw:
        return ""

    if raw in relationship_aliases:
        raw = relationship_aliases[raw]

    raw = DEFAULT_RELATION_ALIASES.get(raw, raw)

    # Synset fallback only for a few safe cases.
    synset_map = {
        "wear.v.01": "wearing",
        "hold.v.01": "holding",
        "ride.v.01": "riding",
        "attach.v.01": "attached to",
        "cover.v.01": "covering",
    }
    if raw not in ALLOWED_RELATIONS and rel_synset in synset_map:
        raw = synset_map[rel_synset]

    return raw


def canonical_attr(a: Any) -> str:
    a = norm_text(a)
    return ATTRIBUTE_ALIASES.get(a, a)


def attr_type(attr: str) -> str:
    a = canonical_attr(attr)
    if not a or a in BAD_ATTRIBUTES:
        return ""

    if a in COLOR_ATTRIBUTES or "colored" in a or "colour" in a or "color" in a:
        return "color"

    if a in MATERIAL_ATTRIBUTES:
        return "material"

    if a in PATTERN_ATTRIBUTES:
        return "pattern"

    if a in SHAPE_ATTRIBUTES:
        return "shape"

    if a in STATE_ATTRIBUTES:
        return "state"

    return ""


def is_good_attribute(attr: str) -> bool:
    return bool(attr_type(attr))


def has_box(obj: Dict[str, Any]) -> bool:
    return all(k in obj for k in ["x", "y"]) and (
        ("w" in obj and "h" in obj) or ("width" in obj and "height" in obj)
    )


def xywh_from_obj(obj: Dict[str, Any]) -> Optional[List[float]]:
    try:
        x = float(obj["x"])
        y = float(obj["y"])
        w = float(obj.get("w", obj.get("width")))
        h = float(obj.get("h", obj.get("height")))
        if w <= 0 or h <= 0:
            return None
        return [x, y, x + w, y + h]
    except Exception:
        return None


# ============================================================
# Geometry
# ============================================================

def box_area(box: List[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def union_box(boxes: List[List[float]]) -> List[float]:
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def pad_and_clip_box(box: List[float], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = box
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    px = bw * CROP_PADDING_RATIO
    py = bh * CROP_PADDING_RATIO

    out = [
        int(max(0, math.floor(x1 - px))),
        int(max(0, math.floor(y1 - py))),
        int(min(width, math.ceil(x2 + px))),
        int(min(height, math.ceil(y2 + py))),
    ]

    if out[2] <= out[0]:
        out[2] = min(width, out[0] + 1)
    if out[3] <= out[1]:
        out[3] = min(height, out[1] + 1)

    return out


def center_inside(box: List[float], crop: List[float]) -> bool:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return crop[0] <= cx <= crop[2] and crop[1] <= cy <= crop[3]


def valid_box(box: List[float], width: int, height: int) -> bool:
    area = box_area(box)
    if area <= 0:
        return False
    ratio = area / float(max(width * height, 1))
    return MIN_BOX_AREA_RATIO <= ratio <= MAX_BOX_AREA_RATIO


def valid_crop(crop: List[float], width: int, height: int) -> bool:
    area = box_area(crop)
    if area <= 0:
        return False
    ratio = area / float(max(width * height, 1))
    return MIN_CROP_AREA_RATIO <= ratio <= MAX_CROP_AREA_RATIO


# ============================================================
# Image path
# ============================================================

def resolve_image_path(image_id: str, image_dir: str) -> Optional[str]:
    base = Path(image_dir)

    candidates = [
        base / f"{image_id}.jpg",
        base / f"{image_id}.jpeg",
        base / f"{image_id}.png",
        base / "VG_100K" / f"{image_id}.jpg",
        base / "VG_100K_2" / f"{image_id}.jpg",
    ]

    for p in candidates:
        if p.exists():
            return str(p)

    return None


# ============================================================
# Streaming collection for selected images
# ============================================================

def collect_objects(
    objects_path: str,
    max_images: int,
    object_aliases: Dict[str, str],
    require_image_size: bool = False,
) -> Tuple[List[str], Dict[str, Dict[str, Dict[str, Any]]], Counter]:
    image_ids = []
    objects_by_image = {}
    object_counts = Counter()

    for item in iter_json_array(objects_path):
        image_id = get_image_id(item)
        if image_id is None:
            continue

        if len(image_ids) >= max_images:
            break

        raw_objects = item.get("objects", [])
        if not isinstance(raw_objects, list):
            continue

        obj_map = {}
        for obj in raw_objects:
            if not isinstance(obj, dict):
                continue

            oid = get_object_id(obj)
            if not oid or not has_box(obj):
                continue

            name, raw_name, synset = canonical_object_name(obj, object_aliases)
            if not name:
                continue

            bbox = xywh_from_obj(obj)
            if bbox is None:
                continue

            obj_map[oid] = {
                "object_id": oid,
                "name": name,
                "raw_name": raw_name,
                "synset": synset,
                "bbox": bbox,
                "attributes": [],
                "raw": obj,
            }
            object_counts[name] += 1

        if obj_map:
            image_ids.append(image_id)
            objects_by_image[image_id] = obj_map

    return image_ids, objects_by_image, object_counts


def attach_attributes(
    attributes_path: str,
    selected_ids: Set[str],
    objects_by_image: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Counter]:
    stats = {
        "alignment": Counter(),
        "attribute_counts_raw": Counter(),
        "attribute_counts_kept": Counter(),
        "attribute_type_counts": Counter(),
        "attribute_by_object_type": defaultdict(Counter),
        "attribute_examples": defaultdict(list),
    }

    for item in iter_json_array(attributes_path):
        image_id = get_image_id(item)
        if image_id not in selected_ids:
            continue

        entries = item.get("attributes", [])
        if not isinstance(entries, list):
            continue

        obj_map = objects_by_image.get(image_id, {})

        for entry in entries:
            if not isinstance(entry, dict):
                continue

            oid = get_object_id(entry)
            if not oid:
                continue

            if oid in obj_map:
                stats["alignment"]["matched_object_id"] += 1
            else:
                stats["alignment"]["unmatched_object_id"] += 1
                continue

            attrs = entry.get("attributes", [])
            if isinstance(attrs, str):
                attrs = [attrs]
            if not isinstance(attrs, list):
                continue

            for raw_attr in attrs:
                a = canonical_attr(raw_attr)
                stats["attribute_counts_raw"][a] += 1

                t = attr_type(a)
                if not t:
                    continue

                obj_map[oid]["attributes"].append({
                    "type": t,
                    "value": a,
                })

                obj_name = obj_map[oid]["name"]
                stats["attribute_counts_kept"][a] += 1
                stats["attribute_type_counts"][t] += 1
                stats["attribute_by_object_type"][a][obj_name] += 1

                if len(stats["attribute_examples"][a]) < 10:
                    stats["attribute_examples"][a].append({
                        "image_id": image_id,
                        "object_id": oid,
                        "object": obj_name,
                    })

    for image_id in objects_by_image:
        for oid in objects_by_image[image_id]:
            attrs = objects_by_image[image_id][oid]["attributes"]
            uniq = sorted({(x["type"], x["value"]) for x in attrs})
            objects_by_image[image_id][oid]["attributes"] = [
                {"type": t, "value": v} for t, v in uniq
            ]

    return stats


def collect_relationships(
    relationships_path: str,
    selected_ids: Set[str],
    objects_by_image: Dict[str, Dict[str, Dict[str, Any]]],
    relationship_aliases: Dict[str, str],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Counter]]:
    rels_by_image = defaultdict(list)

    stats = {
        "alignment": Counter(),
        "relation_counts_raw": Counter(),
        "relation_counts_canonical": Counter(),
        "relation_counts_kept": Counter(),
        "relation_synset_counts": Counter(),
        "relation_examples": defaultdict(list),
        "rejected_relation_counts": Counter(),
    }

    for item in iter_json_array(relationships_path):
        image_id = get_image_id(item)
        if image_id not in selected_ids:
            continue

        rels = item.get("relationships", [])
        if not isinstance(rels, list):
            continue

        obj_map = objects_by_image.get(image_id, {})

        for rel in rels:
            if not isinstance(rel, dict):
                continue

            raw_pred = norm_text(rel.get("predicate", ""))
            rel_synset = get_first_synset(rel)
            pred = canonical_relation(raw_pred, rel_synset, relationship_aliases)

            stats["relation_counts_raw"][raw_pred] += 1
            stats["relation_counts_canonical"][pred] += 1
            if rel_synset:
                stats["relation_synset_counts"][rel_synset] += 1

            subj = rel.get("subject")
            obj = rel.get("object")
            if not isinstance(subj, dict) or not isinstance(obj, dict):
                continue

            sid = get_object_id(subj)
            oid = get_object_id(obj)

            if sid in obj_map and oid in obj_map:
                stats["alignment"]["matched_subject_object_ids"] += 1
            else:
                stats["alignment"]["unmatched_subject_object_ids"] += 1
                continue

            if not pred or pred.endswith("_reverse"):
                stats["rejected_relation_counts"][raw_pred] += 1
                continue

            if rel_synset in BAD_RELATION_SYNSETS:
                stats["rejected_relation_counts"][f"bad_synset:{rel_synset}"] += 1
                continue

            if pred not in ALLOWED_RELATIONS:
                stats["rejected_relation_counts"][pred] += 1
                continue

            sname = obj_map[sid]["name"]
            oname = obj_map[oid]["name"]

            if sid == oid or sname == oname:
                stats["rejected_relation_counts"]["same_subject_object"] += 1
                continue

            out = {
                "relationship_id": str(rel.get("relationship_id", "")),
                "subject_id": sid,
                "object_id": oid,
                "predicate": pred,
                "raw_predicate": raw_pred,
                "predicate_synset": rel_synset,
                "subject_name": sname,
                "object_name": oname,
                "subject_raw_name": obj_map[sid]["raw_name"],
                "object_raw_name": obj_map[oid]["raw_name"],
                "subject_synset": obj_map[sid]["synset"],
                "object_synset": obj_map[oid]["synset"],
            }
            rels_by_image[image_id].append(out)

            stats["relation_counts_kept"][pred] += 1
            if len(stats["relation_examples"][pred]) < 20:
                stats["relation_examples"][pred].append({
                    "image_id": image_id,
                    "subject": sname,
                    "predicate": pred,
                    "object": oname,
                    "raw_predicate": raw_pred,
                    "predicate_synset": rel_synset,
                })

    return dict(rels_by_image), stats


# ============================================================
# Motif construction
# ============================================================

def make_graph(
    objects: List[str],
    attributes: List[Tuple[str, str, str]],
    relations: List[Tuple[str, str, str]],
    counts: List[Tuple[str, int]],
) -> Dict[str, Any]:
    return {
        "objects": sorted(set(objects)),
        "attributes": [list(x) for x in sorted(set(attributes))],
        "relations": [list(x) for x in sorted(set(relations))],
        "counts": [[name, int(count)] for name, count in sorted(set(counts))],
    }


def build_crop_for_ids(
    object_ids: List[str],
    obj_map: Dict[str, Dict[str, Any]],
    width: int,
    height: int,
) -> Optional[List[int]]:
    boxes = [obj_map[x]["bbox"] for x in object_ids if x in obj_map]
    if not boxes:
        return None
    crop = pad_and_clip_box(union_box(boxes), width, height)
    if not valid_crop(crop, width, height):
        return None
    return crop


def count_extra_objects(crop: List[int], motif_ids: Set[str], obj_map: Dict[str, Dict[str, Any]]) -> int:
    n = 0
    for oid, obj in obj_map.items():
        if oid in motif_ids:
            continue
        if center_inside(obj["bbox"], crop):
            n += 1
    return n


def crop_context_ok(crop: List[int], motif_ids: Set[str], obj_map: Dict[str, Dict[str, Any]]) -> bool:
    return count_extra_objects(crop, motif_ids, obj_map) <= MAX_EXTRA_OBJECTS_IN_CROP


def enumerate_attribute_motifs(
    image_id: str,
    obj_map: Dict[str, Dict[str, Any]],
    width: int,
    height: int,
) -> List[Dict[str, Any]]:
    motifs = []

    for oid, obj in obj_map.items():
        if not valid_box(obj["bbox"], width, height):
            continue

        for attr in obj["attributes"]:
            t = attr["type"]
            v = attr["value"]

            # Avoid redundant cases like object_name="white top", attr=white.
            if t == "color" and object_name_contains_attribute(obj["name"], v):
                continue

            crop = build_crop_for_ids([oid], obj_map, width, height)
            if crop is None or not crop_context_ok(crop, {oid}, obj_map):
                continue

            graph = make_graph(
                objects=[obj["name"]],
                attributes=[(obj["name"], t, v)],
                relations=[],
                counts=[],
            )

            motifs.append({
                "motif_type": "attribute",
                "object_ids": [oid],
                "crop_box_xyxy": crop,
                "target_graph": graph,
                "source": {
                    "object_id": oid,
                    "object_name": obj["name"],
                    "object_raw_name": obj["raw_name"],
                    "object_synset": obj["synset"],
                    "attribute_type": t,
                    "attribute_value": v,
                },
            })

    return motifs


def enumerate_relation_motifs(
    image_id: str,
    obj_map: Dict[str, Dict[str, Any]],
    rels: List[Dict[str, Any]],
    width: int,
    height: int,
) -> List[Dict[str, Any]]:
    motifs = []

    for rel in rels:
        sid = rel["subject_id"]
        oid = rel["object_id"]

        crop = build_crop_for_ids([sid, oid], obj_map, width, height)
        if crop is None or not crop_context_ok(crop, {sid, oid}, obj_map):
            continue

        graph = make_graph(
            objects=[rel["subject_name"], rel["object_name"]],
            attributes=[],
            relations=[(rel["subject_name"], rel["predicate"], rel["object_name"])],
            counts=[],
        )

        motifs.append({
            "motif_type": "relation",
            "object_ids": [sid, oid],
            "crop_box_xyxy": crop,
            "target_graph": graph,
            "source": dict(rel),
        })

    return motifs


def enumerate_relation_attribute_motifs(
    image_id: str,
    obj_map: Dict[str, Dict[str, Any]],
    rels: List[Dict[str, Any]],
    width: int,
    height: int,
) -> List[Dict[str, Any]]:
    motifs = []

    for rel in rels:
        sid = rel["subject_id"]
        oid = rel["object_id"]
        target_obj = obj_map[oid]

        for attr in target_obj.get("attributes", []):
            t = attr["type"]
            v = attr["value"]

            if t == "color" and object_name_contains_attribute(rel["object_name"], v):
                continue

            crop = build_crop_for_ids([sid, oid], obj_map, width, height)
            if crop is None or not crop_context_ok(crop, {sid, oid}, obj_map):
                continue

            graph = make_graph(
                objects=[rel["subject_name"], rel["object_name"]],
                attributes=[(rel["object_name"], t, v)],
                relations=[(rel["subject_name"], rel["predicate"], rel["object_name"])],
                counts=[],
            )

            motifs.append({
                "motif_type": "relation_attribute",
                "object_ids": [sid, oid],
                "crop_box_xyxy": crop,
                "target_graph": graph,
                "source": {
                    **dict(rel),
                    "attribute_object_id": oid,
                    "attribute_object_name": rel["object_name"],
                    "attribute_type": t,
                    "attribute_value": v,
                },
            })

    return motifs


def enumerate_multi_relation_anchor_motifs(
    image_id: str,
    obj_map: Dict[str, Dict[str, Any]],
    rels: List[Dict[str, Any]],
    width: int,
    height: int,
) -> List[Dict[str, Any]]:
    motifs = []
    by_subject = defaultdict(list)

    for rel in rels:
        by_subject[rel["subject_id"]].append(rel)

    for sid, subject_rels in by_subject.items():
        if len(subject_rels) < 2:
            continue

        subject_rels = subject_rels[:4]

        for i in range(len(subject_rels)):
            for j in range(i + 1, len(subject_rels)):
                r1, r2 = subject_rels[i], subject_rels[j]
                ids = list(dict.fromkeys([sid, r1["object_id"], r2["object_id"]]))

                if len(ids) > MAX_OBJECTS_PER_TARGET:
                    continue

                crop = build_crop_for_ids(ids, obj_map, width, height)
                if crop is None or not crop_context_ok(crop, set(ids), obj_map):
                    continue

                attrs = []
                for oid in [r1["object_id"], r2["object_id"]]:
                    obj_name = obj_map[oid]["name"]
                    for a in obj_map[oid].get("attributes", []):
                        if a["type"] == "color" and object_name_contains_attribute(obj_name, a["value"]):
                            continue
                        attrs.append((obj_name, a["type"], a["value"]))
                        break

                graph = make_graph(
                    objects=[obj_map[x]["name"] for x in ids],
                    attributes=attrs[:MAX_ATTRIBUTES_PER_TARGET],
                    relations=[
                        (r1["subject_name"], r1["predicate"], r1["object_name"]),
                        (r2["subject_name"], r2["predicate"], r2["object_name"]),
                    ],
                    counts=[],
                )

                motifs.append({
                    "motif_type": "multi_relation_anchor",
                    "object_ids": ids,
                    "crop_box_xyxy": crop,
                    "target_graph": graph,
                    "source": {
                        "anchor_id": sid,
                        "anchor_name": obj_map[sid]["name"],
                        "relations": [dict(r1), dict(r2)],
                    },
                })

    return motifs


def enumerate_count_motifs(
    image_id: str,
    obj_map: Dict[str, Dict[str, Any]],
    width: int,
    height: int,
) -> List[Dict[str, Any]]:
    motifs = []

    by_name = defaultdict(list)
    for oid, obj in obj_map.items():
        by_name[obj["name"]].append(oid)

    for name, ids in by_name.items():
        if is_bad_count_name(name):
            continue

        if not (MIN_COUNT <= len(ids) <= MAX_COUNT):
            continue

        crop = build_crop_for_ids(ids, obj_map, width, height)
        if crop is None or not crop_context_ok(crop, set(ids), obj_map):
            continue

        indexed = [f"{name}#{i + 1}" for i in range(len(ids))]

        graph = make_graph(
            objects=indexed,
            attributes=[],
            relations=[],
            counts=[(name, len(ids))],
        )

        motifs.append({
            "motif_type": "count",
            "object_ids": ids,
            "crop_box_xyxy": crop,
            "target_graph": graph,
            "source": {
                "object_name": name,
                "count": len(ids),
                "object_ids": ids,
            },
        })

    return motifs


# ============================================================
# Crop / visualization
# ============================================================

def save_crop(image_path: str, crop_box: List[int], out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        return True

    try:
        img = Image.open(image_path).convert("RGB")
        crop = img.crop(tuple(crop_box))
        crop.save(out_path, quality=95)
        return True
    except Exception:
        return False


def draw_visualization(
    image_path: str,
    motif: Dict[str, Any],
    obj_map: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> bool:
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return False

    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    crop = motif["crop_box_xyxy"]
    draw.rectangle(crop, outline=(255, 0, 0), width=4)

    for oid in motif["object_ids"]:
        if oid not in obj_map:
            continue
        box = obj_map[oid]["bbox"]
        label = f"{obj_map[oid]['name']} ({obj_map[oid].get('synset', '')})"
        draw.rectangle(box, outline=(0, 255, 0), width=3)
        draw.text((box[0], max(0, box[1] - 18)), label[:80], fill=(0, 255, 0), font=font)

    title = motif["motif_type"]
    draw.rectangle([0, 0, img.width, 28], fill=(255, 255, 255))
    draw.text((5, 5), title, fill=(255, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)
    return True


# ============================================================
# Record
# ============================================================

def build_prompt() -> str:
    return (
        "<image>\n"
        "Describe this visual region in one factual sentence.\n"
        "Mention only visible objects, attributes, and relations in the region.\n"
        "Do not list tuples or use bullet points.\n"
        "Return exactly:\n"
        "<caption>...</caption>"
    )


def make_record(image_id: str, crop_path: str, motif: Dict[str, Any], idx: int) -> Dict[str, Any]:
    return {
        "data_source": f"vg_graph_caption/{motif['motif_type']}",
        "prompt": [
            {
                "role": "user",
                "content": build_prompt(),
            }
        ],
        "images": [crop_path],
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "target_graph": motif["target_graph"],
                "reward_types": {
                    "caption_parse": "controlled_caption_parser",
                    "graph": "active_weighted_graph_f1",
                    "hallucination": "extra_tuple_penalty",
                    "caption_validity": "one_sentence_caption",
                },
            },
        },
        "ability": "graph_verifiable_region_captioning",
        "extra_info": {
            "question_id": f"vg_graph_caption_{idx:08d}",
            "source_dataset": "visual_genome_v1.2",
            "image_id": image_id,
            "motif_type": motif["motif_type"],
            "crop_box_xyxy": motif["crop_box_xyxy"],
            "source_object_ids": motif["object_ids"],
            "source": motif["source"],
            "num_target_objects": len(motif["target_graph"]["objects"]),
            "num_target_attributes": len(motif["target_graph"]["attributes"]),
            "num_target_relations": len(motif["target_graph"]["relations"]),
            "num_target_counts": len(motif["target_graph"]["counts"]),
        },
    }


# ============================================================
# Audit command
# ============================================================

def run_audit(args: argparse.Namespace) -> None:
    object_aliases = DEFAULT_OBJECT_ALIASES.copy()
    object_aliases.update(load_alias_file(args.object_aliases))

    relationship_aliases = {}
    relationship_aliases.update(load_alias_file(args.relationship_aliases))

    print("[1/3] Streaming objects")
    image_ids, objects_by_image, object_counts = collect_objects(
        objects_path=args.objects,
        max_images=args.max_images,
        object_aliases=object_aliases,
    )

    selected = set(image_ids)
    print(f"Selected images with clean objects: {len(image_ids)}")

    print("[2/3] Streaming attributes")
    attr_stats = attach_attributes(args.attributes, selected, objects_by_image)

    print("[3/3] Streaming relationships")
    rels_by_image, rel_stats = collect_relationships(
        args.relationships,
        selected,
        objects_by_image,
        relationship_aliases,
    )

    motif_counts = Counter()
    examples = defaultdict(list)

    for image_id, obj_map in objects_by_image.items():
        attr_count = 0
        for oid, obj in obj_map.items():
            attr_count += len(obj.get("attributes", []))
            if obj.get("attributes") and len(examples["attribute"]) < 20:
                examples["attribute"].append({
                    "image_id": image_id,
                    "object": obj["name"],
                    "raw_name": obj["raw_name"],
                    "synset": obj["synset"],
                    "attributes": obj["attributes"][:5],
                })

        motif_counts["attribute"] += attr_count

        rels = rels_by_image.get(image_id, [])
        motif_counts["relation"] += len(rels)

        by_subject = defaultdict(list)
        for rel in rels:
            by_subject[rel["subject_id"]].append(rel)

            oid = rel["object_id"]
            target_attrs = obj_map[oid].get("attributes", [])
            motif_counts["relation_attribute"] += len(target_attrs)

            if len(examples["relation"]) < 20:
                examples["relation"].append({
                    "image_id": image_id,
                    "subject": rel["subject_name"],
                    "predicate": rel["predicate"],
                    "object": rel["object_name"],
                    "raw_predicate": rel["raw_predicate"],
                    "predicate_synset": rel["predicate_synset"],
                })

            if target_attrs and len(examples["relation_attribute"]) < 20:
                examples["relation_attribute"].append({
                    "image_id": image_id,
                    "subject": rel["subject_name"],
                    "predicate": rel["predicate"],
                    "object": rel["object_name"],
                    "object_attrs": target_attrs[:5],
                })

        for sid, rs in by_subject.items():
            if len(rs) >= 2:
                motif_counts["multi_relation_anchor"] += len(rs) * (len(rs) - 1) // 2

        by_name = defaultdict(list)
        for oid, obj in obj_map.items():
            by_name[obj["name"]].append(oid)
        for name, ids in by_name.items():
            if MIN_COUNT <= len(ids) <= MAX_COUNT:
                motif_counts["count"] += 1
                if len(examples["count"]) < 20:
                    examples["count"].append({
                        "image_id": image_id,
                        "name": name,
                        "count": len(ids),
                    })

    attr_by_type = {}
    for a, c in attr_stats["attribute_counts_raw"].most_common():
        t = attr_type(a)
        attr_by_type[a] = {
            "count": c,
            "type": t if t else "untyped_or_rejected",
            "top_object_types": attr_stats["attribute_by_object_type"][a].most_common(20),
            "examples": attr_stats["attribute_examples"].get(a, []),
        }

    out = {
        "num_selected_images": len(image_ids),
        "alignment": {
            "attributes": dict(attr_stats["alignment"]),
            "relationships": dict(rel_stats["alignment"]),
        },
        "motif_counts": dict(motif_counts),
        "top_objects": object_counts.most_common(100),
        "attribute_audit": {
            "top_raw_attributes": attr_stats["attribute_counts_raw"].most_common(300),
            "top_kept_attributes": attr_stats["attribute_counts_kept"].most_common(300),
            "attribute_type_counts": dict(attr_stats["attribute_type_counts"]),
            "attributes_by_type_guess": attr_by_type,
        },
        "relation_audit": {
            "top_raw_relations": rel_stats["relation_counts_raw"].most_common(300),
            "top_canonical_relations": rel_stats["relation_counts_canonical"].most_common(300),
            "top_kept_relations": rel_stats["relation_counts_kept"].most_common(300),
            "top_relation_synsets": rel_stats["relation_synset_counts"].most_common(300),
            "top_rejected_relations": rel_stats["rejected_relation_counts"].most_common(300),
            "examples_by_relation": dict(rel_stats["relation_examples"]),
        },
        "examples": dict(examples),
        "active_filters": {
            "bad_attributes": sorted(BAD_ATTRIBUTES),
            "color_attributes": sorted(COLOR_ATTRIBUTES),
            "material_attributes": sorted(MATERIAL_ATTRIBUTES),
            "pattern_attributes": sorted(PATTERN_ATTRIBUTES),
            "shape_attributes": sorted(SHAPE_ATTRIBUTES),
            "state_attributes": sorted(STATE_ATTRIBUTES),
            "allowed_relations": sorted(ALLOWED_RELATIONS),
            "bad_relation_synsets": sorted(BAD_RELATION_SYNSETS),
        },
    }

    write_json(out, Path(args.out))

    print(f"\nWrote audit: {args.out}")
    print("\nAlignment:")
    print(json.dumps(out["alignment"], indent=2))

    print("\nMotif counts:")
    for k, v in motif_counts.most_common():
        print(f"  {k}: {v}")

    print("\nAttribute type counts:")
    for k, v in attr_stats["attribute_type_counts"].most_common():
        print(f"  {k}: {v}")

    print("\nTop kept relations:")
    for k, v in rel_stats["relation_counts_kept"].most_common(30):
        print(f"  {k}: {v}")

    print("\nTop rejected relations:")
    for k, v in rel_stats["rejected_relation_counts"].most_common(30):
        print(f"  {k}: {v}")


# ============================================================
# Build command
# ============================================================

def run_build(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)

    out_dir = Path(args.out_dir)
    image_out_dir = out_dir / "images"
    vis_dir = out_dir / "visualize"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    object_aliases = DEFAULT_OBJECT_ALIASES.copy()
    object_aliases.update(load_alias_file(args.object_aliases))

    relationship_aliases = {}
    relationship_aliases.update(load_alias_file(args.relationship_aliases))

    print("[1/3] Streaming objects")
    image_ids, objects_by_image, object_counts = collect_objects(
        args.objects,
        args.max_images,
        object_aliases,
    )
    selected = set(image_ids)

    print("[2/3] Streaming attributes")
    attr_stats = attach_attributes(args.attributes, selected, objects_by_image)

    print("[3/3] Streaming relationships")
    rels_by_image, rel_stats = collect_relationships(
        args.relationships,
        selected,
        objects_by_image,
        relationship_aliases,
    )

    rng.shuffle(image_ids)

    records = []
    audit = {
        "motif_counts": Counter(),
        "record_counts": Counter(),
        "skip_counts": Counter(),
        "examples": [],
        "alignment": {
            "attributes": dict(attr_stats["alignment"]),
            "relationships": dict(rel_stats["alignment"]),
        },
    }

    for image_id in image_ids:
        if len(records) >= args.max_records:
            break

        image_path = resolve_image_path(image_id, args.image_dir)
        if image_path is None:
            audit["skip_counts"]["missing_image"] += 1
            continue

        try:
            with Image.open(image_path) as im:
                width, height = im.size
        except Exception:
            audit["skip_counts"]["cannot_open_image"] += 1
            continue

        obj_map = objects_by_image.get(image_id, {})
        rels = rels_by_image.get(image_id, [])

        motifs = []
        motifs.extend(enumerate_attribute_motifs(image_id, obj_map, width, height))
        motifs.extend(enumerate_relation_motifs(image_id, obj_map, rels, width, height))
        motifs.extend(enumerate_relation_attribute_motifs(image_id, obj_map, rels, width, height))
        motifs.extend(enumerate_multi_relation_anchor_motifs(image_id, obj_map, rels, width, height))
        motifs.extend(enumerate_count_motifs(image_id, obj_map, width, height))

        if not motifs:
            audit["skip_counts"]["no_motifs"] += 1
            continue

        rng.shuffle(motifs)

        per_image = 0
        per_family = Counter()
        global_family = Counter(r["extra_info"]["motif_type"] for r in records)

        for motif in motifs:
            if len(records) >= args.max_records:
                break

            motif_type = motif["motif_type"]

            # v1: skip count unless explicitly added to TARGET_FAMILY_QUOTAS.
            if motif_type not in TARGET_FAMILY_QUOTAS:
                continue

            if global_family[motif_type] >= TARGET_FAMILY_QUOTAS[motif_type]:
                continue

            if per_image >= MAX_MOTIFS_PER_IMAGE:
                break

            if per_family[motif_type] >= MAX_MOTIFS_PER_IMAGE_PER_FAMILY:
                continue

            idx = len(records)
            crop_name = f"{int(image_id):08d}_{motif['motif_type']}_{idx:08d}.jpg"
            crop_path = image_out_dir / crop_name

            if not save_crop(image_path, motif["crop_box_xyxy"], crop_path):
                audit["skip_counts"]["save_crop_failed"] += 1
                continue

            record = make_record(image_id, str(crop_path), motif, idx)
            records.append(record)

            global_family[motif_type] += 1
            per_image += 1
            per_family[motif["motif_type"]] += 1
            audit["motif_counts"][motif["motif_type"]] += 1
            audit["record_counts"][record["data_source"]] += 1

            if len(audit["examples"]) < 50:
                audit["examples"].append({
                    "image_id": image_id,
                    "motif_type": motif["motif_type"],
                    "crop_path": str(crop_path),
                    "target_graph": motif["target_graph"],
                    "source": motif["source"],
                })

            if args.visualize_n > 0 and idx < args.visualize_n:
                vis_path = vis_dir / f"{idx:04d}_{int(image_id):08d}_{motif['motif_type']}.jpg"
                draw_visualization(image_path, motif, obj_map, vis_path)
            
            if all(
                Counter(r["extra_info"]["motif_type"] for r in records)[k] >= v
                for k, v in TARGET_FAMILY_QUOTAS.items()
            ):
                break

            if all(global_family[k] >= v for k, v in TARGET_FAMILY_QUOTAS.items()):
                break

    rng.shuffle(records)
    for idx, r in enumerate(records):
        qid = f"vg_graph_caption_{idx:08d}"
        r["extra_info"]["question_id"] = qid

    train_path = out_dir / "train.jsonl"
    latest_path = out_dir / "latest.jsonl"
    audit_path = out_dir / "audit.json"

    write_jsonl(records, train_path)
    write_jsonl(records, latest_path)

    audit_json = {
        "num_records": len(records),
        "paths": {
            "train": str(train_path),
            "latest": str(latest_path),
            "images": str(image_out_dir),
            "visualize": str(vis_dir),
        },
        "alignment": audit["alignment"],
        "motif_counts": dict(audit["motif_counts"]),
        "record_counts": dict(audit["record_counts"]),
        "skip_counts": dict(audit["skip_counts"]),
        "examples": audit["examples"],
        "filters": {
            "allowed_relations": sorted(ALLOWED_RELATIONS),
            "bad_attributes": sorted(BAD_ATTRIBUTES),
            "bad_relation_synsets": sorted(BAD_RELATION_SYNSETS),
        },
    }
    write_json(audit_json, audit_path)

    print(f"\nDone.")
    print(f"Records: {len(records)}")
    print(f"Train: {train_path}")
    print(f"Audit: {audit_path}")
    print(f"Images: {image_out_dir}")
    print(f"Visualize: {vis_dir}")

    print("\nMotif counts:")
    for k, v in audit["motif_counts"].most_common():
        print(f"  {k}: {v}")

    print("\nSkip counts:")
    for k, v in audit["skip_counts"].most_common():
        print(f"  {k}: {v}")


# ============================================================
# CLI
# ============================================================

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--objects", required=True)
    common.add_argument("--attributes", required=True)
    common.add_argument("--relationships", required=True)
    common.add_argument("--object_aliases", default=None)
    common.add_argument("--relationship_aliases", default=None)
    common.add_argument("--max_images", type=int, default=5000)

    p_audit = sub.add_parser("audit", parents=[common])
    p_audit.add_argument("--out", required=True)

    p_build = sub.add_parser("build", parents=[common])
    p_build.add_argument("--image_dir", required=True)
    p_build.add_argument("--out_dir", required=True)
    p_build.add_argument("--max_records", type=int, default=1000)
    p_build.add_argument("--visualize_n", type=int, default=100)
    p_build.add_argument("--seed", type=int, default=SEED)

    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.cmd == "audit":
        run_audit(args)
    elif args.cmd == "build":
        run_build(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()


'''
python data_factory/audit_vg_graph_caption_html.py \
  --jsonl /scratch/eunwooim/rlpt/src/visualize/caption/vg_graph_caption_debug_v14_balanced/train.jsonl \
  --out_html /scratch/eunwooim/rlpt/src/visualize/caption/vg_graph_caption_debug_v14_balanced/audit.html \
  --per_family 50
'''