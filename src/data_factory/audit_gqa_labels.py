import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict


SCENE_GRAPH_PATH = "/scratch/eunwooim/train_sceneGraphs.json"
OUT_PATH = "/scratch/eunwooim/rlpt/src/data_factory/gqa_label_vocab.json"

MIN_ATTR_FREQ = 100
MIN_ATTR_OBJECT_TYPES = 3

MIN_REL_FREQ = 100
MIN_REL_PAIR_TYPES = 3

TOP_PRINT = 100


STUFF_OBJECTS = {
    "water", "sky", "grass", "road", "street", "sidewalk", "floor", "ground",
    "wall", "ceiling", "field", "snow", "sand", "sea", "ocean", "river",
    "lake", "background", "room", "area", "place", "window", "building",
    "cloud", "clouds", "tree leaves", "leaves"
}


def normalize(x: Any) -> str:
    return str(x).strip().lower()


def load_scene_graphs(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): v for k, v in raw.items()}


def main() -> None:
    scene_graphs = load_scene_graphs(SCENE_GRAPH_PATH)

    attr_counts = Counter()
    attr_object_types = defaultdict(Counter)

    rel_counts = Counter()
    rel_pair_types = defaultdict(Counter)
    rel_subject_types = defaultdict(Counter)
    rel_object_types = defaultdict(Counter)

    object_name_counts = Counter()

    for image_id, sg in scene_graphs.items():
        objects = sg.get("objects", {})
        if not isinstance(objects, dict):
            continue

        id_to_name = {}
        for object_id, obj in objects.items():
            if not isinstance(obj, dict):
                continue

            name = normalize(obj.get("name", ""))
            if not name or name in STUFF_OBJECTS:
                continue

            id_to_name[str(object_id)] = name
            object_name_counts[name] += 1

        for object_id, obj in objects.items():
            if not isinstance(obj, dict):
                continue

            subject_name = id_to_name.get(str(object_id))
            if not subject_name:
                continue

            attrs = obj.get("attributes", [])
            if isinstance(attrs, list):
                for attr in attrs:
                    attr = normalize(attr)
                    if not attr:
                        continue
                    attr_counts[attr] += 1
                    attr_object_types[attr][subject_name] += 1

            rels = obj.get("relations", [])
            if isinstance(rels, list):
                for rel in rels:
                    if not isinstance(rel, dict):
                        continue

                    rel_name = normalize(rel.get("name", ""))
                    target_id = str(rel.get("object", ""))

                    object_name = id_to_name.get(target_id)

                    if not rel_name or not object_name:
                        continue

                    rel_counts[rel_name] += 1
                    rel_pair_types[rel_name][f"{subject_name}->{object_name}"] += 1
                    rel_subject_types[rel_name][subject_name] += 1
                    rel_object_types[rel_name][object_name] += 1

    selected_attributes = {
        attr: {
            "count": count,
            "num_object_types": len(attr_object_types[attr]),
            "top_object_types": attr_object_types[attr].most_common(20),
        }
        for attr, count in attr_counts.items()
        if count >= MIN_ATTR_FREQ and len(attr_object_types[attr]) >= MIN_ATTR_OBJECT_TYPES
    }

    selected_relations = {
        rel: {
            "count": count,
            "num_pair_types": len(rel_pair_types[rel]),
            "num_subject_types": len(rel_subject_types[rel]),
            "num_object_types": len(rel_object_types[rel]),
            "top_pair_types": rel_pair_types[rel].most_common(20),
            "top_subject_types": rel_subject_types[rel].most_common(20),
            "top_object_types": rel_object_types[rel].most_common(20),
        }
        for rel, count in rel_counts.items()
        if count >= MIN_REL_FREQ and len(rel_pair_types[rel]) >= MIN_REL_PAIR_TYPES
    }

    out = {
        "config": {
            "scene_graph_path": SCENE_GRAPH_PATH,
            "min_attr_freq": MIN_ATTR_FREQ,
            "min_attr_object_types": MIN_ATTR_OBJECT_TYPES,
            "min_rel_freq": MIN_REL_FREQ,
            "min_rel_pair_types": MIN_REL_PAIR_TYPES,
        },
        "summary": {
            "num_images": len(scene_graphs),
            "num_object_names": len(object_name_counts),
            "num_attributes_raw": len(attr_counts),
            "num_relations_raw": len(rel_counts),
            "num_attributes_selected": len(selected_attributes),
            "num_relations_selected": len(selected_relations),
        },
        "object_names": {
            name: count for name, count in object_name_counts.most_common()
        },
        "attributes_raw": {
            attr: {
                "count": count,
                "num_object_types": len(attr_object_types[attr]),
                "top_object_types": attr_object_types[attr].most_common(20),
            }
            for attr, count in attr_counts.most_common()
        },
        "relations_raw": {
            rel: {
                "count": count,
                "num_pair_types": len(rel_pair_types[rel]),
                "num_subject_types": len(rel_subject_types[rel]),
                "num_object_types": len(rel_object_types[rel]),
                "top_pair_types": rel_pair_types[rel].most_common(20),
                "top_subject_types": rel_subject_types[rel].most_common(20),
                "top_object_types": rel_object_types[rel].most_common(20),
            }
            for rel, count in rel_counts.most_common()
        },
        "selected_attributes": selected_attributes,
        "selected_relations": selected_relations,
    }

    out_path = Path(OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Wrote: {out_path}")
    print()
    print("Summary:")
    print(json.dumps(out["summary"], indent=2))
    print()
    print(f"Top {TOP_PRINT} raw attributes:")
    for attr, count in attr_counts.most_common(TOP_PRINT):
        print(f"  {attr}: {count} | object_types={len(attr_object_types[attr])}")

    print()
    print(f"Top {TOP_PRINT} raw relations:")
    for rel, count in rel_counts.most_common(TOP_PRINT):
        print(f"  {rel}: {count} | pair_types={len(rel_pair_types[rel])}")


if __name__ == "__main__":
    main()