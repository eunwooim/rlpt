"""Streaming smoke test for gqa.py.

Validates the extract -> generate -> make_record pipeline on ~10 records
without downloading the full ~25GB lmms-lab/GQA image config.

Walks the streaming IterableDataset, intersects image_ids with scene_graphs,
runs each image through the same per-image logic used in generate_dataset(),
and writes a small JSONL plus a summary.
"""

import os
import json
import random
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HOME", "/scratch/sghos104/rlpt/data/hf_cache/")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gqa  # noqa: E402  (sets HF_HOME via its own setdefault)
from datasets import load_dataset  # noqa: E402


TARGET_RECORDS = 10
MAX_IMAGES_TO_SCAN = 200
SMOKE_OUT = Path("/scratch/sghos104/rlpt/data/smoke/smoke.jsonl")
SMOKE_IMG_DIR = "/scratch/sghos104/rlpt/data/smoke/images/"


def main() -> int:
    rng = random.Random(42)
    SMOKE_OUT.parent.mkdir(parents=True, exist_ok=True)

    print(f"[smoke] loading scene graphs: {gqa.DEFAULT_SCENE_GRAPH_PATH}")
    scene_graphs = gqa.load_scene_graphs(gqa.DEFAULT_SCENE_GRAPH_PATH)
    print(f"[smoke] scene graphs loaded: {len(scene_graphs)}")

    print("[smoke] opening streaming dataset lmms-lab/GQA / train_balanced_images")
    ds = load_dataset("lmms-lab/GQA", "train_balanced_images", split="train", streaming=True)

    records = []
    scanned = 0
    skip_stats = Counter()

    for row in ds:
        if scanned >= MAX_IMAGES_TO_SCAN or len(records) >= TARGET_RECORDS:
            break
        scanned += 1

        img_id = str(row.get("id") or row.get("imageId") or "")
        if not img_id or img_id not in scene_graphs:
            skip_stats["not_in_scene_graphs"] += 1
            continue

        image_obj = row.get("image")
        if image_obj is None:
            skip_stats["no_image"] += 1
            continue

        width, height = image_obj.size
        image_path = gqa.save_image(image_obj, img_id, SMOKE_IMG_DIR)

        objects = gqa.extract_objects(scene_graphs[img_id], width, height)
        if len(objects) < 2:
            skip_stats["too_few_objects"] += 1
            continue

        candidates = []
        candidates.extend(gqa.generate_spatial_candidates(
            img_id, image_path, objects, rng, "train", gqa.TRAIN_TEMPLATES))
        candidates.extend(gqa.generate_size_candidates(
            img_id, image_path, objects, rng, "train", gqa.TRAIN_TEMPLATES))
        candidates.extend(gqa.generate_count_candidates(
            img_id, image_path, objects, rng, "train", gqa.TRAIN_TEMPLATES))

        if not candidates:
            skip_stats["no_candidates"] += 1
            continue

        cand = candidates[0]
        qid = f"smoke_{len(records):04d}"
        rec = gqa.make_record(
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
            split="smoke",
            operation=cand["operation"],
            extra_info={"question_id": qid, **cand["extra_info"]},
        )
        records.append(rec)
        print(f"[smoke] record {len(records)}/{TARGET_RECORDS} "
              f"img={img_id} schema={cand['schema']} op={cand['operation']} "
              f"answer={cand['target_answer']!r}")

    gqa.write_jsonl(records, SMOKE_OUT)

    print(f"\n[smoke] scanned={scanned} produced={len(records)} -> {SMOKE_OUT}")
    print(f"[smoke] skip_stats: {dict(skip_stats)}")

    if not records:
        print("[smoke] FAIL: produced 0 records")
        return 1

    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
