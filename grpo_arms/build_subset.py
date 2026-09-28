#!/usr/bin/env python
"""Build the frozen 6,000-prompt GRPO training subset from VisualPRM400K-v1.1-Raw.

Filter (per the six-arm brief):
  - answer-correct: steps_with_score[-1].score > 0.5
  - >= 5 body steps (body = steps_with_score[:-1], i.e. excluding the final-answer step)
  - exclude nlvr2 (two images per record; conflicts with limit_mm_per_prompt={"image":1})
  - dedup by (image, question): keep the trace with the highest min body MC score,
    tie-broken by (source_file, line_index) for determinism
  - cap 300 per source file (seeded shuffle), then sample 6,000 from the pooled cap
    (seeded), sort by (source_file, line_index) for a stable on-disk order

Image resolution (verified convention): strip leading 'VisualPRM400K-v1.1-Raw/',
join under data/visualprm_v11_raw/images_extracted/images/. Every path is
asserted to exist; missing paths abort the build.

Outputs (in grpo_arms/data/):
  train_subset.jsonl    one row per prompt:
      {source_file, line_index, question, question_orig, answer, image (resolved,
       repo-relative), gold_steps, gold_step_scores, min_body_score, n_body_steps,
       response}
  train_subset.sha256   sha256 of the jsonl (the file every arm must read)
  subset_manifest.json  filter funnel counts, per-source composition, seed, spec
"""
import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter

REPO = "/scratch/sghos104/rlpt"
ANNO_DIR = os.path.join(REPO, "data/visualprm_v11_raw/annos/annotations")
IMG_ROOT = os.path.join(REPO, "data/visualprm_v11_raw/images_extracted/images")
RAW_PREFIX = "VisualPRM400K-v1.1-Raw/"
EXCLUDE_FILES = {"nlvr2_en_20240910.jsonl"}


def resolve_image(ref: str) -> str:
    if isinstance(ref, list):
        raise ValueError(f"multi-image record leaked past the nlvr2 exclusion: {ref}")
    rel = ref[len(RAW_PREFIX):] if ref.startswith(RAW_PREFIX) else ref
    return os.path.join(IMG_ROOT, rel)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per_source_cap", type=int, default=300)
    ap.add_argument("--n_prompts", type=int, default=6000)
    ap.add_argument("--min_body_steps", type=int, default=5)
    ap.add_argument("--out_dir", default=os.path.join(REPO, "grpo_arms/data"))
    args = ap.parse_args()

    funnel = Counter()
    per_source_eligible = {}
    files = sorted(f for f in os.listdir(ANNO_DIR) if f.endswith(".jsonl"))
    excluded = [f for f in files if f in EXCLUDE_FILES]
    assert excluded == sorted(EXCLUDE_FILES), f"expected nlvr2 file present, got {excluded}"
    files = [f for f in files if f not in EXCLUDE_FILES]
    print(f"[build] {len(files)} source files (excluded: {sorted(EXCLUDE_FILES)})")

    for fn in files:
        # dedup within source by (image, question): keep max min-body-score
        best = {}
        with open(os.path.join(ANNO_DIR, fn)) as f:
            for line_index, line in enumerate(f):
                funnel["total_rows"] += 1
                r = json.loads(line)
                sws = r.get("steps_with_score") or []
                if not sws:
                    funnel["no_steps"] += 1
                    continue
                if not (float(sws[-1]["score"]) > 0.5):
                    funnel["answer_incorrect"] += 1
                    continue
                body = sws[:-1]
                if len(body) < args.min_body_steps:
                    funnel["too_few_body_steps"] += 1
                    continue
                if isinstance(r["image"], list):
                    funnel["multi_image_unexpected"] += 1
                    continue
                funnel["eligible_traces"] += 1
                key = (r["image"], r["question"])
                min_body = min(float(s["score"]) for s in body)
                row = {
                    "source_file": fn,
                    "line_index": line_index,
                    "question": r["question"],
                    "question_orig": r.get("question_orig", ""),
                    "answer": r["answer"],
                    "image": r["image"],
                    "gold_steps": [s["step"] for s in body],
                    "gold_step_scores": [float(s["score"]) for s in body],
                    "min_body_score": min_body,
                    "n_body_steps": len(body),
                    "response": r["response"],
                }
                prev = best.get(key)
                # keep strictly-better min_body_score; ties keep the earlier line
                if prev is None or min_body > prev["min_body_score"]:
                    best[key] = row
        per_source_eligible[fn] = list(best.values())

    funnel["eligible_unique_prompts"] = sum(len(v) for v in per_source_eligible.values())

    rng = random.Random(args.seed)
    pooled = []
    per_source_taken = {}
    for fn in files:
        rows = per_source_eligible[fn]
        rows.sort(key=lambda r: r["line_index"])  # deterministic pre-shuffle order
        rng.shuffle(rows)
        take = rows[: args.per_source_cap]
        per_source_taken[fn] = len(take)
        pooled.extend(take)
    funnel["pooled_after_cap"] = len(pooled)

    if len(pooled) < args.n_prompts:
        print(f"FATAL: pooled {len(pooled)} < requested {args.n_prompts}", file=sys.stderr)
        sys.exit(2)

    pooled.sort(key=lambda r: (r["source_file"], r["line_index"]))  # canonical order pre-sample
    sample = rng.sample(pooled, args.n_prompts)
    sample.sort(key=lambda r: (r["source_file"], r["line_index"]))

    # resolve + assert every image path
    missing = 0
    for r in sample:
        p = resolve_image(r["image"])
        if not os.path.exists(p):
            print(f"MISSING IMAGE: {r['image']} -> {p}", file=sys.stderr)
            missing += 1
        r["image"] = os.path.relpath(p, REPO)
    if missing:
        print(f"FATAL: {missing} image paths missing; aborting.", file=sys.stderr)
        sys.exit(2)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "train_subset.jsonl")
    with open(out_path, "w") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sha = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
    with open(os.path.join(args.out_dir, "train_subset.sha256"), "w") as f:
        f.write(sha + "\n")

    comp = Counter(r["source_file"] for r in sample)
    manifest = {
        "spec": {
            "seed": args.seed,
            "per_source_cap": args.per_source_cap,
            "n_prompts": args.n_prompts,
            "min_body_steps": args.min_body_steps,
            "answer_correct_rule": "steps_with_score[-1].score > 0.5",
            "dedup": "(image, question) -> max min_body_score, ties earlier line",
            "excluded_files": sorted(EXCLUDE_FILES),
        },
        "funnel": dict(funnel),
        "per_source_capped": per_source_taken,
        "sampled_composition": dict(sorted(comp.items())),
        "n_body_steps_stats": {
            "min": min(r["n_body_steps"] for r in sample),
            "max": max(r["n_body_steps"] for r in sample),
            "mean": round(sum(r["n_body_steps"] for r in sample) / len(sample), 3),
        },
        "sha256": sha,
        "out_path": os.path.relpath(out_path, REPO),
    }
    with open(os.path.join(args.out_dir, "subset_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest["funnel"], indent=2))
    print(f"[build] wrote {len(sample)} prompts -> {out_path}")
    print(f"[build] sha256 {sha}")


if __name__ == "__main__":
    main()
