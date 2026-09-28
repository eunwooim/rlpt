#!/usr/bin/env python
"""Build the frozen question set for the granularity generation experiment.

Draws 200 questions from VisualProcessBench test.jsonl restricted to records
that supply EXACTLY ONE image and reference AT MOST ONE <imageN> tag
(2,831 of 2,866 eligible), stratified by data_source with largest-remainder
allocation, seed 0.

Why frozen to disk: both models x all four conditions must hit byte-identical
inputs. The generation script READS this file and never re-samples, so no
upstream change can silently drift the inputs between runs.

<imageN> tags are stripped from the question text here (once, at build time) —
Qwen's chat template has no notion of them, and the image is carried by the
{"type": "image"} content block instead.

Also records the 25 records whose question references MORE distinct <imageN>
tags than the record supplies images -> vpb_multiimage_defect.json. That is a
property of VisualProcessBench as published, not of this pipeline.

Read-only on test.jsonl. Outputs:
  granularity/qwen_question_set.json
  granularity/vpb_multiimage_defect.json
"""
import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict

VPB_DIR = "/scratch/sghos104/rlpt/data/visualprocessbench"
TEST = os.path.join(VPB_DIR, "test.jsonl")
IMG_ROOT = os.path.join(VPB_DIR, "images_extracted")
OUTDIR = "/scratch/sghos104/rlpt/granularity"
TAG_RE = re.compile(r"<image(\d*)>")


def strip_tags(text):
    """Remove <image>/<imageN> placeholders and tidy the whitespace they leave."""
    out = TAG_RE.sub("", text)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r" +([,.;:?!])", r"\1", out)
    return out.strip()


def _self_test():
    assert strip_tags("the ant <image1> walks <image2> home") == "the ant walks home"
    assert strip_tags("see <image>") == "see"
    # tag removal leaves "what  ?"; the tidy rules close the gap and the space
    # before punctuation -> "what?"
    assert strip_tags("what <image1> ?") == "what?"
    # largest remainder sums exactly
    q = allocate({"a": 1026, "b": 687, "c": 570, "d": 291, "e": 257}, 200)
    assert sum(q.values()) == 200, q
    print("[build] self-test OK", flush=True)


def allocate(counts, total):
    """Proportional allocation with largest-remainder rounding."""
    grand = sum(counts.values())
    exact = {k: total * v / grand for k, v in counts.items()}
    base = {k: int(v) for k, v in exact.items()}
    short = total - sum(base.values())
    order = sorted(counts, key=lambda k: (-(exact[k] - base[k]), k))
    for k in order[:short]:
        base[k] += 1
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=OUTDIR)
    args = ap.parse_args()
    _self_test()

    eligible = defaultdict(list)
    defects = []
    n_total = 0
    excluded = Counter()

    with open(TEST) as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            d = json.loads(line)
            n_total += 1
            imgs = d["image"] if isinstance(d["image"], list) else [d["image"]]
            tags = sorted(set(TAG_RE.findall(d["question"])))
            qid = f"vpb_{idx:05d}"
            if len(tags) > len(imgs):
                defects.append({
                    "qid": qid, "line_index": idx,
                    "data_source": d["data_source"],
                    "n_images_supplied": len(imgs),
                    "n_distinct_tags": len(tags),
                    "tags": [f"<image{t}>" for t in tags],
                    "images": imgs,
                    "question_head": d["question"][:300]})
            if len(imgs) != 1:
                excluded["multi_image"] += 1
                continue
            if len(tags) > 1:
                excluded["tags_gt_images"] += 1
                continue
            eligible[d["data_source"]].append({
                "qid": qid, "line_index": idx,
                "data_source": d["data_source"],
                "image": imgs[0],
                "question": strip_tags(d["question"]),
                "question_raw": d["question"],
                "answer": d.get("answer"),
            })

    counts = {k: len(v) for k, v in eligible.items()}
    quota = allocate(counts, args.n)
    rng = random.Random(args.seed)
    picked = []
    for src in sorted(eligible):
        pool = sorted(eligible[src], key=lambda r: r["qid"])   # deterministic order
        picked.extend(rng.sample(pool, quota[src]))
    picked.sort(key=lambda r: r["qid"])

    # integrity: every image must resolve
    missing = [r["image"] for r in picked
               if not os.path.exists(os.path.join(IMG_ROOT, r["image"]))]
    if missing:
        raise SystemExit(f"[build] {len(missing)} images unresolved: {missing[:3]}")

    payload = {
        "source": TEST,
        "image_root": IMG_ROOT,
        "n_questions": len(picked),
        "seed": args.seed,
        "eligibility_rule": "exactly 1 image supplied AND <=1 distinct <imageN> tag",
        "eligible_pool": counts,
        "quota": quota,
        "excluded": dict(excluded),
        "records_total": n_total,
        "tag_handling": "<imageN> stripped from question text at build time; "
                        "image carried by the chat template image block",
        "questions": picked,
    }
    payload["question_set_sha256"] = hashlib.sha256(
        json.dumps(picked, sort_keys=True).encode()).hexdigest()

    os.makedirs(args.outdir, exist_ok=True)
    qs_path = os.path.join(args.outdir, "qwen_question_set.json")
    with open(qs_path, "w") as f:
        json.dump(payload, f, indent=1)
    df_path = os.path.join(args.outdir, "vpb_multiimage_defect.json")
    with open(df_path, "w") as f:
        json.dump({
            "note": "VisualProcessBench records whose question references more "
                    "distinct <imageN> placeholders than the record supplies "
                    "images. Property of the published benchmark.",
            "source": TEST, "records_total": n_total,
            "n_defective": len(defects), "records": defects}, f, indent=1)

    print(f"[build] eligible pool: {sum(counts.values())} of {n_total} "
          f"(excluded {dict(excluded)})")
    print(f"[build] quota: {quota}")
    print(f"[build] picked {len(picked)} | sha256={payload['question_set_sha256'][:16]}…")
    print(f"[build] all {len(picked)} images resolve")
    print(f"[build] -> {qs_path}")
    print(f"[build] defects: {len(defects)} -> {df_path}")


if __name__ == "__main__":
    main()
