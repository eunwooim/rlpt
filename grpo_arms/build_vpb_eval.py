#!/usr/bin/env python
"""Freeze the VPB validation set for the six-arm campaign.

VPB questions + gold answers ONLY (pre-generated solutions and step labels
discarded). Single-image rows only (2,856 of 2,866) — the validated generation
path uses limit_mm_per_prompt={"image": 1}. Same frozen file for all six arms.

Prompt: the VisualPRM policy wrapper (dominant training template):
  <image>\nYour task is to answer the question below. Give step by step
  reasoning before you answer, and when you're ready to answer, please use
  the format "Final answer: .."\n\nQuestion:\n\n{question}
"""
import hashlib
import json
import os

REPO = "/scratch/sghos104/rlpt"
SRC = os.path.join(REPO, "data/visualprocessbench/test.jsonl")
OUT = os.path.join(REPO, "grpo_arms/data/vpb_eval.jsonl")

WRAPPER = ('<image>\nYour task is to answer the question below. Give step by step '
           'reasoning before you answer, and when you\'re ready to answer, please '
           'use the format "Final answer: .."\n\nQuestion:\n\n{q}')


def main():
    rows = []
    n_multi = 0
    with open(SRC) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            imgs = r["image"] if isinstance(r["image"], list) else [r["image"]]
            if len(imgs) != 1:
                n_multi += 1
                continue
            img = os.path.join("data/visualprocessbench/images_extracted", imgs[0])
            assert os.path.exists(os.path.join(REPO, img)), img
            q = r["question"]
            # strip the VPB <imageN> placeholders' base <image> if present; the
            # wrapper supplies exactly one <image> token for the chat template
            rows.append({
                "qid": i,
                "data_source": r["data_source"],
                "question": q,
                "prompt": WRAPPER.format(q=q),
                "answer": r["answer"],
                "image": img,
            })
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
    with open(OUT.replace(".jsonl", ".sha256"), "w") as f:
        f.write(sha + "\n")
    print(f"[vpb] {len(rows)} single-image questions ({n_multi} multi-image dropped)")
    print(f"[vpb] sha256 {sha}")


if __name__ == "__main__":
    main()
