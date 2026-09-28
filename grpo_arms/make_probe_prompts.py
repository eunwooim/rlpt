#!/usr/bin/env python3
"""Slice 200 train-subset prompts (stratified by source_file, seed 0) into the vpb_eval schema for vpb_generate.py."""
import json, random, os, collections
REPO = "/scratch/sghos104/rlpt"
random.seed(0)
ref = json.loads(open(f"{REPO}/grpo_arms/data/vpb_eval.jsonl").readline())
rows = [json.loads(l) for l in open(f"{REPO}/grpo_arms/data/train_subset.jsonl") if l.strip()]
by = collections.defaultdict(list)
for r in rows:
    by[r["source_file"]].append(r)
per = max(1, round(200 / len(by)))
pick = []
for src in sorted(by):
    rs = by[src]; random.shuffle(rs); pick += rs[:per]
random.shuffle(pick); pick = pick[:200]
out = []
for i, r in enumerate(pick):
    img = r["image"] if os.path.isabs(r["image"]) else os.path.join(REPO, r["image"])
    out.append({"qid": i, "data_source": r["source_file"], "question": r["question_orig"],
                "prompt": r["question"], "image": img, "answer": r["answer"],
                "gold_steps": r["gold_steps"], "source_file": r["source_file"], "line_index": r["line_index"]})
missing = [k for k in ref if k not in out[0]]
print("vpb_eval keys:", sorted(ref)); print("probe keys   :", sorted(out[0]))
print("keys in vpb_eval missing from probe:", missing, "(empty = schema-compatible)")
with open(f"{REPO}/grpo_arms/data/probe_prompts.jsonl", "w") as f:
    for rec in out:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"wrote {len(out)} prompts across {len(by)} sources -> grpo_arms/data/probe_prompts.jsonl")
