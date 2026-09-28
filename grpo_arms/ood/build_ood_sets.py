#!/usr/bin/env python
"""build_ood_sets.py — materialise the seven OOD benchmarks into vpb_generate.py's input format (CPU job; raw parquets downloaded on the login node).
Output per benchmark: data/ood_eval/sets/<bench>.jsonl (+ .sha256, required by vpb_generate) with rows
  {qid, data_source, question, prompt, answer, image, meta{question_type, answer_type, precision, choices, subject, seed_id, variant, n_images}}
and images under data/ood_eval/images/<bench>/. The prompt is the VPB template used for vpb_test (same "Final answer: .." instruction):
  "<image>\\nYour task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format \\"Final answer: ..\\"\\n\\nQuestion:\\n\\n" + question
Benchmarks: mathvista (testmini 1,000; official `query` text), mmk12 (the 1,024-row balanced split of docs/results_scienceqa_mmk12.md, rebuilt as
lettered choices), mathverse (testmini Vision Only + Vision Dominant, 788 each; official `query_cot` text), mathvision (full test 3,040),
wemath (testmini 1,740), dynamath (10 variants × 501 seeds = 5,010), mmmu (validation 900; image_1 only — 43 rows have extra images, flagged).
Gold answers are letters for multiple choice (choice text mapped to its letter) and the raw value otherwise. Extraction/scoring is rule-based (report_ood.py).
"""
import ast
import glob
import hashlib
import io
import json
import os
import re

import pyarrow.parquet as pq
from PIL import Image

REPO = "/scratch/sghos104/rlpt"
RAW = f"{REPO}/data/ood_eval/raw"
SETS = f"{REPO}/data/ood_eval/sets"
IMG = f"{REPO}/data/ood_eval/images"
PRE = ('Your task is to answer the question below. Give step by step reasoning before you answer, and when you\'re ready to answer, '
       'please use the format "Final answer: .."\n\nQuestion:\n\n')
HINT_MC = "Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end."
HINT_FREE = "Hint: Please answer the question and provide the final answer value at the end."
LET = "ABCDEFGHIJ"


def save_image(bench, qid, blob, hint_path=None):
    if isinstance(blob, dict):
        hint_path = hint_path or blob.get("path")
        blob = blob["bytes"]
    if blob is None:
        return None
    ext = (os.path.splitext(hint_path or "")[1] or "").lower()
    if ext not in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
        ext = ".png"
    d = f"{IMG}/{bench}"
    os.makedirs(d, exist_ok=True)
    p = f"{d}/{qid}{ext}"
    with open(p, "wb") as f:
        f.write(blob)
    try:
        with Image.open(p) as im:
            im.verify()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"{bench} qid {qid}: bad image ({exc})")
    return os.path.relpath(p, REPO)


def choices_block(choices):
    return "Choices:\n" + "\n".join(f"({LET[i]}) {c}" for i, c in enumerate(choices))


def write_set(bench, rows):
    p = f"{SETS}/{bench}.jsonl"
    with open(p, "w") as f:
        for r in rows:
            r["prompt"] = "<image>\n" + PRE + r["question"]
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
    open(p.replace(".jsonl", ".sha256"), "w").write(sha + "\n")
    print(f"[build] {bench}: {len(rows)} rows -> {p} sha {sha[:16]}", flush=True)


def mathvista():
    t = pq.read_table(glob.glob(f"{RAW}/AI4Math__MathVista/data/testmini-*.parquet")[0]).to_pylist()
    rows = []
    for r in t:
        qid = int(r["pid"])
        img = save_image("mathvista", qid, r["decoded_image"], r["image"])
        if r["question_type"] == "multi_choice":
            ch = list(r["choices"]); gold = LET[ch.index(r["answer"])]
        else:
            ch = None; gold = str(r["answer"])
        rows.append(dict(qid=qid, data_source="MathVista", question=r["query"], answer=gold, image=img,
                         meta=dict(question_type=r["question_type"], answer_type=r["answer_type"], precision=r.get("precision"), choices=ch,
                                   subject=r["metadata"].get("category"), unit=r.get("unit"), raw_answer=str(r["answer"]))))
    write_set("mathvista", rows)


def mmk12():
    t = pq.read_table(f"{REPO}/data/mmk12_eval/mmk12_test1024.parquet").to_pylist()
    rows = []
    for i, r in enumerate(t):
        text = r["prompt"][0]["content"] if isinstance(r["prompt"], list) else r["prompt"]
        m = re.search(r"Question:\s*(.*?)\n\s*Choices:", text, flags=re.S)
        q = m.group(1).strip() if m else text.replace("<image>", "").strip()
        gt = r["reward_model"]["ground_truth"]; ch = list(gt["choices"]); gold = LET[ch.index(gt["answer"])]
        img = save_image("mmk12", i, r["images"][0])
        rows.append(dict(qid=i, data_source="MMK12", question=f"{HINT_MC}\nQuestion: {q}\n{choices_block(ch)}", answer=gold, image=img,
                         meta=dict(question_type="multi_choice", answer_type="text", precision=None, choices=ch, subject=r["extra_info"]["subject"],
                                   question_id=r["extra_info"]["question_id"])))
    write_set("mmk12", rows)


def mathverse():
    t = pq.read_table(f"{RAW}/AI4Math__MathVerse/testmini.parquet").to_pylist()
    rows = []
    for r in t:
        if r["problem_version"] not in ("Vision Only", "Vision Dominant"):
            continue
        qid = int(r["sample_index"])
        img = save_image("mathverse", qid, r["image"])
        src = "MathVerse_VisionOnly" if r["problem_version"] == "Vision Only" else "MathVerse_VisionDominant"
        rows.append(dict(qid=qid, data_source=src, question=r["query_cot"], answer=str(r["answer"]), image=img,
                         meta=dict(question_type=r["question_type"], answer_type="text" if r["question_type"] == "multi-choice" else "value",
                                   precision=None, choices=None, subject=r["metadata"].get("subject"), problem_index=r["problem_index"])))
    write_set("mathverse", rows)


def mathvision():
    t = pq.read_table(glob.glob(f"{RAW}/MathLLMs__MathVision/data/test-*.parquet")[0]).to_pylist()
    rows = []
    for r in t:
        qid = int(r["id"])
        img = save_image("mathvision", qid, r["decoded_image"], r["image"])
        opts = list(r["options"] or [])
        if opts:
            q = f"{HINT_MC}\nQuestion: {r['question']}\n{choices_block(opts)}"; qt = "multi_choice"
        else:
            q = f"{HINT_FREE}\nQuestion: {r['question']}"; qt = "free_form"
        rows.append(dict(qid=qid, data_source="MathVision", question=q, answer=str(r["answer"]), image=img,
                         meta=dict(question_type=qt, answer_type="text" if opts else "value", precision=None, choices=opts or None,
                                   subject=r["subject"], level=r["level"])))
    write_set("mathvision", rows)


def wemath():
    t = pq.read_table(glob.glob(f"{RAW}/We-Math__We-Math/data/testmini-*.parquet")[0]).to_pylist()
    rows = []
    for i, r in enumerate(t):
        img = save_image("wemath", i, r["image_path"])
        parts = [p.strip() for p in re.split(r";\s*(?=[A-H]\.)", r["option"].strip()) if p.strip()]
        ch = [re.sub(r"^[A-H]\.\s*", "", p) for p in parts]
        rows.append(dict(qid=i, data_source="WeMath", question=f"{HINT_MC}\nQuestion: {r['question']}\n{choices_block(ch)}", answer=str(r["answer"]).strip(),
                         image=img, meta=dict(question_type="multi_choice", answer_type="text", precision=None, choices=ch, subject=r["knowledge concept"],
                                              wemath_id=r["ID"], key=r["key"])))
    write_set("wemath", rows)


def dynamath():
    rows = []
    for v in range(1, 11):
        fs = sorted(glob.glob(f"{RAW}/DynaMath__DynaMath_Sample/data/sample{v}-*.parquet"))
        t = []
        for f in fs:
            t += pq.read_table(f).to_pylist()
        for r in t:
            seed = int(r["id"]); qid = v * 1000 + seed
            img = save_image("dynamath", qid, r["decoded_image"], r["image"])
            at = r["answer_type"]
            rows.append(dict(qid=qid, data_source=f"DynaMath_v{v}", question=r["question"], answer=str(r["ground_truth"]), image=img,
                             meta=dict(question_type="multi_choice" if "choice" in at else "free_form", answer_type=at, precision=None, choices=None,
                                       subject=r["subject"], seed_id=seed, variant=v, level=r["knowledge_level"])))
    write_set("dynamath", rows)


def mmmu():
    rows = []
    n_multi = 0
    for f in sorted(glob.glob(f"{RAW}/MMMU__MMMU/*/validation-*.parquet")):
        for r in pq.read_table(f).to_pylist():
            qid = len(rows)
            img = save_image("mmmu", qid, r["image_1"])
            n_img = sum(1 for k in range(1, 8) if r.get(f"image_{k}"))
            n_multi += n_img > 1
            q = r["question"]
            if r["question_type"] == "multiple-choice":
                opts = ast.literal_eval(r["options"]) if isinstance(r["options"], str) else list(r["options"])
                text = f"{HINT_MC}\nQuestion: {q}\n{choices_block(opts)}"; qt = "multi_choice"
            else:
                opts = None; text = f"{HINT_FREE}\nQuestion: {q}"; qt = "free_form"
            rows.append(dict(qid=qid, data_source="MMMU", question=text, answer=str(r["answer"]), image=img,
                             meta=dict(question_type=qt, answer_type="text" if opts else "value", precision=None, choices=opts,
                                       subject=r["id"].split("_")[1], mmmu_id=r["id"], n_images=n_img)))
    print(f"[build] mmmu: {n_multi} rows have >1 image (only image_1 is shown)", flush=True)
    write_set("mmmu", rows)


if __name__ == "__main__":
    for fn in (mathvista, mmk12, mathverse, mathvision, wemath, dynamath, mmmu):
        fn()
    print("[build] ALL SETS DONE", flush=True)
