#!/usr/bin/env python
"""Targeted invariant checks for the tau=0.90 emission variant
(emit_tau0.90/sft_train.jsonl, rlvr_prompts.jsonl):
  1. verbatim steps vs raw annotations (streaming merge-join)
  2. no duplicate (image-key, normalized question) in sft_train
  3. all image refs resolve in images.zip
  4. counts: sft == n_groups, rlvr == n_unique_questions, sft <= stage2 kept
"""
import json
import sys
import zipfile
from pathlib import Path

FILTERED = Path("/scratch/sghos104/rlpt/data/visualprm_v11_filtered")
EMIT = FILTERED / "emit_tau0.90"
RAW = Path("/scratch/sghos104/rlpt/data/visualprm_v11_raw")
RAW_PREFIX = "VisualPRM400K-v1.1-Raw/"


def iter_jsonl(path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def candidate_members(ref):
    if ref.startswith(RAW_PREFIX):
        return ["images/" + ref[len(RAW_PREFIX):]]
    if ref.startswith("images/"):
        return [ref, "images/nlvr2/" + ref]
    return ["images/" + ref]


def normalize_question(q):
    return " ".join(str(q).split()).casefold()


failures = 0


def report(name, ok, detail=""):
    global failures
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures += 1


# Check 1: verbatim steps via streaming merge-join over raw annotations
streams = [{"path": EMIT / n, "n": 0, "mismatch": 0, "missing": 0}
           for n in ("sft_train.jsonl", "rlvr_prompts.jsonl")]
for s in streams:
    s["it"] = iter_jsonl(s["path"])
    s["head"] = next(s["it"], None)

for f in sorted((RAW / "annos" / "annotations").glob("*.jsonl")):
    source_file = f"annotations/{f.name}"
    with open(f) as fh:
        for line_index, line in enumerate(fh):
            tid = (source_file, line_index)
            hot = [s for s in streams if s["head"] is not None
                   and (s["head"]["source_file"], s["head"]["line_index"]) == tid]
            if not hot:
                continue
            rec = json.loads(line)
            sws = rec.get("steps_with_score") or []
            body = [x["step"] for x in (sws[:-1] if len(sws) > 1 else sws)]
            for s in hot:
                if s["head"]["steps"] != body:
                    s["mismatch"] += 1
                s["n"] += 1
                s["head"] = next(s["it"], None)
for s in streams:
    while s["head"] is not None:
        s["missing"] += 1
        s["head"] = next(s["it"], None)
    report(f"verbatim steps: emit_tau0.90/{s['path'].name}",
           s["mismatch"] == 0 and s["missing"] == 0,
           f"{s['n']} rows checked, {s['mismatch']} mismatched, "
           f"{s['missing']} not found in raw")

# Checks 2 + 3: dedup uniqueness and image resolution
with zipfile.ZipFile(RAW / "images.zip") as z:
    member_key = {i.filename: (i.CRC, i.file_size) for i in z.infolist()}


def image_key(image):
    refs = image if isinstance(image, list) else [image]
    keys = []
    for ref in refs:
        for cand in candidate_members(ref):
            if cand in member_key:
                keys.append(member_key[cand])
                break
        else:
            return None
    return tuple(keys)


seen = set()
dupes = unresolved = n_rows = 0
for row in iter_jsonl(EMIT / "sft_train.jsonl"):
    n_rows += 1
    ik = image_key(row["image"])
    if ik is None:
        unresolved += 1
        continue
    gk = (ik, normalize_question(row["question_orig"]))
    if gk in seen:
        dupes += 1
    seen.add(gk)
report("no duplicate (image-key, question) in emit_tau0.90/sft_train.jsonl",
       dupes == 0, f"{n_rows} rows, {dupes} duplicate keys")
unresolved_rlvr = sum(1 for row in iter_jsonl(EMIT / "rlvr_prompts.jsonl")
                      if image_key(row["image"]) is None)
report("all image refs resolve in images.zip",
       unresolved == 0 and unresolved_rlvr == 0,
       f"sft unresolved={unresolved}, rlvr unresolved={unresolved_rlvr}")

# Check 4: count consistency
em = json.loads((EMIT / "emission_stats.json").read_text())
s2 = json.loads((FILTERED / "stage2_stats_tau0.90.json").read_text())
n_rlvr = sum(1 for _ in iter_jsonl(EMIT / "rlvr_prompts.jsonl"))
ok = (em["sft_rows"] == n_rows == em["n_groups_image_question"]
      and em["rlvr_rows"] == n_rlvr == em["n_unique_questions"]
      and em["sft_rows"] <= s2["kept"] and em["rlvr_rows"] <= em["sft_rows"])
report("counts: stage2 kept >= sft >= rlvr, rows == group counts", ok,
       f"stage2 {s2['kept']} >= sft {n_rows} >= rlvr {n_rlvr}; "
       f"groups {em['n_groups_image_question']}, uniqueQ {em['n_unique_questions']}")

print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
sys.exit(0 if failures == 0 else 1)
