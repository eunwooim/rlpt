#!/usr/bin/env python
"""EXP 1 supplement 2 — proper fence analysis of the MARKER path.

exp1b's per-chunk ``` parity conflates two things: (a) a chunk boundary
inside a fenced region, and (b) a source blob whose fences are themselves
unbalanced (unclosed final fence). This pass separates them with the same
cumulative line-start-parity method as chunk_canonical/verify_fences.py,
applied to marker (and gold) records:

  per record: marks_k = count of line-start ``` markers in chunk k
              (fence_guard.FENCE_RE — inline ``` never counts);
  boundary after chunk k is IN-FENCE iff cumulative parity is odd;
  record source-unbalanced iff total marks is odd.

Reported per bucket: records with any fence marker, records with >= 1
in-fence boundary, total in-fence boundaries, source-unbalanced records.
Only lines containing b'```' are parsed (cheap pre-filter).
"""
import json
import sys
from collections import Counter
from multiprocessing import Pool

import orjson

sys.path.insert(0, "/scratch/sghos104/rlpt/chunk_canonical")
from fence_guard import FENCE_RE  # noqa: E402


def work(batch):
    c = Counter()
    ex = []
    for line in batch:
        rec = orjson.loads(line)
        prov = rec["metadata"]["chunk_provenance"]
        m = prov["method"]
        if m == "passthrough":
            if len(rec["steps"]) <= 1:
                continue
            b = "gold"
        elif m == "marker":
            b = "marker/" + prov["tier"]
        elif m in ("model", "model_skipped_nonprose"):
            b = m
        else:
            continue
        chunks = rec["steps"]
        marks = [len(FENCE_RE.findall(s)) for s in chunks]
        total = sum(marks)
        if not total:
            continue
        c[(b, "recs_with_fence")] += 1
        if total % 2:
            c[(b, "source_unbalanced")] += 1
        cum, bad = 0, 0
        for k in range(len(chunks) - 1):
            cum += marks[k]
            if cum % 2:
                bad += 1
        if bad:
            c[(b, "recs_infence_boundary")] += 1
            c[(b, "infence_boundaries")] += bad
            if b.startswith("marker") and len(ex) < 3:
                ex.append({"sid": rec["metadata"]["source_sample_id"],
                           "tier": prov.get("tier"), "marks": marks,
                           "first_bad_chunk": next(
                               chunks[k][:400] for k in range(len(chunks))
                               if sum(marks[:k + 1]) % 2)})
    return c, ex


def main():
    path = "/scratch/sghos104/rlpt/canonical_chunked_v2.jsonl"
    tot = Counter()
    examples = []

    def batches(f):
        buf = []
        for line in f:
            if b"```" in line:
                buf.append(line)
            if len(buf) >= 5000:
                yield buf
                buf = []
        if buf:
            yield buf

    with open(path, "rb") as f, Pool(14) as pool:
        for c, ex in pool.imap_unordered(work, batches(f), chunksize=1):
            tot.update(c)
            if len(examples) < 12:
                examples.extend(ex)

    buckets = ("marker/step", "marker/list", "marker/para", "model",
               "model_skipped_nonprose", "gold")
    out = {}
    print(f"{'bucket':24s} {'w/fence':>9s} {'srcUnbal':>9s} {'recsInFence':>12s} {'boundaries':>11s}")
    for b in buckets:
        row = {k: tot[(b, k)] for k in
               ("recs_with_fence", "source_unbalanced",
                "recs_infence_boundary", "infence_boundaries")}
        out[b] = row
        print(f"{b:24s} {row['recs_with_fence']:>9,d} {row['source_unbalanced']:>9,d} "
              f"{row['recs_infence_boundary']:>12,d} {row['infence_boundaries']:>11,d}")
    with open("/scratch/sghos104/rlpt/chunk_eval/exp1c_marker_fences.json", "w") as f:
        json.dump({"buckets": out, "examples": examples}, f, indent=2)
    print("[exp1c] DONE")


if __name__ == "__main__":
    main()
