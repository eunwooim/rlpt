#!/usr/bin/env python
"""EXP 4c — production relevance of a list-tier policy change (CPU, no writes).

Over canonical_chunked_v2.jsonl: count marker/list records (expected
6,370,908) and reservoir-sample 20K of them (seed 0). For the sample,
fetch the ORIGINAL blobs from canonical.jsonl (read-only) and apply the
coarser variant split (step > para, list off — the same hierarchy as
ablation arm A) to measure what the coarser alternative looks like at
production scale:
  - chunks/record before (list tier, from the shipped file) vs after
  - resolution mix after (para / step / falls-through-to-model-path)
  - token-length p50/p99 of the after-chunks (raw tokenizers backend —
    transformers 4.56 cannot load the release tokenizer config)
Falls-through records are counted as 1 chunk here (the model leaves ~92%
of its inputs whole in production); flagged separately.
"""
import json
import random
import sys
from collections import Counter

import numpy as np
import orjson

sys.path.insert(0, "/scratch/sghos104/rlpt/chunk_eval")
from marker_variants import marker_split_tiers  # noqa: E402

V2 = "/scratch/sghos104/rlpt/canonical_chunked_v2.jsonl"
CANON = "/scratch/sghos104/rlpt/canonical.jsonl"
TOK_JSON = ("/scratch/sghos104/rlpt/chunker/release/chunker-deberta-v3-small-v1"
            "/model/tokenizer.json")
N_SAMPLE = 20_000


def main():
    rng = random.Random(0)

    # ---- pass 1 (v2): count list-tier records, reservoir-sample keys ----
    n_list = 0
    res = []          # (sid, idx, n_chunks_before)
    with open(V2, "rb") as f:
        for i, line in enumerate(f):
            if i % 2_000_000 == 0:
                print(f"[exp4c pass1] {i:,} list={n_list:,}", flush=True)
            if b'"tier":"list"' not in line and b'"tier": "list"' not in line:
                continue
            rec = orjson.loads(line)
            prov = rec["metadata"]["chunk_provenance"]
            if prov["method"] != "marker" or prov.get("tier") != "list":
                continue
            n_list += 1
            item = (rec["metadata"]["source_sample_id"],
                    rec["metadata"]["source_index"], len(rec["steps"]))
            if len(res) < N_SAMPLE:
                res.append(item)
            else:
                j = rng.randrange(n_list)
                if j < N_SAMPLE:
                    res[j] = item
    print(f"[exp4c] marker/list records: {n_list:,} (expected 6,370,908)",
          flush=True)
    assert n_list == 6_370_908, n_list
    wanted = {(sid, idx): nb for sid, idx, nb in res}

    # ---- pass 2 (canonical.jsonl): fetch original blobs ----
    blobs = {}
    with open(CANON, "rb") as f:
        for i, line in enumerate(f):
            if i % 2_000_000 == 0:
                print(f"[exp4c pass2] {i:,} found={len(blobs):,}", flush=True)
            if len(blobs) == len(wanted):
                break
            rec = orjson.loads(line)
            key = (rec["metadata"]["source_sample_id"],
                   rec["metadata"]["source_index"])
            if key in wanted:
                blobs[key] = rec["steps"][0]
    assert len(blobs) == len(wanted), (len(blobs), len(wanted))

    # ---- apply coarser variant + tokenize ----
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(TOK_JSON)
    before, after = [], []
    mix = Counter()
    tok_lens = []
    for key, blob in blobs.items():
        before.append(wanted[key])
        tier, chunks = marker_split_tiers(blob, ("step", "para"))
        if tier is None:
            mix["falls_to_model"] += 1
            chunks = [blob]
        else:
            mix[tier] += 1
        after.append(len(chunks))
        for enc in tk.encode_batch(chunks, add_special_tokens=False):
            tok_lens.append(len(enc.ids))

    before, after = np.array(before), np.array(after)
    tok_lens = np.array(tok_lens)
    out = {
        "n_list_records": n_list, "sampled": len(blobs),
        "resolution_mix_after": dict(mix),
        "chunks_per_record_before": {
            "mean": float(before.mean()), "p50": int(np.median(before)),
            "p90": int(np.percentile(before, 90))},
        "chunks_per_record_after": {
            "mean": float(after.mean()), "p50": int(np.median(after)),
            "p90": int(np.percentile(after, 90))},
        "total_chunks_before_sample": int(before.sum()),
        "total_chunks_after_sample": int(after.sum()),
        "after_token_len": {
            "mean": float(tok_lens.mean()), "p50": int(np.median(tok_lens)),
            "p90": int(np.percentile(tok_lens, 90)),
            "p99": int(np.percentile(tok_lens, 99)),
            "pct_gt_220": float(100 * (tok_lens > 220).mean())},
    }
    with open("/scratch/sghos104/rlpt/chunk_eval/exp4c_production_relevance.json",
              "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print("[exp4c] DONE")


if __name__ == "__main__":
    main()
