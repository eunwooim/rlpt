#!/usr/bin/env python
"""Convert PRM800K + ProcessBench to the Exp-4 harness format.

Output (xdataset/): prm800k.jsonl, processbench.jsonl — one record per line:
  {"sid", "dataset", "subset", "n", "steps": [...]}
plus convert_stats.json with per-source filter funnel, step-count and
chunker-token-length distributions, and the SAME distributions computed on
the VisualPRM gold sample (exp2_populations/gold.jsonl — 20K passthrough
records >=3 steps, stratified by census; identical measurement code) so
convention differences are visible up front.

PRM800K gold path (per trajectory record):
  for each step in label.steps: text = completions[chosen_completion].text
  if chosen_completion is not None else human_completion (human-authored).
  Filters (all counted): drop is_quality_control_question /
  is_initial_screening_question; require finish_reason == "solution";
  require every step resolved (no unresolved step mid-path); strip steps,
  drop records with any empty step; require >= 3 steps; dedupe exact
  (problem, joined-steps) across train+test (multiple labeler records can
  cover the same generator solution).

ProcessBench: the pre-split "steps" field, >= 3 steps, one subset per file
(gsm8k / math / olympiadbench / omnimath).
"""
import glob
import hashlib
import json
import os
import sys
from collections import Counter

import numpy as np

XDIR = "/scratch/sghos104/rlpt/chunk_eval/xdataset"
SRC = "/scratch/sghos104/rlpt/data/xdataset_eval"
GOLD_SAMPLE = "/scratch/sghos104/rlpt/chunk_eval/exp2_populations/gold.jsonl"
TOK_JSON = ("/scratch/sghos104/rlpt/chunker/release/"
            "chunker-deberta-v3-small-v1/model/tokenizer.json")


def prm_extract(rec):
    """Returns (steps, reason). reason != 'ok' means dropped."""
    if rec.get("is_quality_control_question"):
        return None, "qc_question"
    if rec.get("is_initial_screening_question"):
        return None, "screening_question"
    label = rec.get("label") or {}
    if label.get("finish_reason") != "solution":
        return None, f"finish_{label.get('finish_reason')}"
    steps = []
    for st in label.get("steps") or []:
        txt = None
        cc = st.get("chosen_completion")
        if cc is not None and st.get("completions"):
            txt = st["completions"][cc].get("text")
        elif st.get("human_completion"):
            hc = st["human_completion"]
            txt = hc.get("text") if isinstance(hc, dict) else hc
        if txt is None:
            return None, "unresolved_step"
        steps.append(txt.strip())
    if any(not s for s in steps):
        return None, "empty_step"
    if len(steps) < 3:
        return None, "lt3_steps"
    return steps, "ok"


def load_pb_file(path):
    if path.endswith(".jsonl"):
        with open(path) as f:
            return [json.loads(x) for x in f if x.strip()]
    if path.endswith(".json"):
        with open(path) as f:
            obj = json.load(f)
        return obj if isinstance(obj, list) else obj.get("data", [])
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    raise ValueError(path)


def dist(vals):
    a = np.asarray(vals, dtype=np.float64)
    if not len(a):
        return None
    return {"n": int(len(a)), "mean": float(a.mean()),
            "p10": float(np.percentile(a, 10)), "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)), "p99": float(np.percentile(a, 99)),
            "max": float(a.max())}


def token_lengths(tk, all_steps, batch=2000):
    out = []
    for i in range(0, len(all_steps), batch):
        for e in tk.encode_batch(all_steps[i:i + batch]):
            out.append(len(e.ids))
    return out


def main():
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(TOK_JSON)
    stats = {}
    os.makedirs(XDIR, exist_ok=True)

    # ---------------- PRM800K ----------------
    funnel = Counter()
    seen_hash = set()
    kept = []
    for subset in ("train", "test"):
        path = os.path.join(SRC, "prm800k", f"phase2_{subset}.jsonl")
        with open(path) as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                rec = json.loads(line)
                funnel[f"{subset}_total"] += 1
                steps, reason = prm_extract(rec)
                if steps is None:
                    funnel[reason] += 1
                    continue
                problem = (rec.get("question") or {}).get("problem", "")
                h = hashlib.sha1(
                    (problem + "\x00" + "\x00".join(steps)).encode()).hexdigest()
                if h in seen_hash:
                    funnel["dup_exact"] += 1
                    continue
                seen_hash.add(h)
                funnel["kept"] += 1
                kept.append({"sid": f"prm800k_{subset}_{i}",
                             "dataset": "prm800k", "subset": subset,
                             "n": len(steps), "steps": steps})
    with open(os.path.join(XDIR, "prm800k.jsonl"), "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    all_steps = [s for r in kept for s in r["steps"]]
    stats["prm800k"] = {
        "funnel": dict(funnel), "records": len(kept),
        "steps_per_solution": dist([r["n"] for r in kept]),
        "step_tokens": dist(token_lengths(tk, all_steps)),
        "subset_mix": dict(Counter(r["subset"] for r in kept))}
    print(f"[convert] prm800k: {len(kept):,} records; funnel={dict(funnel)}",
          flush=True)

    # ---------------- ProcessBench ----------------
    pb_dir = os.path.join(SRC, "processbench")
    files = sorted(
        glob.glob(os.path.join(pb_dir, "**", "*.json"), recursive=True) +
        glob.glob(os.path.join(pb_dir, "**", "*.jsonl"), recursive=True) +
        glob.glob(os.path.join(pb_dir, "**", "*.parquet"), recursive=True))
    files = [p for p in files if "tokenizer" not in p
             and os.path.basename(p) != "acquisition.json"]
    pb_kept, pb_funnel = [], Counter()
    for path in files:
        subset = os.path.splitext(os.path.basename(path))[0]
        for j, rec in enumerate(load_pb_file(path)):
            pb_funnel[f"{subset}_total"] += 1
            steps = [str(s).strip() for s in (rec.get("steps") or [])]
            steps = [s for s in steps if s]
            if len(steps) < 3:
                pb_funnel["lt3_steps"] += 1
                continue
            pb_funnel["kept"] += 1
            pb_kept.append({"sid": f"pb_{subset}_{rec.get('id', j)}",
                            "dataset": "processbench", "subset": subset,
                            "n": len(steps), "steps": steps})
    with open(os.path.join(XDIR, "processbench.jsonl"), "w") as f:
        for r in pb_kept:
            f.write(json.dumps(r) + "\n")
    pb_steps = [s for r in pb_kept for s in r["steps"]]
    stats["processbench"] = {
        "funnel": dict(pb_funnel), "records": len(pb_kept),
        "steps_per_solution": dist([r["n"] for r in pb_kept]),
        "step_tokens": dist(token_lengths(tk, pb_steps)),
        "subset_mix": dict(Counter(r["subset"] for r in pb_kept))}
    print(f"[convert] processbench: {len(pb_kept):,} records; "
          f"funnel={dict(pb_funnel)}", flush=True)

    # ---------------- VisualPRM gold reference (same measurement) ----------
    vp_n, vp_steps = [], []
    with open(GOLD_SAMPLE) as f:
        for line in f:
            r = json.loads(line)
            vp_n.append(len(r["chunks"]))
            vp_steps.extend(r["chunks"])
    stats["visualprm_gold_sample"] = {
        "source": GOLD_SAMPLE, "records": len(vp_n),
        "steps_per_solution": dist(vp_n),
        "step_tokens": dist(token_lengths(tk, vp_steps))}
    print(f"[convert] visualprm gold sample: {len(vp_n):,} records", flush=True)

    with open(os.path.join(XDIR, "convert_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    for name in ("prm800k", "processbench", "visualprm_gold_sample"):
        s = stats[name]
        print(f"[convert] {name}: steps/sol mean={s['steps_per_solution']['mean']:.2f} "
              f"p50={s['steps_per_solution']['p50']:.0f}; step tokens "
              f"mean={s['step_tokens']['mean']:.1f} p50={s['step_tokens']['p50']:.0f} "
              f"p99={s['step_tokens']['p99']:.0f}")
    print("[convert] DONE")


if __name__ == "__main__":
    sys.exit(main())
