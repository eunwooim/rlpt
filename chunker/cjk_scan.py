#!/usr/bin/env python
"""Measure the CJK multi-step pool, and how much of it could ever form an
UNSEEN eval slice for the step-2000 best checkpoint.

Replays prepare_data.py's pass-1 + grouping + split assignment VERBATIM (same
seed, same rng call order) so the train/val/test assignment matches the data the
model was actually built from. Then reports, for CJK multi-step trajectories:
  * how many exist corpus-wide and per split,
  * for the train ones, the size of their question group (total rollouts and
    train windows-worth of records) -- because a group is only usable as an
    unseen slice if the checkpoint touched NONE of its records.

The checkpoint saw 256,000 of 702,450 train windows (36.4%), drawn uniformly at
random, so P(a group is entirely unseen) ~= (1-0.364)^(records in that group).
This script reports that survival probability per candidate group, which tells us
whether the exact (fragile) dataloader-order replay is worth doing at all.

Writes cjk_pool_report.json + cjk_multi_candidates.jsonl (train-side candidates).
"""
import argparse, json, os, random
from collections import defaultdict

os.environ.setdefault("HF_HOME", "/scratch/sghos104/.hf_cache")
import chunker_common as cc

CANONICAL = "/scratch/sghos104/rlpt/canonical.jsonl"
SEEN_FRAC = 256000 / 702450.0     # step 2000 * (64*2) / train windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=CANONICAL)
    ap.add_argument("--out_dir", default="/scratch/sghos104/rlpt/chunker")
    # identical defaults to prepare_data.py -- required for an exact replay
    ap.add_argument("--single_total", type=int, default=128000)
    ap.add_argument("--val_multi", type=int, default=15000)
    ap.add_argument("--val_single", type=int, default=4000)
    ap.add_argument("--test_multi", type=int, default=15000)
    ap.add_argument("--test_single", type=int, default=4000)
    ap.add_argument("--large_group", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # ---------- pass 1: VERBATIM from prepare_data.py (same rng consumption) ----------
    q_total = defaultdict(int)
    multi_recs, single_res = [], []
    single_seen = 0
    n_lines = 0
    print("[pass1] scanning...", flush=True)
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            n_lines += 1
            steps = r.get("steps") or []
            q = r.get("question") or ""
            qh = cc.qhash(q)
            q_total[qh] += 1
            rec = {"steps": steps, "qh": qh}
            if len(steps) >= 2:
                multi_recs.append(rec)
            else:
                single_seen += 1
                if len(single_res) < args.single_total:
                    single_res.append(rec)
                else:
                    j = rng.randint(0, single_seen - 1)
                    if j < args.single_total:
                        single_res[j] = rec
            if n_lines % 1_000_000 == 0:
                print(f"  ...{n_lines:,} lines (multi {len(multi_recs):,})", flush=True)
    print(f"[pass1 done] lines={n_lines:,} multi={len(multi_recs):,} single={len(single_res):,}", flush=True)

    # ---------- grouping + assignment: VERBATIM ----------
    groups = defaultdict(lambda: {"multi": [], "single": []})
    for r in multi_recs:
        groups[r["qh"]]["multi"].append(r)
    for r in single_res:
        groups[r["qh"]]["single"].append(r)

    large, eligible = [], []
    for qh, g in groups.items():
        (large if q_total[qh] > args.large_group else eligible).append(qh)

    rng.shuffle(eligible)
    multi_groups = sorted([q for q in eligible if groups[q]["multi"]],
                          key=lambda q: (len(groups[q]["multi"]), -len(groups[q]["single"])), reverse=True)
    pure_single = [q for q in eligible if not groups[q]["multi"]]
    rng.shuffle(pure_single)

    assign = {}
    counts = {s: {"multi": 0, "single": 0} for s in ("val", "test", "train")}

    def take_multi(split, need_multi):
        for qh in multi_groups:
            if qh in assign:
                continue
            if counts[split]["multi"] >= need_multi:
                break
            assign[qh] = split
            counts[split]["multi"] += len(groups[qh]["multi"])
            counts[split]["single"] += len(groups[qh]["single"])

    def top_up_single(split, need_single):
        for qh in pure_single:
            if qh in assign:
                continue
            if counts[split]["single"] >= need_single:
                break
            assign[qh] = split
            counts[split]["single"] += len(groups[qh]["single"])

    take_multi("val", args.val_multi);   top_up_single("val", args.val_single)
    take_multi("test", args.test_multi); top_up_single("test", args.test_single)
    for qh in groups:
        if qh not in assign:
            assign[qh] = "train"
            counts["train"]["multi"] += len(groups[qh]["multi"])
            counts["train"]["single"] += len(groups[qh]["single"])

    print("[split composition replay]", {s: counts[s] for s in ("train", "val", "test")}, flush=True)

    # ---------- find CJK multi-step trajectories ----------
    by_split = defaultdict(int)
    lang_multi = defaultdict(int)
    cands = []
    for qh, g in groups.items():
        sp = assign[qh]
        for r in g["multi"]:
            text = cc.join_steps(r["steps"])
            lang = cc.lang_bucket(text)
            lang_multi[lang] += 1
            if lang == "cjk":
                by_split[sp] += 1
                if sp == "train":
                    # records of this question that live in train (all of them, since
                    # a question never spans splits): the whole group must be unseen.
                    n_group_recs = len(g["multi"]) + len(g["single"])
                    cands.append({
                        "qh": qh, "n_steps": len(r["steps"]),
                        "group_multi": len(g["multi"]), "group_single": len(g["single"]),
                        "group_records_in_train": n_group_recs,
                        "q_total_rollouts": q_total[qh],
                        "p_group_entirely_unseen": round((1 - SEEN_FRAC) ** n_group_recs, 6),
                        "steps": r["steps"],
                    })

    cands.sort(key=lambda c: c["group_records_in_train"])
    expected_survivors = sum(c["p_group_entirely_unseen"] for c in cands)

    report = {
        "n_lines": n_lines,
        "multi_total": len(multi_recs),
        "multi_by_language": dict(lang_multi),
        "cjk_multi_by_split": {k: by_split.get(k, 0) for k in ("train", "val", "test")},
        "cjk_multi_total": sum(by_split.values()),
        "checkpoint_seen_fraction_of_train_windows": round(SEEN_FRAC, 4),
        "n_train_cjk_multi_candidates": len(cands),
        "expected_unseen_survivors_by_group": round(expected_survivors, 3),
        "smallest_candidate_groups": cands[:10] and [
            {k: v for k, v in c.items() if k != "steps"} for c in cands[:10]],
    }
    with open(os.path.join(args.out_dir, "cjk_pool_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.out_dir, "cjk_multi_candidates.jsonl"), "w") as f:
        for c in cands:
            f.write(json.dumps(c) + "\n")
    print("\n" + json.dumps(report, indent=2))
    print(f"\n[written] cjk_pool_report.json ({len(cands)} train candidates)")


if __name__ == "__main__":
    main()
