#!/usr/bin/env python
"""jb_run.py — one shard of one judge-benchmark task on one GPU. --task 3a|3b|3c_nb|3c_para|3d|3e_select|3e_score --model cur|v3 --shard i --nshards K
Outputs out/<task>_<model>_<shard>.jsonl (resume-safe: skipped when present). Logs NLI calls and throughput."""
import argparse
import collections
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jb_lib as jb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True); ap.add_argument("--model", required=True, choices=["cur", "v3"])
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--device", default=None); ap.add_argument("--bs", type=int, default=32)
a = ap.parse_args()
OUT = jb.OUT
out_path = f"{OUT}/{a.task}_{a.model}_{a.shard}.jsonl"
if os.path.exists(out_path) and os.path.getsize(out_path) > 0 and not a.limit:
    print(f"[jb] {out_path} exists; nothing to do"); sys.exit(0)


def shard(rows):
    rows = rows[a.shard::a.nshards]
    return rows[:a.limit] if a.limit else rows


def load(name):
    return [json.loads(l) for l in open(f"{OUT}/{name}.jsonl")]


t0 = time.time()
J = jb.Judge(a.model, batch_size=a.bs, device=a.device)
emb = jb.Embedder(a.device) if a.task in ("3a", "3b", "3e_score") else None
print(f"[jb] task={a.task} model={a.model} ({J.name}) shard={a.shard}/{a.nshards} loaded in {time.time()-t0:.0f}s", flush=True)
results = []

if a.task == "3a":
    rows = shard(load("sample_3a"))
    ed = jb.score_pairs(J, emb, [(r["neg_step"], r["pos_step"]) for r in rows])
    bl = jb.score_pairs(J, emb, [(r["base_step"], r["pos_step"]) for r in rows])
    for r, e, b in zip(rows, ed, bl):
        results.append({"idx": r["idx"], "source": r["source"], "edit": r["edit"], "category": r["category"], "k": r["k"], "n_steps": r["n_steps"], "edited": e, "baseline": b})
elif a.task == "3b":
    rows = shard(load("sample_3b"))
    sc = jb.score_pairs(J, emb, [(r["paraphrase"], r["pos"]) for r in rows])
    for r, s in zip(rows, sc):
        results.append({"idx": r["idx"], "para": s})
elif a.task in ("3c_nb", "3c_para"):
    S = jb.RewardAdapter(J)
    rows = shard(load("sample_3a" if a.task == "3c_nb" else "sample_3b"))
    for n, r in enumerate(rows):
        gold, _ = jb.rv.split_steps(r["pos"])
        rec = {"idx": r["idx"]}
        conds = {"neg": r["neg"], "base": r["baseline"]} if a.task == "3c_nb" else {"para": r["paraphrase"]}
        for cond, text in conds.items():
            b, pairs, kept, dup = jb.reward_match(S, text, gold)
            rec[f"match_{cond}"] = b.match; rec[f"n_pairs_{cond}"] = len(pairs); rec[f"n_segs_{cond}"] = b.n_segs; rec[f"gated_{cond}"] = int(b.gated)
            if cond == "neg":
                k = r["k"]
                hit = [p for p in pairs if p[0] == k]
                rec["edited_in_dup"] = int(k in dup)
                rec["edited_s"] = hit[0][2] if hit else 0.0
                rec["edited_assigned_to_k"] = int(bool(hit) and hit[0][1] == k)
                rec["edited_credited"] = int(bool(hit) and hit[0][2] >= 0.5)
        results.append(rec)
        if n % 100 == 0:
            print(f"[jb] {n}/{len(rows)} rows, {J.calls} calls, {J.calls/max(J.secs,1e-9):.1f} pairs/s", flush=True)
elif a.task == "3d":
    rows = shard(load("sample_3d"))
    reqs = [(r["premise"], r["hypothesis"]) for r in rows] + [(r["hypothesis"], r["premise"]) for r in rows]
    J.flush(reqs)
    for r in rows:
        e, c, n = J.get(r["premise"], r["hypothesis"])
        pred = max((("entailment", e), ("contradiction", c), ("neutral", n)), key=lambda x: x[1])[0]
        results.append({"subset": r["subset"], "i": r["i"], "gold": r["gold"], "E": e, "C": c, "N": n, "pred": pred, "s": J.s(r["premise"], r["hypothesis"])})
elif a.task == "3e_select":
    assert a.model == "cur", "pair selection uses the current judge (the reward's actual assignment)"
    S = jb.RewardAdapter(J)
    run = f"{jb.REPO}/grpo_arms/runs/arm1_3b_matchv4_softgate/rollouts"
    cells = collections.defaultdict(list)
    for step in range(1, 184):
        p = f"{run}/{step}.jsonl"
        if not os.path.exists(p):
            continue
        bucket = "1-60" if step <= 60 else ("61-120" if step <= 120 else "121-183")
        for line in open(p, errors="replace"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            cells[(bucket, int(round(float(r.get("acc", 0)))))].append((step, r))
    rnd = random.Random(0)
    pid = 0
    for (bucket, acc), rows in sorted(cells.items()):
        rnd.shuffle(rows)
        got = 0
        for step, r in rows:
            if got >= 50:
                break
            try:
                gold = json.loads(r["gts"])["gold_steps"]
            except Exception:  # noqa: BLE001
                continue
            segs, _ = jb.rv.split_steps(r["output"])
            if not segs or not gold:
                continue
            b, pairs, kept, dup = jb.reward_match(S, r["output"], gold)
            if not pairs:
                continue
            i, j, s = rnd.choice(pairs)
            gold_clean = [jb.rv.THINK_RE.sub("", g).strip() for g in gold if not jb.rv.FINAL_RE.match(g.strip())]
            gold_clean = [g for g in gold_clean if g]
            results.append({"pair_id": pid, "step": step, "bucket": bucket, "acc": acc, "rollout_step": segs[i], "ref_step": gold_clean[j],
                            "i": i, "j": j, "s_assign_cur": s, "match_cur": b.match, "n_segs": len(segs), "n_gold": len(gold_clean)})
            pid += 1; got += 1
        print(f"[jb] cell {bucket} acc={acc}: {got} pairs", flush=True)
    out_path = f"{OUT}/3e_pairs.jsonl"
elif a.task == "3e_score":
    rows = load("3e_pairs")
    sc = jb.score_pairs(J, emb, [(r["rollout_step"], r["ref_step"]) for r in rows])
    for r, s in zip(rows, sc):
        results.append({"pair_id": r["pair_id"], "scores": s})
else:
    sys.exit(f"unknown task {a.task}")

with open(out_path, "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[jb] DONE task={a.task} model={a.model} shard={a.shard}: {len(results)} rows, NLI calls={J.calls}, nli_secs={J.secs:.1f}, "
      f"throughput={J.calls/max(J.secs,1e-9):.1f} pairs/s (bs {a.bs}), wall={time.time()-t0:.0f}s -> {out_path}", flush=True)
