#!/usr/bin/env python
"""Prepare training data for the chunk-boundary classifier.

One linear pass over canonical.jsonl to:
  * keep ALL multi-step trajectories (primary signal),
  * seed-sample single-step trajectories as negatives-only (~20% of train),
  * count full per-question rollouts (for the >1000 large-group rule),
then split GROUPED BY QUESTION (no question spans two splits):
  * question-groups with >LARGE_GROUP total rollouts -> TRAIN only,
  * val/test drawn from small/medium groups, preferring multi-rich groups so a
    single question can't dominate evaluation, topped up with pure-single groups
    to reach the single target,
then tokenize + sliding-window each trajectory and cache arrow datasets on
/scratch, and report the 1-vs-0 class balance under the expanded candidate rule.

Outputs under --out_dir:
  train/  val/  test/                 (arrow: windowed input_ids + labels)
  val_traj.jsonl  test_traj.jsonl     (trajectory-level: steps, lang, kind, ...)
  prepare_report.json                 (split sizes, class balance, mix)
"""
import argparse, json, os, random, sys
from collections import defaultdict

os.environ.setdefault("HF_HOME", "/scratch/sghos104/.hf_cache")
import chunker_common as cc

CANONICAL = "/scratch/sghos104/rlpt/canonical.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=CANONICAL)
    ap.add_argument("--out_dir", default="/scratch/sghos104/rlpt/chunker/data")
    ap.add_argument("--single_total", type=int, default=128000,
                    help="single-step negatives to sample (train ~120k + val/test 4k+4k)")
    ap.add_argument("--val_multi", type=int, default=15000)
    ap.add_argument("--val_single", type=int, default=4000)
    ap.add_argument("--test_multi", type=int, default=15000)
    ap.add_argument("--test_single", type=int, default=4000)
    ap.add_argument("--large_group", type=int, default=1000,
                    help="question-groups with more total rollouts than this go to train only")
    ap.add_argument("--max_multi_per_q", type=int, default=20,
                    help="HARD cap: a question contributing more multi rollouts than this is "
                         "ineligible for val/test, so no single question can dominate eval")
    ap.add_argument("--val_cjk_multi", type=int, default=400,
                    help="CJK multi-step trajectories to reserve for val (~corpus rate of 2.6%)")
    ap.add_argument("--test_cjk_multi", type=int, default=400)
    ap.add_argument("--min_questions_warn", type=int, default=500,
                    help="warn loudly if a val/test split has fewer distinct questions than this")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny fast subset for the smoke train (caps multi & single)")
    ap.add_argument("--smoke_multi", type=int, default=50000)
    ap.add_argument("--smoke_single", type=int, default=12000)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cc.MODEL)
    cls_id, sep_id = tok.cls_token_id, tok.sep_token_id

    single_target = args.smoke_single if args.smoke else args.single_total
    multi_cap = args.smoke_multi if args.smoke else None

    # ---------- pass 1: collect selected records + full question rollout counts ----------
    q_total = defaultdict(int)                 # rollouts per question over the WHOLE file
    multi_recs = []                            # full multi-step records kept
    single_res = []                            # reservoir of single-step records
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
                if multi_cap is None or len(multi_recs) < multi_cap:
                    multi_recs.append(rec)
            else:
                single_seen += 1
                if len(single_res) < single_target:
                    single_res.append(rec)
                else:
                    j = rng.randint(0, single_seen - 1)
                    if j < single_target:
                        single_res[j] = rec
            if n_lines % 1_000_000 == 0:
                print(f"  ...{n_lines:,} lines  (multi kept {len(multi_recs):,})", flush=True)
    print(f"[pass1 done] lines={n_lines:,}  multi={len(multi_recs):,}  single={len(single_res):,}", flush=True)

    # ---------- group selected records by question ----------
    # language is resolved here (cheap, no tokenizer) so CJK can be stratified into
    # val/test -- previously every CJK multi-step trajectory landed in train.
    groups = defaultdict(lambda: {"multi": [], "single": []})
    for r in multi_recs:
        r["lang"] = cc.lang_bucket(cc.join_steps(r["steps"]))
        groups[r["qh"]]["multi"].append(r)
    for r in single_res:
        groups[r["qh"]]["single"].append(r)

    def n_cjk_multi(qh):
        return sum(1 for r in groups[qh]["multi"] if r["lang"] == "cjk")

    large, eligible = [], []
    for qh, g in groups.items():
        (large if q_total[qh] > args.large_group else eligible).append(qh)
    # HARD per-question cap: a question with more than max_multi_per_q multi rollouts
    # can never enter val/test. Without this, the old "prefer multi-rich groups"
    # ordering produced val/test of 31 and 72 DISTINCT QUESTIONS (~490 near-duplicate
    # rollouts each) -- the exact domination this was meant to prevent.
    capped = [q for q in eligible if 0 < len(groups[q]["multi"]) <= args.max_multi_per_q]
    print(f"[groups] total={len(groups):,}  large(>{args.large_group})={len(large):,}  "
          f"eligible={len(eligible):,}  eval-eligible(multi<={args.max_multi_per_q})={len(capped):,}", flush=True)

    # ---------- assign splits (grouped by question) ----------
    # Random order over capped groups (NOT sorted by size): sorting descending is what
    # collapsed eval onto a handful of questions; sorting ascending would hand val/test
    # every singleton group -- which is exactly where all the CJK lives -- and strip CJK
    # out of train instead. A shuffle keeps a representative mix of small/medium groups.
    cjk_groups = [q for q in capped if n_cjk_multi(q) > 0]
    non_cjk_groups = [q for q in capped if n_cjk_multi(q) == 0]
    rng.shuffle(cjk_groups)
    rng.shuffle(non_cjk_groups)
    pure_single = [q for q in eligible if not groups[q]["multi"]]
    rng.shuffle(pure_single)
    print(f"[groups] eval-eligible with CJK multi={len(cjk_groups):,}  without={len(non_cjk_groups):,}", flush=True)

    assign = {}   # qh -> split
    counts = {s: {"multi": 0, "single": 0, "cjk_multi": 0} for s in ("val", "test", "train")}

    def _take(split, qh):
        assign[qh] = split
        counts[split]["multi"] += len(groups[qh]["multi"])
        counts[split]["single"] += len(groups[qh]["single"])
        counts[split]["cjk_multi"] += n_cjk_multi(qh)

    def take_cjk(split, need_cjk):
        for qh in cjk_groups:
            if qh in assign:
                continue
            if counts[split]["cjk_multi"] >= need_cjk:
                break
            _take(split, qh)

    def take_multi(split, need_multi):
        for qh in non_cjk_groups:
            if qh in assign:
                continue
            if counts[split]["multi"] >= need_multi:
                break
            _take(split, qh)

    def top_up_single(split, need_single):
        for qh in pure_single:
            if qh in assign:
                continue
            if counts[split]["single"] >= need_single:
                break
            assign[qh] = split
            counts[split]["single"] += len(groups[qh]["single"])

    # CJK quota first (so per-language F1 is measurable at all), then fill with non-CJK
    take_cjk("val", args.val_cjk_multi);   take_multi("val", args.val_multi)
    top_up_single("val", args.val_single)
    take_cjk("test", args.test_cjk_multi); take_multi("test", args.test_multi)
    top_up_single("test", args.test_single)
    for qh in groups:                      # everything else -> train (incl. all large groups)
        if qh not in assign:
            _take("train", qh)

    print("[split composition]")
    for s in ("train", "val", "test"):
        print(f"  {s:5s}: multi={counts[s]['multi']:,}  single={counts[s]['single']:,}  "
              f"cjk_multi={counts[s]['cjk_multi']:,}")

    # ---------- tokenize + window + write ----------
    # Stream windows straight to arrow via from_generator so peak memory stays
    # bounded (the interactive node is capped at 4 GB; accumulating all windows in
    # Python lists OOMs). Class balance is counted inside the generator.
    from datasets import Dataset, Features, Sequence, Value
    os.makedirs(args.out_dir, exist_ok=True)
    feat = Features({"input_ids": Sequence(Value("int32")), "labels": Sequence(Value("int16"))})
    bal = {s: {"1": 0, "0": 0} for s in ("train", "val", "test")}
    bal_by_kind = {"multi": {"1": 0, "0": 0}, "single": {"1": 0, "0": 0}}

    def iter_split(split):
        for qh, g in groups.items():
            if assign[qh] != split:
                continue
            for kind in ("multi", "single"):
                for r in g[kind]:
                    yield r, kind

    def process(split, want_traj):
        traj_rows = []
        prog = {"n": 0}

        def gen():
            for r, kind in iter_split(split):
                ex = cc.build_token_labels(r["steps"], tok)
                for l in ex["labels"]:
                    if l == 1: bal[split]["1"] += 1; bal_by_kind[kind]["1"] += 1
                    elif l == 0: bal[split]["0"] += 1; bal_by_kind[kind]["0"] += 1
                if want_traj:
                    traj_rows.append({"steps": r["steps"], "text": ex["text"],
                                      "kind": kind, "lang": cc.lang_bucket(ex["text"]),
                                      "qh": r["qh"], "n_steps": len(r["steps"])})
                prog["n"] += 1
                if prog["n"] % 50000 == 0:
                    print(f"    [{split}] {prog['n']:,} trajectories tokenized", flush=True)
                for w in cc.make_windows(ex["input_ids"], ex["offsets"], ex["labels"], cls_id, sep_id):
                    yield {"input_ids": w["input_ids"], "labels": w["labels"]}

        ds = Dataset.from_generator(gen, features=feat, keep_in_memory=False)
        ds.save_to_disk(os.path.join(args.out_dir, split))
        if want_traj:
            with open(os.path.join(args.out_dir, f"{split}_traj.jsonl"), "w") as fo:
                for row in traj_rows:
                    fo.write(json.dumps(row) + "\n")
        print(f"  [{split}] windows={len(ds):,}  trajectories={prog['n']:,}", flush=True)
        return len(ds), prog["n"]

    print("[tokenize+window] train ...", flush=True)
    tr_w, tr_n = process("train", want_traj=False)
    print("[tokenize+window] val ...", flush=True)
    va_w, va_n = process("val", want_traj=True)
    print("[tokenize+window] test ...", flush=True)
    te_w, te_n = process("test", want_traj=True)

    def ratio(b): return round(b["0"] / max(1, b["1"]), 2)

    # ---------- split-health diagnostics (FIRST-CLASS CHECKS) ----------
    # These exist because trajectory counts alone looked perfect (15,210/4,005) while
    # val/test actually covered 31 and 72 distinct questions. Never judge a split by
    # trajectory count again -- judge it by distinct questions and rollout spread.
    def pct(xs, p):
        return sorted(xs)[min(len(xs) - 1, int(p * len(xs)))] if xs else 0

    health = {}
    for s in ("train", "val", "test"):
        qs = [qh for qh in groups if assign[qh] == s]
        multi_qs = [qh for qh in qs if groups[qh]["multi"]]
        roll = [len(groups[qh]["multi"]) for qh in multi_qs]
        health[s] = {
            "distinct_questions": len(qs),
            "distinct_questions_with_multi": len(multi_qs),
            "multi_rollouts_per_question": {
                "mean": round(sum(roll) / max(1, len(roll)), 2), "p50": pct(roll, 0.5),
                "p90": pct(roll, 0.9), "max": max(roll) if roll else 0},
            "cjk_multi_trajectories": counts[s]["cjk_multi"],
        }

    flags = []
    for s in ("val", "test"):
        nq = health[s]["distinct_questions_with_multi"]
        if nq < args.min_questions_warn:
            flags.append(f"{s}: only {nq} distinct multi-step questions "
                         f"(< {args.min_questions_warn}) -- eval may be dominated by few questions")
        if health[s]["cjk_multi_trajectories"] == 0:
            flags.append(f"{s}: ZERO CJK multi-step trajectories -- per-language CJK F1 unmeasurable")
        mx = health[s]["multi_rollouts_per_question"]["max"]
        if mx > args.max_multi_per_q:
            flags.append(f"{s}: a question contributes {mx} multi rollouts (> cap {args.max_multi_per_q})")
        # the cap can starve the target if few small/medium groups exist
        want = args.val_multi if s == "val" else args.test_multi
        got = counts[s]["multi"]
        if got < 0.9 * want:
            flags.append(f"{s}: only {got:,} multi trajectories vs target {want:,} — the "
                         f"max_multi_per_q={args.max_multi_per_q} cap starved it; raise the cap")

    report = {
        "n_lines": n_lines,
        "smoke": args.smoke,
        "split_health": health,
        "split_health_flags": flags or ["none -- all split-health checks passed"],
        "mix": {"multi_kept": len(multi_recs), "single_sampled": len(single_res),
                "single_frac_of_selected": round(len(single_res) / max(1, len(multi_recs) + len(single_res)), 3)},
        "groups": {"total": len(groups), "large": len(large), "eligible": len(eligible)},
        "split_trajectories": {"train": {"multi": counts["train"]["multi"], "single": counts["train"]["single"]},
                               "val": {"multi": counts["val"]["multi"], "single": counts["val"]["single"]},
                               "test": {"multi": counts["test"]["multi"], "single": counts["test"]["single"]}},
        "split_windows": {"train": tr_w, "val": va_w, "test": te_w},
        "class_balance_over_candidate_tokens": {
            s: {"label1": bal[s]["1"], "label0": bal[s]["0"], "ratio_1_to_0": ratio(bal[s])}
            for s in ("train", "val", "test")},
        "class_balance_by_kind": {
            k: {"label1": bal_by_kind[k]["1"], "label0": bal_by_kind[k]["0"], "ratio_1_to_0": ratio(bal_by_kind[k])}
            for k in ("multi", "single")},
    }
    with open(os.path.join(args.out_dir, "prepare_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\n" + json.dumps(report, indent=2))
    print(f"\n[written] {args.out_dir}/{{train,val,test}} + val_traj/test_traj.jsonl + prepare_report.json")
    # loud warning if balance worse than ~1:10, per brief
    tr = report["class_balance_over_candidate_tokens"]["train"]["ratio_1_to_0"]
    if tr > 10:
        print(f"\n[WARN] train class balance 1:{tr} is worse than 1:10 — review before training.", file=sys.stderr)
    for fl in flags:
        print(f"\n[SPLIT-HEALTH WARN] {fl}", file=sys.stderr)
    if not flags:
        print("\n[split-health] all checks passed "
              f"(val {health['val']['distinct_questions_with_multi']} / "
              f"test {health['test']['distinct_questions_with_multi']} distinct multi-step questions; "
              f"CJK multi val {health['val']['cjk_multi_trajectories']} / "
              f"test {health['test']['cjk_multi_trajectories']})")


if __name__ == "__main__":
    main()
