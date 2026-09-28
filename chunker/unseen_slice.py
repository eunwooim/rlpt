#!/usr/bin/env python
"""Build a CJK eval slice the step-2000 best checkpoint NEVER SAW.

Why this is needed: all 13,103 CJK multi-step trajectories were assigned to
train, so a naive CJK slice would measure memorization, not generalization. But
the best checkpoint is step 2000 = 256,000 of 702,450 windows (36.4% of one
epoch), so most train windows were never actually consumed.

Three stages:
  1. Replay prepare_data.py's seeded grouping + train iteration order VERBATIM,
     tokenizing in the same order, to recover each trajectory's window index
     range (the prepared arrow keeps only input_ids/labels -- no ids survive).
  2. Replay train.py's setup VERBATIM to recover the RandomSampler permutation,
     and take the first N_SEEN indices as the windows the checkpoint consumed.
  3. Emit CJK multi-step trajectories that are (a) entirely unseen AND (b) whose
     whole question group is unseen -- plus a matched SEEN control set.

The sampler in transformers 5.14 is a bare RandomSampler with NO explicit
generator: its permutation is seeded off the global torch RNG state, so this
replay is only valid if RNG consumption matches exactly. That cannot be proven,
only argued -- which is why we emit the SEEN control. If the replay is sound and
memorization is real, F1(seen) >> F1(unseen). If the two are equal, either there
is no memorization or the replay is wrong; both readings say do not trust a rosy
unseen number by itself.

Outputs: cjk_unseen_traj.jsonl, cjk_seen_traj.jsonl (val_traj schema, so
evaluate.py consumes them directly) + unseen_slice_report.json
"""
import argparse, json, os, random
from collections import defaultdict

os.environ.setdefault("HF_HOME", "/scratch/sghos104/.hf_cache")
import chunker_common as cc

CANONICAL = "/scratch/sghos104/rlpt/canonical.jsonl"


def replay_groups(input_path, seed, single_total, val_multi, val_single,
                  test_multi, test_single, large_group):
    """VERBATIM replay of prepare_data.py pass1 + grouping + assignment."""
    rng = random.Random(seed)
    q_total = defaultdict(int)
    multi_recs, single_res = [], []
    single_seen = 0
    n_lines = 0
    print("[stage1] pass1 scanning...", flush=True)
    with open(input_path, "r", encoding="utf-8") as f:
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
            qh = cc.qhash(r.get("question") or "")
            q_total[qh] += 1
            rec = {"steps": steps, "qh": qh}
            if len(steps) >= 2:
                multi_recs.append(rec)
            else:
                single_seen += 1
                if len(single_res) < single_total:
                    single_res.append(rec)
                else:
                    j = rng.randint(0, single_seen - 1)
                    if j < single_total:
                        single_res[j] = rec
            if n_lines % 2_000_000 == 0:
                print(f"  ...{n_lines:,} lines", flush=True)
    print(f"[stage1] lines={n_lines:,} multi={len(multi_recs):,} single={len(single_res):,}", flush=True)

    groups = defaultdict(lambda: {"multi": [], "single": []})
    for r in multi_recs:
        groups[r["qh"]]["multi"].append(r)
    for r in single_res:
        groups[r["qh"]]["single"].append(r)

    large, eligible = [], []
    for qh in groups:
        (large if q_total[qh] > large_group else eligible).append(qh)
    rng.shuffle(eligible)
    multi_groups = sorted([q for q in eligible if groups[q]["multi"]],
                          key=lambda q: (len(groups[q]["multi"]), -len(groups[q]["single"])), reverse=True)
    pure_single = [q for q in eligible if not groups[q]["multi"]]
    rng.shuffle(pure_single)

    assign = {}
    counts = {s: {"multi": 0, "single": 0} for s in ("val", "test", "train")}

    def take_multi(split, need):
        for qh in multi_groups:
            if qh in assign:
                continue
            if counts[split]["multi"] >= need:
                break
            assign[qh] = split
            counts[split]["multi"] += len(groups[qh]["multi"])
            counts[split]["single"] += len(groups[qh]["single"])

    def top_up_single(split, need):
        for qh in pure_single:
            if qh in assign:
                continue
            if counts[split]["single"] >= need:
                break
            assign[qh] = split
            counts[split]["single"] += len(groups[qh]["single"])

    take_multi("val", val_multi);   top_up_single("val", val_single)
    take_multi("test", test_multi); top_up_single("test", test_single)
    for qh in groups:
        if qh not in assign:
            assign[qh] = "train"
            counts["train"]["multi"] += len(groups[qh]["multi"])
            counts["train"]["single"] += len(groups[qh]["single"])
    print(f"[stage1] split replay: {dict(counts)}", flush=True)
    return groups, assign, counts, q_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=CANONICAL)
    ap.add_argument("--data_dir", default="/scratch/sghos104/rlpt/chunker/data")
    ap.add_argument("--out_dir", default="/scratch/sghos104/rlpt/chunker")
    ap.add_argument("--best_step", type=int, default=2000)
    ap.add_argument("--per_device_batch", type=int, default=64)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--eval_subset", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--single_total", type=int, default=128000)
    ap.add_argument("--val_multi", type=int, default=15000)
    ap.add_argument("--val_single", type=int, default=4000)
    ap.add_argument("--test_multi", type=int, default=15000)
    ap.add_argument("--test_single", type=int, default=4000)
    ap.add_argument("--large_group", type=int, default=1000)
    ap.add_argument("--max_emit", type=int, default=4000)
    args = ap.parse_args()

    n_seen = args.best_step * args.per_device_batch * args.grad_accum

    groups, assign, counts, q_total = replay_groups(
        args.input, args.seed, args.single_total, args.val_multi, args.val_single,
        args.test_multi, args.test_single, args.large_group)

    # ---------- stage 1b: window ranges, in the exact prepare order ----------
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cc.MODEL)
    cls_id, sep_id = tok.cls_token_id, tok.sep_token_id

    print("[stage1b] tokenizing train in prepare order to recover window ranges...", flush=True)
    traj = []          # (qh, lang_is_cjk, kind, w_start, w_count, idx_in_group)
    cjk_steps = {}     # traj_index -> steps (only for CJK multi; keeps memory bounded)
    w = 0
    n = 0
    for qh, g in groups.items():
        if assign[qh] != "train":
            continue
        for kind in ("multi", "single"):
            for r in g[kind]:
                ex = cc.build_token_labels(r["steps"], tok)
                cnt = sum(1 for _ in cc.make_windows(ex["input_ids"], ex["offsets"], ex["labels"], cls_id, sep_id))
                is_cjk_multi = (kind == "multi" and cc.lang_bucket(ex["text"]) == "cjk")
                traj.append((qh, is_cjk_multi, kind, w, cnt))
                if is_cjk_multi:
                    cjk_steps[len(traj) - 1] = r["steps"]
                w += cnt
                n += 1
                if n % 100000 == 0:
                    print(f"    {n:,} trajectories, {w:,} windows", flush=True)
    print(f"[stage1b] train trajectories={n:,} windows={w:,} (expect 702,450)", flush=True)
    total_windows = w

    # ---------- stage 2: replay train.py setup to recover sampler order ----------
    print("[stage2] replaying train.py setup for the sampler permutation...", flush=True)
    import torch
    from datasets import load_from_disk
    from transformers import (AutoModelForTokenClassification, DataCollatorForTokenClassification,
                              TrainingArguments, Trainer)

    model = AutoModelForTokenClassification.from_pretrained(
        cc.MODEL, num_labels=2, id2label={0: "no_split", 1: "split"}, label2id={"no_split": 0, "split": 1})
    model = model.float()
    train_ds = load_from_disk(os.path.join(args.data_dir, "train"))
    val_ds = load_from_disk(os.path.join(args.data_dir, "val"))
    eval_ds = val_ds.shuffle(seed=args.seed).select(range(args.eval_subset)) \
        if args.eval_subset and len(val_ds) > args.eval_subset else val_ds
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    collator = DataCollatorForTokenClassification(tok, label_pad_token_id=cc.LABEL_IGNORE, max_length=cc.MAX_LEN)
    targs = TrainingArguments(
        output_dir=os.path.join(args.out_dir, "runs/_replay_tmp"),
        num_train_epochs=3.0, learning_rate=3e-5, warmup_ratio=0.05, weight_decay=0.01,
        lr_scheduler_type="linear",
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        bf16=False, fp16=False,
        eval_strategy="steps", eval_steps=1000, save_strategy="steps", save_steps=1000,
        save_total_limit=3, load_best_model_at_end=True, metric_for_best_model="best_f1",
        greater_is_better=True, logging_steps=100, report_to="none", seed=args.seed,
        dataloader_num_workers=min(4, os.cpu_count() or 1), eval_accumulation_steps=8,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=collator)
    dl = trainer.get_train_dataloader()
    sampler = getattr(dl, "sampler", None)
    if sampler is None and getattr(dl, "batch_sampler", None) is not None:
        sampler = dl.batch_sampler.sampler
    print(f"[stage2] sampler={type(sampler).__name__}", flush=True)
    order = []
    for i, idx in enumerate(sampler):
        order.append(int(idx))
        if len(order) >= n_seen:
            break
    seen = set(order)
    print(f"[stage2] recovered {len(order):,} sampled window indices; unique={len(seen):,} "
          f"(target {n_seen:,}, {len(seen)/max(1,total_windows):.4f} of train)", flush=True)

    # ---------- stage 3: unseen CJK slice + seen control ----------
    q_has_seen = defaultdict(bool)
    traj_seen = []
    for (qh, is_cjk, kind, s, cnt) in traj:
        t_seen = any((s + k) in seen for k in range(cnt))
        traj_seen.append(t_seen)
        if t_seen:
            q_has_seen[qh] = True

    unseen_rows, seen_rows = [], []
    for i, (qh, is_cjk, kind, s, cnt) in enumerate(traj):
        if not is_cjk:
            continue
        steps = cjk_steps.get(i)
        if steps is None:
            continue
        text = cc.join_steps(steps)
        row = {"steps": steps, "text": text, "kind": "multi", "lang": "cjk",
               "qh": qh, "n_steps": len(steps)}
        if (not traj_seen[i]) and (not q_has_seen[qh]):
            unseen_rows.append(row)
        elif traj_seen[i]:
            seen_rows.append(row)

    unseen_rows = unseen_rows[:args.max_emit]
    seen_rows = seen_rows[:args.max_emit]
    for name, rows in (("cjk_unseen_traj.jsonl", unseen_rows), ("cjk_seen_traj.jsonl", seen_rows)):
        with open(os.path.join(args.out_dir, name), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    n_cjk = sum(1 for t in traj if t[1])
    report = {
        "best_step": args.best_step,
        "windows_seen_by_checkpoint": n_seen,
        "train_windows_total": total_windows,
        "seen_fraction": round(len(seen) / max(1, total_windows), 4),
        "train_cjk_multi_total": n_cjk,
        "cjk_unseen_and_group_clean": len(unseen_rows),
        "cjk_seen_control": len(seen_rows),
        "note": ("unseen = no window of the trajectory in the sampled prefix AND no record "
                 "of its question group seen. Validity depends on RNG-consumption parity in "
                 "the sampler replay; the seen control set is the check."),
    }
    with open(os.path.join(args.out_dir, "unseen_slice_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
