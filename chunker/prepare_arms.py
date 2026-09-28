#!/usr/bin/env python
"""Build train sets for the single-step negative-policy arms.

WHY: 83% of single-step VisualPRM records carry step markers, numbered lists,
or blank-line breaks (audit 2026-08-07). They were emitted as negative-only
examples, i.e. "do not split here" at real boundaries. These arms measure the
cost.

WHY NOT just re-run prepare_data.py per arm: it draws val/test from the SAME
rng stream as the single-step reservoir (one random.Random(seed); ~8.98M
randint calls in pass 1, then rng.shuffle for group assignment). Changing the
single policy shifts every later draw, so each arm would get a DIFFERENT
val/test. Instead we freeze val/test by question hash, read from the existing
data/{val,test}_traj.jsonl, and exclude those questions from train entirely.

ARMS
  keep   all sampled singles, labeled as now      (baseline / anchor)
  drop   no single-step records in train
  clean  only structurally clean singles (~18%)
  mask   all singles, marker-adjacent candidates forced to -100

Output: <out_dir>/<arm>/{train, val -> frozen, test -> frozen}
"""
import argparse, json, os, random
from collections import defaultdict

os.environ.setdefault("HF_HOME", "/scratch/sghos104/.hf_cache")
import chunker_common as cc
from single_step_policy import is_clean_single, apply_policy

CANONICAL = "/scratch/sghos104/rlpt/canonical.jsonl"
IGNORE = -100


def frozen_qhashes(split_dir):
    """Question hashes reserved for val/test in the v2 split. Train excludes them."""
    qs = set()
    for name in ("val_traj.jsonl", "test_traj.jsonl"):
        p = os.path.join(split_dir, name)
        with open(p, encoding="utf-8") as f:
            for line in f:
                qs.add(json.loads(line)["qh"])
        print(f"  {name}: cumulative {len(qs):,} frozen question hashes", flush=True)
    return qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=CANONICAL)
    ap.add_argument("--split_dir", default="/scratch/sghos104/rlpt/chunker/data",
                    help="frozen v2 split; val/test are read and symlinked, never rebuilt")
    ap.add_argument("--out_dir", default="/scratch/sghos104/rlpt/chunker/data_arms")
    ap.add_argument("--single_total", type=int, default=120000,
                    help="single-step negatives for TRAIN (v2 train had ~120k)")
    ap.add_argument("--policies", default="keep,drop,clean,mask")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke_multi", type=int, default=0, help=">0 caps multi records")
    args = ap.parse_args()
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    rng = random.Random(args.seed)

    from transformers import AutoTokenizer
    from datasets import Dataset, Features, Sequence, Value, concatenate_datasets
    tok = AutoTokenizer.from_pretrained(cc.MODEL)
    cls_id, sep_id = tok.cls_token_id, tok.sep_token_id
    feat = Features({"input_ids": Sequence(Value("int32")),
                     "labels": Sequence(Value("int16"))})

    print("[freeze] loading val/test question hashes", flush=True)
    frozen = frozen_qhashes(args.split_dir)

    # ---------- pass 1 ----------
    # Train-eligible records only: anything whose question is in val/test is
    # skipped outright, so no arm can influence evaluation data.
    multi_recs, single_res = [], []
    single_seen = n_lines = skipped = 0
    print("[pass1] scanning canonical.jsonl", flush=True)
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            n_lines += 1
            qh = cc.qhash(r.get("question") or "")
            if qh in frozen:
                skipped += 1
            else:
                steps = r.get("steps") or []
                if len(steps) >= 2:
                    if not args.smoke_multi or len(multi_recs) < args.smoke_multi:
                        multi_recs.append({"steps": steps, "qh": qh})
                else:
                    single_seen += 1
                    if len(single_res) < args.single_total:
                        single_res.append({"steps": steps, "qh": qh})
                    else:
                        j = rng.randint(0, single_seen - 1)
                        if j < args.single_total:
                            single_res[j] = {"steps": steps, "qh": qh}
            if n_lines % 1_000_000 == 0:
                print(f"  ...{n_lines:,} lines  multi={len(multi_recs):,} "
                      f"single_seen={single_seen:,} frozen_skipped={skipped:,}", flush=True)
    print(f"[pass1 done] lines={n_lines:,}  multi={len(multi_recs):,}  "
          f"singles_sampled={len(single_res):,} of {single_seen:,}  "
          f"frozen_skipped={skipped:,}", flush=True)

    # how many sampled singles survive each record-level filter
    n_clean = sum(1 for r in single_res if is_clean_single(cc.join_steps(r["steps"])))
    print(f"[policy] structurally clean singles: {n_clean:,}/{len(single_res):,} "
          f"({100*n_clean/max(1,len(single_res)):.1f}%)  -- expect ~18%", flush=True)

    bal = defaultdict(lambda: {"1": 0, "0": 0})

    def emit(records, policy, tag):
        """Tokenize -> label -> (policy) -> window -> arrow."""
        prog = {"n": 0}

        def gen():
            for r in records:
                ex = cc.build_token_labels(r["steps"], tok)
                labels = ex["labels"]
                if policy == "mask":
                    labels = apply_policy(labels, ex["offsets"], ex["text"], "mask",
                                          ignore_index=IGNORE)
                for l in labels:
                    if l == 1:   bal[tag]["1"] += 1
                    elif l == 0: bal[tag]["0"] += 1
                prog["n"] += 1
                if prog["n"] % 50000 == 0:
                    print(f"    [{tag}] {prog['n']:,} trajectories", flush=True)
                for w in cc.make_windows(ex["input_ids"], ex["offsets"], labels,
                                         cls_id, sep_id):
                    yield {"input_ids": w["input_ids"], "labels": w["labels"]}

        ds = Dataset.from_generator(gen, features=feat, keep_in_memory=False)
        print(f"  [{tag}] windows={len(ds):,}  trajectories={prog['n']:,}", flush=True)
        return ds

    os.makedirs(args.out_dir, exist_ok=True)
    cache = os.path.join(args.out_dir, "_cache")
    os.makedirs(cache, exist_ok=True)

    # multi half: identical across every arm, so build it once
    multi_path = os.path.join(cache, "multi")
    if os.path.exists(os.path.join(multi_path, "state.json")):
        from datasets import load_from_disk
        ds_multi = load_from_disk(multi_path)
        print(f"[multi] reusing cache: {len(ds_multi):,} windows", flush=True)
    else:
        print("[multi] tokenizing (once, shared by all arms)", flush=True)
        ds_multi = emit(multi_recs, "none", "multi")
        ds_multi.save_to_disk(multi_path)

    report = {"n_lines": n_lines, "frozen_questions": len(frozen),
              "frozen_records_skipped": skipped,
              "multi_records": len(multi_recs), "multi_windows": len(ds_multi),
              "singles_sampled": len(single_res), "singles_clean": n_clean,
              "arms": {}}

    for pol in policies:
        arm_dir = os.path.join(args.out_dir, pol)
        os.makedirs(arm_dir, exist_ok=True)
        if pol == "drop":
            ds = ds_multi
            n_single = 0
        else:
            recs = single_res
            if pol == "clean":
                recs = [r for r in single_res
                        if is_clean_single(cc.join_steps(r["steps"]))]
            n_single = len(recs)
            ds_single = emit(recs, pol, f"single_{pol}")
            ds = concatenate_datasets([ds_multi, ds_single])
        ds.save_to_disk(os.path.join(arm_dir, "train"))

        # frozen val/test, byte-identical across arms
        for name in ("val", "test"):
            link = os.path.join(arm_dir, name)
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link) if os.path.islink(link) else None
            os.symlink(os.path.join(args.split_dir, name), link)
        for name in ("val_traj.jsonl", "test_traj.jsonl"):
            link = os.path.join(arm_dir, name)
            if os.path.islink(link):
                os.remove(link)
            if not os.path.exists(link):
                os.symlink(os.path.join(args.split_dir, name), link)

        b = bal[f"single_{pol}"] if pol != "drop" else {"1": 0, "0": 0}
        tot1 = bal["multi"]["1"] + b["1"]
        tot0 = bal["multi"]["0"] + b["0"]
        report["arms"][pol] = {
            "train_windows": len(ds), "single_records": n_single,
            "label1": tot1, "label0": tot0,
            "ratio_0_per_1": round(tot0 / max(1, tot1), 2)}
        print(f"[arm {pol}] windows={len(ds):,}  singles={n_single:,}  "
              f"balance 1:{report['arms'][pol]['ratio_0_per_1']}", flush=True)

    with open(os.path.join(args.out_dir, "prepare_arms_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\n[done] " + json.dumps(report["arms"], indent=2))
    print("v2 train balance was 1:6.64 -- a large shift here is expected for "
          "drop/clean (fewer label-0) and for mask (label-0 -> -100).")


if __name__ == "__main__":
    main()
