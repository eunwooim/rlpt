"""Accuracy/format scoring for MCQ-only eval generations (no GT rationales, so
no bipartite match — e.g. MMK12). One TSV row per generation file: overall acc,
format rate, and per-subject accuracy.

Usage:
  python tools/score_mcq_eval.py --gen-dir outputs/validation_hard \
      --pattern "mmk12_*.jsonl" --out data/logs/validation_results/validation_mmk12.tsv
"""

import argparse
import glob
import json
import os
import sys

os.environ.setdefault("RLPT_REWARD_DEVICE", "cpu")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from graph_match_reward import answer_correct, parse_output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--pattern", default="mmk12_*.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.gen_dir, args.pattern)))
    if not paths:
        sys.exit(f"no files matching {args.pattern} in {args.gen_dir}")

    subjects = set()
    results = []
    for path in paths:
        with open(path) as f:
            rows = [json.loads(l) for l in f]
        accs, fmts, by_subj = [], [], {}
        for r in rows:
            _, answer, format_ok = parse_output(r["output"])
            ok = 1.0 if answer_correct(answer, r["answer"], r["choices"]) else 0.0
            accs.append(ok)
            fmts.append(1.0 if format_ok else 0.0)
            by_subj.setdefault(r.get("subject") or r.get("topic") or "_", []).append(ok)
        subjects.update(by_subj)
        results.append((os.path.basename(path), len(rows), float(np.mean(accs)),
                        float(np.mean(fmts)), {s: float(np.mean(v)) for s, v in by_subj.items()}))

    subjects = sorted(subjects)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("file\tn\tacc\tformat\t" + "\t".join(f"acc_{s}" for s in subjects) + "\n")
        for name, n, acc, fmt, subj in results:
            cols = "\t".join(f"{subj.get(s, float('nan')):.4f}" for s in subjects)
            f.write(f"{name}\t{n}\t{acc:.4f}\t{fmt:.4f}\t{cols}\n")
            print(f"{name}: n={n} acc={acc:.4f} format={fmt:.4f} "
                  + " ".join(f"{s}={subj.get(s, float('nan')):.3f}" for s in subjects),
                  flush=True)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
