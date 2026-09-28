#!/usr/bin/env python
"""Score VPB generations: answer accuracy overall and per source.

Extraction + correctness are arm_reward's (same cascade the training reward
used: "Final answer:" -> \\boxed{} -> <answer> -> last line; normalized exact /
MCQ letter / numeric-tolerance compare). One row per generation file.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_reward import answer_correct, extract_answer  # noqa: E402


def score_file(path):
    total = defaultdict(int)
    correct = defaultdict(int)
    n_marker = 0
    n = 0
    for line in open(path):
        r = json.loads(line)
        pred, has_marker = extract_answer(r["output"])
        ok = answer_correct(pred, r["answer"])
        n += 1
        n_marker += has_marker
        total[r["data_source"]] += 1
        correct[r["data_source"]] += ok
        total["ALL"] += 1
        correct["ALL"] += ok
    return {
        "file": os.path.basename(path), "n": n,
        "accuracy": round(correct["ALL"] / max(1, total["ALL"]), 4),
        "format_marker_rate": round(n_marker / max(1, n), 4),
        "per_source": {k: {"n": total[k], "acc": round(correct[k] / total[k], 4)}
                       for k in sorted(total) if k != "ALL"},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen_dir", default="/scratch/sghos104/rlpt/grpo_arms/evals")
    ap.add_argument("--pattern", default="vpb_gen_*.jsonl")
    ap.add_argument("--out", default="/scratch/sghos104/rlpt/grpo_arms/evals/vpb_scores.json")
    args = ap.parse_args()

    results = [score_file(p) for p in sorted(glob.glob(os.path.join(args.gen_dir, args.pattern)))]
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    for r in results:
        print(f"{r['file']:40s} n={r['n']:5d} acc={r['accuracy']:.4f} "
              f"marker={r['format_marker_rate']:.3f}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
