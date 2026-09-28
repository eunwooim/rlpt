#!/usr/bin/env python
"""Score eval generations (accuracy / format / per-subject) into one TSV.

Answer extraction (uniform for every model — fairness control):
  1. <answer>...</answer> (last occurrence), as in the earlier validation track;
  2. else "Final answer: ..." tail (the VisualPRM SFT trace style);
  3. else the last non-empty line of the completion.
`answer_correct` replicates tools/graph_match_reward.py semantics from the
scienceqa-grpo-validation branch (exact text, letter, 0/1-based index, or
unambiguous substring of the gold choice). format_strict = exactly one
<think> and one <answer> block (the old track's definition); format_any also
accepts a "Final answer:" line.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_FINAL_RE = re.compile(r"final answer\s*[:\-]\s*(.+)", re.IGNORECASE)
_LETTER_RE = re.compile(r"^\(?([a-z])\)?[.):]?$", re.IGNORECASE)


def _norm(s):
    return re.sub(r"[^\w\s]", "", str(s)).strip().lower()


def extract_answer(text: str) -> tuple[str, bool, bool]:
    """-> (answer_text, format_strict, format_any)"""
    text = text or ""
    thinks = _THINK_RE.findall(text)
    answers = _ANSWER_RE.findall(text)
    format_strict = len(thinks) == 1 and len(answers) == 1
    if answers:
        return answers[-1].strip(), format_strict, True
    finals = _FINAL_RE.findall(text)
    if finals:
        return finals[-1].strip().splitlines()[0].strip(), False, True
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return (lines[-1] if lines else ""), False, False


def answer_correct(pred_answer, gold_text, choices=None):
    if not pred_answer:
        return False
    p, g = _norm(pred_answer), _norm(gold_text)
    if not g:
        return False
    if p == g:
        return True
    if choices:
        norm_choices = [_norm(c) for c in choices]
        gold_idx = norm_choices.index(g) if g in norm_choices else None
        m = _LETTER_RE.match(pred_answer.strip())
        if m and gold_idx is not None:
            if (ord(m.group(1).lower()) - ord("a")) == gold_idx:
                return True
        if p.isascii() and p.isdigit() and gold_idx is not None:
            if int(p) == gold_idx or int(p) == gold_idx + 1:
                return True
        if gold_idx is not None and g and g in p and g not in [
            c for i, c in enumerate(norm_choices) if i != gold_idx and c and c in p
        ]:
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--pattern", default="*.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.gen_dir, args.pattern)))
    if not paths:
        raise SystemExit(f"no files matching {args.pattern} in {args.gen_dir}")

    all_subjects: set[str] = set()
    results = []
    for path in paths:
        with open(path) as f:
            rows = [json.loads(l) for l in f]
        accs, strict, anyf = [], [], []
        by_subj = defaultdict(list)
        for r in rows:
            ans, fs, fa = extract_answer(r["output"])
            ok = 1.0 if answer_correct(ans, r["answer"], r.get("choices")) else 0.0
            accs.append(ok)
            strict.append(1.0 if fs else 0.0)
            anyf.append(1.0 if fa else 0.0)
            by_subj[r.get("subject") or "_"].append(ok)
        all_subjects.update(by_subj)
        mean = lambda v: sum(v) / len(v) if v else float("nan")
        results.append((os.path.basename(path), len(rows), mean(accs), mean(strict),
                        mean(anyf), {s: mean(v) for s, v in by_subj.items()}))

    subjects = sorted(all_subjects)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("file\tn\tacc\tformat_strict\tformat_any\t"
                + "\t".join(f"acc_{s}" for s in subjects) + "\n")
        for name, n, acc, fs, fa, subj in results:
            cols = "\t".join(f"{subj.get(s, float('nan')):.4f}" for s in subjects)
            f.write(f"{name}\t{n}\t{acc:.4f}\t{fs:.4f}\t{fa:.4f}\t{cols}\n")
            print(f"{name}: n={n} acc={acc:.4f} format_strict={fs:.4f} format_any={fa:.4f}",
                  flush=True)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
