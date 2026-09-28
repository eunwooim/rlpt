#!/usr/bin/env python
"""Stage 1: MC trace-level filter for VisualPRM400K-v1.1-Raw.

Streams the 38 extracted annotation jsonl files (never loads the corpus into
memory) and keeps a trace iff:
  - steps_with_score[-1].score > 0.5  (final answer correct), AND
  - min(score of body steps) >= tau, body = steps_with_score[:-1]
    (a 1-step trace's body is all of its steps)

Semantics replicate data/visualprm_v11_raw/check_step_scores.py exactly
(strict > 0.5 on the last step; rows with empty/missing steps_with_score are
counted in totals but can never pass), so the stats must reproduce the
previously measured numbers (tau=0.85 -> 338,685 kept; answer-correct 503,885).

Traces are kept or dropped WHOLE. The only step removal is stripping the
trailing final-answer step (the body/steps field of output rows).

Optionally reservoir-samples the "plausible victim" band (answer-correct but
min body score in [band_lo, band_hi)) for the Stage-3 agreement study.

Deterministic: files processed in sorted order, lines in file order, seed=0
reservoir; band output re-sorted by (source_file, line_index).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

RAW_DEFAULT = Path("/scratch/sghos104/rlpt/data/visualprm_v11_raw")
OUT_DEFAULT = Path("/scratch/sghos104/rlpt/data/visualprm_v11_filtered")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annos_dir", type=Path, default=RAW_DEFAULT / "annos" / "annotations",
                   help="Directory with the 38 extracted .jsonl files (read-only)")
    p.add_argument("--out_dir", type=Path, default=OUT_DEFAULT)
    p.add_argument("--tau", type=str, default="0.85",
                   help="Min body-step score threshold (string, reused verbatim in filenames)")
    p.add_argument("--strategy", choices=["flat", "per_source_topk", "best_per_question"],
                   default="flat")
    p.add_argument("--band_sample_out", type=Path, default=None,
                   help="If set, reservoir-sample answer-correct traces with min body score "
                        "in [band_lo, band_hi) into this jsonl (Stage-3 input)")
    p.add_argument("--band_lo", type=float, default=0.5)
    p.add_argument("--band_hi", type=float, default=0.85)
    p.add_argument("--band_k", type=int, default=15000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def make_row(source_file: str, line_index: int, rec: dict, body: list[dict],
             extra_step_scores: bool = False) -> dict:
    scores = [s["score"] for s in body]
    row = {
        "source_file": source_file,
        "line_index": line_index,
        "image": rec.get("image"),
        "question_orig": rec.get("question_orig", rec.get("question")),
        "answer": rec.get("answer"),
        "steps": [s["step"] for s in body],
        "min_score": min(scores),
        "mean_score": sum(scores) / len(scores),
        "n_steps": len(body),
    }
    if extra_step_scores:
        row["step_scores"] = scores
    return row


def main() -> int:
    args = parse_args()
    if args.strategy != "flat":
        raise NotImplementedError(
            f"--strategy {args.strategy} is stubbed pending supervisor decision")
    tau = float(args.tau)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    kept_path = args.out_dir / f"stage1_kept_tau{args.tau}.jsonl"
    stats_path = args.out_dir / f"stage1_stats_tau{args.tau}.json"

    files = sorted(args.annos_dir.glob("*.jsonl"))
    if len(files) != 38:
        print(f"FATAL: expected exactly 38 jsonl files, found {len(files)}: "
              f"{[f.name for f in files]}", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    band: list[tuple[int, dict]] = []  # reservoir of (arrival_index, row)
    band_seen = 0
    total = answer_correct = kept = 0
    missing_sws = missing_qorig = 0
    per_source: dict[str, dict[str, int]] = {}

    with open(kept_path, "w") as out:
        for f in files:
            source_file = f"annotations/{f.name}"  # zip-relative provenance path
            src_total = src_kept = 0
            with open(f) as fh:
                for line_index, line in enumerate(fh):
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    total += 1
                    src_total += 1
                    sws = rec.get("steps_with_score") or []
                    if not sws:
                        missing_sws += 1
                        continue
                    if sws[-1]["score"] <= 0.5:
                        continue
                    answer_correct += 1
                    if "question_orig" not in rec:
                        missing_qorig += 1
                    body = sws[:-1] if len(sws) > 1 else sws
                    mn = min(s["score"] for s in body)
                    if mn >= tau:
                        kept += 1
                        src_kept += 1
                        out.write(json.dumps(
                            make_row(source_file, line_index, rec, body),
                            ensure_ascii=False) + "\n")
                    elif (args.band_sample_out is not None
                          and args.band_lo <= mn < args.band_hi):
                        row = make_row(source_file, line_index, rec, body,
                                       extra_step_scores=True)
                        if band_seen < args.band_k:
                            band.append((band_seen, row))
                        else:
                            j = rng.randint(0, band_seen)
                            if j < args.band_k:
                                band[j] = (band_seen, row)
                        band_seen += 1
            per_source[source_file] = {"total": src_total, "kept": src_kept}

    stats = {
        "tau": tau,
        "strategy": "flat",
        "n_source_files": len(files),
        "total_rows": total,
        "answer_correct": answer_correct,
        "kept": kept,
        "rows_missing_steps_with_score": missing_sws,
        "rows_missing_question_orig_fellback_to_question": missing_qorig,
        "band_definition": (f"answer-correct AND min body score in "
                            f"[{args.band_lo}, {args.band_hi})"),
        "band_population": band_seen if args.band_sample_out else None,
        "band_sampled": len(band) if args.band_sample_out else None,
        "per_source": per_source,
    }
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)

    if args.band_sample_out is not None:
        rows = [r for _, r in band]
        rows.sort(key=lambda r: (r["source_file"], r["line_index"]))
        with open(args.band_sample_out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps({k: v for k, v in stats.items() if k != "per_source"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
