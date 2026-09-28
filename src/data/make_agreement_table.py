#!/usr/bin/env python
"""Stage 3: MC x NLI agreement table + off-diagonal example dumps.

MC-pass row of the 2x2 comes from the FULL Stage-2 run on the tau-kept set;
MC-fail row comes from the 15,000-trace seed=0 sample of the "plausible
victim" band (answer-correct, min body score in [0.5, 0.85)) and is labeled
as a sample estimate.

Off-diagonal examples (20 each, seed=0 reservoir):
  MC-pass / NLI-fail  <- stage2_rejected_tau{tau}.jsonl
  MC-fail / NLI-pass  <- stage2_band_kept.jsonl
Each dump shows the question, every body step with BOTH its MC score and its
P(contradiction), and provenance. MC per-step scores for the kept-set rows are
looked up in the raw annotations (read-only) via source_file + line_index.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

FILTERED = Path("/scratch/sghos104/rlpt/data/visualprm_v11_filtered")
RAW_ANNOS = Path("/scratch/sghos104/rlpt/data/visualprm_v11_raw/annos")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--filtered_dir", type=Path, default=FILTERED)
    p.add_argument("--raw_annos", type=Path, default=RAW_ANNOS)
    p.add_argument("--tau", type=str, default="0.85")
    p.add_argument("--n_examples", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def iter_jsonl(path: Path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_lines(path: Path) -> int:
    n = 0
    with open(path) as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def reservoir(path: Path, k: int, rng: random.Random) -> list[dict]:
    sample: list[dict] = []
    for i, row in enumerate(iter_jsonl(path)):
        if i < k:
            sample.append(row)
        else:
            j = rng.randint(0, i)
            if j < k:
                sample[j] = row
    sample.sort(key=lambda r: (r["source_file"], r["line_index"]))
    return sample


def fetch_step_scores(rows: list[dict], raw_annos: Path) -> None:
    """Fill row['step_scores'] from the raw annotations for rows lacking it."""
    need: dict[str, dict[int, dict]] = {}
    for r in rows:
        if "step_scores" not in r:
            need.setdefault(r["source_file"], {})[r["line_index"]] = r
    for source_file, wanted in need.items():
        with open(raw_annos / source_file) as fh:
            for idx, line in enumerate(fh):
                if idx in wanted:
                    rec = json.loads(line)
                    sws = rec["steps_with_score"]
                    body = sws[:-1] if len(sws) > 1 else sws
                    wanted[idx]["step_scores"] = [s["score"] for s in body]


def dump_example(row: dict, path: Path) -> None:
    lines = [
        f"source_file: {row['source_file']}   line_index: {row['line_index']}",
        f"image: {row['image']}",
        f"min MC score: {row['min_score']:.4f}   mean: {row['mean_score']:.4f}   "
        f"max P(contra): {row['max_p_contra']:.4f}",
        f"answer: {row['answer']}",
        "",
        "QUESTION:",
        str(row["question_orig"]),
        "",
        "STEPS (MC score | P(contradiction)):",
    ]
    for mc, pc, step in zip(row["step_scores"], row["p_contra_per_step"], row["steps"]):
        lines.append(f"  [MC {mc:.4f} | contra {pc:.4f}] {step}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    d = args.filtered_dir
    stats = json.loads((d / f"stage2_stats_tau{args.tau}.json").read_text())
    band_stats = json.loads((d / "stage2_band_stats.json").read_text())
    s1_stats = json.loads((d / "stage1_stats_tau0.85.json").read_text())

    a = stats["kept"]            # MC pass / NLI pass
    b = stats["rejected"]        # MC pass / NLI fail
    c = band_stats["kept"]       # MC fail / NLI pass (sample)
    e = band_stats["rejected"]   # MC fail / NLI fail (sample)
    band_pop = s1_stats["band_population"]

    def pct(x: int, tot: int) -> str:
        return f"{100 * x / tot:.1f}%"

    table = f"""# MC x NLI Agreement Table (tau={args.tau})

NLI screen: max over steps of P(contradiction) > {stats['contra_threshold']}
=> fail. Model: {stats['model']}.

- **MC pass** row = full Stage-2 run on the stage1 tau={args.tau} kept set
  ({a + b:,} traces, exact counts).
- **MC fail** row = SAMPLE ESTIMATE from {c + e:,} traces drawn (seed=0
  reservoir) out of the {band_pop:,}-trace "plausible victim" band
  (answer-correct, min body step score in [0.5, 0.85)). It does NOT cover
  MC-fail traces below 0.5 or with a wrong final answer.

|                       | NLI pass | NLI fail | total |
|-----------------------|---------:|---------:|------:|
| MC pass (full set)    | {a:,} ({pct(a, a + b)}) | {b:,} ({pct(b, a + b)}) | {a + b:,} |
| MC fail (band sample) | {c:,} ({pct(c, c + e)}) | {e:,} ({pct(e, c + e)}) | {c + e:,} |

Off-diagonal reading:
- MC-pass / NLI-fail ({b:,}): every step survived MC sampling, yet the NLI
  model flags a contradiction somewhere — candidate label noise OR NLI false
  alarms (cf. the known neutral-trap / numeric-contradiction failure modes).
- MC-fail / NLI-pass ({c:,} of sample): a mid-scored MC step but no textual
  contradiction — candidate MC harshness (correct step, unlucky rollouts).

20 random examples from each off-diagonal cell: `agreement_examples/`.
"""
    (d / "agreement_table.md").write_text(table)

    rng = random.Random(args.seed)
    ex_dir = d / "agreement_examples"
    ex_dir.mkdir(exist_ok=True)
    cells = [
        ("mcpass_nlifail", d / f"stage2_rejected_tau{args.tau}.jsonl"),
        ("mcfail_nlipass", d / "stage2_band_kept.jsonl"),
    ]
    for label, src in cells:
        rows = reservoir(src, args.n_examples, rng)
        fetch_step_scores(rows, args.raw_annos)
        for i, row in enumerate(rows, 1):
            name = (f"{label}_{i:02d}_"
                    f"{Path(row['source_file']).stem[:40]}_L{row['line_index']}.txt")
            dump_example(row, ex_dir / name)
        print(f"{label}: {len(rows)} examples -> {ex_dir}")
    print(table)


if __name__ == "__main__":
    main()
