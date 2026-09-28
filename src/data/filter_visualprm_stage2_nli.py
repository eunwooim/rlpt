#!/usr/bin/env python
"""Stage 2: NLI contradiction screen over Stage-1 filtered VisualPRM traces.

For each trace, for each step k (1-indexed over the kept body steps):
  premise    = question_orig + steps 1..k-1, truncated to the most recent steps
               that fit the token budget (keep latest steps, drop the oldest;
               the question is kept only if it fits alongside ALL prior steps —
               when space is short the question is dropped first, then the
               oldest steps)
  hypothesis = step k
  record P(contradiction) from the MNLI classifier (single direction).

Trace-level decision: FAILS iff max_k P(contradiction) > --contra_threshold.

Model: microsoft/deberta-xlarge-mnli (off-the-shelf MNLI checkpoint; loading
pattern reused from src/metrics/scorers/nli.py + transformer_utils.py; the
xlarge variant is used per instruction, upgraded from deberta-large-mnli).

Engineering: streams the input jsonl (never holds the corpus in memory),
fixed-order batching (deterministic), fp16 on GPU, resumable — on restart,
(source_file, line_index) pairs already present in the output files are
skipped and outputs are appended. Progress checkpoint (flush+fsync+json)
every --checkpoint_every completed traces.

--derive_from_tau mode: for a stricter tau whose Stage-1 kept set is a subset
of an already-scored run (min-score thresholds nest), reuse the existing
per-trace NLI scores instead of re-running the model. Exact, not an estimate:
P(contradiction) depends only on trace content. Fails loudly if any input
trace is missing from the donor run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

FILTERED = Path("/scratch/sghos104/rlpt/data/visualprm_v11_filtered")
MODEL_NAME = "microsoft/deberta-xlarge-mnli"
SPECIAL_MARGIN = 8  # specials + join-token slack when budgeting the premise


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out_kept", type=Path, required=True)
    p.add_argument("--out_rejected", type=Path, required=True)
    p.add_argument("--stats", type=Path, required=True)
    p.add_argument("--contra_threshold", type=float, default=0.5)
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_hyp_tokens", type=int, default=400,
                   help="Defensive cap so the hypothesis can never crowd out the premise")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--checkpoint_every", type=int, default=10000)
    p.add_argument("--derive_from_kept", type=Path, default=None,
                   help="Donor stage2 kept jsonl from a looser-tau run (no GPU needed)")
    p.add_argument("--derive_from_rejected", type=Path, default=None)
    return p.parse_args()


def iter_jsonl(path: Path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def trace_id(row: dict) -> tuple[str, int]:
    return (row["source_file"], row["line_index"])


def write_stats(args, kept, rejected, n_pairs, hist, per_source, extra) -> None:
    n_bins = 20
    histogram = {
        f"[{i*0.05:.2f},{(i+1)*0.05:.2f}{']' if i == n_bins-1 else ')'}": hist[i]
        for i in range(n_bins)
    }
    stats = {
        "model": args.model,
        "contra_threshold": args.contra_threshold,
        "input": str(args.input),
        "kept": kept,
        "rejected": rejected,
        "n_pairs_scored": n_pairs,
        "max_p_contra_histogram_bin0.05": histogram,
        "per_source_rejection": {
            src: {"total": v["total"], "rejected": v["rejected"],
                  "rejection_rate": round(v["rejected"] / v["total"], 4)}
            for src, v in sorted(per_source.items())
        },
        **extra,
    }
    with open(args.stats, "w") as fh:
        json.dump(stats, fh, indent=2)


def finalize(row: dict, p_contras: list[float], threshold: float,
             hist, per_source, files) -> bool:
    """Attach NLI fields, route to kept/rejected, update stats. True if kept."""
    mx = max(p_contras)
    row["max_p_contra"] = mx
    row["p_contra_per_step"] = p_contras
    ok = not (mx > threshold)
    files["kept" if ok else "rejected"].write(
        json.dumps(row, ensure_ascii=False) + "\n")
    hist[min(int(mx / 0.05), 19)] += 1
    src = per_source.setdefault(row["source_file"], {"total": 0, "rejected": 0})
    src["total"] += 1
    src["rejected"] += 0 if ok else 1
    return ok


def run_derive(args) -> int:
    donor: dict[tuple[str, int], tuple[float, list[float]]] = {}
    for path in (args.derive_from_kept, args.derive_from_rejected):
        for row in iter_jsonl(path):
            donor[trace_id(row)] = (row["max_p_contra"], row["p_contra_per_step"])
    kept = rejected = n_pairs = 0
    hist = [0] * 20
    per_source: dict[str, dict[str, int]] = {}
    missing = []
    with open(args.out_kept, "w") as fk, open(args.out_rejected, "w") as fr:
        files = {"kept": fk, "rejected": fr}
        for row in iter_jsonl(args.input):
            tid = trace_id(row)
            if tid not in donor:
                missing.append(tid)
                if len(missing) > 5:
                    break
                continue
            _, per_step = donor[tid]
            if len(per_step) != row["n_steps"]:
                print(f"FATAL: step-count mismatch for {tid}", file=sys.stderr)
                return 2
            n_pairs += len(per_step)
            if finalize(row, per_step, args.contra_threshold, hist, per_source, files):
                kept += 1
            else:
                rejected += 1
    if missing:
        print(f"FATAL: {len(missing)}+ input traces absent from donor run, e.g. "
              f"{missing[:5]} — stricter tau must be a subset; investigate.",
              file=sys.stderr)
        return 2
    write_stats(args, kept, rejected, n_pairs, hist, per_source,
                {"derived_from": [str(args.derive_from_kept),
                                  str(args.derive_from_rejected)]})
    print(f"[derive] kept={kept} rejected={rejected} pairs={n_pairs}")
    return 0


def main() -> int:
    args = parse_args()
    args.out_kept.parent.mkdir(parents=True, exist_ok=True)
    if args.derive_from_kept or args.derive_from_rejected:
        if not (args.derive_from_kept and args.derive_from_rejected):
            print("FATAL: derive mode needs both donor files", file=sys.stderr)
            return 2
        return run_derive(args)

    import torch
    from tqdm.auto import tqdm
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.truncation_side = "left"  # residual premise overflow drops OLDEST tokens
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    if device.type == "cuda":
        model = model.half()
    model.to(device)
    model.eval()
    id2label = {int(i): str(l).lower() for i, l in model.config.id2label.items()}
    contra_idx = next(i for i, l in id2label.items() if "contradiction" in l)

    # Resume support: skip traces already present in either output, and fold
    # them back into the stats accumulators so final stats cover the full run.
    kept = rejected = n_pairs = n_done = 0
    hist = [0] * 20
    per_source: dict[str, dict[str, int]] = {}
    n_q_dropped = n_steps_dropped = n_hyp_truncated = 0
    done: set[tuple[str, int]] = set()
    for path, is_kept in ((args.out_kept, True), (args.out_rejected, False)):
        if not path.exists():
            continue
        good_bytes = 0
        with open(path) as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    if fh.readline():  # partial line NOT at EOF: not a crash tail
                        print(f"FATAL: corrupt non-final line in {path}", file=sys.stderr)
                        return 2
                    print(f"[resume] dropping partial final line of {path}")
                    break
                good_bytes += len(line.encode())
                done.add(trace_id(row))
                mx = row["max_p_contra"]
                hist[min(int(mx / 0.05), 19)] += 1
                src = per_source.setdefault(row["source_file"], {"total": 0, "rejected": 0})
                src["total"] += 1
                src["rejected"] += 0 if is_kept else 1
                n_pairs += len(row["p_contra_per_step"])
                kept += 1 if is_kept else 0
                rejected += 0 if is_kept else 1
        if good_bytes < path.stat().st_size:
            with open(path, "r+") as fh:
                fh.truncate(good_bytes)
    if done:
        print(f"[resume] {len(done)} traces already scored, appending")

    def count_tokens(texts: list[str]) -> list[int]:
        return [len(ids) for ids in
                tokenizer(texts, add_special_tokens=False)["input_ids"]]

    @torch.inference_mode()
    def score_batch(batch: list[tuple[dict, str, str]]) -> None:
        nonlocal n_pairs
        enc = tokenizer(
            [p for _, p, _ in batch], [h for _, _, h in batch],
            padding=True, truncation="only_first",
            max_length=args.max_length, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        probs = torch.softmax(model(**enc).logits.float(), dim=-1)[:, contra_idx]
        for (trace, _, _), p in zip(batch, probs.tolist()):
            trace["p_contras"].append(p)
        n_pairs += len(batch)

    with open(args.out_kept, "a") as fk, open(args.out_rejected, "a") as fr:
        files = {"kept": fk, "rejected": fr}

        def drain(pending: deque) -> None:
            """Write out head-of-queue traces whose pairs are all scored."""
            nonlocal kept, rejected, n_done
            while pending and len(pending[0]["p_contras"]) == pending[0]["row"]["n_steps"]:
                t = pending.popleft()
                if finalize(t["row"], t["p_contras"], args.contra_threshold,
                            hist, per_source, files):
                    kept += 1
                else:
                    rejected += 1
                n_done += 1
                if n_done % args.checkpoint_every == 0:
                    for f in files.values():
                        f.flush()
                        os.fsync(f.fileno())

        pending: deque = deque()
        buffer: list[tuple[dict, str, str]] = []
        for row in tqdm(iter_jsonl(args.input), desc="traces", unit="trace",
                        mininterval=30, maxinterval=60):
            if trace_id(row) in done:
                continue
            question = row["question_orig"] or ""
            steps = row["steps"]
            lens = count_tokens([question] + steps)
            q_len, step_lens = lens[0], lens[1:]
            trace = {"row": row, "p_contras": []}
            pending.append(trace)
            for k in range(len(steps)):
                hyp = steps[k]
                hyp_len = step_lens[k]
                if hyp_len > args.max_hyp_tokens:
                    n_hyp_truncated += 1
                    hyp = tokenizer.decode(
                        tokenizer(hyp, add_special_tokens=False)
                        ["input_ids"][:args.max_hyp_tokens])
                    hyp_len = args.max_hyp_tokens
                budget = args.max_length - hyp_len - SPECIAL_MARGIN
                prior_total = sum(step_lens[:k])
                if q_len + prior_total <= budget:
                    parts = [question] + steps[:k]
                else:
                    if k == 0:
                        parts = [question]  # lone over-long question: left-truncated
                    else:
                        n_q_dropped += 1
                        acc, start = 0, k
                        for i in range(k - 1, -1, -1):
                            if acc + step_lens[i] > budget:
                                break
                            acc += step_lens[i]
                            start = i
                        if start < k:
                            if start > 0:
                                n_steps_dropped += 1
                            parts = steps[start:k]
                        else:
                            n_steps_dropped += 1
                            parts = [steps[k - 1]]  # over-long step: left-truncated
                buffer.append((trace, "\n".join(parts), hyp))
                if len(buffer) >= args.batch_size:
                    score_batch(buffer[:args.batch_size])
                    buffer = buffer[args.batch_size:]
                    drain(pending)
        if buffer:
            score_batch(buffer)
        drain(pending)
        assert not pending, f"{len(pending)} traces left unscored"

    write_stats(args, kept, rejected, n_pairs, hist, per_source, {
        "resumed_traces_skipped": len(done),
        "premise_question_dropped": n_q_dropped,
        "premise_oldest_steps_dropped": n_steps_dropped,
        "hypotheses_truncated": n_hyp_truncated,
    })
    print(f"[stage2] kept={kept} rejected={rejected} pairs={n_pairs} "
          f"(+{len(done)} previously done)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
