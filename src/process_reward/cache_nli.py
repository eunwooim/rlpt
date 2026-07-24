#!/usr/bin/env python
"""Cache frozen bidirectional NLI probabilities for generated pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from process_reward.common import (
    append_log,
    create_stage_layout,
    default_output_dir,
    iter_jsonl,
    print_config,
    write_json,
    write_jsonl,
)


DEFAULT_NLI_MODEL = "microsoft/deberta-xlarge-mnli"
PROVENANCE_FIELDS = (
    "pair_id",
    "anchor",
    "generated",
    "compatibility_label",
    "generation_backend",
    "generation_model",
    "generation_type",
    "source_sample_id",
    "step_index",
    "prompt_version",
    "generation_seed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs_jsonl", required=True)
    parser.add_argument("--nli_model", default=DEFAULT_NLI_MODEL)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--config_only", action="store_true")
    return parser.parse_args()


def validate_pair(row: dict[str, Any], line_index: int) -> None:
    missing = [field for field in PROVENANCE_FIELDS if field not in row]
    if missing:
        raise ValueError(f"Pair row {line_index} is missing fields: {missing}")
    if not isinstance(row["anchor"], str) or not row["anchor"].strip():
        raise ValueError(f"Pair row {line_index} has an invalid anchor")
    if not isinstance(row["generated"], str) or not row["generated"].strip():
        raise ValueError(f"Pair row {line_index} has invalid generated text")
    if row["compatibility_label"] not in {0, 1}:
        raise ValueError(f"Pair row {line_index} has a non-binary compatibility_label")


def _label_map(model: Any) -> dict[str, int]:
    mapped: dict[str, int] = {}
    for index, label in model.config.id2label.items():
        normalized = str(label).lower()
        for name in ("entailment", "contradiction", "neutral"):
            if name in normalized:
                mapped[name] = int(index)
    missing = sorted({"entailment", "contradiction", "neutral"} - set(mapped))
    if missing:
        raise ValueError(f"Could not infer NLI labels {missing} from {model.config.id2label}")
    return mapped


def _resolve_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch.device("cuda:0" if requested == "cuda" else "cpu")


def _directional_probabilities(
    rows: list[dict[str, Any]], reverse: bool, tokenizer: Any, model: Any, torch: Any, device: Any, args: argparse.Namespace
) -> list[dict[str, float]]:
    labels = _label_map(model)
    outputs: list[dict[str, float]] = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        first = [row["generated"] if reverse else row["anchor"] for row in batch]
        second = [row["anchor"] if reverse else row["generated"] for row in batch]
        encoded = tokenizer(
            first,
            second,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            probabilities = torch.softmax(model(**encoded).logits, dim=-1).cpu()
        for probability in probabilities:
            outputs.append(
                {
                    "E": float(probability[labels["entailment"]]),
                    "C": float(probability[labels["contradiction"]]),
                    "N": float(probability[labels["neutral"]]),
                }
            )
    return outputs


def main() -> None:
    args = parse_args()
    resolved_output = Path(args.output_dir) if args.output_dir else default_output_dir("nli")
    config = {**vars(args), "output_dir": str(resolved_output)}
    if args.config_only:
        print_config(config)
        return
    if args.batch_size <= 0 or args.max_length <= 0:
        raise ValueError("--batch_size and --max_length must be positive")

    rows = list(iter_jsonl(args.pairs_jsonl))
    for line_index, row in enumerate(rows, start=1):
        validate_pair(row, line_index)
    if not rows:
        raise ValueError("No generated pairs found")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = _resolve_device(torch, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.nli_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.nli_model)
    model.to(device)
    model.eval()
    ab = _directional_probabilities(rows, False, tokenizer, model, torch, device, args)
    ba = _directional_probabilities(rows, True, tokenizer, model, torch, device, args)

    cached_rows: list[dict[str, Any]] = []
    for row, forward, reverse in zip(rows, ab, ba):
        cached_rows.append(
            {
                **{field: row[field] for field in PROVENANCE_FIELDS},
                "E_ab": forward["E"],
                "E_ba": reverse["E"],
                "C_ab": forward["C"],
                "C_ba": reverse["C"],
                "N_ab": forward["N"],
                "N_ba": reverse["N"],
                "nli_model": args.nli_model,
            }
        )

    output_dir = create_stage_layout(resolved_output)
    write_json(output_dir / "configs" / "args.json", config)
    write_jsonl(output_dir / "data" / "nli_features.jsonl", cached_rows)
    write_jsonl(output_dir / "errors" / "nli_errors.jsonl", [])
    summary = {
        "cached_pairs": len(cached_rows),
        "nli_model": args.nli_model,
        "device": str(device),
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "metrics" / "summary.json", summary)
    append_log(output_dir, json.dumps(summary, sort_keys=True))
    print_config(summary)


if __name__ == "__main__":
    main()
