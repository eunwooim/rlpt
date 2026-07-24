#!/usr/bin/env python
"""Convert VisualPRM400K-v1.1 into the canonical process-reward schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from process_reward.common import append_log, create_stage_layout, default_output_dir, print_config, write_json, write_jsonl


DEFAULT_DATASET = "OpenGVLab/VisualPRM400K-v1.1"
QUESTION_STEP_PATTERN = re.compile(
    r"^\s*###\s*Question:\s*(.*?)\s*###\s*Solution\s+Process:\s*(.*)\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)
LABEL_MAP = {
    "+": 1,
    "positive": 1,
    "correct": 1,
    "-": -1,
    "negative": -1,
    "incorrect": -1,
    "0": 0,
    "=": 0,
    "neutral": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET)
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--shuffle_buffer_size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--config_only", action="store_true")
    return parser.parse_args()


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _conversation_content(turn: dict[str, Any]) -> str:
    return _text(turn.get("value")) or _text(turn.get("content")) or _text(turn.get("text"))


def _conversation_role(turn: dict[str, Any]) -> str:
    return str(turn.get("from") or turn.get("role") or "").strip().lower()


def _normalize_images(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    images: list[str] = []
    for item in values:
        if item is None:
            continue
        if isinstance(item, str) and item.strip():
            images.append(item.strip())
        elif isinstance(item, dict):
            path = item.get("path") or item.get("image_path") or item.get("filename")
            if isinstance(path, str) and path.strip():
                images.append(path.strip())
        else:
            filename = getattr(item, "filename", None)
            if isinstance(filename, str) and filename:
                images.append(filename)
    return list(dict.fromkeys(images))


def _stable_sample_id(row: dict[str, Any], question: str, images: list[str], steps: list[str]) -> str:
    for key in ("id", "uid", "sample_id"):
        value = row.get(key)
        if value not in {None, "", -1, "-1"}:
            return str(value)
    payload = json.dumps({"question": question, "images": images, "steps": steps}, ensure_ascii=False, sort_keys=True)
    return "visualprm_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _parse_conversations(row: dict[str, Any]) -> tuple[str, list[str], list[int]]:
    conversations = row.get("conversations") or row.get("messages")
    if not isinstance(conversations, list):
        raise ValueError("Missing conversations list")

    question = ""
    steps: list[str] = []
    labels: list[int] = []
    pending_step: str | None = None
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = _conversation_role(turn)
        content = _conversation_content(turn)
        if not content or role == "system":
            continue
        if role in {"human", "user"}:
            if pending_step is not None:
                raise ValueError("Encountered a new reasoning step before its label")
            match = QUESTION_STEP_PATTERN.match(content)
            if match:
                if question and question != match.group(1).strip():
                    raise ValueError("Conflicting questions in conversation")
                question = match.group(1).strip()
                pending_step = match.group(2).strip()
            else:
                pending_step = content.strip()
            if not pending_step:
                raise ValueError("Empty reasoning step")
            continue
        if role in {"gpt", "assistant", "model"}:
            if pending_step is None:
                continue
            label_key = content.strip().lower()
            if label_key not in LABEL_MAP:
                raise ValueError(f"Unsupported step label token: {content!r}")
            steps.append(pending_step)
            labels.append(LABEL_MAP[label_key])
            pending_step = None

    if pending_step is not None:
        raise ValueError("Final reasoning step has no label")
    if not question:
        question = _text(row.get("question"))
    if not question:
        raise ValueError("Missing question")
    if not steps:
        raise ValueError("No labeled reasoning steps")
    return question, steps, labels


def canonicalize_visualprm_row(
    row: dict[str, Any], dataset_name: str, dataset_config: str | None, split: str, source_index: int
) -> dict[str, Any]:
    question, steps, labels = _parse_conversations(row)
    if len(steps) != len(labels):
        raise ValueError(f"Step/label length mismatch: {len(steps)} != {len(labels)}")
    images = _normalize_images(row.get("images", row.get("image")))
    source_sample_id = _stable_sample_id(row, question, images, steps)
    answer = _text(row.get("answer"))
    return {
        "question": question,
        "images": images,
        "steps": steps,
        "step_labels": labels,
        "answer": answer,
        "metadata": {
            "source_sample_id": source_sample_id,
            "dataset": dataset_name,
            "dataset_config": dataset_config,
            "split": split,
            "source_index": source_index,
        },
    }


def main() -> None:
    args = parse_args()
    resolved_output = Path(args.output_dir) if args.output_dir else default_output_dir("canonical")
    config = {**vars(args), "output_dir": str(resolved_output)}
    if args.config_only:
        print_config(config)
        return
    if args.max_samples == 0 or args.max_samples < -1:
        raise ValueError("--max_samples must be -1 or a positive integer")
    if args.shuffle_buffer_size < 0:
        raise ValueError("--shuffle_buffer_size must be non-negative")

    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": args.split, "streaming": args.streaming}
    if args.dataset_config:
        kwargs["name"] = args.dataset_config
    dataset = load_dataset(args.dataset_name, **kwargs)
    if args.shuffle_buffer_size > 0:
        if args.streaming:
            dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)
        else:
            dataset = dataset.shuffle(seed=args.seed)

    output_dir = create_stage_layout(resolved_output)
    write_json(output_dir / "configs" / "args.json", config)
    canonical_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    label_counts: Counter[int] = Counter()
    for source_index, row in enumerate(dataset):
        try:
            if not isinstance(row, dict):
                raise TypeError(f"Expected dict row, got {type(row).__name__}")
            canonical = canonicalize_visualprm_row(
                row, args.dataset_name, args.dataset_config, args.split, source_index
            )
            canonical_rows.append(canonical)
            label_counts.update(canonical["step_labels"])
        except Exception as exc:  # noqa: BLE001 - errors are persisted for audit.
            errors.append(
                {
                    "source_index": source_index,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
        if args.max_samples > 0 and len(canonical_rows) >= args.max_samples:
            break

    write_jsonl(output_dir / "data" / "canonical.jsonl", canonical_rows)
    write_jsonl(output_dir / "errors" / "load_errors.jsonl", errors)
    summary = {
        "canonical_samples": len(canonical_rows),
        "load_errors": len(errors),
        "step_label_counts": {str(key): value for key, value in sorted(label_counts.items())},
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "metrics" / "summary.json", summary)
    append_log(output_dir, json.dumps(summary, sort_keys=True))
    print_config(summary)


if __name__ == "__main__":
    main()
