#!/usr/bin/env python
"""Normalize pre-defined VisualProcessBench text pairs for judge evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from process_reward.common import print_config, write_jsonl


LABEL_FIELDS = ("compatibility_label", "process_correctness", "label")
METADATA_FIELDS = (
    "category",
    "subcategory",
    "error_type",
    "transformation_category",
    "generation_type",
)
STRING_LABELS = {
    "+": 1,
    "correct": 1,
    "compatible": 1,
    "positive": 1,
    "-": 0,
    "incorrect": 0,
    "incompatible": 0,
    "negative": 0,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_jsonl",
        "--input-jsonl",
        "--data_path",
        "--data-path",
        required=True,
        help="JSONL whose records already define anchor/generated text pairs.",
    )
    parser.add_argument(
        "--output_jsonl",
        "--output-jsonl",
        required=True,
        help="New evaluator-compatible pair JSONL; existing files are not overwritten.",
    )
    parser.add_argument(
        "--label_field",
        "--label-field",
        choices=LABEL_FIELDS,
        default=None,
        help="Explicit input label field; by default all supported fields are reconciled.",
    )
    parser.add_argument(
        "--neutral_policy",
        "--neutral-policy",
        choices=("skip", "error"),
        default="skip",
        help="How to handle process_correctness == 0.",
    )
    parser.add_argument(
        "--max_samples",
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of output pairs after neutral filtering.",
    )
    parser.add_argument("--config_only", "--config-only", action="store_true")
    args = parser.parse_args(argv)
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max_samples must be positive when provided")
    return args


def _nested_pair(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("pair")
    return value if isinstance(value, dict) else {}


def _value(row: dict[str, Any], field: str) -> Any:
    if field in row:
        return row[field]
    return _nested_pair(row).get(field)


def _required_text(
    row: dict[str, Any], field: str, source: Path, line_number: int
) -> str:
    value = _value(row, field)
    if not isinstance(value, str) or not value.strip():
        response = row.get("response")
        if (
            isinstance(response, dict)
            and isinstance(response.get("steps"), list)
            and isinstance(response.get("process_correctness"), list)
        ):
            raise ValueError(
                f"{source}:{line_number} is a raw VisualProcessBench record. "
                "It does not define an anchor/generated reference-candidate pair."
            )
        raise ValueError(
            f"{source}:{line_number} has no non-empty string '{field}'"
        )
    return value.strip()


def _normalize_label(value: Any, field: str) -> int | None:
    if isinstance(value, bool):
        raise ValueError(f"{field} must not be boolean")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in STRING_LABELS:
            return STRING_LABELS[normalized]
        if normalized in {"0", "1", "-1"}:
            value = int(normalized)
        else:
            raise ValueError(f"unsupported {field} value {value!r}")
    if not isinstance(value, int):
        raise ValueError(f"unsupported {field} value {value!r}")

    if field == "process_correctness":
        if value == 1:
            return 1
        if value == -1:
            return 0
        if value == 0:
            return None
    elif value in {0, 1}:
        return value
    raise ValueError(f"unsupported {field} value {value!r}")


def _label(
    row: dict[str, Any],
    source: Path,
    line_number: int,
    label_field: str | None,
) -> int | None:
    fields = [label_field] if label_field else [
        field for field in LABEL_FIELDS if _value(row, field) is not None
    ]
    if not fields:
        raise ValueError(
            f"{source}:{line_number} has none of the supported label fields: "
            f"{', '.join(LABEL_FIELDS)}"
        )

    normalized: list[tuple[str, int | None]] = []
    for field in fields:
        value = _value(row, field)
        if value is None:
            raise ValueError(f"{source}:{line_number} has no '{field}' value")
        try:
            normalized.append((field, _normalize_label(value, field)))
        except ValueError as exc:
            raise ValueError(f"{source}:{line_number}: {exc}") from exc

    binary = {value for _, value in normalized if value is not None}
    if len(binary) > 1:
        raise ValueError(
            f"{source}:{line_number} has conflicting labels: {normalized}"
        )
    return next(iter(binary)) if binary else None


def _optional_identifier(
    row: dict[str, Any], field: str, source: Path, line_number: int
) -> str | int | None:
    value = _value(row, field)
    if value is None:
        metadata = row.get("metadata")
        value = metadata.get(field) if isinstance(metadata, dict) else None
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(
            f"{source}:{line_number} has invalid '{field}' {value!r}"
        )
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{source}:{line_number} has empty '{field}'")
    return value


def _data_source(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    for value in (
        row.get("data_source"),
        row.get("source_dataset"),
        row.get("dataset"),
        metadata.get("data_source"),
        metadata.get("source_dataset"),
        metadata.get("dataset"),
    ):
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            text = str(value).strip()
            if text:
                return text
    return "UNKNOWN"


def _stable_id(prefix: str, payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return prefix + hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:20]


def _source_sample_id(
    row: dict[str, Any],
    anchor: str,
    data_source: str,
    source: Path,
    line_number: int,
) -> str | int:
    provided = _optional_identifier(
        row, "source_sample_id", source, line_number
    )
    if provided is not None:
        return provided
    return _stable_id(
        "vpb_",
        {
            "data_source": data_source,
            "question": row.get("question"),
            "image": row.get("image", row.get("images")),
            "policy_model": row.get("policy_model"),
            "anchor": anchor,
        },
    )


def _step_index(
    row: dict[str, Any], source: Path, line_number: int
) -> int | None:
    value = _value(row, "step_index")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{source}:{line_number} has invalid 'step_index' {value!r}"
        )
    return value


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    nested = row.get("metadata")
    nested = nested if isinstance(nested, dict) else {}
    output: dict[str, Any] = {}
    for field in METADATA_FIELDS:
        value = row.get(field, nested.get(field))
        scalar = value is None or isinstance(value, (str, int, float, bool))
        scalar_list = isinstance(value, list) and all(
            item is None or isinstance(item, (str, int, float, bool))
            for item in value
        )
        if value is not None and (scalar or scalar_list):
            output[field] = value
    return output


def normalize_pair(
    row: dict[str, Any],
    *,
    source: Path,
    line_number: int,
    label_field: str | None,
) -> dict[str, Any] | None:
    anchor = _required_text(row, "anchor", source, line_number)
    generated = _required_text(row, "generated", source, line_number)
    compatibility_label = _label(row, source, line_number, label_field)
    if compatibility_label is None:
        return None

    data_source = _data_source(row)
    source_sample_id = _source_sample_id(
        row, anchor, data_source, source, line_number
    )
    step_index = _step_index(row, source, line_number)
    pair_id = _optional_identifier(row, "pair_id", source, line_number)
    if pair_id is None:
        pair_id = _stable_id(
            "pair_",
            {
                "source_sample_id": source_sample_id,
                "step_index": step_index,
                "anchor": anchor,
                "generated": generated,
            },
        )

    output: dict[str, Any] = {
        "pair_id": pair_id,
        "source_sample_id": source_sample_id,
        "step_index": step_index,
        "data_source": data_source,
        "anchor": anchor,
        "generated": generated,
        "compatibility_label": compatibility_label,
    }
    output.update(_metadata(row))
    return output


def process_pairs(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.input_jsonl).expanduser()
    output = Path(args.output_jsonl).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Input JSONL does not exist: {source}")
    if source.resolve() == output.resolve():
        raise ValueError("--input_jsonl and --output_jsonl must be different")

    pairs: list[dict[str, Any]] = []
    rows_read = 0
    neutral_skipped = 0
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            rows_read += 1
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {source}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected a JSON object at {source}:{line_number}"
                )
            pair = normalize_pair(
                row,
                source=source,
                line_number=line_number,
                label_field=args.label_field,
            )
            if pair is None:
                if args.neutral_policy == "error":
                    raise ValueError(
                        f"{source}:{line_number} has neutral process_correctness"
                    )
                neutral_skipped += 1
                continue
            pairs.append(pair)
            if args.max_samples is not None and len(pairs) >= args.max_samples:
                break

    if not pairs:
        raise ValueError(f"No binary pair records were produced from {source}")
    write_jsonl(output, pairs)
    return {
        "input_jsonl": str(source),
        "output_jsonl": str(output),
        "rows_read": rows_read,
        "pairs_written": len(pairs),
        "neutral_skipped": neutral_skipped,
        "positive_pairs": sum(row["compatibility_label"] == 1 for row in pairs),
        "negative_pairs": sum(row["compatibility_label"] == 0 for row in pairs),
        "data_sources": sorted({str(row["data_source"]) for row in pairs}),
    }


def main() -> None:
    try:
        args = parse_args()
        if args.config_only:
            print_config(vars(args))
            return
        print_config(process_pairs(args))
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"{Path(__file__).name}: error: {exc}") from None


if __name__ == "__main__":
    main()
