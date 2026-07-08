from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from metrics.data_loaders.common import read_jsonl, write_csv


SUMMARY_FIELDS = {
    "sbert": ["raw_score"],
    "bertscore": ["bertscore_f1"],
    "nli": ["nli_heuristic"],
    "cross_nli": ["raw_score"],
    "bge_reranker": ["sigmoid_raw_score"],
}


def _safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _scale_reward(score: float, kappa: float) -> float:
    return max(0.0, score - kappa) / (1.0 - kappa)


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _raw_values(row: dict[str, Any]) -> dict[str, Any]:
    values = row.get("all_raw_values")
    return values if isinstance(values, dict) else row


def _nli_heuristic(row: dict[str, Any]) -> float | None:
    values = _raw_values(row)
    e_ab = _num(values.get("E_ab"))
    e_ba = _num(values.get("E_ba"))
    c_ab = _num(values.get("C_ab"))
    c_ba = _num(values.get("C_ba"))
    if e_ab is None or e_ba is None or c_ab is None or c_ba is None:
        return None
    return 0.5 * e_ab + 0.5 * e_ba - max(c_ab, c_ba)


def reward_value(row: dict[str, Any], field: str) -> float | None:
    if row.get("scorer") == "nli" and field == "nli_heuristic":
        return _nli_heuristic(row)
    if row.get("scorer") == "bge_reranker" and field == "sigmoid_raw_score":
        raw_score = _num(row.get("score"))
        return _sigmoid(raw_score) if raw_score is not None else None
    if row.get("score_field") == field:
        return _num(row.get("score"))
    return None


def score_fields_for_row(row: dict[str, Any]) -> list[str]:
    scorer = str(row.get("scorer", ""))
    if scorer == "nli":
        return ["nli_heuristic"]
    if scorer == "bge_reranker":
        return ["sigmoid_raw_score"]
    score_field = row.get("score_field")
    if isinstance(score_field, str) and score_field:
        return [score_field]
    return SUMMARY_FIELDS.get(scorer, ["raw_score"])


def pair_type(row: dict[str, Any]) -> str:
    value = row.get("pair_type")
    if value in {"positive", "positive_pair"}:
        return "positive"
    if value in {"negative", "negative_pair"}:
        return "negative"
    return str(value or "")


def group_value(value: Any) -> Any:
    return "" if value is None else value


def write_counts_and_preview(output_dir: Path, cases: list[dict[str, Any]]) -> None:
    by_subcategory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_subcategory[str(case.get("subcategory", "unknown"))].append(case)

    count_rows: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []
    for subcategory in sorted(by_subcategory):
        rows = by_subcategory[subcategory]
        negative_type_counts: dict[str, int] = defaultdict(int)
        trace_ids = set()
        for case in rows:
            trace_ids.add(case.get("trace_id"))
            for negative in case.get("negatives") or []:
                if isinstance(negative, dict):
                    negative_type_counts[str(negative.get("type", "unknown"))] += 1
        count_rows.append(
            {
                "dataset": "visualprm",
                "subcategory": subcategory,
                "num_traces": len(trace_ids),
                "num_cases": len(rows),
                "num_positive_pairs": len(rows),
                "num_negative_pairs": sum(negative_type_counts.values()),
                "num_value_flip": negative_type_counts.get("value_flip", 0),
                "num_entity_flip": negative_type_counts.get("entity_flip", 0),
                "num_relation_flip": negative_type_counts.get("relation_flip", 0),
            }
        )
        for case in rows[:5]:
            negatives = case.get("negatives") or []
            preview_rows.append(
                {
                    "case_id": case.get("case_id", ""),
                    "trace_id": case.get("trace_id", ""),
                    "subcategory": subcategory,
                    "segment_index": case.get("segment_index", ""),
                    "reference": case.get("reference", ""),
                    "positive": case.get("positive", ""),
                    "negative_0_type": negatives[0].get("type", "") if negatives and isinstance(negatives[0], dict) else "",
                    "negative_0": negatives[0].get("text", "") if negatives and isinstance(negatives[0], dict) else "",
                }
            )

    write_csv(
        output_dir / "tables" / "numeric_subcategory_counts.csv",
        count_rows,
        [
            "dataset",
            "subcategory",
            "num_traces",
            "num_cases",
            "num_positive_pairs",
            "num_negative_pairs",
            "num_value_flip",
            "num_entity_flip",
            "num_relation_flip",
        ],
    )
    write_csv(
        output_dir / "tables" / "numeric_cases_preview.csv",
        preview_rows,
        [
            "case_id",
            "trace_id",
            "subcategory",
            "segment_index",
            "reference",
            "positive",
            "negative_0_type",
            "negative_0",
        ],
    )


def _dedupe_key(row: dict[str, Any], field: str) -> tuple[Any, ...]:
    values = _raw_values(row)
    return (
        row.get("case_id"),
        row.get("trace_id"),
        row.get("negative_type"),
        row.get("pair_type"),
        row.get("scorer"),
        field,
        values.get("E_ab") if field == "nli_heuristic" else row.get("score"),
        values.get("E_ba") if field == "nli_heuristic" else None,
        values.get("C_ab") if field == "nli_heuristic" else None,
        values.get("C_ba") if field == "nli_heuristic" else None,
    )


def _aggregate_rewards(scores: list[dict[str, Any]], keys: list[str], kappa: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(lambda: {"positive": [], "negative": []})
    seen: set[tuple[Any, ...]] = set()
    for row in scores:
        for field in score_fields_for_row(row):
            dedupe_key = _dedupe_key(row, field)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            value = reward_value(row, field)
            if value is None:
                continue
            value = _scale_reward(value, kappa)
            item = {key: group_value(row.get(key, "")) for key in keys}
            item["scorer"] = row.get("scorer")
            item["score_field"] = field
            group_key = tuple(item[key] for key in [*keys, "scorer", "score_field"])
            ptype = pair_type(row)
            if ptype in {"positive", "negative"}:
                grouped[group_key][ptype].append(value)

    out: list[dict[str, Any]] = []
    columns = [*keys, "scorer", "score_field"]
    for key_values in sorted(grouped):
        values = grouped[key_values]
        positives = values["positive"]
        negatives = values["negative"]
        mean_positive = _safe_mean(positives)
        mean_negative = _safe_mean(negatives)
        item = {key: value for key, value in zip(columns, key_values)}
        item.update(
            {
                "num_positive_pairs": len(positives),
                "num_negative_pairs": len(negatives),
                "mean_positive_reward": mean_positive,
                "mean_negative_reward": mean_negative,
                "reward_margin": (mean_positive - mean_negative) if mean_positive is not None and mean_negative is not None else None,
            }
        )
        out.append(item)
    return out


def aggregate(output_dir: Path, kappa: float) -> None:
    cases = read_jsonl(output_dir / "raw" / "numeric_cases.jsonl")
    scores = read_jsonl(output_dir / "raw" / "numeric_scores.jsonl")
    output_dir.joinpath("tables").mkdir(parents=True, exist_ok=True)
    write_counts_and_preview(output_dir, cases)

    metric_columns = [
        "num_positive_pairs",
        "num_negative_pairs",
        "mean_positive_reward",
        "mean_negative_reward",
        "reward_margin",
    ]
    write_csv(
        output_dir / "tables" / "table_numeric_overall.csv",
        _aggregate_rewards(scores, ["dataset"], kappa),
        ["dataset", "scorer", "score_field", *metric_columns],
    )
    write_csv(
        output_dir / "tables" / "table_numeric_by_subcategory.csv",
        _aggregate_rewards(scores, ["dataset", "subcategory"], kappa),
        ["dataset", "subcategory", "scorer", "score_field", *metric_columns],
    )
    write_csv(
        output_dir / "tables" / "table_numeric_by_perturbation.csv",
        _aggregate_rewards(scores, ["dataset", "negative_type"], kappa),
        ["dataset", "negative_type", "scorer", "score_field", *metric_columns],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--kappa", type=float, default=0.0, help="Reward scaling threshold in [0, 1).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.kappa < 1.0:
        raise ValueError("--kappa must be in [0, 1)")
    aggregate(Path(args.output_dir), args.kappa)


if __name__ == "__main__":
    main()
