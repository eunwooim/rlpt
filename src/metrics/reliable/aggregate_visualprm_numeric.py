from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from metrics.reliable.data_loaders.common import read_jsonl, write_csv


SUMMARY_FIELDS = {
    "sbert": ["raw_score"],
    "bertscore": ["bertscore_f1"],
    "nli": ["nli_score_coverage", "nli_score_equiv", "E_ab", "E_ba"],
    "cross_nli": ["raw_score"],
    "bge_reranker": ["raw_score"],
}


def _safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _summary_rows(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positives: dict[tuple[str, str, str], dict[str, Any]] = {}
    negatives: list[dict[str, Any]] = []
    for row in scores:
        score = _num(row.get("score"))
        if score is None:
            continue
        key = (str(row.get("case_id")), str(row.get("scorer")), str(row.get("score_field")))
        if row.get("pair_type") == "positive":
            positives[key] = row
        elif row.get("pair_type") == "negative":
            negatives.append(row)

    rows: list[dict[str, Any]] = []
    for negative in negatives:
        key = (str(negative.get("case_id")), str(negative.get("scorer")), str(negative.get("score_field")))
        positive = positives.get(key)
        if positive is None:
            continue
        positive_score = _num(positive.get("score"))
        negative_score = _num(negative.get("score"))
        if positive_score is None or negative_score is None:
            continue
        rows.append(
            {
                "dataset": negative.get("dataset", "visualprm"),
                "subcategory": negative.get("subcategory", "unknown"),
                "negative_type": negative.get("negative_type", "unknown"),
                "scorer": negative.get("scorer"),
                "score_field": negative.get("score_field"),
                "positive_score": positive_score,
                "negative_score": negative_score,
                "margin": positive_score - negative_score,
            }
        )
    return rows


def _aggregate(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)

    out: list[dict[str, Any]] = []
    for key_values in sorted(grouped):
        group = grouped[key_values]
        margins = [float(row["margin"]) for row in group]
        positives = [float(row["positive_score"]) for row in group]
        negatives = [float(row["negative_score"]) for row in group]
        item = {key: value for key, value in zip(keys, key_values)}
        item.update(
            {
                "num_pairs": len(group),
                "ranking_acc": sum(1 for margin in margins if margin > 0) / len(margins),
                "fpr_at_0_6": sum(1 for score in negatives if score >= 0.6) / len(negatives),
                "mean_positive_score": _safe_mean(positives),
                "mean_negative_score": _safe_mean(negatives),
                "mean_margin": _safe_mean(margins),
            }
        )
        out.append(item)
    return out


def aggregate(output_dir: Path) -> None:
    cases = read_jsonl(output_dir / "raw" / "numeric_cases.jsonl")
    scores = read_jsonl(output_dir / "raw" / "numeric_scores.jsonl")
    output_dir.joinpath("tables").mkdir(parents=True, exist_ok=True)
    write_counts_and_preview(output_dir, cases)
    summary_rows = _summary_rows(scores)

    metric_columns = [
        "num_pairs",
        "ranking_acc",
        "fpr_at_0_6",
        "mean_positive_score",
        "mean_negative_score",
        "mean_margin",
    ]
    write_csv(
        output_dir / "tables" / "table_numeric_overall.csv",
        _aggregate(summary_rows, ["dataset", "scorer", "score_field"]),
        ["dataset", "scorer", "score_field", *metric_columns],
    )
    write_csv(
        output_dir / "tables" / "table_numeric_by_subcategory.csv",
        _aggregate(summary_rows, ["dataset", "subcategory", "scorer", "score_field"]),
        ["dataset", "subcategory", "scorer", "score_field", *metric_columns],
    )
    write_csv(
        output_dir / "tables" / "table_numeric_by_perturbation.csv",
        _aggregate(summary_rows, ["dataset", "negative_type", "scorer", "score_field"]),
        ["dataset", "negative_type", "scorer", "score_field", *metric_columns],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate(Path(args.output_dir))


if __name__ == "__main__":
    main()

