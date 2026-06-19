from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from metrics.reliable.data_loaders.common import read_jsonl, write_csv


NLI_FIELDS = ["nli_score_coverage", "nli_score_equiv", "E_ab", "E_ba", "C_ab", "C_ba"]
BERTSCORE_FIELDS = ["bertscore_precision", "bertscore_recall", "bertscore_f1"]
TAUS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def score_fields_for_scorer(scorer: str) -> list[str]:
    if scorer == "nli":
        return NLI_FIELDS
    if scorer == "bertscore":
        return BERTSCORE_FIELDS
    return ["raw_score"]


def _safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_counts_and_preview(output_dir: Path, cases: list[dict[str, Any]], scores: list[dict[str, Any]]) -> None:
    grouped_cases: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped_cases[(case["dataset"], case["subcategory"], case["mode"])].append(case)

    count_rows: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []
    for key in sorted(grouped_cases):
        dataset, subcategory, mode = key
        rows = grouped_cases[key]
        num_positive_pairs = len(rows) if mode == "ranking" else 0
        num_negative_pairs = sum(len(case.get("negatives") or []) for case in rows)
        count_rows.append(
            {
                "dataset": dataset,
                "subcategory": subcategory,
                "mode": mode,
                "num_cases": len(rows),
                "num_positive_pairs": num_positive_pairs,
                "num_negative_pairs": num_negative_pairs,
            }
        )
        for case in rows[:5]:
            negatives = case.get("negatives") or []
            preview_rows.append(
                {
                    "dataset": dataset,
                    "subcategory": subcategory,
                    "mode": mode,
                    "anchor": case.get("anchor", ""),
                    "positive": case.get("positive", ""),
                    "negative_0": negatives[0] if negatives else "",
                    "metadata": json.dumps(case.get("metadata", {}), ensure_ascii=False),
                }
            )

    write_csv(
        output_dir / "subcategory_counts.csv",
        count_rows,
        ["dataset", "subcategory", "mode", "num_cases", "num_positive_pairs", "num_negative_pairs"],
    )
    write_csv(
        output_dir / "cases_preview.csv",
        preview_rows,
        ["dataset", "subcategory", "mode", "anchor", "positive", "negative_0", "metadata"],
    )


def ranking_table(output_dir: Path, scores: list[dict[str, Any]]) -> None:
    rows = [row for row in scores if row["mode"] == "ranking"]
    by_key_case: dict[tuple[str, str, str, str], dict[str, Any]] = defaultdict(dict)
    for row in rows:
        for field in score_fields_for_scorer(row["scorer"]):
            value = _num(row.get(field))
            if value is None:
                continue
            key = (row["dataset"], row["subcategory"], row["scorer"], field, row["case_id"])
            if row["pair_type"] == "positive_pair":
                by_key_case[key]["positive"] = value
            elif row["pair_type"] == "negative_pair":
                by_key_case[key].setdefault("negative", []).append(value)

    grouped: dict[tuple[str, str, str, str], list[tuple[float, float]]] = defaultdict(list)
    for key, values in by_key_case.items():
        dataset, subcategory, scorer, field, _case_id = key
        if "positive" not in values or not values.get("negative"):
            continue
        negative = max(values["negative"])
        grouped[(dataset, subcategory, scorer, field)].append((values["positive"], negative))

    out: list[dict[str, Any]] = []
    for key in sorted(grouped):
        dataset, subcategory, scorer, field = key
        pairs = grouped[key]
        positives = [p for p, _ in pairs]
        negatives = [n for _, n in pairs]
        margins = [p - n for p, n in pairs]
        out.append(
            {
                "dataset": dataset,
                "subcategory": subcategory,
                "scorer": scorer,
                "score_field": field,
                "num_cases": len(pairs),
                "ranking_acc": sum(1 for margin in margins if margin > 0) / len(margins),
                "mean_positive_score": _safe_mean(positives),
                "mean_negative_score": _safe_mean(negatives),
                "mean_margin": _safe_mean(margins),
            }
        )
    write_csv(
        output_dir / "table_ranking_by_subcategory.csv",
        out,
        [
            "dataset",
            "subcategory",
            "scorer",
            "score_field",
            "num_cases",
            "ranking_acc",
            "mean_positive_score",
            "mean_negative_score",
            "mean_margin",
        ],
    )


def separation_table(output_dir: Path, scores: list[dict[str, Any]]) -> None:
    rows = [row for row in scores if row["mode"] == "separation" and row["pair_type"] == "negative_pair"]
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        for field in score_fields_for_scorer(row["scorer"]):
            value = _num(row.get(field))
            if value is not None:
                grouped[(row["dataset"], row["subcategory"], row["scorer"], field)].append(value)

    out: list[dict[str, Any]] = []
    for key in sorted(grouped):
        dataset, subcategory, scorer, field = key
        values = grouped[key]
        for tau in TAUS:
            margins = [tau - value for value in values]
            out.append(
                {
                    "dataset": dataset,
                    "subcategory": subcategory,
                    "scorer": scorer,
                    "score_field": field,
                    "tau": tau,
                    "num_pairs": len(values),
                    "fpr_at_tau": sum(1 for value in values if value >= tau) / len(values),
                    "mean_score": _safe_mean(values),
                    "mean_margin_to_tau": _safe_mean(margins),
                }
            )
    write_csv(
        output_dir / "table_separation_by_subcategory.csv",
        out,
        [
            "dataset",
            "subcategory",
            "scorer",
            "score_field",
            "tau",
            "num_pairs",
            "fpr_at_tau",
            "mean_score",
            "mean_margin_to_tau",
        ],
    )


def overall_table(output_dir: Path, scores: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    keys = sorted({(row["dataset"], row["scorer"], field) for row in scores for field in score_fields_for_scorer(row["scorer"])})
    for dataset, scorer, field in keys:
        relevant = [row for row in scores if row["dataset"] == dataset and row["scorer"] == scorer]
        ranking_pairs: dict[str, dict[str, Any]] = defaultdict(dict)
        sep_values: list[float] = []
        pos_values: list[float] = []
        neg_values: list[float] = []
        for row in relevant:
            value = _num(row.get(field))
            if value is None:
                continue
            if row["pair_type"] == "positive_pair":
                pos_values.append(value)
            elif row["pair_type"] == "negative_pair":
                neg_values.append(value)
            if row["mode"] == "ranking":
                bucket = ranking_pairs[row["case_id"]]
                if row["pair_type"] == "positive_pair":
                    bucket["positive"] = value
                elif row["pair_type"] == "negative_pair":
                    bucket.setdefault("negative", []).append(value)
            elif row["mode"] == "separation" and row["pair_type"] == "negative_pair":
                sep_values.append(value)

        margins: list[float] = []
        for bucket in ranking_pairs.values():
            if "positive" in bucket and bucket.get("negative"):
                margins.append(bucket["positive"] - max(bucket["negative"]))

        sep_margins = [0.6 - value for value in sep_values]
        combined_margins = margins + sep_margins
        rows.append(
            {
                "dataset": dataset,
                "scorer": scorer,
                "score_field": field,
                "ranking_acc": (sum(1 for margin in margins if margin > 0) / len(margins)) if margins else "",
                "separation_fpr_at_0_6": (sum(1 for value in sep_values if value >= 0.6) / len(sep_values)) if sep_values else "",
                "mean_positive_score": _safe_mean(pos_values),
                "mean_negative_score": _safe_mean(neg_values),
                "mean_margin": _safe_mean(combined_margins),
            }
        )
    write_csv(
        output_dir / "table_scorer_overall.csv",
        rows,
        [
            "dataset",
            "scorer",
            "score_field",
            "ranking_acc",
            "separation_fpr_at_0_6",
            "mean_positive_score",
            "mean_negative_score",
            "mean_margin",
        ],
    )


def aggregate(output_dir: Path) -> None:
    cases = read_jsonl(output_dir / "cases_top3.jsonl")
    scores = read_jsonl(output_dir / "scores_top3.jsonl")
    write_counts_and_preview(output_dir, cases, scores)
    ranking_table(output_dir, scores)
    separation_table(output_dir, scores)
    overall_table(output_dir, scores)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="src/outputs/reliable")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate(Path(args.output_dir))


if __name__ == "__main__":
    main()
