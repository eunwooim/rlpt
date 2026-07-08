from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from metrics.data_loaders.common import read_jsonl, write_csv


NLI_HEURISTIC_FIELD = "nli_heuristic"
BERTSCORE_FIELDS = ["bertscore_f1"]
RAW_SCORE_FIELDS = {
    "sbert": ["raw_score"],
    "cross_nli": ["raw_score"],
    "bge_reranker": ["sigmoid_raw_score"],
}


def score_fields_for_scorer(scorer: str) -> list[str]:
    if scorer == "nli":
        return [NLI_HEURISTIC_FIELD]
    if scorer == "bertscore":
        return BERTSCORE_FIELDS
    if scorer in RAW_SCORE_FIELDS:
        return RAW_SCORE_FIELDS[scorer]
    return ["raw_score"]


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


def _nli_heuristic(row: dict[str, Any]) -> float | None:
    e_ab = _num(row.get("E_ab"))
    e_ba = _num(row.get("E_ba"))
    c_ab = _num(row.get("C_ab"))
    c_ba = _num(row.get("C_ba"))
    if e_ab is None or e_ba is None or c_ab is None or c_ba is None:
        return None
    return 0.5 * e_ab + 0.5 * e_ba - max(c_ab, c_ba)


def reward_value(row: dict[str, Any], field: str) -> float | None:
    if row.get("scorer") == "nli" and field == NLI_HEURISTIC_FIELD:
        return _nli_heuristic(row)
    if row.get("scorer") == "bge_reranker" and field == "sigmoid_raw_score":
        raw_score = _num(row.get("raw_score"))
        return _sigmoid(raw_score) if raw_score is not None else None
    return _num(row.get(field))


def pair_type(row: dict[str, Any]) -> str:
    value = row.get("pair_type")
    if value in {"positive", "positive_pair"}:
        return "positive"
    if value in {"negative", "negative_pair"}:
        return "negative"
    return str(value or "")


def group_value(value: Any) -> Any:
    return "" if value is None else value


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


def reward_rows(scores: list[dict[str, Any]], keys: list[str], kappa: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(lambda: {"positive": [], "negative": []})
    for row in scores:
        for field in score_fields_for_scorer(row["scorer"]):
            value = reward_value(row, field)
            if value is None:
                continue
            value = _scale_reward(value, kappa)
            item = {key: group_value(row.get(key, "")) for key in keys}
            item["scorer"] = row["scorer"]
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
        row = {key: value for key, value in zip(columns, key_values)}
        row.update(
            {
                "num_positive_pairs": len(positives),
                "num_negative_pairs": len(negatives),
                "mean_positive_reward": mean_positive,
                "mean_negative_reward": mean_negative,
                "reward_margin": (mean_positive - mean_negative) if mean_positive is not None and mean_negative is not None else None,
            }
        )
        out.append(row)
    return out


def reward_tables(output_dir: Path, scores: list[dict[str, Any]], kappa: float) -> None:
    metric_columns = [
        "num_positive_pairs",
        "num_negative_pairs",
        "mean_positive_reward",
        "mean_negative_reward",
        "reward_margin",
    ]
    write_csv(
        output_dir / "table_scorer_overall.csv",
        reward_rows(scores, ["dataset"], kappa),
        ["dataset", "scorer", "score_field", *metric_columns],
    )
    write_csv(
        output_dir / "table_scorer_by_subcategory.csv",
        reward_rows(scores, ["dataset", "subcategory"], kappa),
        ["dataset", "subcategory", "scorer", "score_field", *metric_columns],
    )
    write_csv(
        output_dir / "table_scorer_by_mode.csv",
        reward_rows(scores, ["dataset", "mode"], kappa),
        ["dataset", "mode", "scorer", "score_field", *metric_columns],
    )


def aggregate(output_dir: Path, kappa: float) -> None:
    cases = read_jsonl(output_dir / "cases_top3.jsonl")
    scores = read_jsonl(output_dir / "scores_top3.jsonl")
    write_counts_and_preview(output_dir, cases, scores)
    reward_tables(output_dir, scores, kappa)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="src/outputs/reliable")
    parser.add_argument("--kappa", type=float, default=0.0, help="Reward scaling threshold in [0, 1).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.kappa < 1.0:
        raise ValueError("--kappa must be in [0, 1)")
    aggregate(Path(args.output_dir), args.kappa)


if __name__ == "__main__":
    main()
