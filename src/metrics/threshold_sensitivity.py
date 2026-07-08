from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001 - tqdm is optional.
    def tqdm(iterable: Iterable[Any], **_kwargs: Any) -> Iterable[Any]:  # type: ignore[no-redef]
        return iterable


TAUS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
NLI_FIELDS = ["nli_score_coverage", "nli_score_equiv", "E_ab", "E_ba", "C_ab", "C_ba"]
BERTSCORE_FIELDS = ["bertscore_precision", "bertscore_recall", "bertscore_f1"]
RAW_SCORE_FIELDS = {"sbert": ["raw_score"], "cross_nli": ["raw_score"], "bge_reranker": ["sigmoid_raw_score"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed-threshold reliability sensitivity tables from cached score JSONL files.")
    parser.add_argument("--caption_scores", default="src/outputs/reliable/caption_negation/initial_run/scores_top3.jsonl")
    parser.add_argument("--numeric_scores", default="src/outputs/reliable/visualprm_numeric/initial_run/raw/numeric_scores.jsonl")
    parser.add_argument("--calibration_predictions", default="src/outputs/reliable/calibration/run_20260706_224211/raw/nli_calibration_predictions.jsonl")
    parser.add_argument("--calibration_best", default="src/outputs/reliable/calibration/run_20260706_224211/configs/nli_calibration_best.json")
    parser.add_argument("--output_dir", default="src/outputs/reliable/threshold_sensitivity/initial_run")
    return parser.parse_args()


def score_fields_for_scorer(scorer: str) -> list[str]:
    if scorer == "nli":
        return NLI_FIELDS
    if scorer == "bertscore":
        return BERTSCORE_FIELDS
    if scorer == "bge_reranker":
        return ["sigmoid_raw_score"]
    return RAW_SCORE_FIELDS.get(scorer, ["raw_score"])


def num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def score_value(row: dict[str, Any], field: str) -> float | None:
    if row.get("scorer") == "bge_reranker" and field == "sigmoid_raw_score":
        raw_score = num(row.get("raw_score"))
        return sigmoid(raw_score) if raw_score is not None else None
    return num(row.get(field))


def numeric_score_field(row: dict[str, Any]) -> str:
    if row.get("scorer") == "bge_reranker":
        return "sigmoid_raw_score"
    return str(row.get("score_field") or "score")


def numeric_score_value(row: dict[str, Any], field: str) -> float | None:
    if row.get("scorer") == "bge_reranker" and field == "sigmoid_raw_score":
        raw_score = num(row.get("score"))
        return sigmoid(raw_score) if raw_score is not None else None
    return num(row.get("score"))


def normalize_pair_type(value: Any) -> str:
    text = str(value or "")
    if text in {"positive", "positive_pair"}:
        return "positive"
    if text in {"negative", "negative_pair"}:
        return "negative"
    return text


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Reading {path.name}", unit="line"):
            line = line.strip()
            if line:
                yield json.loads(line)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_markdown(path: Path, rows: list[dict[str, Any]], columns: list[str], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for row in rows:
            values = []
            for column in columns:
                value = row.get(column, "")
                if isinstance(value, float):
                    value = f"{value:.6g}"
                values.append(str(value))
            f.write("| " + " | ".join(values) + " |\n")


def score_key(row: dict[str, Any], field: str) -> str:
    return f"{row.get('scorer')}:{field}"


def add_score(
    buckets: dict[str, list[float]],
    ranking: dict[str, dict[tuple[str, str], dict[str, list[float] | float]]],
    family: str,
    scorer_field: str,
    mode: str,
    pair_type: str,
    case_id: str,
    value: float,
) -> None:
    if pair_type == "negative":
        buckets[f"{scorer_field}_{family}_negative"].append(value)
    if family in {"hn", "num"} and mode == "ranking":
        case_bucket = ranking[scorer_field].setdefault((family, case_id), {})
        if pair_type == "positive":
            case_bucket["positive"] = value
        elif pair_type == "negative":
            case_bucket.setdefault("negative", []).append(value)  # type: ignore[union-attr]


def collect_cached_scores(caption_scores: Path, numeric_scores: Path) -> tuple[dict[str, list[float]], dict[str, dict[tuple[str, str], dict[str, Any]]]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    ranking: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)

    for row in read_jsonl(caption_scores):
        scorer = str(row.get("scorer") or "")
        family = "hn" if row.get("mode") == "ranking" else "neg"
        pair_type = normalize_pair_type(row.get("pair_type"))
        for field in score_fields_for_scorer(scorer):
            value = score_value(row, field)
            if value is None:
                continue
            add_score(buckets, ranking, family, score_key(row, field), str(row.get("mode") or ""), pair_type, str(row.get("case_id")), value)

    seen_numeric: set[tuple[Any, ...]] = set()
    for row in read_jsonl(numeric_scores):
        scorer = str(row.get("scorer") or "")
        pair_type = normalize_pair_type(row.get("pair_type"))
        score_field = numeric_score_field(row)
        key = (row.get("case_id"), row.get("negative_type"), pair_type, scorer, score_field, row.get("score"))
        if key in seen_numeric:
            continue
        seen_numeric.add(key)
        value = numeric_score_value(row, score_field)
        if value is None:
            continue
        add_score(buckets, ranking, "num", score_key(row, score_field), str(row.get("mode") or ""), pair_type, str(row.get("case_id")), value)
    return buckets, ranking


def collect_calibrated(predictions: Path) -> tuple[dict[str, list[float]], dict[str, dict[tuple[str, str], dict[str, Any]]]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    ranking: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    scorer_field = "nli_calibrated_logistic:calibrated_score"
    for row in read_jsonl(predictions):
        source_path = str(row.get("source_path") or "")
        if "visualprm_numeric" in source_path:
            family = "num"
        elif row.get("mode") == "ranking":
            family = "hn"
        else:
            family = "neg"
        value = num(row.get("calibrated_score"))
        if value is None:
            continue
        add_score(
            buckets,
            ranking,
            family,
            scorer_field,
            str(row.get("mode") or ""),
            normalize_pair_type(row.get("pair_type")),
            str(row.get("case_id")),
            value,
        )
    return buckets, ranking


def merge_nested_lists(dst: dict[str, list[float]], src: dict[str, list[float]]) -> None:
    for key, values in src.items():
        dst[key].extend(values)


def merge_ranking(dst: dict[str, dict[tuple[str, str], dict[str, Any]]], src: dict[str, dict[tuple[str, str], dict[str, Any]]]) -> None:
    for scorer, cases in src.items():
        dst[scorer].update(cases)


def ranking_acc(cases: dict[tuple[str, str], dict[str, Any]], family: str | None = None) -> float | None:
    margins = []
    for (case_family, _case_id), values in cases.items():
        if family is not None and case_family != family:
            continue
        negatives = values.get("negative") or []
        if "positive" in values and negatives:
            margins.append(float(values["positive"]) - max(float(value) for value in negatives))
    return (sum(1 for margin in margins if margin > 0) / len(margins)) if margins else None


def fpr(values: list[float], tau: float) -> float | None:
    return (sum(1 for value in values if value >= tau) / len(values)) if values else None


def build_sensitivity_rows(
    buckets: dict[str, list[float]],
    ranking: dict[str, dict[tuple[str, str], dict[str, Any]]],
    selected_tau: float | None,
) -> list[dict[str, Any]]:
    scorer_fields = set(ranking)
    for key in buckets:
        for suffix in ["_hn_negative", "_neg_negative", "_num_negative"]:
            if key.endswith(suffix):
                scorer_fields.add(key[: -len(suffix)])
    rows: list[dict[str, Any]] = []
    for scorer_field in sorted(scorer_fields):
        scorer, field = scorer_field.split(":", 1)
        rank_acc_hn = ranking_acc(ranking.get(scorer_field, {}), "hn")
        rank_acc_num = ranking_acc(ranking.get(scorer_field, {}), "num")
        rank_acc_any = ranking_acc(ranking.get(scorer_field, {}))
        for tau in TAUS:
            rows.append(
                {
                    "scorer": scorer,
                    "score_field": field,
                    "tau": tau,
                    "hn_fpr": fpr(buckets.get(f"{scorer_field}_hn_negative", []), tau),
                    "neg_fpr": fpr(buckets.get(f"{scorer_field}_neg_negative", []), tau),
                    "num_fpr": fpr(buckets.get(f"{scorer_field}_num_negative", []), tau),
                    "ranking_acc": rank_acc_any,
                    "hn_ranking_acc": rank_acc_hn,
                    "num_ranking_acc": rank_acc_num,
                    "selected_operating_tau": selected_tau if scorer == "nli_calibrated_logistic" else "",
                }
            )
    return rows


def load_selected_tau(path: Path) -> float | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return num(json.load(f).get("tau"))


def main() -> None:
    args = parse_args()
    caption_scores = Path(args.caption_scores)
    numeric_scores = Path(args.numeric_scores)
    calibration_predictions = Path(args.calibration_predictions)
    calibration_best = Path(args.calibration_best)
    output_dir = Path(args.output_dir)

    buckets, ranking = collect_cached_scores(caption_scores, numeric_scores)
    if calibration_predictions.exists():
        cal_buckets, cal_ranking = collect_calibrated(calibration_predictions)
        merge_nested_lists(buckets, cal_buckets)
        merge_ranking(ranking, cal_ranking)
    selected_tau = load_selected_tau(calibration_best)
    rows = build_sensitivity_rows(buckets, ranking, selected_tau)

    columns = [
        "scorer",
        "score_field",
        "tau",
        "hn_fpr",
        "neg_fpr",
        "num_fpr",
        "ranking_acc",
        "hn_ranking_acc",
        "num_ranking_acc",
        "selected_operating_tau",
    ]
    write_csv(output_dir / "appendix_threshold_sensitivity.csv", rows, columns)
    write_markdown(output_dir / "appendix_threshold_sensitivity.md", rows, columns, "Tau Threshold Sensitivity")
    selected_rows = [
        {
            "scorer": "nli_calibrated_logistic",
            "selected_operating_tau": selected_tau,
            "calibration_best": str(calibration_best),
            "calibration_predictions": str(calibration_predictions),
        }
    ]
    write_csv(
        output_dir / "appendix_calibrated_operating_threshold.csv",
        selected_rows,
        ["scorer", "selected_operating_tau", "calibration_best", "calibration_predictions"],
    )
    write_markdown(
        output_dir / "appendix_calibrated_operating_threshold.md",
        selected_rows,
        ["scorer", "selected_operating_tau", "calibration_best", "calibration_predictions"],
        "Calibrated Operating Threshold",
    )
    print(json.dumps({"output_dir": str(output_dir), "num_rows": len(rows)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
