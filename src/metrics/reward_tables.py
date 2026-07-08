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


NLI_HEURISTIC_FIELD = "nli_heuristic"
BERTSCORE_FIELDS = ["bertscore_f1"]
RAW_SCORE_FIELDS = {"sbert": ["raw_score"], "cross_nli": ["raw_score"], "bge_reranker": ["sigmoid_raw_score"]}
CALIBRATED_SCORER = "nli_calibrated_logistic"
CALIBRATED_FIELD = "calibrated_score"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reward-summary reliability tables from cached score JSONL files.")
    parser.add_argument("--caption_scores", default="src/outputs/reliable/caption_negation/initial_run/scores_top3.jsonl")
    parser.add_argument("--numeric_scores", default="src/outputs/reliable/visualprm_numeric/initial_run/raw/numeric_scores.jsonl")
    parser.add_argument("--calibration_predictions", default="src/outputs/reliable/calibration/run_20260706_224211/raw/nli_calibration_predictions.jsonl")
    parser.add_argument("--calibration_weights", default="src/ckpts/calibration/v1/nli_calibration_weights.json")
    parser.add_argument("--output_dir", default="src/outputs/reliable/reward_tables/initial_run")
    parser.add_argument("--kappa", type=float, default=0.0, help="Reward scaling threshold in [0, 1).")
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Reading {path.name}", unit="line"):
            line = line.strip()
            if line:
                yield json.loads(line)


def num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def group_value(value: Any) -> Any:
    return "" if value is None else value


def scale_reward(score: float, kappa: float) -> float:
    return max(0.0, score - kappa) / (1.0 - kappa)


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


def score_fields_for_scorer(scorer: str) -> list[str]:
    if scorer == "nli":
        return [NLI_HEURISTIC_FIELD]
    if scorer == "bertscore":
        return BERTSCORE_FIELDS
    if scorer == "bge_reranker":
        return ["sigmoid_raw_score"]
    return RAW_SCORE_FIELDS.get(scorer, ["raw_score"])


def raw_values(row: dict[str, Any]) -> dict[str, Any]:
    values = row.get("all_raw_values")
    return values if isinstance(values, dict) else row


def nli_heuristic(row: dict[str, Any]) -> float | None:
    values = raw_values(row)
    e_ab = num(values.get("E_ab"))
    e_ba = num(values.get("E_ba"))
    c_ab = num(values.get("C_ab"))
    c_ba = num(values.get("C_ba"))
    if e_ab is None or e_ba is None or c_ab is None or c_ba is None:
        return None
    return 0.5 * e_ab + 0.5 * e_ba - max(c_ab, c_ba)


def nli_features(row: dict[str, Any]) -> dict[str, float] | None:
    values = raw_values(row)
    e_ab = num(values.get("E_ab"))
    e_ba = num(values.get("E_ba"))
    c_ab = num(values.get("C_ab"))
    c_ba = num(values.get("C_ba"))
    n_ab = num(values.get("N_ab"))
    n_ba = num(values.get("N_ba"))
    if None in {e_ab, e_ba, c_ab, c_ba, n_ab, n_ba}:
        return None
    return {
        "E_ab": float(e_ab),
        "E_ba": float(e_ba),
        "C_max": max(float(c_ab), float(c_ba)),
        "N_max": max(float(n_ab), float(n_ba)),
    }


def load_calibration_weights(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    weights = payload.get("weights")
    if not isinstance(weights, dict):
        return None
    return payload


def calibrated_logistic_score(row: dict[str, Any], payload: dict[str, Any]) -> float | None:
    features = nli_features(row)
    if features is None:
        return None
    weights = payload.get("weights", {})
    bias = num(payload.get("bias")) or 0.0
    logit = bias
    for name, value in features.items():
        weight = num(weights.get(name))
        if weight is None:
            return None
        logit += weight * value
    if logit >= 0:
        z = math.exp(-logit)
        return 1.0 / (1.0 + z)
    z = math.exp(logit)
    return z / (1.0 + z)


def pair_type(value: Any) -> str:
    text = str(value or "")
    if text in {"positive", "positive_pair"}:
        return "positive"
    if text in {"negative", "negative_pair"}:
        return "negative"
    return text


def caption_reward_value(row: dict[str, Any], field: str) -> float | None:
    if row.get("scorer") == "nli" and field == NLI_HEURISTIC_FIELD:
        return nli_heuristic(row)
    if row.get("scorer") == "bge_reranker" and field == "sigmoid_raw_score":
        raw_score = num(row.get("raw_score"))
        return sigmoid(raw_score) if raw_score is not None else None
    return num(row.get(field))


def numeric_reward_value(row: dict[str, Any], field: str) -> float | None:
    if row.get("scorer") == "nli" and field == NLI_HEURISTIC_FIELD:
        return nli_heuristic(row)
    if row.get("scorer") == "bge_reranker" and field == "sigmoid_raw_score":
        raw_score = num(row.get("score"))
        return sigmoid(raw_score) if raw_score is not None else None
    if row.get("score_field") == field:
        return num(row.get("score"))
    return None


def numeric_fields(row: dict[str, Any]) -> list[str]:
    if row.get("scorer") == "nli":
        return [NLI_HEURISTIC_FIELD]
    if row.get("scorer") == "bge_reranker":
        return ["sigmoid_raw_score"]
    score_field = row.get("score_field")
    return [score_field] if isinstance(score_field, str) and score_field else score_fields_for_scorer(str(row.get("scorer", "")))


def append_reward(
    rows: list[dict[str, Any]],
    dataset: Any,
    subcategory: Any,
    source: str,
    scorer: str,
    score_field: str,
    ptype: str,
    reward: float | None,
    kappa: float,
) -> None:
    if reward is None or ptype not in {"positive", "negative"}:
        return
    reward = scale_reward(reward, kappa)
    rows.append(
        {
            "dataset": dataset or "",
            "subcategory": subcategory or "",
            "source": source,
            "scorer": scorer,
            "score_field": score_field,
            "pair_type": ptype,
            "reward": reward,
        }
    )


def collect_caption(path: Path, kappa: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        scorer = str(row.get("scorer", ""))
        for field in score_fields_for_scorer(scorer):
            append_reward(
                rows,
                row.get("dataset"),
                row.get("subcategory"),
                "caption_negation",
                scorer,
                field,
                pair_type(row.get("pair_type")),
                caption_reward_value(row, field),
                kappa,
            )
    return rows


def numeric_dedupe_key(row: dict[str, Any], field: str) -> tuple[Any, ...]:
    values = raw_values(row)
    use_nli_values = field in {NLI_HEURISTIC_FIELD, CALIBRATED_FIELD}
    return (
        row.get("case_id"),
        row.get("trace_id"),
        row.get("negative_type"),
        row.get("pair_type"),
        row.get("scorer"),
        field,
        values.get("E_ab") if use_nli_values else row.get("score"),
        values.get("E_ba") if use_nli_values else None,
        values.get("C_ab") if use_nli_values else None,
        values.get("C_ba") if use_nli_values else None,
    )


def collect_numeric(path: Path, kappa: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in read_jsonl(path):
        scorer = str(row.get("scorer", ""))
        for field in numeric_fields(row):
            key = numeric_dedupe_key(row, field)
            if key in seen:
                continue
            seen.add(key)
            append_reward(
                rows,
                row.get("dataset", "visualprm"),
                row.get("subcategory"),
                "visualprm_numeric",
                scorer,
                field,
                pair_type(row.get("pair_type")),
                numeric_reward_value(row, field),
                kappa,
            )
    return rows


def collect_calibrated(path: Path, kappa: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for row in read_jsonl(path):
        source_path = str(row.get("source_path") or "")
        source = "visualprm_numeric" if "visualprm_numeric" in source_path else "caption_negation"
        append_reward(
            rows,
            row.get("dataset"),
            row.get("subcategory"),
            source,
            CALIBRATED_SCORER,
            CALIBRATED_FIELD,
            pair_type(row.get("pair_type")),
            num(row.get("calibrated_score")),
            kappa,
        )
    return rows


def collect_caption_calibrated(path: Path, weights: dict[str, Any], kappa: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        if row.get("scorer") != "nli":
            continue
        append_reward(
            rows,
            row.get("dataset"),
            row.get("subcategory"),
            "caption_negation",
            CALIBRATED_SCORER,
            CALIBRATED_FIELD,
            pair_type(row.get("pair_type")),
            calibrated_logistic_score(row, weights),
            kappa,
        )
    return rows


def collect_numeric_calibrated(path: Path, weights: dict[str, Any], kappa: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in read_jsonl(path):
        if row.get("scorer") != "nli":
            continue
        key = numeric_dedupe_key(row, CALIBRATED_FIELD)
        if key in seen:
            continue
        seen.add(key)
        append_reward(
            rows,
            row.get("dataset", "visualprm"),
            row.get("subcategory"),
            "visualprm_numeric",
            CALIBRATED_SCORER,
            CALIBRATED_FIELD,
            pair_type(row.get("pair_type")),
            calibrated_logistic_score(row, weights),
            kappa,
        )
    return rows


def aggregate(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(lambda: {"positive": [], "negative": []})
    for row in rows:
        group_key = tuple(group_value(row.get(key, "")) for key in keys)
        grouped[group_key][row["pair_type"]].append(float(row["reward"]))

    out: list[dict[str, Any]] = []
    for key_values in sorted(grouped):
        values = grouped[key_values]
        positives = values["positive"]
        negatives = values["negative"]
        mean_positive = safe_mean(positives)
        mean_negative = safe_mean(negatives)
        item = {key: value for key, value in zip(keys, key_values)}
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


def main() -> None:
    args = parse_args()
    caption_scores = Path(args.caption_scores)
    numeric_scores = Path(args.numeric_scores)
    calibration_predictions = Path(args.calibration_predictions)
    calibration_weights = Path(args.calibration_weights)
    output_dir = Path(args.output_dir)
    if not 0.0 <= args.kappa < 1.0:
        raise ValueError("--kappa must be in [0, 1)")

    rows = []
    rows.extend(collect_caption(caption_scores, args.kappa))
    rows.extend(collect_numeric(numeric_scores, args.kappa))
    calibrated_rows = collect_calibrated(calibration_predictions, args.kappa)
    if calibrated_rows:
        rows.extend(calibrated_rows)
    else:
        weights = load_calibration_weights(calibration_weights)
        if weights is not None:
            rows.extend(collect_caption_calibrated(caption_scores, weights, args.kappa))
            rows.extend(collect_numeric_calibrated(numeric_scores, weights, args.kappa))

    metric_columns = [
        "num_positive_pairs",
        "num_negative_pairs",
        "mean_positive_reward",
        "mean_negative_reward",
        "reward_margin",
    ]
    by_dataset = aggregate(rows, ["dataset", "scorer", "score_field"])
    by_source_dataset = aggregate(rows, ["source", "dataset", "scorer", "score_field"])
    by_subcategory = aggregate(rows, ["source", "dataset", "subcategory", "scorer", "score_field"])

    write_csv(output_dir / "table_reward_by_dataset.csv", by_dataset, ["dataset", "scorer", "score_field", *metric_columns])
    write_csv(
        output_dir / "table_reward_by_source_dataset.csv",
        by_source_dataset,
        ["source", "dataset", "scorer", "score_field", *metric_columns],
    )
    write_csv(
        output_dir / "table_reward_by_subcategory.csv",
        by_subcategory,
        ["source", "dataset", "subcategory", "scorer", "score_field", *metric_columns],
    )
    write_markdown(
        output_dir / "table_reward_by_dataset.md",
        by_dataset,
        ["dataset", "scorer", "score_field", *metric_columns],
        f"Compatibility Reward By Dataset (kappa={args.kappa:g})",
    )
    print(json.dumps({"output_dir": str(output_dir), "num_reward_rows": len(rows)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
