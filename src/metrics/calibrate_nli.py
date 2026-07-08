from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001 - tqdm is optional for this CPU-only utility.
    def tqdm(iterable: Iterable[Any], **_kwargs: Any) -> Iterable[Any]:  # type: ignore[no-redef]
        return iterable


FEATURES = ["E_ab", "E_ba", "C_max", "N_max"]
METHOD = "dataset_label_balanced_logistic_calibration"
ERRORS_REL = Path("errors") / "calibration_errors.jsonl"
PREDICTIONS_REL = Path("raw") / "nli_calibration_predictions.jsonl"


@dataclass
class CalibrationRow:
    row_id: str
    source_path: str
    source_line: int
    case_id: str
    dataset: str
    subcategory: str
    negative_type: str
    pair_type: str
    label: int
    mode: str
    features: list[float]
    split: str = ""
    calibrated_score: float = 0.0
    negative_index: Any = None
    trace_id: Any = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit dataset-balanced logistic calibration over cached NLI reliability scores.",
    )
    parser.add_argument("--score_jsonl", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--calib_frac", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_cases_per_dataset", type=int, default=None)
    parser.add_argument("--max_pairs_per_dataset", type=int, default=None)
    parser.add_argument("--use_all_for_calibration", action="store_true")
    parser.add_argument(
        "--tau_grid",
        default="0.50:0.95:0.05",
        help="Tau candidates as start:end:step or comma-separated values. Default: 0.50:0.95:0.05.",
    )
    parser.add_argument(
        "--tau_selection_metric",
        choices=["fpr_at_fixed_tpr", "balanced_threshold", "fixed_tau"],
        default="fpr_at_fixed_tpr",
    )
    parser.add_argument("--target_tpr", type=float, default=0.90)
    parser.add_argument("--fixed_tau", type=float, default=0.60)
    parser.add_argument(
        "--exp_name",
        default=None,
        help="Checkpoint subdirectory under src/ckpts/calibration; defaults to output_dir name.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def safe_mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return mean(values) if values else None


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(output):
        return None
    return output


def parse_tau_grid(value: str) -> list[float]:
    text = value.strip()
    if not text:
        raise ValueError("--tau_grid must not be empty")
    if ":" in text:
        parts = [to_float(part.strip()) for part in text.split(":")]
        if len(parts) != 3 or any(part is None for part in parts):
            raise ValueError("--tau_grid range must use start:end:step, for example 0.50:0.95:0.05")
        start, end, step = parts
        if step <= 0:
            raise ValueError("--tau_grid step must be positive")
        values: list[float] = []
        current = start
        epsilon = step / 1000.0
        while current <= end + epsilon:
            values.append(round(current, 10))
            current += step
    else:
        values = []
        for part in text.split(","):
            parsed = to_float(part.strip())
            if parsed is None:
                raise ValueError(f"Invalid --tau_grid value: {part}")
            values.append(round(parsed, 10))
    values = sorted(set(values))
    if not values:
        raise ValueError("--tau_grid produced no tau candidates")
    out_of_range = [tau for tau in values if tau < 0.0 or tau > 1.0]
    if out_of_range:
        raise ValueError(f"--tau_grid values must be in [0, 1], got: {out_of_range}")
    return values


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def log_line(output_dir: Path, message: str) -> None:
    path = output_dir / "logs" / "run.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{utc_now()}] {message.rstrip()}\n")


def require_fresh_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output_dir: {output_dir}")
    for name in ["raw", "tables", "logs", "errors", "configs"]:
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def row_error(source_path: Path, source_line: int, error_type: str, message: str, row: dict[str, Any] | None = None) -> dict[str, Any]:
    error = {
        "source_path": str(source_path),
        "source_line": source_line,
        "error_type": error_type,
        "error_message": message,
    }
    if row is not None:
        error.update(
            {
                "case_id": row.get("case_id"),
                "dataset": row.get("dataset"),
                "scorer": row.get("scorer"),
                "pair_type": row.get("pair_type"),
                "score_field": row.get("score_field"),
            }
        )
    return error


def normalize_pair_type(value: Any) -> tuple[str, int] | None:
    text = str(value or "").strip()
    if text in {"positive", "positive_pair"}:
        return "positive", 1
    if text in {"negative", "negative_pair"}:
        return "negative", 0
    return None


def feature_source(row: dict[str, Any]) -> dict[str, Any]:
    nested = row.get("all_raw_values")
    if isinstance(nested, dict):
        merged = dict(nested)
        merged.update({key: value for key, value in row.items() if key in {"E_ab", "E_ba", "C_ab", "C_ba", "N_ab", "N_ba"}})
        return merged
    return row


def parse_nli_row(row: dict[str, Any], source_path: Path, source_line: int) -> tuple[CalibrationRow | None, dict[str, Any] | None]:
    if row.get("scorer") != "nli":
        return None, None
    pair = normalize_pair_type(row.get("pair_type"))
    if pair is None:
        return None, row_error(source_path, source_line, "ValueError", "Unsupported pair_type", row)

    values = feature_source(row)
    required = {name: to_float(values.get(name)) for name in ["E_ab", "E_ba", "C_ab", "C_ba", "N_ab", "N_ba"]}
    missing = [name for name, value in required.items() if value is None]
    if missing:
        return None, row_error(source_path, source_line, "ValueError", f"Missing NLI fields: {','.join(missing)}", row)

    dataset = str(row.get("dataset") or "").strip()
    case_id = str(row.get("case_id") or "").strip()
    if not dataset or not case_id:
        return None, row_error(source_path, source_line, "ValueError", "Missing dataset or case_id", row)

    pair_type, label = pair
    mode = str(row.get("mode") or ("ranking" if label == 1 else "separation")).strip()
    c_max = max(required["C_ab"], required["C_ba"])  # type: ignore[arg-type]
    n_max = max(required["N_ab"], required["N_ba"])  # type: ignore[arg-type]
    negative_type = row.get("negative_type")
    return (
        CalibrationRow(
            row_id="",
            source_path=str(source_path),
            source_line=source_line,
            case_id=case_id,
            dataset=dataset,
            subcategory=str(row.get("subcategory") or "unknown"),
            negative_type="" if negative_type is None else str(negative_type),
            pair_type=pair_type,
            label=label,
            mode=mode,
            features=[required["E_ab"], required["E_ba"], c_max, n_max],  # type: ignore[list-item]
            negative_index=row.get("negative_index"),
            trace_id=row.get("trace_id"),
        ),
        None,
    )


def dedupe_key(row: dict[str, Any], source_path: Path) -> tuple[Any, ...]:
    values = feature_source(row)
    return (
        str(source_path),
        row.get("dataset"),
        row.get("case_id"),
        row.get("mode"),
        row.get("pair_type"),
        row.get("negative_type"),
        row.get("negative_index"),
        row.get("text_a"),
        row.get("text_b"),
        values.get("E_ab"),
        values.get("E_ba"),
        values.get("C_ab"),
        values.get("C_ba"),
        values.get("N_ab"),
        values.get("N_ba"),
    )


def load_rows(paths: list[str]) -> tuple[list[CalibrationRow], list[dict[str, Any]]]:
    rows: list[CalibrationRow] = []
    errors: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for path_text in paths:
        path = Path(path_text)
        if not path.exists():
            errors.append(row_error(path, 0, "FileNotFoundError", f"Missing score_jsonl: {path}"))
            continue
        with path.open("r", encoding="utf-8") as f:
            lines = tqdm(f, desc=f"Reading {path.name}", unit="line")
            for line_no, line in enumerate(lines, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(row_error(path, line_no, type(exc).__name__, str(exc)))
                    continue
                key = dedupe_key(raw, path)
                if key in seen:
                    continue
                parsed, error = parse_nli_row(raw, path, line_no)
                if error is not None:
                    errors.append(error)
                if parsed is not None:
                    seen.add(key)
                    parsed.row_id = f"{len(rows):012d}"
                    rows.append(parsed)
    return rows, errors


def apply_caps(rows: list[CalibrationRow], seed: int, max_cases: int | None, max_pairs: int | None) -> list[CalibrationRow]:
    rng = random.Random(seed)
    kept = rows
    if max_cases is not None and max_cases > 0:
        allowed_cases: set[tuple[str, str]] = set()
        cases_by_dataset: dict[str, list[str]] = defaultdict(list)
        for row in kept:
            key = (row.dataset, row.case_id)
            if row.case_id not in cases_by_dataset[row.dataset]:
                cases_by_dataset[row.dataset].append(row.case_id)
        for dataset, case_ids in cases_by_dataset.items():
            case_ids = sorted(case_ids)
            rng.shuffle(case_ids)
            for case_id in case_ids[:max_cases]:
                allowed_cases.add((dataset, case_id))
        kept = [row for row in kept if (row.dataset, row.case_id) in allowed_cases]

    if max_pairs is not None and max_pairs > 0:
        selected_ids: set[str] = set()
        rows_by_dataset: dict[str, list[CalibrationRow]] = defaultdict(list)
        for row in kept:
            rows_by_dataset[row.dataset].append(row)
        for dataset_rows in rows_by_dataset.values():
            pool = sorted(dataset_rows, key=lambda item: item.row_id)
            rng.shuffle(pool)
            selected_ids.update(row.row_id for row in pool[:max_pairs])
        kept = [row for row in kept if row.row_id in selected_ids]
    return kept


def assign_splits(rows: list[CalibrationRow], calib_frac: float, seed: int, use_all: bool) -> None:
    if use_all:
        for row in rows:
            row.split = "calib"
        return
    if not 0.0 < calib_frac < 1.0:
        raise ValueError("--calib_frac must be between 0 and 1 for held-out calibration")

    rng = random.Random(seed)
    cases_by_dataset: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row.case_id not in cases_by_dataset[row.dataset]:
            cases_by_dataset[row.dataset].append(row.case_id)

    calib_cases: set[tuple[str, str]] = set()
    for dataset, case_ids in cases_by_dataset.items():
        case_ids = sorted(case_ids)
        rng.shuffle(case_ids)
        if len(case_ids) == 1:
            n_calib = 1
        else:
            n_calib = min(len(case_ids) - 1, max(1, int(round(len(case_ids) * calib_frac))))
        for case_id in case_ids[:n_calib]:
            calib_cases.add((dataset, case_id))

    for row in rows:
        row.split = "calib" if (row.dataset, row.case_id) in calib_cases else "heldout"


def infer_case_modes(rows: list[CalibrationRow]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[CalibrationRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.dataset, row.case_id)].append(row)

    rows_changed = 0
    case_counts = Counter()
    row_counts = Counter()
    for case_rows in grouped.values():
        has_positive = any(row.pair_type == "positive" for row in case_rows)
        has_negative = any(row.pair_type == "negative" for row in case_rows)
        inferred_mode = "ranking" if has_positive and has_negative else "separation"
        case_counts[inferred_mode] += 1
        row_counts[inferred_mode] += len(case_rows)
        for row in case_rows:
            if row.mode != inferred_mode:
                rows_changed += 1
            row.mode = inferred_mode

    return {
        "rule": "ranking if a (dataset, case_id) group has at least one positive and one negative row; otherwise separation",
        "group_key": ["dataset", "case_id"],
        "num_cases": len(grouped),
        "num_ranking_cases": case_counts.get("ranking", 0),
        "num_separation_cases": case_counts.get("separation", 0),
        "num_ranking_rows": row_counts.get("ranking", 0),
        "num_separation_rows": row_counts.get("separation", 0),
        "num_rows_changed": rows_changed,
    }


def sample_weights(rows: list[CalibrationRow]) -> list[float]:
    counts = Counter((row.dataset, row.label) for row in rows)
    raw = [1.0 / counts[(row.dataset, row.label)] for row in rows]
    total = sum(raw)
    if total <= 0:
        return [1.0 for _ in rows]
    scale = len(rows) / total
    return [weight * scale for weight in raw]


def fit_with_sklearn(rows: list[CalibrationRow], weights: list[float]) -> tuple[list[float], float] | None:
    try:
        from sklearn.linear_model import LogisticRegression  # type: ignore
    except Exception:
        return None
    model = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0)
    model.fit([row.features for row in rows], [row.label for row in rows], sample_weight=weights)
    return [float(value) for value in model.coef_[0]], float(model.intercept_[0])


def fit_with_python(rows: list[CalibrationRow], weights: list[float], max_iter: int = 6000) -> tuple[list[float], float]:
    coef = [0.0, 0.0, 0.0, 0.0]
    bias = 0.0
    total_weight = sum(weights) or 1.0
    l2 = 1.0 / total_weight
    steps = tqdm(range(max_iter), desc="Fitting fallback logistic", unit="iter")
    for step in steps:
        grad = [0.0, 0.0, 0.0, 0.0]
        grad_bias = 0.0
        for row, sample_weight in zip(rows, weights):
            z = bias + sum(w * x for w, x in zip(coef, row.features))
            err = (sigmoid(z) - row.label) * sample_weight / total_weight
            for i, value in enumerate(row.features):
                grad[i] += err * value
            grad_bias += err
        for i in range(len(coef)):
            grad[i] += l2 * coef[i]
        lr = 0.8 / (1.0 + 0.001 * step)
        for i in range(len(coef)):
            coef[i] -= lr * grad[i]
        bias -= lr * grad_bias
    return coef, bias


def fit_logistic(rows: list[CalibrationRow]) -> tuple[list[float], float, str]:
    labels = {row.label for row in rows}
    if labels != {0, 1}:
        raise ValueError("Calibration split must contain both positive and negative labels")
    weights = sample_weights(rows)
    sklearn_fit = fit_with_sklearn(rows, weights)
    if sklearn_fit is not None:
        coef, bias = sklearn_fit
        return coef, bias, "sklearn"
    coef, bias = fit_with_python(rows, weights)
    return coef, bias, "python_fallback"


def score_rows(rows: list[CalibrationRow], coef: list[float], bias: float) -> None:
    for row in tqdm(rows, desc="Scoring calibrated rows", unit="row"):
        row.calibrated_score = sigmoid(bias + sum(w * x for w, x in zip(coef, row.features)))


def ranking_pairs(rows: list[CalibrationRow]) -> list[tuple[CalibrationRow, list[CalibrationRow]]]:
    positives: dict[tuple[str, str], CalibrationRow] = {}
    negatives: dict[tuple[str, str, str], list[CalibrationRow]] = defaultdict(list)
    for row in rows:
        if row.mode != "ranking":
            continue
        case_key = (row.dataset, row.case_id)
        if row.pair_type == "positive":
            positives.setdefault(case_key, row)
        elif row.pair_type == "negative":
            negative_type = row.negative_type or "__all__"
            negatives[(row.dataset, row.case_id, negative_type)].append(row)
    pairs: list[tuple[CalibrationRow, list[CalibrationRow]]] = []
    for dataset, case_id, _negative_type in sorted(negatives):
        positive = positives.get((dataset, case_id))
        if positive is not None:
            pairs.append((positive, negatives[(dataset, case_id, _negative_type)]))
    return pairs


def metrics_for_rows(rows: list[CalibrationRow], tau: float) -> dict[str, Any]:
    cases = {(row.dataset, row.case_id) for row in rows}
    positives = [row.calibrated_score for row in rows if row.label == 1]
    negatives = [row.calibrated_score for row in rows if row.label == 0]
    rank_pairs = ranking_pairs(rows)
    margins: list[float] = []
    for positive, negative_rows in rank_pairs:
        hardest_negative = max(row.calibrated_score for row in negative_rows)
        margins.append(positive.calibrated_score - hardest_negative)
    ranking_acc = (sum(1 for margin in margins if margin > 0) / len(margins)) if margins else None
    tpr_at_tau = (sum(1 for score in positives if score >= tau) / len(positives)) if positives else None
    fpr_at_tau = (sum(1 for score in negatives if score >= tau) / len(negatives)) if negatives else None
    threshold_balanced_acc = None
    if tpr_at_tau is not None and fpr_at_tau is not None:
        threshold_balanced_acc = 0.5 * tpr_at_tau + 0.5 * (1.0 - fpr_at_tau)
    if ranking_acc is not None:
        mode = "ranking"
    elif positives and negatives:
        mode = "mixed"
    elif fpr_at_tau is not None:
        mode = "separation"
    elif tpr_at_tau is not None:
        mode = "positive_only"
    else:
        mode = ""
    if threshold_balanced_acc is not None:
        component = threshold_balanced_acc
    elif fpr_at_tau is not None:
        component = 1.0 - fpr_at_tau
    else:
        component = tpr_at_tau
    return {
        "mode": mode,
        "num_cases": len(cases),
        "num_pairs": len(rows),
        "ranking_acc": ranking_acc,
        "tpr_at_tau": tpr_at_tau,
        "positive_pass_rate": tpr_at_tau,
        "fpr_at_tau": fpr_at_tau,
        "threshold_balanced_acc": threshold_balanced_acc,
        "mean_positive_score": safe_mean(positives),
        "mean_negative_score": safe_mean(negatives),
        "mean_margin": safe_mean(margins),
        "component": component,
    }


def dataset_metrics(rows: list[CalibrationRow], tau: float) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[CalibrationRow]] = defaultdict(list)
    for row in rows:
        grouped[row.dataset].append(row)
    return {dataset: metrics_for_rows(grouped[dataset], tau) for dataset in sorted(grouped)}


def threshold_summary_from_dataset_metrics(by_dataset: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    tprs = [metrics["tpr_at_tau"] for metrics in by_dataset.values() if metrics["tpr_at_tau"] is not None]
    fprs = [metrics["fpr_at_tau"] for metrics in by_dataset.values() if metrics["fpr_at_tau"] is not None]
    balanced = [
        metrics["threshold_balanced_acc"]
        for metrics in by_dataset.values()
        if metrics["threshold_balanced_acc"] is not None
    ]
    return {
        "selection_dataset_tpr": mean(tprs) if tprs else None,
        "selection_dataset_fpr": mean(fprs) if fprs else None,
        "selection_dataset_threshold_balanced_acc": mean(balanced) if balanced else None,
    }


def choose_tau(
    rows: list[CalibrationRow],
    candidates: list[float],
    tau_selection_metric: str,
    target_tpr: float,
    fixed_tau: float,
) -> tuple[float, dict[str, Any]]:
    if tau_selection_metric == "fixed_tau":
        best_tau = fixed_tau
        metrics = summarize_split(rows, best_tau)
        metrics["selection_score"] = metrics["selection_dataset_fpr"]
        return best_tau, metrics

    best_tau = candidates[0]
    best_key: tuple[float, ...] | None = None
    for tau in tqdm(candidates, desc="Sweeping tau", unit="tau"):
        metrics = summarize_split(rows, tau)
        dataset_tpr = metrics["selection_dataset_tpr"]
        dataset_fpr = metrics["selection_dataset_fpr"]
        balanced_acc = metrics["selection_dataset_threshold_balanced_acc"]
        tpr_key = dataset_tpr if dataset_tpr is not None else float("-inf")
        fpr_key = dataset_fpr if dataset_fpr is not None else float("inf")
        balanced_key = balanced_acc if balanced_acc is not None else float("-inf")

        if tau_selection_metric == "fpr_at_fixed_tpr":
            if dataset_tpr is not None and dataset_tpr >= target_tpr:
                key = (1.0, -fpr_key, tpr_key)
            else:
                key = (0.0, tpr_key, -fpr_key)
        elif tau_selection_metric == "balanced_threshold":
            key = (balanced_key, -fpr_key, tpr_key)
        else:
            raise ValueError(f"Unsupported tau_selection_metric: {tau_selection_metric}")

        if best_key is None or key > best_key:
            best_key = key
            best_tau = tau

    metrics = summarize_split(rows, best_tau)
    if tau_selection_metric == "fpr_at_fixed_tpr":
        dataset_tpr = metrics["selection_dataset_tpr"]
        dataset_fpr = metrics["selection_dataset_fpr"]
        if dataset_tpr is not None and dataset_tpr >= target_tpr:
            metrics["selection_score"] = None if dataset_fpr is None else 1.0 - dataset_fpr
        else:
            metrics["selection_score"] = dataset_tpr
    else:
        metrics["selection_score"] = metrics["selection_dataset_threshold_balanced_acc"]
    return best_tau, metrics


def summarize_split(rows: list[CalibrationRow], tau: float) -> dict[str, Any]:
    by_dataset = dataset_metrics(rows, tau)
    components = [metrics["component"] for metrics in by_dataset.values() if metrics["component"] is not None]
    overall = metrics_for_rows(rows, tau)
    overall["mean_dataset_component"] = mean(components) if components else None
    threshold_summary = threshold_summary_from_dataset_metrics(by_dataset)
    return {
        "overall": overall,
        "by_dataset": by_dataset,
        **threshold_summary,
    }


def table_row(
    split: str,
    dataset: str,
    metrics: dict[str, Any],
    coef: list[float],
    bias: float,
    tau: float,
    tau_selection_metric: str,
    target_tpr: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "split": split,
        "dataset": dataset,
        "mode": metrics.get("mode", ""),
        "num_cases": metrics.get("num_cases", ""),
        "num_pairs": metrics.get("num_pairs", ""),
        "weight_E_ab": coef[0],
        "weight_E_ba": coef[1],
        "weight_C_max": coef[2],
        "weight_N_max": coef[3],
        "bias": bias,
        "tau": tau,
        "ranking_acc": metrics.get("ranking_acc"),
        "tpr_at_tau": metrics.get("tpr_at_tau"),
        "positive_pass_rate": metrics.get("positive_pass_rate"),
        "fpr_at_tau": metrics.get("fpr_at_tau"),
        "threshold_balanced_acc": metrics.get("threshold_balanced_acc"),
        "mean_positive_score": metrics.get("mean_positive_score"),
        "mean_negative_score": metrics.get("mean_negative_score"),
        "mean_margin": metrics.get("mean_margin"),
        "component": metrics.get("component"),
        "tau_selection_metric": tau_selection_metric,
        "target_tpr": target_tpr,
    }
    if extra:
        row.update(extra)
    return row


def build_tables(
    rows: list[CalibrationRow],
    coef: list[float],
    bias: float,
    tau: float,
    tau_selection_metric: str,
    target_tpr: float,
) -> dict[str, list[dict[str, Any]]]:
    tables: dict[str, list[dict[str, Any]]] = {"overall": [], "by_dataset": [], "by_subcategory": [], "by_negative_type": []}
    for split in sorted({row.split for row in rows}):
        split_rows = [row for row in rows if row.split == split]
        summary = summarize_split(split_rows, tau)
        overall_metrics = dict(summary["overall"])
        overall_metrics["component"] = overall_metrics.get("mean_dataset_component")
        tables["overall"].append(table_row(split, "__overall__", overall_metrics, coef, bias, tau, tau_selection_metric, target_tpr))
        for dataset, metrics in summary["by_dataset"].items():
            tables["by_dataset"].append(table_row(split, dataset, metrics, coef, bias, tau, tau_selection_metric, target_tpr))

        subgroups: dict[tuple[str, str], list[CalibrationRow]] = defaultdict(list)
        for row in split_rows:
            subgroups[(row.dataset, row.subcategory)].append(row)
        for (dataset, subcategory), group_rows in sorted(subgroups.items()):
            tables["by_subcategory"].append(
                table_row(
                    split,
                    dataset,
                    metrics_for_rows(group_rows, tau),
                    coef,
                    bias,
                    tau,
                    tau_selection_metric,
                    target_tpr,
                    {"subcategory": subcategory},
                )
            )

        neg_groups: dict[tuple[str, str], list[CalibrationRow]] = defaultdict(list)
        positive_by_case: dict[tuple[str, str], list[CalibrationRow]] = defaultdict(list)
        for row in split_rows:
            if row.pair_type == "positive":
                positive_by_case[(row.dataset, row.case_id)].append(row)
            elif row.pair_type == "negative":
                neg_groups[(row.dataset, row.negative_type or "unknown")].append(row)
        for (dataset, negative_type), negatives in sorted(neg_groups.items()):
            case_keys = {(row.dataset, row.case_id) for row in negatives}
            group_rows = list(negatives)
            for case_key in case_keys:
                group_rows.extend(positive_by_case.get(case_key, []))
            tables["by_negative_type"].append(
                table_row(
                    split,
                    dataset,
                    metrics_for_rows(group_rows, tau),
                    coef,
                    bias,
                    tau,
                    tau_selection_metric,
                    target_tpr,
                    {"negative_type": negative_type},
                )
            )
    return tables


def prediction_dict(row: CalibrationRow, tau: float) -> dict[str, Any]:
    return {
        "case_id": row.case_id,
        "dataset": row.dataset,
        "subcategory": row.subcategory,
        "negative_type": row.negative_type,
        "pair_type": row.pair_type,
        "label": row.label,
        "split": row.split,
        "E_ab": row.features[0],
        "E_ba": row.features[1],
        "C_max": row.features[2],
        "N_max": row.features[3],
        "calibrated_score": row.calibrated_score,
        "tau": tau,
        "predicted_compatible": row.calibrated_score >= tau,
        "mode": row.mode,
        "source_path": row.source_path,
        "source_line": row.source_line,
        "negative_index": row.negative_index,
        "trace_id": row.trace_id,
    }


def save_tables(output_dir: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    base_columns = [
        "split",
        "dataset",
        "mode",
        "num_cases",
        "num_pairs",
        "weight_E_ab",
        "weight_E_ba",
        "weight_C_max",
        "weight_N_max",
        "bias",
        "tau",
        "ranking_acc",
        "tpr_at_tau",
        "positive_pass_rate",
        "fpr_at_tau",
        "threshold_balanced_acc",
        "mean_positive_score",
        "mean_negative_score",
        "mean_margin",
        "component",
        "tau_selection_metric",
        "target_tpr",
    ]
    write_csv(output_dir / "tables" / "table_nli_calibration_overall.csv", tables["overall"], base_columns)
    write_csv(output_dir / "tables" / "table_nli_calibration_by_dataset.csv", tables["by_dataset"], base_columns)
    write_csv(
        output_dir / "tables" / "table_nli_calibration_by_subcategory.csv",
        tables["by_subcategory"],
        ["subcategory", *base_columns],
    )
    write_csv(
        output_dir / "tables" / "table_nli_calibration_by_negative_type.csv",
        tables["by_negative_type"],
        ["negative_type", *base_columns],
    )


def checkpoint_path(repo_root: Path, exp_name: str) -> Path:
    return repo_root / "src" / "ckpts" / "calibration" / exp_name / "nli_calibration_weights.json"


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite calibration checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)


def main() -> None:
    args = parse_args()
    tau_grid = parse_tau_grid(args.tau_grid)
    if not 0.0 <= args.target_tpr <= 1.0:
        raise ValueError("--target_tpr must be in [0, 1]")
    if not 0.0 <= args.fixed_tau <= 1.0:
        raise ValueError("--fixed_tau must be in [0, 1]")
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir)
    require_fresh_output_dir(output_dir)
    write_json(output_dir / "configs" / "args.json", vars(args))
    log_line(output_dir, "Starting NLI calibration")

    rows, errors = load_rows(args.score_jsonl)
    rows = apply_caps(rows, args.seed, args.max_cases_per_dataset, args.max_pairs_per_dataset)
    if not rows:
        errors.append({"error_type": "ValueError", "error_message": "No usable nli rows found"})
        write_jsonl(output_dir / ERRORS_REL, errors)
        raise SystemExit("No usable nli rows found")
    mode_inference_metadata = infer_case_modes(rows)
    assign_splits(rows, args.calib_frac, args.seed, args.use_all_for_calibration)

    calib_rows = [row for row in rows if row.split == "calib"]
    coef, bias, backend = fit_logistic(calib_rows)
    pre_tau_weights = {
        "method": METHOD,
        "features": FEATURES,
        "weights": {name: coef[i] for i, name in enumerate(FEATURES)},
        "bias": bias,
        "backend": backend,
        "created_at": utc_now(),
    }
    write_json(output_dir / "configs" / "nli_calibration_weights_pre_tau.json", pre_tau_weights)
    score_rows(rows, coef, bias)
    tau, calib_metrics = choose_tau(
        calib_rows,
        tau_grid,
        args.tau_selection_metric,
        args.target_tpr,
        args.fixed_tau,
    )
    heldout_rows = [row for row in rows if row.split == "heldout"]
    heldout_metrics = None if args.use_all_for_calibration else summarize_split(heldout_rows, tau)

    datasets = sorted({row.dataset for row in rows})
    best = {
        "method": METHOD,
        "features": FEATURES,
        "weights": {name: coef[i] for i, name in enumerate(FEATURES)},
        "bias": bias,
        "tau": tau,
        "selection_metric": args.tau_selection_metric,
        "tau_selection_metric": args.tau_selection_metric,
        "target_tpr": args.target_tpr,
        "fixed_tau": args.fixed_tau,
        "tau_grid": tau_grid,
        "selection_dataset_tpr": calib_metrics["selection_dataset_tpr"],
        "selection_dataset_fpr": calib_metrics["selection_dataset_fpr"],
        "selection_dataset_threshold_balanced_acc": calib_metrics["selection_dataset_threshold_balanced_acc"],
        "calib_metrics": calib_metrics,
        "heldout_metrics": heldout_metrics,
        "datasets": datasets,
        "score_jsonl": args.score_jsonl,
        "use_all_for_calibration": args.use_all_for_calibration,
        "backend": backend,
    }
    write_json(output_dir / "configs" / "nli_calibration_best.json", best)

    exp_name = args.exp_name or output_dir.name
    ckpt = checkpoint_path(repo_root, exp_name)
    save_checkpoint(
        ckpt,
        {
            "method": METHOD,
            "features": FEATURES,
            "weights": best["weights"],
            "bias": bias,
            "tau": tau,
            "backend": backend,
            "source_output_dir": str(output_dir),
            "created_at": utc_now(),
        },
    )

    prediction_rows = (prediction_dict(row, tau) for row in tqdm(rows, desc="Writing predictions", unit="row"))
    write_jsonl(output_dir / PREDICTIONS_REL, prediction_rows)
    write_jsonl(output_dir / ERRORS_REL, errors)
    save_tables(output_dir, build_tables(rows, coef, bias, tau, args.tau_selection_metric, args.target_tpr))
    write_json(
        output_dir / "configs" / "run_metadata.json",
        {
            "created_at": utc_now(),
            "num_rows": len(rows),
            "num_calib_rows": len(calib_rows),
            "num_heldout_rows": len(heldout_rows),
            "num_errors": len(errors),
            "datasets": datasets,
            "backend": backend,
            "checkpoint_path": str(ckpt),
            "tau_grid": tau_grid,
            "mode_inference": mode_inference_metadata,
        },
    )
    log_line(output_dir, f"Finished NLI calibration with backend={backend}")
    print(json.dumps({"output_dir": str(output_dir), "checkpoint_path": str(ckpt), "backend": backend}, indent=2))


if __name__ == "__main__":
    main()
