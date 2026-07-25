#!/usr/bin/env python
"""Evaluate a trained process-reward judge on VisualProcessBench."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from process_reward.common import default_output_dir, print_config, write_json, write_jsonl

try:
    from sklearn.metrics import roc_auc_score

    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


INPUT_MODES = ("pair", "context_step")
METADATA_FIELDS = (
    "category",
    "subcategory",
    "error_type",
    "transformation_category",
    "generation_type",
)


@dataclass(frozen=True)
class EvaluationExample:
    row_index: int
    anchor: str
    generated: str
    compatibility_label: int
    data_source: str
    policy_model: str
    pair_id: str | int | None
    source_sample_id: str | int | None
    step_index: int | None
    input_mode: str
    metadata: dict[str, Any]


def _nested_metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata")
    return value if isinstance(value, dict) else {}


def _group_value(row: dict[str, Any], *fields: str) -> str:
    nested = _nested_metadata(row)
    for field in fields:
        value = row.get(field, nested.get(field))
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            text = str(value).strip()
            if text:
                return text
    return "UNKNOWN"


def _data_source(row: dict[str, Any]) -> str:
    return _group_value(row, "data_source", "source_dataset", "dataset")


def _policy_model(row: dict[str, Any]) -> str:
    return _group_value(row, "policy_model")


def _compact_metadata(row: dict[str, Any]) -> dict[str, Any]:
    nested = _nested_metadata(row)
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


def _optional_identifier(
    row: dict[str, Any], field: str, source: Path, line_number: int
) -> str | int | None:
    value = row.get(field, _nested_metadata(row).get(field))
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(
            f"{source}:{line_number} has invalid '{field}' {value!r}; "
            "expected a string, integer, or null"
        )
    if isinstance(value, str) and not value.strip():
        raise ValueError(
            f"{source}:{line_number} has an empty '{field}'; "
            "expected a non-empty string, integer, or null"
        )
    return value


def _stable_id(prefix: str, payload: dict[str, Any]) -> str:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return prefix + hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def _context_anchor(question: str, preceding_steps: Sequence[str]) -> str:
    parts = [f"Question:\n{question}"]
    if preceding_steps:
        rendered = "\n".join(
            f"{index}. {step}" for index, step in enumerate(preceding_steps, start=1)
        )
        parts.append(f"Previous response steps:\n{rendered}")
    return "\n\n".join(parts)


def _pair_example(
    row: dict[str, Any], source: Path, line_number: int
) -> EvaluationExample:
    anchor = row.get("anchor")
    generated = row.get("generated")
    label = row.get("compatibility_label")
    if not isinstance(anchor, str) or not anchor.strip():
        raise ValueError(f"{source}:{line_number} has no non-empty string 'anchor'")
    if not isinstance(generated, str) or not generated.strip():
        raise ValueError(f"{source}:{line_number} has no non-empty string 'generated'")
    if isinstance(label, bool) or not isinstance(label, int) or label not in {0, 1}:
        raise ValueError(
            f"{source}:{line_number} has invalid 'compatibility_label' "
            f"{label!r}; expected 0 or 1"
        )
    step_index = row.get("step_index")
    if isinstance(step_index, bool) or (
        step_index is not None and not isinstance(step_index, int)
    ):
        raise ValueError(
            f"{source}:{line_number} has invalid 'step_index' {step_index!r}; "
            "expected an integer or null"
        )
    return EvaluationExample(
        row_index=line_number,
        anchor=anchor.strip(),
        generated=generated.strip(),
        compatibility_label=1 if label == 1 else -1,
        data_source=_data_source(row),
        policy_model=_policy_model(row),
        pair_id=_optional_identifier(row, "pair_id", source, line_number),
        source_sample_id=_optional_identifier(
            row, "source_sample_id", source, line_number
        ),
        step_index=step_index,
        input_mode="pair",
        metadata=_compact_metadata(row),
    )


def _context_step_examples(
    row: dict[str, Any],
    source: Path,
    line_number: int,
) -> list[EvaluationExample]:
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(
            f"{source}:{line_number} has no non-empty string 'question'"
        )
    question_text = question.strip()

    response = row.get("response")
    if not isinstance(response, dict):
        raise ValueError(f"{source}:{line_number} has no response object")

    raw_steps = response.get("steps")
    human_labels = response.get("process_correctness")

    if not isinstance(raw_steps, list):
        raise ValueError(
            f"{source}:{line_number} response.steps must be a list"
        )
    if not isinstance(human_labels, list):
        raise ValueError(
            f"{source}:{line_number} "
            "response.process_correctness must be a list"
        )
    if len(raw_steps) != len(human_labels):
        raise ValueError(
            f"{source}:{line_number} step/label length mismatch: "
            f"{len(raw_steps)} steps != {len(human_labels)} labels"
        )
    if not raw_steps:
        raise ValueError(
            f"{source}:{line_number} contains no response steps"
        )

    data_source = _data_source(row)
    policy_model = _policy_model(row)
    provided_source_id = _optional_identifier(
        row,
        "source_sample_id",
        source,
        line_number,
    )
    source_sample_id = provided_source_id or _stable_id(
        "vpb_",
        {
            "question": question_text,
            "image": row.get("image", row.get("images")),
            "policy_model": policy_model,
            "data_source": data_source,
        },
    )
    metadata = _compact_metadata(row)

    examples: list[EvaluationExample] = []
    preceding_steps: list[str] = []

    for original_step_index, (raw_step, human_label) in enumerate(
        zip(raw_steps, human_labels)
    ):
        if not isinstance(raw_step, str):
            raise ValueError(
                f"{source}:{line_number} has an invalid step "
                f"at index {original_step_index}"
            )

        if (
            isinstance(human_label, bool)
            or not isinstance(human_label, int)
            or human_label not in {-1, 0, 1}
        ):
            raise ValueError(
                f"{source}:{line_number} has invalid "
                f"process_correctness {human_label!r} at step index "
                f"{original_step_index}; expected -1, 0, or 1"
            )

        current_step = raw_step.strip()

        if not current_step:
            if human_label == 0:
                # Empty neutral steps contain no reasoning to score.
                continue

            raise ValueError(
                f"{source}:{line_number} has an empty non-neutral "
                f"response step at index {original_step_index}"
            )

        examples.append(
            EvaluationExample(
                row_index=line_number,
                anchor=_context_anchor(
                    question_text,
                    preceding_steps,
                ),
                generated=current_step,
                compatibility_label=human_label,
                data_source=data_source,
                policy_model=policy_model,
                pair_id=_stable_id(
                    "vpb_step_",
                    {
                        "source_sample_id": source_sample_id,
                        "step_index": original_step_index,
                        "step": current_step,
                    },
                ),
                source_sample_id=source_sample_id,
                step_index=original_step_index,
                input_mode="context_step",
                metadata=metadata,
            )
        )

        preceding_steps.append(current_step)

    return examples


def load_evaluation_examples(
    data_path: str | Path,
    *,
    input_mode: str = "pair",
    max_samples: int | None = None,
) -> list[EvaluationExample]:
    """Load pair records or flatten raw VisualProcessBench response steps."""
    source = Path(data_path).expanduser()
    if input_mode not in INPUT_MODES:
        raise ValueError(
            f"Unsupported input_mode {input_mode!r}; choose one of {INPUT_MODES}"
        )
    if not source.is_file():
        raise FileNotFoundError(f"Evaluation data does not exist or is not a file: {source}")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided")

    examples: list[EvaluationExample] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {source}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {source}:{line_number}")
            row_examples = (
                [_pair_example(row, source, line_number)]
                if input_mode == "pair"
                else _context_step_examples(row, source, line_number)
            )
            remaining = (
                None if max_samples is None else max_samples - len(examples)
            )
            examples.extend(row_examples if remaining is None else row_examples[:remaining])
            if max_samples is not None and len(examples) >= max_samples:
                break
    if not examples:
        raise ValueError(f"Evaluation data contains no records: {source}")
    return examples


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def _macro_f1_binary(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    tp_pos = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    fp_pos = sum(t != 1 and p == 1 for t, p in zip(y_true, y_pred))
    fn_pos = sum(t == 1 and p != 1 for t, p in zip(y_true, y_pred))
    f1_pos = _f1_from_counts(tp_pos, fp_pos, fn_pos)

    tp_neg = sum(t == -1 and p == -1 for t, p in zip(y_true, y_pred))
    fp_neg = sum(t != -1 and p == -1 for t, p in zip(y_true, y_pred))
    fn_neg = sum(t == -1 and p != -1 for t, p in zip(y_true, y_pred))
    f1_neg = _f1_from_counts(tp_neg, fp_neg, fn_neg)
    return {
        "f1_positive": f1_pos,
        "f1_negative": f1_neg,
        "macro_f1": (f1_pos + f1_neg) / 2.0,
        "counts": {
            "tp_pos": tp_pos,
            "fp_pos": fp_pos,
            "fn_pos": fn_pos,
            "tp_neg": tp_neg,
            "fp_neg": fp_neg,
            "fn_neg": fn_neg,
        },
    }


def _binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    fp = sum(t == -1 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == -1 for t, p in zip(y_true, y_pred))
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return {
        "accuracy": correct / len(y_true) if y_true else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": _f1_from_counts(tp, fp, fn),
    }


def _auc(y_true: list[int], scores: list[float]) -> float | None:
    if not _HAS_SKLEARN or not y_true:
        return None
    try:
        return float(roc_auc_score([int(y == 1) for y in y_true], scores))
    except Exception:
        return None



def _grouped_metrics(
    examples: Sequence[EvaluationExample],
    positive_probabilities: Sequence[float],
    predictions: Sequence[int],
    group_getter: Callable[[EvaluationExample], str],
) -> dict[str, dict[str, Any]]:
    grouped_indices: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        grouped_indices.setdefault(group_getter(example), []).append(index)

    output: dict[str, dict[str, Any]] = {}
    for group, indices in sorted(grouped_indices.items()):
        evaluated = [
            index for index in indices if examples[index].compatibility_label != 0
        ]
        y_true = [examples[index].compatibility_label for index in evaluated]
        y_pred = [predictions[index] for index in evaluated]
        scores = [positive_probabilities[index] for index in evaluated]
        macro = _macro_f1_binary(y_true, y_pred)
        output[group] = {
            **_binary_metrics(y_true, y_pred),
            "macro_f1": macro["macro_f1"],
            "f1_positive": macro["f1_positive"],
            "f1_negative": macro["f1_negative"],
            "counts": macro["counts"],
            "auc": _auc(y_true, scores),
            "total_steps": len(indices),
            "evaluated_steps": len(evaluated),
            "neutral_steps": len(indices) - len(evaluated),
        }
    return output


def compute_metrics(
    examples: Sequence[EvaluationExample],
    scores: Sequence[float],
    *,
    threshold: float,
    runtime: float,
) -> tuple[dict[str, Any], list[int]]:
    """Compute preserved F1 metrics from binary softmax probability margins."""
    if len(examples) != len(scores):
        raise ValueError(
            f"Example/score length mismatch: {len(examples)} examples, {len(scores)} scores"
        )
    if not examples:
        raise ValueError("Cannot compute metrics for an empty evaluation set")
    probabilities = [float(score) for score in scores]
    if any(
        not math.isfinite(score) or not 0.0 <= score <= 1.0
        for score in probabilities
    ):
        raise ValueError("All judge scores must be finite probabilities in [0, 1]")

    margins = [p_positive - (1.0 - p_positive) for p_positive in probabilities]
    evaluated_indices = [
        index
        for index, example in enumerate(examples)
        if example.compatibility_label != 0
    ]
    y_true = [examples[index].compatibility_label for index in evaluated_indices]
    predictions = [1 if margin > threshold else -1 for margin in margins]
    evaluated_predictions = [predictions[index] for index in evaluated_indices]

    epsilon = 1e-12
    losses: list[float] = []
    for index in evaluated_indices:
        score = min(max(probabilities[index], epsilon), 1.0 - epsilon)
        losses.append(
            -math.log(score)
            if examples[index].compatibility_label == 1
            else -math.log(1.0 - score)
        )

    grouped_data_source = _grouped_metrics(
        examples, probabilities, predictions, lambda item: item.data_source
    )
    grouped_policy_model = _grouped_metrics(
        examples, probabilities, predictions, lambda item: item.policy_model
    )
    pooled = _macro_f1_binary(y_true, evaluated_predictions)
    evaluated_count = len(evaluated_indices)
    weighted_macro = sum(
        group["macro_f1"] * group["evaluated_steps"]
        for group in grouped_data_source.values()
    )
    metrics: dict[str, Any] = {
        "evaluation_loss": sum(losses) / len(losses) if losses else 0.0,
        **_binary_metrics(y_true, evaluated_predictions),
        "f1": pooled["macro_f1"],
        "runtime": runtime,
        "samples_per_second": len(examples) / runtime if runtime > 0 else 0.0,
        "per_source": grouped_data_source,
        "grouped_metrics": {
            "policy_model": grouped_policy_model,
            "data_source": grouped_data_source,
        },
        "overall": {
            "weighted_macro_f1_over_sources": (
                weighted_macro / evaluated_count if evaluated_count else 0.0
            ),
            "macro_f1_pooled": pooled["macro_f1"],
            "f1_positive": pooled["f1_positive"],
            "f1_negative": pooled["f1_negative"],
            "total_steps": len(examples),
            "evaluated_steps": evaluated_count,
            "neutral_steps": len(examples) - evaluated_count,
            "threshold_used": threshold,
            "threshold_definition": "p_positive_minus_p_negative_strictly_greater",
            "auc": _auc(
                y_true,
                [probabilities[index] for index in evaluated_indices],
            ),
        },
    }
    return metrics, predictions


def _prediction_rows(
    examples: Sequence[EvaluationExample],
    scores: Sequence[float],
    predictions: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item, p_positive, prediction in zip(examples, scores, predictions):
        p_positive = float(p_positive)
        p_negative = 1.0 - p_positive
        neutral = item.compatibility_label == 0
        row: dict[str, Any] = {
            "row_index": item.row_index,
            "input_mode": item.input_mode,
            "data_source": item.data_source,
            "policy_model": item.policy_model,
            "process_correctness": item.compatibility_label,
            "compatibility_label": (
                None if neutral else int(item.compatibility_label == 1)
            ),
            "p_positive": p_positive,
            "p_negative": p_negative,
            "probability_margin": p_positive - p_negative,
            "predicted_label": prediction,
            "is_neutral": neutral,
            "correct": None if neutral else prediction == item.compatibility_label,
        }
        for key in ("pair_id", "source_sample_id", "step_index"):
            value = getattr(item, key)
            if value is not None:
                row[key] = value
        row.update(item.metadata)
        rows.append(row)
    return rows


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.max_length <= 0:
        raise ValueError("--max_length must be positive")
    if not -1.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [-1, 1]")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max_samples must be positive when provided")


def _output_path(args: argparse.Namespace) -> Path:
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else default_output_dir("evaluation")
    )
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {output_dir}"
        )
    if output_dir.parent.exists() and not output_dir.parent.is_dir():
        raise NotADirectoryError(
            f"Evaluation output parent is not a directory: {output_dir.parent}"
        )
    return output_dir


def _model_path(value: str | Path) -> Path:
    model_path = Path(value).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Judge checkpoint does not exist or is not a directory: {model_path}"
        )
    model_config = model_path / "model_config.json"
    if not model_config.is_file():
        raise FileNotFoundError(
            f"Judge checkpoint is missing model_config.json: {model_config}"
        )
    return model_path


def _class_indices(model_config: dict[str, Any]) -> tuple[int, int]:
    num_labels = model_config.get("num_labels")
    positive_index = model_config.get("reward_class_index")
    if isinstance(num_labels, bool) or not isinstance(num_labels, int):
        raise ValueError("Judge model_config.json must define integer num_labels")
    if num_labels != 2:
        raise ValueError(
            f"Margin evaluation requires exactly two labels, found {num_labels}"
        )
    if isinstance(positive_index, bool) or not isinstance(positive_index, int):
        raise ValueError(
            "Judge model_config.json must define integer reward_class_index"
        )
    if positive_index not in range(num_labels):
        raise ValueError(
            f"reward_class_index {positive_index} is invalid for {num_labels} labels"
        )
    negative_index = next(
        index for index in range(num_labels) if index != positive_index
    )
    return positive_index, negative_index


def _score_with_progress(
    scorer: Any,
    examples: Sequence[EvaluationExample],
    *,
    batch_size: int,
    disable: bool,
) -> list[float]:
    from tqdm.auto import tqdm

    scores: list[float] = []
    with tqdm(
        total=len(examples),
        desc="VisualProcessBench inference",
        unit="step",
        disable=disable,
    ) as progress:
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            batch_scores = scorer.score_pairs(
                [(item.anchor, item.generated) for item in batch],
                batch_size=len(batch),
            )
            if len(batch_scores) != len(batch):
                raise RuntimeError(
                    "Judge returned an unexpected number of scores: "
                    f"{len(batch_scores)} for {len(batch)} inputs"
                )
            scores.extend(float(score) for score in batch_scores)
            progress.update(len(batch))
    return scores


def evaluate(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    examples = load_evaluation_examples(
        args.data_path,
        input_mode=args.input_mode,
        max_samples=args.max_samples,
    )
    model_path = _model_path(args.model_path)
    output_dir = _output_path(args)

    from process_reward.inference import ProcessRewardScorer

    scorer = ProcessRewardScorer.from_pretrained(
        model_path, device=args.device, max_length=args.max_length
    )
    positive_class_index, negative_class_index = _class_indices(scorer.config)
    started = time.perf_counter()
    scores = _score_with_progress(
        scorer,
        examples,
        batch_size=args.batch_size,
        disable=args.no_progress,
    )
    runtime = time.perf_counter() - started
    metrics, predictions = compute_metrics(
        examples,
        scores,
        threshold=args.threshold,
        runtime=runtime,
    )

    output_dir.mkdir(parents=True)
    distribution_shift = args.input_mode == "context_step"
    config = {
        **vars(args),
        "data_path": str(Path(args.data_path).expanduser()),
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "steps_loaded": len(examples),
        "neutral_steps": sum(item.compatibility_label == 0 for item in examples),
        "positive_class_index": positive_class_index,
        "negative_class_index": negative_class_index,
        "sklearn_auc_available": _HAS_SKLEARN,
        "distribution_shift": distribution_shift,
        "distribution_shift_note": (
            "context_step uses (question + preceding response steps, current step), "
            "while the judge was trained on (original step, generated variant) pairs"
            if distribution_shift
            else None
        ),
    }
    write_json(output_dir / "run_config.json", config)
    write_json(output_dir / "metrics.json", metrics)
    write_jsonl(
        output_dir / "predictions.jsonl",
        _prediction_rows(examples, scores, predictions),
    )
    return output_dir, metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_path",
        "--data-path",
        required=True,
        help="Pair JSONL or raw VisualProcessBench JSONL, selected by --input_mode.",
    )
    parser.add_argument(
        "--input_mode",
        "--input-mode",
        choices=INPUT_MODES,
        default="pair",
        help=(
            "'pair' evaluates anchor/generated records. 'context_step' flattens raw "
            "VisualProcessBench as (question + preceding steps, current step) and is "
            "an explicit distribution-shift evaluation."
        ),
    )
    parser.add_argument(
        "--model_path",
        "--model-path",
        required=True,
        help="Trained process-reward checkpoint directory.",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        default=None,
        help="New run directory; defaults to a timestamped evaluation run.",
    )
    parser.add_argument("--batch_size", "--batch-size", type=int, default=32)
    parser.add_argument("--max_length", "--max-length", type=int, default=256)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Strict threshold on p_positive - p_negative, in [-1, 1].",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--no_progress",
        "--no-progress",
        action="store_true",
        help="Disable the tqdm inference progress bar.",
    )
    parser.add_argument(
        "--max_samples",
        "--max-samples",
        "--limit",
        type=int,
        default=None,
        help="Maximum pairs or flattened steps to evaluate.",
    )
    parser.add_argument(
        "--validate_data_only", "--validate-data-only", action="store_true"
    )
    parser.add_argument("--config_only", "--config-only", action="store_true")
    args = parser.parse_args(argv)
    _validate_args(args)
    return args


def _run(args: argparse.Namespace) -> None:
    if args.config_only:
        print_config(vars(args))
        return
    if args.validate_data_only:
        examples = load_evaluation_examples(
            args.data_path,
            input_mode=args.input_mode,
            max_samples=args.max_samples,
        )
        print_config(
            {
                "data_path": args.data_path,
                "input_mode": args.input_mode,
                "steps_validated": len(examples),
                "correct_steps": sum(
                    item.compatibility_label == 1 for item in examples
                ),
                "incorrect_steps": sum(
                    item.compatibility_label == -1 for item in examples
                ),
                "neutral_steps": sum(
                    item.compatibility_label == 0 for item in examples
                ),
                "policy_models": sorted({item.policy_model for item in examples}),
                "data_sources": sorted({item.data_source for item in examples}),
                "distribution_shift": args.input_mode == "context_step",
            }
        )
        return
    output_dir, metrics = evaluate(args)
    print_config(
        {
            key: metrics[key]
            for key in (
                "evaluation_loss",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "runtime",
                "samples_per_second",
            )
        }
    )
    print(f"Outputs saved to {output_dir}")


def main() -> None:
    try:
        _run(parse_args())
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"{Path(__file__).name}: error: {exc}") from None


if __name__ == "__main__":
    main()

"""
CUDA_VISIBLE_DEVICES=4 python src/process_reward/visualprocessbench.py \
  --input_mode context_step \
  --data_path /mnt/data1/eunwooim/VisualProcessBench/test.jsonl \
  --model_path src/outputs/process_reward_judge/v1_1_0/checkpoints/final \
  --batch_size 32 \
  --max_length 256 \
  --threshold 0.0 \
  --device cuda
"""
